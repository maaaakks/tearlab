"""Live (never-pause) PPO trainer.

The game runs continuously at full speed. A background collector thread steps
the environment and produces transitions; the main thread periodically runs
PPO updates on the freshest buffered transitions. Transitions collected while
an update is running are at most one update stale, which is acceptable for
PPO (this is the standard async-PPO tradeoff, cf. IMPALA/async A3C lineage).

Protocol requirements (mod side): FREE_RUNNING = true so Lua never blocks
waiting for Python; frames Python misses are simply skipped by the game.
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from live_checkpoint import CHECKPOINT_FORMAT_VERSION, LiveCheckpointManager
from network import IsaacFeatureExtractor

log = logging.getLogger("live_trainer")

ROLLING_EPISODE_WINDOW = 20


class IsaacPolicy(nn.Module):
    """Actor-critic policy for MultiDiscrete action spaces with Dict observations.

    One categorical head per MultiDiscrete dimension (e.g. [9, 9, 2, 2, 2] =
    move, shoot, bomb, active item, card/pill). Optional (config-gated, C4/C5):

      - a GRU over the fused feature embedding for partially-observable runs;
      - auxiliary representation heads predicting room type, local danger,
        next destination and object reachability.
    """

    ROOM_TYPE_VOCAB = 17   # one-hot length for the room-type classifier
    DEST_VOCAB = 9         # door slot 0..7 + "none"
    NEXT_STATE_DIM = 11    # next-state summary vector (see next_state_targets)

    def __init__(
        self,
        observation_space,
        action_nvec,
        features_dim: int = 256,
        hidden=(128, 64),
        recurrent: bool = False,
        recurrent_hidden_size: int = 128,
        aux_coef: float = 0.0,
        next_state_coef: float = 0.0,
    ):
        super().__init__()
        self.features = IsaacFeatureExtractor(observation_space, features_dim)
        self.action_nvec = list(int(n) for n in action_nvec)
        self.recurrent = bool(recurrent)
        self.recurrent_hidden_size = int(recurrent_hidden_size)
        self.aux_enabled = bool(aux_coef and aux_coef > 0.0)
        # S1 world-model embryo: supervised f(state, action) -> next-state
        # summary head. Never consumed by the policy itself; its loss probes
        # whether the trunk encodes action consequences (go/no-go for stage 2).
        self.next_state_enabled = bool(
            next_state_coef and next_state_coef > 0.0
        )

        if self.recurrent:
            self.gru = nn.GRU(
                features_dim,
                self.recurrent_hidden_size,
                batch_first=True,
            )
            policy_in = self.recurrent_hidden_size
        else:
            policy_in = features_dim

        self.pi = self._mlp(policy_in, hidden)
        self.vf = self._mlp(policy_in, hidden)
        self.action_heads = nn.ModuleList(
            nn.Linear(hidden[-1], n) for n in self.action_nvec
        )
        self.value_head = nn.Linear(hidden[-1], 1)

        if self.aux_enabled:
            shared_dim = hidden[-1]
            self.aux_room_type = nn.Linear(shared_dim, self.ROOM_TYPE_VOCAB)
            self.aux_danger = nn.Linear(shared_dim, 1)
            self.aux_next_dest = nn.Linear(shared_dim, self.DEST_VOCAB)
            self.aux_reachable = nn.Linear(shared_dim, 1)

        if self.next_state_enabled:
            # S1 v2: input is [pi features | one-hot taken action] — the doc
            # asks for f(state, action); without action conditioning the best
            # achievable prediction is the transition marginal mean, which is
            # exactly the chance baseline (observed: ratio plateau at 1.0).
            # Additive head: legacy checkpoints resume with fresh weights here
            # via _adapt_state_dict + load_state_dict(strict=False).
            self.aux_next_state = nn.Linear(
                hidden[-1] + sum(self.action_nvec), self.NEXT_STATE_DIM
            )

    @staticmethod
    def _mlp(in_dim, hidden):
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        return nn.Sequential(*layers)

    @staticmethod
    def _new_hidden(batch_size: int, hidden_size: int, device) -> torch.Tensor:
        return torch.zeros(1, batch_size, hidden_size, device=device)

    def _one_hot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, K) LongTensor -> flat float one-hot (B, sum(nvec))."""
        parts = []
        for k, n in enumerate(self.action_nvec):
            parts.append(torch.nn.functional.one_hot(actions[:, k], n).float())
        return torch.cat(parts, dim=-1)

    def forward(self, obs, hidden=None, actions=None):
        feat = self.features(obs)  # (B, F)
        if self.recurrent:
            if hidden is None:
                hidden = self._new_hidden(
                    feat.shape[0], self.recurrent_hidden_size, feat.device
                )
            # GRU expects (B, seq, F); we process one step at a time.
            feat_seq = feat.unsqueeze(1)
            _, hidden_out = self.gru(feat_seq, hidden)
            context = hidden_out.squeeze(0)  # (B, H)
        else:
            context = feat
            hidden_out = None

        shared_pi = self.pi(context)
        logits = [head(shared_pi) for head in self.action_heads]
        value = self.value_head(self.vf(context)).squeeze(-1)

        aux = None
        if self.aux_enabled:
            aux = {
                "room_type": self.aux_room_type(shared_pi),
                "danger": self.aux_danger(shared_pi).squeeze(-1),
                "next_dest": self.aux_next_dest(shared_pi),
                "reachable": self.aux_reachable(shared_pi).squeeze(-1),
            }
        if self.next_state_enabled and actions is not None:
            # Only computed when the taken action is supplied (training path);
            # action-free inference (collector) has no next-state to predict.
            one_hot = self._one_hot_actions(actions)
            pred = self.aux_next_state(
                torch.cat([shared_pi, one_hot], dim=-1)
            )
            aux = {**(aux or {}), "next_state": pred}
        return value, logits, hidden_out, aux

    @staticmethod
    def _dist(logits, masks=None):
        if masks is not None:
            logits = torch.where(
                masks, logits, torch.full_like(logits, float("-inf"))
            )
        return torch.distributions.Categorical(logits=logits)

    @staticmethod
    def split_masks(masks, action_nvec):
        """Flat mask (B, sum(nvec)) -> list of per-dim BoolTensors (B, n_k)."""
        parts = []
        offset = 0
        for n in action_nvec:
            parts.append(masks[:, offset:offset + n])
            offset += n
        return parts

    def evaluate_actions(self, obs, actions, masks=None, hidden=None):
        """actions: LongTensor (B, K).

        Returns values, logprobs, entropy, hidden_out, logits. Keeps the
        recurrent contract: hidden is the state ACTIVE before this observation
        (stored in each transition at collection time), hidden_out is what
        follows; logits are returned so auxiliary objectives can supervise the
        existing action heads.
        """
        value, logits, hidden_out, _aux = self.forward(obs, hidden)
        logprob = None
        entropy = None
        mb_masks = (
            self.split_masks(masks, self.action_nvec)
            if masks is not None else None
        )
        for k, head_logits in enumerate(logits):
            dist = self._dist(head_logits, mb_masks[k] if mb_masks else None)
            term = dist.log_prob(actions[:, k])
            ent = dist.entropy()
            logprob = term if logprob is None else logprob + term
            entropy = ent if entropy is None else entropy + ent
        return value, logprob, entropy, hidden_out, logits

    def aux_losses(
        self, obs, aux_targets, hidden=None, next_targets=None, actions=None
    ) -> dict[str, torch.Tensor]:
        """C5 + S1 auxiliary losses evaluated on the current parameters.

        The next-state head needs the taken action (S1 v2); without it no
        next_state loss is produced even when the head exists.
        """
        if not self.aux_enabled and not (
            self.next_state_enabled
            and next_targets is not None
            and actions is not None
        ):
            return {}
        _value, _logits, hidden_out, aux = self.forward(
            obs, hidden, actions=actions
        )
        losses = {}
        if self.aux_enabled and aux_targets is not None:
            room_type = aux_targets["room_type"]
            losses["room_type"] = torch.nn.functional.cross_entropy(
                aux["room_type"], room_type
            )
            # danger: bounded regression toward normalized enemy density.
            losses["danger"] = torch.nn.functional.mse_loss(
                aux["danger"], aux_targets["danger"]
            )
            losses["next_dest"] = torch.nn.functional.cross_entropy(
                aux["next_dest"], aux_targets["next_dest"]
            )
            losses["reachable"] = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    aux["reachable"], aux_targets["reachable"]
                )
            )
        if (
            self.next_state_enabled
            and next_targets is not None
            and actions is not None
        ):
            losses["next_state"] = torch.nn.functional.mse_loss(
                aux["next_state"], next_targets
            )
        return losses


def _obs_to_tensors(obs, device):
    tensors = {
        "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0),
        "player": torch.as_tensor(obs["player"], dtype=torch.float32, device=device).unsqueeze(0),
    }
    if "map" in obs:
        tensors["map"] = torch.as_tensor(obs["map"], dtype=torch.float32, device=device).unsqueeze(0)
    if "entities" in obs:
        tensors["entities"] = torch.as_tensor(obs["entities"], dtype=torch.float32, device=device).unsqueeze(0)
    if "goal" in obs:
        tensors["goal"] = torch.as_tensor(obs["goal"], dtype=torch.float32, device=device).unsqueeze(0)
    if "decision" in obs:
        tensors["decision"] = torch.as_tensor(obs["decision"], dtype=torch.float32, device=device).unsqueeze(0)
    return tensors


def _mask_to_tensors(mask: np.ndarray, device):
    """Flattened env mask -> BoolTensor (sum(nvec),)."""
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def aim_targets_from_player_features(players: torch.Tensor) -> torch.Tensor:
    """Map the nearest-enemy delta to idle or one of the 8 shoot directions.

    Player features 18/19 contain normalized screen-space dx/dy. A zero vector
    means no visible target and is explicitly labelled as shoot-idle, which
    teaches the same policy head to stop spraying into empty rooms.
    """
    if players.ndim != 2 or players.shape[1] < 20:
        raise ValueError("Aim supervision requires the 22-value player vector")

    deltas = players[:, 18:20]
    lengths = torch.linalg.vector_norm(deltas, dim=1)
    unit = deltas / lengths.clamp_min(1e-8).unsqueeze(1)
    diagonal = 1.0 / math.sqrt(2.0)
    directions = players.new_tensor(
        [
            [0.0, -1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-diagonal, -diagonal],
            [diagonal, -diagonal],
            [-diagonal, diagonal],
            [diagonal, diagonal],
        ]
    )
    targets = torch.argmax(unit @ directions.T, dim=1) + 1
    return torch.where(lengths > 1e-6, targets, torch.zeros_like(targets))


def next_state_targets(
    players: torch.Tensor, entities: torch.Tensor | None
) -> torch.Tensor:
    """S1 world-model probe: compact summary of an observation.

    Deterministic function of the observation tensors (no info/state needed),
    so targets stay computable offline from a replay buffer at stage 2. The
    trainer supervises the DELTA summary(next_obs) - summary(obs): absolute
    positions contain a trivial persistence shortcut, deltas force actual
    dynamics. Per doc DREAMERV4_update.md we deliberately avoid predicting
    the full t+1 grid (stochastic, partially observable). Components:

      [0:2]   player position x/y (normalized)
      [2:8]   nearest pickup / enemy / projectile dx/dy
      [8]     normalized enemy count min(1, n/8)
      [9:11]  mean enemy position over visible enemies (0 when none)

    players: (B, >=22) player feature vectors. entities: optional (B, E, D)
    entity table where column 0 is kind (1 = enemy) and 6:8 the position.
    """
    if players.ndim != 2 or players.shape[1] < 22:
        raise ValueError("Next-state targets require the 22-value player vector")
    parts = [
        players[:, 14:16],
        players[:, 16:18],
        players[:, 18:20],
        players[:, 20:22],
    ]
    if entities is not None:
        enemy_rows = entities[..., 0] == 1.0
        counts = enemy_rows.sum(dim=-1)
        parts.append((counts.float() / 8.0).clamp(max=1.0).unsqueeze(-1))
        positions = entities[..., 6:8]
        summed = (positions * enemy_rows.unsqueeze(-1)).sum(dim=1)
        parts.append(summed / counts.clamp_min(1).float().unsqueeze(-1))
    else:
        parts.append(players.new_zeros(players.shape[0], 3))
    return torch.cat(parts, dim=-1)


class Collector(threading.Thread):
    """Continuously steps the environment, requesting actions from the policy."""

    def __init__(
        self,
        env,
        policy,
        device,
        buffer,
        buffer_lock,
        buffer_cv,
        stop_event,
        total_timesteps,
        initial_steps=0,
        controlled_fraction_fn=None,
        controlled_scenarios=None,
        policy_lock=None,
    ):
        super().__init__(daemon=True, name="isaac-collector")
        self.env = env
        self.policy = policy
        self.device = device
        self.buffer = buffer
        self.buffer_lock = buffer_lock
        self.buffer_cv = buffer_cv
        self.stop_event = stop_event
        self.total_timesteps = total_timesteps
        self.controlled_fraction_fn = controlled_fraction_fn
        self.controlled_scenarios = list(controlled_scenarios or [])
        self.steps = int(initial_steps)
        self.policy_lock = policy_lock or threading.RLock()
        self.episodes = 0
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.last_ep_reward = 0.0
        self.last_ep_length = 0
        self.last_ep_reason = "unknown"
        self.recent_rewards = deque(maxlen=ROLLING_EPISODE_WINDOW)
        self.recent_lengths = deque(maxlen=ROLLING_EPISODE_WINDOW)
        self.recent_wins = deque(maxlen=ROLLING_EPISODE_WINDOW)
        # F5 degenerate-behavior counters, reset each episode.
        self.recent_idle_rate = deque(maxlen=ROLLING_EPISODE_WINDOW)
        self.recent_wall_rate = deque(maxlen=ROLLING_EPISODE_WINDOW)
        self.ep_idle_steps = 0
        self.ep_wall_steps = 0
        self.ep_bomb_uses = 0
        self.last_ep_idle_rate = 0.0
        self.last_ep_wall_rate = 0.0
        self.last_ep_bomb_uses = 0
        # C4: per-episode recurrent hidden state. Reset at episode starts.
        self.hidden = None

    def _maybe_controlled_reset(self) -> None:
        """E6/E7: occasionally restart from a rare scenario instead of natural.

        The fraction is provided by the trainer over the collector steps; when
        the env has no scenario hook it falls back to a natural reset.
        """
        fn = self.controlled_fraction_fn
        if fn is None:
            return
        fraction = float(fn(self.steps))
        if fraction <= 0.0:
            return
        if np.random.random() >= fraction:
            return
        request = getattr(self.env, "request_controlled_reset", None)
        if request is None:
            return
        scenarios = self.controlled_scenarios
        scenario = "natural"
        if scenarios:
            scenario = str(np.random.choice(scenarios))
        request(scenario)

    @staticmethod
    def _aux_targets(info) -> dict | None:
        """C5: derive auxiliary labels from the post-step game state.

        Returns None when the state is unavailable; only running policies with
        aux heads consume these.
        """
        state = info.get("state") if isinstance(info, dict) else None
        if state is None:
            return None
        run = state.run
        enemies = getattr(state, "enemies", None)
        enemy_count = int(getattr(state, "enemy_count", 0) or 0)
        danger = min(1.0, enemy_count / 8.0)
        doors = getattr(state, "doors", None) or []
        open_door_slots = [d.slot for d in doors if not d.locked]
        next_dest = open_door_slots[0] if open_door_slots else 8
        reachable = 1.0 if (open_door_slots or state.pedestal_items) else 0.0
        room_type = min(int(getattr(run, "room_type", 0) or 0), 16)
        return {
            "room_type": room_type,
            "danger": float(danger),
            "next_dest": int(next_dest),
            "reachable": float(reachable),
        }

    def run(self):
        obs = None
        while not self.stop_event.is_set():
            try:
                if obs is None:
                    obs, _ = self.env.reset()
                self.env.update_training_metrics(
                    global_steps=self.steps,
                    total_timesteps=self.total_timesteps,
                    episodes_completed=self.episodes,
                    last_episode_reward=round(self.last_ep_reward, 2),
                    last_episode_length=self.last_ep_length,
                    last_episode_reason=self.last_ep_reason,
                    rolling_window=ROLLING_EPISODE_WINDOW,
                    rolling_reward=(
                        round(sum(self.recent_rewards) / len(self.recent_rewards), 2)
                        if self.recent_rewards else 0.0
                    ),
                    rolling_episode_length=(
                        round(sum(self.recent_lengths) / len(self.recent_lengths), 1)
                        if self.recent_lengths else 0.0
                    ),
                    rolling_win_rate=(
                        round(sum(self.recent_wins) / len(self.recent_wins), 3)
                        if self.recent_wins else 0.0
                    ),
                    rolling_idle_rate=(
                        round(sum(self.recent_idle_rate) / len(self.recent_idle_rate), 3)
                        if self.recent_idle_rate else 0.0
                    ),
                    rolling_wall_rate=(
                        round(sum(self.recent_wall_rate) / len(self.recent_wall_rate), 3)
                        if self.recent_wall_rate else 0.0
                    ),
                    last_episode_idle_rate=round(self.last_ep_idle_rate, 3),
                    last_episode_wall_rate=round(self.last_ep_wall_rate, 3),
                    last_episode_bomb_uses=self.last_ep_bomb_uses,
                )
                tensors = _obs_to_tensors(obs, self.device)
                mask_flat = _mask_to_tensors(self.env.action_masks(), self.device)
                mask_parts = IsaacPolicy.split_masks(
                    mask_flat.unsqueeze(0), self.env.action_space.nvec.tolist()
                )
                hidden_in = self.hidden
                # A collector must never read weights while PPO mutates them.
                # This also makes a CUDA policy safe to share across workers.
                with self.policy_lock, torch.no_grad():
                    value, logits, hidden_out, _aux = self.policy(
                        tensors, hidden=hidden_in
                    )
                    dists = [IsaacPolicy._dist(l, m[0]) for l, m in zip(logits, mask_parts)]
                    samples = [d.sample() for d in dists]
                    logprob = sum(
                        d.log_prob(s).squeeze() for d, s in zip(dists, samples)
                    )

                action = tuple(int(s.item()) for s in samples)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                transition = {
                    "obs": {k: v.copy() for k, v in obs.items()},
                    # Boundary bootstrap value is computed once per batch in
                    # _bootstrap_boundary instead of once per step here.
                    "next_obs": {k: v.copy() for k, v in next_obs.items()},
                    "action": action,
                    "mask": np.asarray(mask_flat.cpu(), dtype=bool).copy(),
                    "logprob": float(logprob.item()),
                    "value": float(value.item()),
                    "reward": float(reward),
                    "done": bool(done),
                    # Bootstrap for the case where this transition ends the batch
                    "next_value": 0.0 if done else None,
                    # C4: per-step recurrent state entered/exited for this step.
                    "hidden_in": (
                        None if hidden_in is None else hidden_in.detach().cpu().numpy()
                    ),
                    "hidden_out": (
                        None if hidden_out is None else hidden_out.detach().cpu().numpy()
                    ),
                    # C5: auxiliary labels for the representation heads.
                    "aux_targets": self._aux_targets(info),
                }

                self.ep_reward += reward
                self.ep_steps += 1
                self.steps += 1

                # F5 degenerate-behavior tracking.
                if int(action[0]) == 0 and int(action[1]) == 0:
                    self.ep_idle_steps += 1
                rc = info.get("reward_components") or {}
                if rc.get("wall_collision", 0.0) < 0.0:
                    self.ep_wall_steps += 1
                if len(action) > 2 and int(action[2]) == 1:
                    self.ep_bomb_uses += 1

                if done:
                    self.episodes += 1
                    self.last_ep_reward = self.ep_reward
                    self.last_ep_length = self.ep_steps
                    self.last_ep_reason = info.get("state").terminal_reason if info.get("state") else "unknown"
                    self.last_ep_idle_rate = (
                        self.ep_idle_steps / max(1, self.ep_steps)
                    )
                    self.last_ep_wall_rate = (
                        self.ep_wall_steps / max(1, self.ep_steps)
                    )
                    self.last_ep_bomb_uses = self.ep_bomb_uses
                    self.recent_rewards.append(float(self.last_ep_reward))
                    self.recent_lengths.append(int(self.last_ep_length))
                    self.recent_wins.append(
                        float(self.last_ep_reason in ("room_cleared", "run_won"))
                    )
                    self.recent_idle_rate.append(self.last_ep_idle_rate)
                    self.recent_wall_rate.append(self.last_ep_wall_rate)
                    log.info(
                        "EP %d (%s) | steps=%d reward=%.1f idle=%.2f wall=%.2f bombs=%d | total collected=%d",
                        self.episodes, self.last_ep_reason, self.ep_steps,
                        self.ep_reward, self.last_ep_idle_rate,
                        self.last_ep_wall_rate, self.ep_bomb_uses, self.steps,
                    )
                    self.ep_reward = 0.0
                    self.ep_steps = 0
                    self.ep_idle_steps = 0
                    self.ep_wall_steps = 0
                    self.ep_bomb_uses = 0
                    self.hidden = None  # new episode -> zero recurrent state
                    self._maybe_controlled_reset()
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs
                    self.hidden = hidden_out

                with self.buffer_lock:
                    self.buffer.append(transition)
                    self.buffer_cv.notify_all()
            except Exception:
                log.exception("Collector crashed")
                time.sleep(1.0)
                obs = None


class LivePPOTrainer:
    def __init__(
        self,
        config: Config,
        env,
        resume: str | None = None,
        config_path: str | None = None,
    ):
        self.config = config
        self.env = env
        tc = config.train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = IsaacPolicy(
            env.observation_space,
            env.action_space.nvec.tolist(),
            recurrent=tc.recurrent,
            recurrent_hidden_size=tc.recurrent_hidden_size,
            aux_coef=tc.aux_coef,
            next_state_coef=tc.next_state_coef,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=tc.learning_rate)
        self.gamma = tc.gamma
        self.gae_lambda = tc.gae_lambda
        self.clip_range = tc.clip_range
        self.ent_coef = tc.ent_coef
        self.ent_coef_final = (
            tc.ent_coef if tc.ent_coef_final is None else tc.ent_coef_final
        )
        self.entropy_decay_fraction = max(0.0, float(tc.entropy_decay_fraction))
        self.aim_aux_coef = max(0.0, float(tc.aim_aux_coef))
        self.aim_aux_final = max(
            0.0,
            self.aim_aux_coef
            if tc.aim_aux_final is None
            else float(tc.aim_aux_final),
        )
        self.aim_aux_decay_fraction = max(
            0.0, float(tc.aim_aux_decay_fraction)
        )
        self.aim_aux_idle_weight = max(0.0, float(tc.aim_aux_idle_weight))
        if self.aim_aux_coef > 0.0 and config.env.mask_distance_vectors:
            raise ValueError(
                "Aim auxiliary loss needs nearest-enemy dx/dy; set "
                "env.mask_distance_vectors: false or train.aim_aux_coef: 0."
            )
        self.vf_coef = tc.vf_coef
        self.max_grad_norm = tc.max_grad_norm
        self.n_steps = tc.n_steps
        self.batch_size = tc.batch_size
        self.n_epochs = tc.n_epochs
        self.recurrent = bool(tc.recurrent)
        self.recurrent_hidden_size = int(tc.recurrent_hidden_size)
        self.aux_coef = float(tc.aux_coef)
        self.next_state_coef = max(0.0, float(tc.next_state_coef))
        if self.next_state_coef > 0.0 and int(config.env.player_features) < 22:
            raise ValueError(
                "Next-state head needs the 22-value player vector; set "
                "env.player_features: 22 or train.next_state_coef: 0."
            )
        self.normalize_returns = bool(tc.normalize_returns)
        self.return_norm_decay = float(tc.return_norm_decay)
        # E9 running statistics for return normalization.
        self._return_mean = 0.0
        self._return_var = 1.0
        self._return_stats_seen = 0
        # E6/E7 controlled rare-reset scheduling.
        self.controlled_reset_fraction = float(tc.controlled_reset_fraction)
        self.controlled_reset_final = float(tc.controlled_reset_final)
        self.controlled_reset_decay_fraction = max(
            0.0, float(tc.controlled_reset_decay_fraction)
        )
        self.controlled_scenarios = list(
            getattr(tc, "controlled_scenarios", None) or []
        )
        self.total_timesteps = tc.total_timesteps
        self.save_interval_seconds = max(1.0, float(tc.save_interval_seconds))

        self.checkpoints = LiveCheckpointManager(
            run_name=config.wandb.run_name or "live-training",
            config_path=config_path,
            resume=resume,
        )
        self.update_count = 0
        initial_steps = 0
        if self.checkpoints.resume_path is not None:
            initial_steps = self._load_checkpoint(self.checkpoints.resume_path)

        self.buffer: list = []
        self.buffer_lock = threading.Lock()
        self.buffer_cv = threading.Condition(self.buffer_lock)
        self.stop_event = threading.Event()
        self.policy_lock = threading.RLock()
        self.collector = Collector(
            env, self.policy, self.device,
            self.buffer, self.buffer_lock, self.buffer_cv, self.stop_event,
            self.total_timesteps,
            initial_steps,
            controlled_fraction_fn=self._current_controlled_fraction,
            controlled_scenarios=self.controlled_scenarios,
            policy_lock=self.policy_lock,
        )
        self.env.update_training_metrics(
            trainer="LIVE PPO",
            run_name=config.wandb.run_name or "training",
            device=str(self.device).upper(),
            learning_rate=tc.learning_rate,
            trainer_state="STARTING",
            update_count=0,
            last_checkpoint=initial_steps,
            checkpoint_run=self.checkpoints.run_id,
            checkpoint_minutes=round(self.save_interval_seconds / 60.0, 1),
        )

    @staticmethod
    def _observation_shapes(observation_space) -> dict[str, list[int]]:
        return {
            key: list(space.shape)
            for key, space in observation_space.spaces.items()
        }

    @staticmethod
    def _adapt_state_dict(stored: dict, current: dict) -> dict:
        """Adapt a checkpoint state dict for additive observation growth.

        The feature extractor concatenates observation streams and projects
        them with ``features.fc.0``; adding a new stream (e.g. ``entities``)
        grows that layer's input dim. ``load_state_dict(strict=False)`` cannot
        bridge a size mismatch, so copy the overlapping slice (pre-existing
        streams come first in the concat, the new stream is appended last) and
        leave the new columns at their fresh init. Any other incompatibility
        raises a descriptive error instead of silently corrupting weights.
        """
        adapted = {}
        for key, stored_tensor in stored.items():
            if key not in current:
                continue  # stale key — dropped (strict=False)
            current_tensor = current[key]
            if stored_tensor.shape == current_tensor.shape:
                adapted[key] = stored_tensor
                continue
            if (
                stored_tensor.ndim == current_tensor.ndim
                and stored_tensor.shape[:-1] == current_tensor.shape[:-1]
                and stored_tensor.shape[-1] < current_tensor.shape[-1]
            ):
                log.warning(
                    "Parameter %s shape grew %s -> %s; copying overlapping "
                    "slice, new columns keep fresh init.",
                    key,
                    tuple(stored_tensor.shape),
                    tuple(current_tensor.shape),
                )
                copy = current_tensor.clone()
                copy[..., : stored_tensor.shape[-1]] = stored_tensor
                adapted[key] = copy
                continue
            raise ValueError(
                f"Cannot adapt checkpoint parameter {key}: stored shape "
                f"{tuple(stored_tensor.shape)} is incompatible with current "
                f"{tuple(current_tensor.shape)}."
            )
        return adapted

    @staticmethod
    def _architecture_matches(stored: dict, current: dict) -> bool:
        return set(stored) == set(current) and all(
            stored[key].shape == current[key].shape for key in stored
        )

    def _load_checkpoint(self, path: Path) -> int:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        expected_shapes = self._observation_shapes(self.env.observation_space)
        stored_shapes = checkpoint.get("observation_shapes")
        if stored_shapes is not None and stored_shapes != expected_shapes:
            # Additive observation extensions keep old checkpoints loadable:
            # the new keys simply get fresh weights (TODO C6).
            additive = {}
            for key, shape in expected_shapes.items():
                if key not in stored_shapes:
                    additive[key] = shape
            if additive and all(
                key not in stored_shapes for key in additive
            ):
                log.warning(
                    "Checkpoint has no %s observation(s); loading pre-%s "
                    "network with strict=False (fresh weights for new branches).",
                    ", ".join(sorted(additive)),
                    ", ".join(sorted(additive)),
                )
            else:
                raise ValueError(
                    "Checkpoint observation shape mismatch: "
                    f"stored={stored_shapes}, current={expected_shapes}. "
                    "Use the same config/frame_stack as the saved run."
                )
        stored_action_nvec = checkpoint.get("action_nvec")
        current_action_nvec = self.env.action_space.nvec.tolist()
        if (
            stored_action_nvec is not None
            and stored_action_nvec != current_action_nvec
        ):
            raise ValueError(
                "Checkpoint action-space mismatch: "
                f"stored={stored_action_nvec}, current={current_action_nvec}."
            )
        stored_preprocessing = checkpoint.get("preprocessing")
        current_preprocessing = {
            "normalize_player_features": bool(
                self.config.env.normalize_player_features
            ),
            "frame_stack": int(self.config.env.frame_stack),
        }
        if (
            stored_preprocessing is not None
            and stored_preprocessing != current_preprocessing
        ):
            raise ValueError(
                "Checkpoint preprocessing mismatch: "
                f"stored={stored_preprocessing}, current={current_preprocessing}."
            )
        if (
            stored_preprocessing is None
            and self.config.env.normalize_player_features
        ):
            raise ValueError(
                "This legacy checkpoint predates player-feature normalization. "
                "Set env.normalize_player_features: false to load it, or start "
                "a new normalized run."
            )

        checkpoint_state = checkpoint["state_dict"]
        self.policy.load_state_dict(
            self._adapt_state_dict(checkpoint_state, self.policy.state_dict()),
            strict=False,
        )
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state and self._architecture_matches(
            checkpoint_state, self.policy.state_dict()
        ):
            self.optimizer.load_state_dict(optimizer_state)
        elif optimizer_state:
            log.warning(
                "Model architecture differs from checkpoint (e.g. a new "
                "observation branch); skipping optimizer state restore — "
                "Adam moments reset."
            )

        self.update_count = int(checkpoint.get("update_count", 0))
        steps = int(checkpoint.get("global_steps", 0))
        if steps == 0:
            stem_parts = path.stem.rsplit("_", 1)
            if len(stem_parts) == 2 and stem_parts[1].isdigit():
                steps = int(stem_parts[1])

        torch_rng_state = checkpoint.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.cpu())
        numpy_rng_state = checkpoint.get("numpy_rng_state")
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)
        cuda_rng_states = checkpoint.get("cuda_rng_state_all")
        if cuda_rng_states is not None and torch.cuda.is_available():
            # torch.load(map_location="cuda") moves these byte tensors too,
            # while set_rng_state_all specifically expects CPU ByteTensors.
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in cuda_rng_states]
            )

        log.info(
            "Resumed live checkpoint %s at step %d (update %d)",
            path,
            steps,
            self.update_count,
        )
        return steps

    def _current_controlled_fraction(self, steps: int) -> float:
        """E7: linear decay of the controlled-reset fraction over training."""
        start = self.controlled_reset_fraction
        end = self.controlled_reset_final
        if start <= 0.0 or self.controlled_reset_decay_fraction <= 0.0:
            return end
        if start == end:
            return end
        decay_steps = max(
            1.0, self.total_timesteps * self.controlled_reset_decay_fraction
        )
        progress = min(1.0, max(0.0, steps / decay_steps))
        return start + progress * (end - start)

    def _current_ent_coef(self, steps: int) -> float:
        if self.entropy_decay_fraction <= 0 or self.ent_coef == self.ent_coef_final:
            return self.ent_coef_final
        decay_steps = max(1.0, self.total_timesteps * self.entropy_decay_fraction)
        progress = min(1.0, max(0.0, steps / decay_steps))
        return self.ent_coef + progress * (self.ent_coef_final - self.ent_coef)

    def _current_aim_aux_coef(self, steps: int) -> float:
        if (
            self.aim_aux_decay_fraction <= 0
            or self.aim_aux_coef == self.aim_aux_final
        ):
            return self.aim_aux_final
        decay_steps = max(
            1.0, self.total_timesteps * self.aim_aux_decay_fraction
        )
        progress = min(1.0, max(0.0, steps / decay_steps))
        return self.aim_aux_coef + progress * (
            self.aim_aux_final - self.aim_aux_coef
        )

    def _wait_for_batch(self, n: int, deadline: float | None = None) -> list:
        with self.buffer_cv:
            while len(self.buffer) < n and not self.stop_event.is_set():
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return []
                    timeout = min(1.0, remaining)
                else:
                    timeout = 1.0
                self.buffer_cv.wait(timeout=timeout)
            take = min(n, len(self.buffer))
            batch = self.buffer[:take]
            del self.buffer[:take]
        return batch

    def _compute_gae(self, batch: list):
        n = len(batch)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            tr = batch[t]
            if tr["done"]:
                bootstrap = 0.0
                gae = 0.0
            elif t == n - 1:
                # Batch boundary: use the bootstrap value captured at collection time
                bootstrap = tr["next_value"]
            else:
                bootstrap = batch[t + 1]["value"]
            delta = tr["reward"] + self.gamma * bootstrap - tr["value"]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        values = np.array([tr["value"] for tr in batch], dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    def _maybe_normalize_returns(self, returns: np.ndarray) -> np.ndarray:
        """E9: optionally whiten PPO return targets with running statistics.

        Long full runs produce returns spanning death (-10) to win (10+) with
        many intermediate magnitudes; global whitening keeps the value head
        scale stable without changing the reward signal itself. Off by default.
        """
        if not self.normalize_returns:
            return returns
        batch_mean = float(returns.mean())
        batch_var = float(returns.var()) if returns.size > 1 else 1.0
        if self._return_stats_seen == 0:
            self._return_mean = batch_mean
            self._return_var = batch_var
        else:
            decay = self.return_norm_decay
            self._return_mean = (
                decay * self._return_mean + (1.0 - decay) * batch_mean
            )
            self._return_var = (
                decay * self._return_var + (1.0 - decay) * batch_var
            )
        self._return_stats_seen += 1
        return (returns - self._return_mean) / (np.sqrt(self._return_var) + 1e-8)

    def _bootstrap_boundary(self, batch: list):
        """Compute the batch-boundary bootstrap value lazily (one forward per
        batch instead of one per step). The collector stores next_obs; GAE only
        needs next_value when the last transition is not done."""
        if not batch:
            return
        last = batch[-1]
        if last["done"] or last.get("next_value") is not None:
            return
        if "next_obs" not in last:
            return
        with torch.no_grad():
            tensors = _obs_to_tensors(last["next_obs"], self.device)
            hidden = None
            if self.recurrent and last.get("hidden_out") is not None:
                hidden = torch.as_tensor(
                    last["hidden_out"], dtype=torch.float32, device=self.device
                )
            value, _logits, _hidden, _aux = self.policy(tensors, hidden=hidden)
        last["next_value"] = float(value.item())

    def _update(self, batch: list):
        # A full PPO update is atomic with respect to policy inference. This
        # trades a few milliseconds of collection time for valid log-probs and
        # avoids simultaneous CUDA reads/writes from worker threads.
        policy_lock = getattr(self, "policy_lock", None)
        if policy_lock is None:
            return self._update_locked(batch)
        with policy_lock:
            return self._update_locked(batch)

    def _update_locked(self, batch: list):
        tc = self.config.train
        self._bootstrap_boundary(batch)
        advantages, returns = self._compute_gae(batch)
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        returns = self._maybe_normalize_returns(returns)

        grids = torch.as_tensor(
            np.stack([tr["obs"]["grid"] for tr in batch]), dtype=torch.float32, device=self.device
        )
        players = torch.as_tensor(
            np.stack([tr["obs"]["player"] for tr in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [tr["action"] for tr in batch], dtype=torch.long, device=self.device
        )
        old_logprobs = torch.as_tensor(
            [tr["logprob"] for tr in batch], dtype=torch.float32, device=self.device
        )
        if all("mask" in tr for tr in batch):
            masks = torch.as_tensor(
                np.stack([tr["mask"] for tr in batch]), dtype=torch.bool, device=self.device
            )
        else:
            masks = None
        advs = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        rets = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        aim_targets = aim_targets_from_player_features(players)

        has_map = "map" in batch[0]["obs"]
        if has_map:
            maps = torch.as_tensor(
                np.stack([tr["obs"]["map"] for tr in batch]), dtype=torch.float32, device=self.device
            )
        has_entities = "entities" in batch[0]["obs"]
        if has_entities:
            entities = torch.as_tensor(
                np.stack([tr["obs"]["entities"] for tr in batch]), dtype=torch.float32, device=self.device
            )
        has_goal = "goal" in batch[0]["obs"]
        if has_goal:
            goals = torch.as_tensor(
                np.stack([tr["obs"]["goal"] for tr in batch]), dtype=torch.float32, device=self.device
            )
        has_decision = "decision" in batch[0]["obs"]
        if has_decision:
            decisions = torch.as_tensor(
                np.stack([tr["obs"]["decision"] for tr in batch]), dtype=torch.float32, device=self.device
            )

        n = len(batch)
        # C4: per-transition recurrent state entered at each step. When
        # recurrent, minibatches must respect temporal order (no shuffle).
        hiddens = None
        if self.recurrent and all(tr.get("hidden_in") is not None for tr in batch):
            # Each stored hidden is (1, 1, H); a full batch is (1, n, H).
            stacked = np.stack([tr["hidden_in"] for tr in batch])
            stacked = stacked.reshape(-1, self.recurrent_hidden_size)
            hiddens = torch.as_tensor(
                stacked, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        aux_enabled = self.aux_coef > 0.0 and self.policy.aux_enabled
        has_aux_targets = (
            aux_enabled and all(tr.get("aux_targets") is not None for tr in batch)
        )
        aux_loss_factor = self.aux_coef if has_aux_targets else 0.0

        # S1 next-state probe: targets come from the post-step observation the
        # collector already stores, so no env/protocol changes are needed.
        # v2 supervises the DELTA summary(next_obs) - summary(obs) and
        # conditions the head on the taken action (see IsaacPolicy).
        ns_enabled = (
            getattr(self, "next_state_coef", 0.0) > 0.0
            and self.policy.next_state_enabled
        )
        ns_targets_full = None
        ns_baseline_var = 0.0
        if ns_enabled:
            try:
                ns_players = torch.as_tensor(
                    np.stack(
                        [tr["next_obs"]["player"] for tr in batch]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                ns_entities = (
                    torch.as_tensor(
                        np.stack(
                            [tr["next_obs"]["entities"] for tr in batch]
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if "entities" in batch[0]["next_obs"]
                    else None
                )
                cur_summary = next_state_targets(
                    players, entities if has_entities else None
                )
                next_summary = next_state_targets(ns_players, ns_entities)
                ns_targets_full = next_summary - cur_summary
                # Chance baseline: MSE of predicting the batch mean equals the
                # target variance. S1 go criterion: loss < 50% of this after
                # ~100k steps (DREAMERV4_update.md).
                ns_baseline_var = float(
                    ns_targets_full.var(dim=0).mean().item()
                )
            except ValueError as exc:
                log.warning("Next-state head disabled: %s", exc)
                ns_enabled = False

        idx = np.arange(n)
        stats = {
            "pi_loss": 0.0,
            "v_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "updates": 0,
            "aux_loss": 0.0,
            "aim_aux_loss": 0.0,
            "next_state_loss": 0.0,
        }
        current_ent_coef = self._current_ent_coef(self.collector.steps)
        current_aim_aux_coef = self._current_aim_aux_coef(
            self.collector.steps
        )
        early_stop = False
        for _ in range(self.n_epochs):
            if not self.recurrent:
                np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                if len(mb) < 2:
                    continue
                mb_obs = {"grid": grids[mb], "player": players[mb]}
                if has_map:
                    mb_obs["map"] = maps[mb]
                if has_entities:
                    mb_obs["entities"] = entities[mb]
                if has_goal:
                    mb_obs["goal"] = goals[mb]
                if has_decision:
                    mb_obs["decision"] = decisions[mb]
                mb_hidden = hiddens[:, mb] if hiddens is not None else None
                (
                    values,
                    logprobs,
                    entropy,
                    _hidden_out,
                    action_logits,
                ) = self.policy.evaluate_actions(
                    mb_obs,
                    actions[mb],
                    masks[mb] if masks is not None else None,
                    hidden=mb_hidden,
                )

                ratio = torch.exp(logprobs - old_logprobs[mb])
                mb_adv = advs[mb]
                pi_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv,
                ).mean()
                v_loss = ((values - rets[mb]) ** 2).mean()
                ent = entropy.mean()

                loss = pi_loss + self.vf_coef * v_loss - current_ent_coef * ent

                # Supervise the existing shoot head inside the same PPO update.
                # Visible enemies receive an 8-way direction label; empty-room
                # samples receive a lower-weight idle label so they curb spam
                # without overwhelming combat examples in full runs.
                mb_aim_targets = aim_targets[mb]
                shoot_logits = action_logits[1]
                if masks is not None:
                    shoot_mask = IsaacPolicy.split_masks(
                        masks[mb], self.policy.action_nvec
                    )[1]
                    target_allowed = shoot_mask.gather(
                        1, mb_aim_targets.unsqueeze(1)
                    ).squeeze(1)
                    mb_aim_targets = torch.where(
                        target_allowed,
                        mb_aim_targets,
                        torch.zeros_like(mb_aim_targets),
                    )
                    shoot_logits = torch.where(
                        shoot_mask,
                        shoot_logits,
                        torch.full_like(shoot_logits, float("-inf")),
                    )
                aim_per_sample = torch.nn.functional.cross_entropy(
                    shoot_logits, mb_aim_targets, reduction="none"
                )
                aim_weights = torch.where(
                    mb_aim_targets == 0,
                    torch.full_like(
                        aim_per_sample, self.aim_aux_idle_weight
                    ),
                    torch.ones_like(aim_per_sample),
                )
                aim_aux_loss = (
                    (aim_per_sample * aim_weights).sum()
                    / aim_weights.sum().clamp_min(1e-8)
                )
                loss = loss + current_aim_aux_coef * aim_aux_loss

                aux_loss_total = 0.0
                ns_loss = None
                if has_aux_targets or ns_enabled:
                    aux_targets = None
                    if has_aux_targets:
                        aux_targets = {
                            key: torch.as_tensor(
                                [tr["aux_targets"][key] for tr in
                                 [batch[i] for i in mb]],
                                dtype=(
                                    torch.long if key in ("room_type", "next_dest")
                                    else torch.float32
                                ),
                                device=self.device,
                            )
                            for key in ("room_type", "danger", "next_dest", "reachable")
                        }
                    aux_losses = self.policy.aux_losses(
                        mb_obs,
                        aux_targets,
                        hidden=mb_hidden,
                        next_targets=(
                            ns_targets_full[mb] if ns_enabled else None
                        ),
                        actions=actions[mb],
                    )
                    ns_loss = aux_losses.pop("next_state", None)
                    for aux_loss in aux_losses.values():
                        aux_loss_total = aux_loss_total + aux_loss
                    loss = loss + aux_loss_factor * aux_loss_total
                    if ns_loss is not None:
                        loss = loss + self.next_state_coef * ns_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = logprobs - old_logprobs[mb]
                    approx_kl = (
                        (torch.exp(log_ratio) - 1.0) - log_ratio
                    ).mean().item()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_range)
                        .float()
                        .mean()
                        .item()
                    )
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl"] += approx_kl
                stats["clip_fraction"] += clip_fraction
                stats["grad_norm"] += float(grad_norm)
                stats["aux_loss"] += (
                    float(aux_loss_total.item()) if has_aux_targets else 0.0
                )
                stats["aim_aux_loss"] += float(aim_aux_loss.item())
                if ns_loss is not None:
                    stats["next_state_loss"] += float(ns_loss.item())
                stats["updates"] += 1

                if tc.target_kl is not None and approx_kl > 1.5 * tc.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        if stats["updates"]:
            u = stats["updates"]
            sample_count = min(n, 512)
            sample_idx = torch.arange(sample_count, device=self.device)
            sample_obs = {
                "grid": grids[sample_idx],
                "player": players[sample_idx],
            }
            if has_map:
                sample_obs["map"] = maps[sample_idx]
            if has_entities:
                sample_obs["entities"] = entities[sample_idx]
            if has_goal:
                sample_obs["goal"] = goals[sample_idx]
            if has_decision:
                sample_obs["decision"] = decisions[sample_idx]
            with torch.no_grad():
                sample_values, sample_logits, _hidden, _aux = self.policy(
                    sample_obs
                )
                sample_features = self.policy.features(sample_obs)

            sample_aim_targets = aim_targets[sample_idx]
            sample_aim_predictions = sample_logits[1].argmax(dim=1)
            visible_targets = sample_aim_targets != 0
            idle_targets = ~visible_targets
            aim_target_accuracy = (
                (sample_aim_predictions == sample_aim_targets)
                .float()
                .mean()
                .item()
            )
            aim_visible_accuracy = (
                (sample_aim_predictions[visible_targets]
                 == sample_aim_targets[visible_targets])
                .float()
                .mean()
                .item()
                if visible_targets.any()
                else 0.0
            )
            aim_idle_accuracy = (
                (sample_aim_predictions[idle_targets] == 0)
                .float()
                .mean()
                .item()
                if idle_targets.any()
                else 0.0
            )

            sample_returns = rets[sample_idx]
            return_variance = torch.var(sample_returns, unbiased=False)
            if return_variance.item() > 1e-8:
                explained_variance = 1.0 - (
                    torch.var(
                        sample_returns - sample_values,
                        unbiased=False,
                    )
                    / return_variance
                ).item()
            else:
                explained_variance = 0.0

            feature_std = torch.std(sample_features, dim=0, unbiased=False)
            dead_feature_fraction = (
                (feature_std < 1e-3).float().mean().item()
            )

            action_array = np.asarray([tr["action"] for tr in batch], dtype=np.int64)

            def normalized_action_entropy(values: np.ndarray) -> float:
                counts = np.bincount(values, minlength=9).astype(np.float64)
                probabilities = counts[counts > 0] / counts.sum()
                entropy = -np.sum(probabilities * np.log(probabilities))
                return float(entropy / math.log(9))

            # S1 v2 diagnostics: per-component delta losses on the sample
            # slice (player pos / proximity deltas / enemy aggregates) so a
            # plateau can be localized to specific target components.
            ns_component_losses = {}
            if ns_enabled and ns_targets_full is not None:
                # sample_idx lives on the training device; the numpy action
                # array must be indexed with host indices.
                host_idx = sample_idx.cpu().numpy()
                with torch.no_grad():
                    sample_actions = torch.as_tensor(
                        action_array[host_idx],
                        dtype=torch.long,
                        device=self.device,
                    )
                    _sv, _sl, _sh, saux = self.policy(
                        sample_obs, actions=sample_actions
                    )
                spred = saux.get("next_state") if saux else None
                if spred is not None:
                    err = (spred - ns_targets_full[host_idx]) ** 2
                    ns_component_losses = {
                        "next_state_loss_player": err[:, 0:2].mean().item(),
                        "next_state_loss_proximity": err[:, 2:8].mean().item(),
                        "next_state_loss_enemies": err[:, 8:11].mean().item(),
                    }

            return {
                "pi_loss": stats["pi_loss"] / u,
                "v_loss": stats["v_loss"] / u,
                "entropy": stats["entropy"] / u,
                "approx_kl": stats["kl"] / u,
                "clip_fraction": stats["clip_fraction"] / u,
                "grad_norm": stats["grad_norm"] / u,
                "aux_loss": stats["aux_loss"] / u,
                "mean_return": float(rets.mean().item()),
                "return_std": float(rets.std().item()),
                "effective_horizon": 1.0 / max(1e-6, 1.0 - self.gamma),
                "explained_variance": explained_variance,
                "feature_abs_mean": sample_features.abs().mean().item(),
                "dead_feature_fraction": dead_feature_fraction,
                "move_action_entropy": normalized_action_entropy(action_array[:, 0]),
                "shoot_action_entropy": normalized_action_entropy(action_array[:, 1]),
                "move_idle_rate": float(np.mean(action_array[:, 0] == 0)),
                "shoot_idle_rate": float(np.mean(action_array[:, 1] == 0)),
                "bomb_use_rate": (
                    float(np.mean(action_array[:, 2] == 1))
                    if action_array.shape[1] > 2 else 0.0
                ),
                "active_use_rate": (
                    float(np.mean(action_array[:, 3] == 1))
                    if action_array.shape[1] > 3 else 0.0
                ),
                "pillcard_use_rate": (
                    float(np.mean(action_array[:, 4] == 1))
                    if action_array.shape[1] > 4 else 0.0
                ),
                "ent_coef": current_ent_coef,
                "early_stop": float(early_stop),
                "aim_aux_loss": stats["aim_aux_loss"] / u,
                "aim_aux_coef": current_aim_aux_coef,
                "aim_target_accuracy": aim_target_accuracy,
                "aim_visible_accuracy": aim_visible_accuracy,
                "aim_idle_accuracy": aim_idle_accuracy,
                "aim_visible_fraction": visible_targets.float().mean().item(),
                "aux_loss": stats["aux_loss"] / u,
                "next_state_loss": stats["next_state_loss"] / u,
                "next_state_baseline_var": ns_baseline_var,
                "next_state_ratio": (
                    (stats["next_state_loss"] / u) / ns_baseline_var
                    if ns_enabled and ns_baseline_var > 1e-12
                    else 0.0
                ),
                **ns_component_losses,
            }
        return None

    def learn(self):
        tc = self.config.train
        log.info(
            "Live training started: device=%s n_steps=%d lr=%g total=%d "
            "checkpoint_every=%.1fmin run_dir=%s",
            self.device,
            self.n_steps,
            tc.learning_rate,
            self.total_timesteps,
            self.save_interval_seconds / 60.0,
            self.checkpoints.run_dir,
        )
        self.collector.start()

        initial_steps = self.collector.steps
        last_update_step = initial_steps
        update_times = []
        started_at = time.monotonic()
        next_checkpoint_at = started_at + self.save_interval_seconds
        while self.collector.steps < self.total_timesteps and not self.stop_event.is_set():
            self.env.update_training_metrics(trainer_state="COLLECTING")
            batch = self._wait_for_batch(self.n_steps, deadline=next_checkpoint_at)
            if not batch:
                if time.monotonic() >= next_checkpoint_at:
                    path = self.save("periodic")
                    log.info("Checkpoint saved: %s", path)
                    self.env.update_training_metrics(
                        last_checkpoint=self.collector.steps
                    )
                    next_checkpoint_at = (
                        time.monotonic() + self.save_interval_seconds
                    )
                continue
            self.env.update_training_metrics(trainer_state="UPDATING")
            t0 = time.monotonic()
            metrics = self._update(batch)
            dt = time.monotonic() - t0
            update_times.append(dt)
            current = self.collector.steps

            if metrics:
                self.update_count += 1
                sps = (current - last_update_step) / max(1e-6, sum(update_times[-10:]) / min(len(update_times), 10))
                wall_sps = (current - initial_steps) / max(
                    1e-6, time.monotonic() - started_at
                )
                self.env.update_training_metrics(
                    trainer_state="COLLECTING",
                    update_count=self.update_count,
                    policy_loss=round(metrics["pi_loss"], 5),
                    value_loss=round(metrics["v_loss"], 4),
                    entropy=round(metrics["entropy"], 4),
                    approx_kl=round(metrics["approx_kl"], 5),
                    clip_fraction=round(metrics["clip_fraction"], 4),
                    explained_variance=round(metrics["explained_variance"], 4),
                    grad_norm=round(metrics["grad_norm"], 3),
                    feature_abs_mean=round(metrics["feature_abs_mean"], 4),
                    dead_feature_fraction=round(metrics["dead_feature_fraction"], 4),
                    move_action_entropy=round(metrics["move_action_entropy"], 4),
                    shoot_action_entropy=round(metrics["shoot_action_entropy"], 4),
                    move_idle_rate=round(metrics["move_idle_rate"], 4),
                    shoot_idle_rate=round(metrics["shoot_idle_rate"], 4),
                    aim_aux_loss=round(metrics["aim_aux_loss"], 4),
                    aim_aux_coef=round(metrics["aim_aux_coef"], 4),
                    aim_target_accuracy=round(
                        metrics["aim_target_accuracy"], 4
                    ),
                    aim_visible_accuracy=round(
                        metrics["aim_visible_accuracy"], 4
                    ),
                    aim_idle_accuracy=round(metrics["aim_idle_accuracy"], 4),
                    next_state_loss=round(
                        metrics.get("next_state_loss", 0.0), 5
                    ),
                    next_state_ratio=round(
                        metrics.get("next_state_ratio", 0.0), 4
                    ),
                    current_ent_coef=round(metrics["ent_coef"], 5),
                    target_kl=tc.target_kl or 0.0,
                    update_seconds=round(dt, 2),
                    steps_per_second=round(wall_sps, 1),
                )
                rolling_count = len(self.collector.recent_rewards)
                self.checkpoints.append_metrics(
                    {
                        "type": "ppo_update",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "global_steps": current,
                        "update_count": self.update_count,
                        "steps_per_second": wall_sps,
                        "last_episode_reward": self.collector.last_ep_reward,
                        "last_episode_length": self.collector.last_ep_length,
                        "last_episode_reason": self.collector.last_ep_reason,
                        "rolling_window": ROLLING_EPISODE_WINDOW,
                        "rolling_episodes": rolling_count,
                        "rolling_reward": (
                            sum(self.collector.recent_rewards) / rolling_count
                            if rolling_count else 0.0
                        ),
                        "rolling_win_rate": (
                            sum(self.collector.recent_wins) / rolling_count
                            if rolling_count else 0.0
                        ),
                        "rolling_idle_rate": (
                            sum(self.collector.recent_idle_rate) / rolling_count
                            if rolling_count else 0.0
                        ),
                        "rolling_wall_rate": (
                            sum(self.collector.recent_wall_rate) / rolling_count
                            if rolling_count else 0.0
                        ),
                        "last_episode_idle_rate": self.collector.last_ep_idle_rate,
                        "last_episode_wall_rate": self.collector.last_ep_wall_rate,
                        "last_episode_bomb_uses": self.collector.last_ep_bomb_uses,
                        **metrics,
                    }
                )
                log.info(
                    "UPDATE @ %d steps | %.2fs/update | ~%.0f steps/s | "
                    "pi=%.4f v=%.3f H=%.3f KL=%.4f clip=%.3f EV=%.3f "
                    "feat_dead=%.1f%% actH=%.2f/%.2f aim=%.3f acc=%.1f%% | "
                    "ep_rew=%.1f ep_len=%d (%s)",
                    current, dt, sps,
                    metrics["pi_loss"], metrics["v_loss"], metrics["entropy"],
                    metrics["approx_kl"], metrics["clip_fraction"],
                    metrics["explained_variance"],
                    metrics["dead_feature_fraction"] * 100.0,
                    metrics["move_action_entropy"], metrics["shoot_action_entropy"],
                    metrics["aim_aux_loss"],
                    metrics["aim_visible_accuracy"] * 100.0,
                    self.collector.last_ep_reward, self.collector.last_ep_length,
                    self.collector.last_ep_reason,
                )
            last_update_step = current

            now = time.monotonic()
            if now >= next_checkpoint_at:
                path = self.save("periodic")
                log.info("Checkpoint saved: %s", path)
                self.env.update_training_metrics(last_checkpoint=current)
                next_checkpoint_at = now + self.save_interval_seconds

        self.stop_collectors()
        final_path = self.save("final")
        log.info("Training complete. Final model: %s", final_path)
        return final_path

    def stop_collectors(self, timeout: float = 10.0) -> None:
        """Request collector shutdown and wait briefly for it."""
        self.stop_event.set()
        with self.buffer_cv:
            self.buffer_cv.notify_all()
        if self.collector.is_alive():
            self.collector.join(timeout=timeout)

    def save(self, label: str) -> Path:
        steps = int(self.collector.steps)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "run_id": self.checkpoints.run_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "global_steps": steps,
            "update_count": int(self.update_count),
            "state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "observation_shapes": self._observation_shapes(
                self.env.observation_space
            ),
            "action_nvec": self.env.action_space.nvec.tolist(),
            "preprocessing": {
                "normalize_player_features": bool(
                    self.config.env.normalize_player_features
                ),
                "frame_stack": int(self.config.env.frame_stack),
            },
            "config_path": self.checkpoints.config_path,
            "config": asdict(self.config),
            "model": {
                "features_dim": 256,
                "hidden": [128, 64],
                "recurrent": self.recurrent,
                "recurrent_hidden_size": self.config.train.recurrent_hidden_size,
                "aux_coef": self.aux_coef,
                "next_state_coef": self.next_state_coef,
            },
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        path = self.checkpoints.save(payload, label, steps)
        return path
