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


class IsaacPolicy(nn.Module):
    """Actor-critic policy for MultiDiscrete action spaces with Dict observations.

    One categorical head per MultiDiscrete dimension (e.g. [9, 9, 2, 2, 2] =
    move, shoot, bomb, active item, card/pill).
    """

    def __init__(self, observation_space, action_nvec, features_dim: int = 256, hidden=(128, 64)):
        super().__init__()
        self.features = IsaacFeatureExtractor(observation_space, features_dim)
        self.action_nvec = list(int(n) for n in action_nvec)
        self.pi = self._mlp(features_dim, hidden)
        self.vf = self._mlp(features_dim, hidden)
        self.action_heads = nn.ModuleList(
            nn.Linear(hidden[-1], n) for n in self.action_nvec
        )
        self.value_head = nn.Linear(hidden[-1], 1)

    @staticmethod
    def _mlp(in_dim, hidden):
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        return nn.Sequential(*layers)

    def forward(self, obs):
        feat = self.features(obs)
        shared_pi = self.pi(feat)
        logits = [head(shared_pi) for head in self.action_heads]
        value = self.value_head(self.vf(feat)).squeeze(-1)
        return value, logits

    @staticmethod
    def _dist(logits):
        return torch.distributions.Categorical(logits=logits)

    def evaluate_actions(self, obs, actions):
        """actions: LongTensor (B, K). Returns values, joint logprobs, joint entropy."""
        value, logits = self.forward(obs)
        logprob = None
        entropy = None
        for k, head_logits in enumerate(logits):
            dist = self._dist(head_logits)
            term = dist.log_prob(actions[:, k])
            ent = dist.entropy()
            logprob = term if logprob is None else logprob + term
            entropy = ent if entropy is None else entropy + ent
        return value, logprob, entropy


def _obs_to_tensors(obs, device):
    return {
        "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0),
        "player": torch.as_tensor(obs["player"], dtype=torch.float32, device=device).unsqueeze(0),
    }


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
        self.steps = int(initial_steps)
        self.episodes = 0
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.last_ep_reward = 0.0
        self.last_ep_length = 0
        self.last_ep_reason = "unknown"
        self.recent_rewards = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
        self.recent_wins = deque(maxlen=100)

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
                )
                tensors = _obs_to_tensors(obs, self.device)
                with torch.no_grad():
                    value, logits = self.policy(tensors)
                    dists = [IsaacPolicy._dist(l) for l in logits]
                    samples = [d.sample() for d in dists]
                    logprob = sum(
                        d.log_prob(s).squeeze() for d, s in zip(dists, samples)
                    )

                action = tuple(int(s.item()) for s in samples)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                with torch.no_grad():
                    nv_tensors = _obs_to_tensors(next_obs, self.device)
                    next_value, _ = self.policy(nv_tensors)

                transition = {
                    "obs": {k: v.copy() for k, v in obs.items()},
                    "action": action,
                    "logprob": float(logprob.item()),
                    "value": float(value.item()),
                    "reward": float(reward),
                    "done": bool(done),
                    # Bootstrap for the case where this transition ends the batch
                    "next_value": 0.0 if done else float(next_value.item()),
                }

                self.ep_reward += reward
                self.ep_steps += 1
                self.steps += 1

                if done:
                    self.episodes += 1
                    self.last_ep_reward = self.ep_reward
                    self.last_ep_length = self.ep_steps
                    self.last_ep_reason = info.get("state").terminal_reason if info.get("state") else "unknown"
                    self.recent_rewards.append(float(self.last_ep_reward))
                    self.recent_lengths.append(int(self.last_ep_length))
                    self.recent_wins.append(float(self.last_ep_reason == "room_cleared"))
                    log.info(
                        "EP %d (%s) | steps=%d reward=%.1f | total collected=%d",
                        self.episodes, self.last_ep_reason, self.ep_steps,
                        self.ep_reward, self.steps,
                    )
                    self.ep_reward = 0.0
                    self.ep_steps = 0
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs

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
            env.observation_space, env.action_space.nvec.tolist()
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
        self.vf_coef = tc.vf_coef
        self.max_grad_norm = tc.max_grad_norm
        self.n_steps = tc.n_steps
        self.batch_size = tc.batch_size
        self.n_epochs = tc.n_epochs
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
        self.collector = Collector(
            env, self.policy, self.device,
            self.buffer, self.buffer_lock, self.buffer_cv, self.stop_event,
            self.total_timesteps,
            initial_steps,
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

    def _load_checkpoint(self, path: Path) -> int:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        expected_shapes = self._observation_shapes(self.env.observation_space)
        stored_shapes = checkpoint.get("observation_shapes")
        if stored_shapes is not None and stored_shapes != expected_shapes:
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

        self.policy.load_state_dict(checkpoint["state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

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
            torch.cuda.set_rng_state_all(cuda_rng_states)

        log.info(
            "Resumed live checkpoint %s at step %d (update %d)",
            path,
            steps,
            self.update_count,
        )
        return steps

    def _current_ent_coef(self, steps: int) -> float:
        if self.entropy_decay_fraction <= 0 or self.ent_coef == self.ent_coef_final:
            return self.ent_coef_final
        decay_steps = max(1.0, self.total_timesteps * self.entropy_decay_fraction)
        progress = min(1.0, max(0.0, steps / decay_steps))
        return self.ent_coef + progress * (self.ent_coef_final - self.ent_coef)

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

    def _update(self, batch: list):
        tc = self.config.train
        advantages, returns = self._compute_gae(batch)
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

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
        advs = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        rets = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = len(batch)
        idx = np.arange(n)
        stats = {
            "pi_loss": 0.0,
            "v_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "updates": 0,
        }
        current_ent_coef = self._current_ent_coef(self.collector.steps)
        early_stop = False
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                if len(mb) < 2:
                    continue
                mb_obs = {"grid": grids[mb], "player": players[mb]}
                values, logprobs, entropy = self.policy.evaluate_actions(mb_obs, actions[mb])

                ratio = torch.exp(logprobs - old_logprobs[mb])
                mb_adv = advs[mb]
                pi_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv,
                ).mean()
                v_loss = ((values - rets[mb]) ** 2).mean()
                ent = entropy.mean()

                loss = pi_loss + self.vf_coef * v_loss - current_ent_coef * ent
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
            with torch.no_grad():
                sample_values, _ = self.policy(sample_obs)
                sample_features = self.policy.features(sample_obs)

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

            return {
                "pi_loss": stats["pi_loss"] / u,
                "v_loss": stats["v_loss"] / u,
                "entropy": stats["entropy"] / u,
                "approx_kl": stats["kl"] / u,
                "clip_fraction": stats["clip_fraction"] / u,
                "grad_norm": stats["grad_norm"] / u,
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
                        "rolling_episodes": rolling_count,
                        "rolling_reward": (
                            sum(self.collector.recent_rewards) / rolling_count
                            if rolling_count else 0.0
                        ),
                        "rolling_win_rate": (
                            sum(self.collector.recent_wins) / rolling_count
                            if rolling_count else 0.0
                        ),
                        **metrics,
                    }
                )
                log.info(
                    "UPDATE @ %d steps | %.2fs/update | ~%.0f steps/s | "
                    "pi=%.4f v=%.3f H=%.3f KL=%.4f clip=%.3f EV=%.3f "
                    "feat_dead=%.1f%% actH=%.2f/%.2f | ep_rew=%.1f ep_len=%d (%s)",
                    current, dt, sps,
                    metrics["pi_loss"], metrics["v_loss"], metrics["entropy"],
                    metrics["approx_kl"], metrics["clip_fraction"],
                    metrics["explained_variance"],
                    metrics["dead_feature_fraction"] * 100.0,
                    metrics["move_action_entropy"], metrics["shoot_action_entropy"],
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

        self.stop_event.set()
        final_path = self.save("final")
        log.info("Training complete. Final model: %s", final_path)
        return final_path

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
            },
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        path = self.checkpoints.save(payload, label, steps)
        return path
