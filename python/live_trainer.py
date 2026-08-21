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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from network import IsaacFeatureExtractor

log = logging.getLogger("live_trainer")


class IsaacPolicy(nn.Module):
    """Actor-critic policy for MultiDiscrete([9, 9]) with Dict observations."""

    def __init__(self, observation_space, features_dim: int = 256, hidden=(128, 64)):
        super().__init__()
        self.features = IsaacFeatureExtractor(observation_space, features_dim)
        self.pi = self._mlp(features_dim, hidden)
        self.vf = self._mlp(features_dim, hidden)
        self.move_head = nn.Linear(hidden[-1], 9)
        self.shoot_head = nn.Linear(hidden[-1], 9)
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
        move_logits = self.move_head(self.pi(feat))
        shoot_logits = self.shoot_head(self.pi(feat))
        value = self.value_head(self.vf(feat)).squeeze(-1)
        return value, move_logits, shoot_logits

    @staticmethod
    def _dist(logits):
        return torch.distributions.Categorical(logits=logits)

    def evaluate_actions(self, obs, actions):
        """actions: LongTensor (B, 2). Returns values, joint logprobs, joint entropy."""
        value, move_logits, shoot_logits = self.forward(obs)
        move_dist = self._dist(move_logits)
        shoot_dist = self._dist(shoot_logits)
        logprob = (
            move_dist.log_prob(actions[:, 0]) + shoot_dist.log_prob(actions[:, 1])
        )
        entropy = move_dist.entropy() + shoot_dist.entropy()
        return value, logprob, entropy


def _obs_to_tensors(obs, device):
    return {
        "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0),
        "player": torch.as_tensor(obs["player"], dtype=torch.float32, device=device).unsqueeze(0),
    }


class Collector(threading.Thread):
    """Continuously steps the environment, requesting actions from the policy."""

    def __init__(self, env, policy, device, buffer, buffer_lock, buffer_cv, stop_event):
        super().__init__(daemon=True, name="isaac-collector")
        self.env = env
        self.policy = policy
        self.device = device
        self.buffer = buffer
        self.buffer_lock = buffer_lock
        self.buffer_cv = buffer_cv
        self.stop_event = stop_event
        self.steps = 0
        self.episodes = 0
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.last_ep_reward = 0.0
        self.last_ep_length = 0
        self.last_ep_reason = "unknown"

    def run(self):
        obs, _ = self.env.reset()
        while not self.stop_event.is_set():
            try:
                tensors = _obs_to_tensors(obs, self.device)
                with torch.no_grad():
                    value, move_logits, shoot_logits = self.policy(tensors)
                    move_dist = IsaacPolicy._dist(move_logits)
                    shoot_dist = IsaacPolicy._dist(shoot_logits)
                    move = move_dist.sample()
                    shoot = shoot_dist.sample()
                    logprob = move_dist.log_prob(move) + shoot_dist.log_prob(shoot)

                action = (int(move.item()), int(shoot.item()))
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                with torch.no_grad():
                    nv_tensors = _obs_to_tensors(next_obs, self.device)
                    next_value, _, _ = self.policy(nv_tensors)

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
                try:
                    obs, _ = self.env.reset()
                except Exception:
                    pass


class LivePPOTrainer:
    def __init__(self, config: Config, env):
        self.config = config
        self.env = env
        tc = config.train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = IsaacPolicy(env.observation_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=tc.learning_rate)
        self.gamma = tc.gamma
        self.gae_lambda = tc.gae_lambda
        self.clip_range = tc.clip_range
        self.ent_coef = tc.ent_coef
        self.vf_coef = tc.vf_coef
        self.max_grad_norm = tc.max_grad_norm
        self.n_steps = tc.n_steps
        self.batch_size = tc.batch_size
        self.n_epochs = tc.n_epochs
        self.total_timesteps = tc.total_timesteps
        self.save_interval = tc.save_interval

        self.buffer: list = []
        self.buffer_lock = threading.Lock()
        self.buffer_cv = threading.Condition(self.buffer_lock)
        self.stop_event = threading.Event()
        self.collector = Collector(
            env, self.policy, self.device,
            self.buffer, self.buffer_lock, self.buffer_cv, self.stop_event,
        )

    def _wait_for_batch(self, n: int) -> list:
        with self.buffer_cv:
            while len(self.buffer) < n and not self.stop_event.is_set():
                self.buffer_cv.wait(timeout=1.0)
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
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "updates": 0}
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

                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logprobs[mb] - logprobs).mean().item()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl"] += approx_kl
                stats["updates"] += 1

        if stats["updates"]:
            u = stats["updates"]
            return {
                "pi_loss": stats["pi_loss"] / u,
                "v_loss": stats["v_loss"] / u,
                "entropy": stats["entropy"] / u,
                "approx_kl": stats["kl"] / u,
            }
        return None

    def learn(self):
        tc = self.config.train
        log.info(
            "Live training started: device=%s n_steps=%d lr=%g total=%d",
            self.device, self.n_steps, tc.learning_rate, self.total_timesteps,
        )
        self.collector.start()

        last_update_step = 0
        last_save_step = 0
        update_times = []
        while self.collector.steps < self.total_timesteps and not self.stop_event.is_set():
            batch = self._wait_for_batch(self.n_steps)
            t0 = time.monotonic()
            metrics = self._update(batch)
            dt = time.monotonic() - t0
            update_times.append(dt)
            current = self.collector.steps

            if metrics:
                sps = (current - last_update_step) / max(1e-6, sum(update_times[-10:]) / min(len(update_times), 10))
                log.info(
                    "UPDATE @ %d steps | %.2fs/update | ~%.0f steps/s | "
                    "pi=%.4f v=%.3f H=%.3f KL=%.4f | ep_rew=%.1f ep_len=%d (%s)",
                    current, dt, sps,
                    metrics["pi_loss"], metrics["v_loss"], metrics["entropy"],
                    metrics["approx_kl"],
                    self.collector.last_ep_reward, self.collector.last_ep_length,
                    self.collector.last_ep_reason,
                )
            last_update_step = current

            if current - last_save_step >= self.save_interval:
                path = self.save(str(current))
                log.info("Checkpoint saved: %s", path)
                last_save_step = current

        self.stop_event.set()
        final_path = self.save("final")
        log.info("Training complete. Final model: %s", final_path)
        return final_path

    def save(self, name) -> Path:
        out_dir = Path("checkpoints") / "live"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"policy_{name}.pt"
        torch.save({
            "state_dict": self.policy.state_dict(),
            "observation_space": self.env.observation_space,
            "config": {
                "features_dim": 256,
                "hidden": [128, 64],
            },
        }, path)
        return path
