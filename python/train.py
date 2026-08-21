import argparse
import logging
import math
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path

try:
    from sb3_contrib import MaskablePPO as PPO
except ImportError as exc:
    raise ImportError(
        "sb3-contrib is required for action masking. "
        "Install dependencies from python/requirements.txt."
    ) from exc
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from checkpoint_manager import CheckpointManager
from config import load_config
from isaac_env import IsaacEnv
from network import IsaacFeatureExtractor
from utils import (
    get_checkpoint_dir,
    get_log_dir,
)


def validate_ppo_checkpoint(checkpoint_path: Path, env) -> None:
    """Raise if a checkpoint cannot be loaded for the provided environment."""
    PPO.load(str(checkpoint_path), env=env)


def resolve_resume_path(
    resume: str | None,
    checkpoint_dir: Path,
    config_path: str | None,
    env,
) -> str | None:
    """Resolve special resume modes to an actual checkpoint path.

    Modes:
        latest              — most recent checkpoint for the same config
        latest-any          — most recent checkpoint across all configs
        latest-compatible   — newest checkpoint (same config) that loads successfully
        run:<wandb_id>      — latest checkpoint from a specific W&B run
        <path>              — explicit path (absolute, relative, or filename in checkpoint_dir)
    """
    if resume is None:
        return None

    if resume.startswith("run:"):
        run_id = resume[4:]
        checkpoint = CheckpointManager.find_latest_for_run(checkpoint_dir, run_id)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoints found for run '{run_id}' in {checkpoint_dir}"
            )
        return str(checkpoint)

    if resume == "latest":
        checkpoint = CheckpointManager.find_latest_for_config(checkpoint_dir, config_path)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoints found for config '{Path(config_path).stem}' in {checkpoint_dir}"
            )
        return str(checkpoint)

    if resume == "latest-any":
        checkpoint = CheckpointManager.find_latest(checkpoint_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return str(checkpoint)

    if resume == "latest-compatible":
        checkpoint = CheckpointManager.find_latest_compatible(
            checkpoint_dir,
            config_path,
            validator=lambda candidate: validate_ppo_checkpoint(candidate, env),
        )
        if checkpoint is None:
            raise FileNotFoundError(
                f"No compatible checkpoints found in {checkpoint_dir}"
            )
        return str(checkpoint)

    resume_path = Path(resume).expanduser()
    if resume_path.is_absolute():
        return str(resume_path)

    if resume_path.exists():
        return str(resume_path.resolve())

    checkpoint_candidate = checkpoint_dir / resume_path
    if checkpoint_candidate.exists():
        return str(checkpoint_candidate.resolve())

    return str(resume_path.resolve())


class ManagedCheckpointCallback(BaseCallback):
    """Save periodic checkpoints via CheckpointManager."""

    def __init__(self, manager: CheckpointManager, save_freq: int, verbose=0):
        super().__init__(verbose)
        self.manager = manager
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            name = f"step_{self.num_timesteps:07d}"
            self.manager.save(self.model, name, self.num_timesteps, "periodic")
        return True


class GamePauseCallback(BaseCallback):
    """Pause the game during PPO training to prevent stale state buffering.

    Works with both single IsaacEnv and vectorized (SubprocVecEnv/DummyVecEnv) envs.
    """

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self._is_vec = hasattr(env, "env_method")

    def _on_rollout_end(self) -> None:
        if self._is_vec:
            self.env.env_method("pause_game")
        else:
            self.env.unwrapped.pause_game()

    def _on_rollout_start(self) -> None:
        if self._is_vec:
            results = self.env.env_method("resume_and_flush")
            total_flushed = sum(r for r in results if r)
        else:
            total_flushed = self.env.unwrapped.resume_and_flush()
        if total_flushed > 0:
            log = logging.getLogger("train")
            log.info("Flushed %d stale bytes from TCP buffer(s) at rollout start", total_flushed)

    def _on_step(self) -> bool:
        return True


class IsaacMetricsCallback(BaseCallback):
    """Log Isaac-specific episode metrics to wandb and console."""

    ROLLING_WINDOW = 50

    def __init__(
        self,
        use_wandb=False,
        wall_collision_penalty: float = 0.0,
        verbose=0,
    ):
        super().__init__(verbose)
        self.use_wandb = use_wandb
        self.wall_collision_penalty = wall_collision_penalty
        self._ep_count = 0
        # Rolling window buffers
        self._recent_rewards: list[float] = []
        self._recent_lengths: list[int] = []
        self._recent_wins: list[float] = []
        self._recent_pickups: list[float] = []
        self._recent_pickup_counts: list[float] = []
        self._recent_wall_collision_steps: list[float] = []
        self._last_train_log_step = 0
        self._nav_stats_by_env: dict[int, dict] = {}
        # Steps-per-second tracking
        self._sps_last_time: float | None = None
        self._sps_last_step: int = 0

    @staticmethod
    def _fresh_nav_stats() -> dict:
        return {
            "steps": 0,
            "pickup_target_steps": 0,
            "pickup_dist_sum": 0.0,
            "pickup_no_target_steps": 0,
            "enemy_no_target_steps": 0,
            "projectile_no_target_steps": 0,
            "alignment_sum": 0.0,
            "alignment_count": 0,
            "prev_pos": None,
        }

    def _update_nav_stats(self, env_idx: int, state) -> None:
        if state is None:
            return
        stats = self._nav_stats_by_env.setdefault(env_idx, self._fresh_nav_stats())
        stats["steps"] += 1

        p = state.player
        pickup_dx = float(p.nearest_pickup_dx)
        pickup_dy = float(p.nearest_pickup_dy)
        enemy_dx = float(p.nearest_enemy_dx)
        enemy_dy = float(p.nearest_enemy_dy)
        projectile_dx = float(p.nearest_projectile_dx)
        projectile_dy = float(p.nearest_projectile_dy)

        pickup_has_target = abs(pickup_dx) > 1e-8 or abs(pickup_dy) > 1e-8
        if pickup_has_target:
            pickup_dist = math.sqrt(pickup_dx * pickup_dx + pickup_dy * pickup_dy)
            stats["pickup_dist_sum"] += pickup_dist
            stats["pickup_target_steps"] += 1
        else:
            stats["pickup_no_target_steps"] += 1

        if abs(enemy_dx) <= 1e-8 and abs(enemy_dy) <= 1e-8:
            stats["enemy_no_target_steps"] += 1

        if abs(projectile_dx) <= 1e-8 and abs(projectile_dy) <= 1e-8:
            stats["projectile_no_target_steps"] += 1

        prev_pos = stats["prev_pos"]
        curr_pos = p.position
        if (
            pickup_has_target
            and prev_pos is not None
            and curr_pos is not None
        ):
            move_dx = float(curr_pos[0] - prev_pos[0])
            move_dy = float(curr_pos[1] - prev_pos[1])
            move_norm = math.sqrt(move_dx * move_dx + move_dy * move_dy)
            target_norm = math.sqrt(pickup_dx * pickup_dx + pickup_dy * pickup_dy)
            if move_norm > 1e-8 and target_norm > 1e-8:
                cosine = (move_dx * pickup_dx + move_dy * pickup_dy) / (move_norm * target_norm)
                cosine = max(-1.0, min(1.0, cosine))
                stats["alignment_sum"] += cosine
                stats["alignment_count"] += 1

        if curr_pos is not None:
            stats["prev_pos"] = curr_pos

    def _compute_nav_metrics(self, env_idx: int) -> dict[str, float]:
        stats = self._nav_stats_by_env.setdefault(env_idx, self._fresh_nav_stats())
        steps = max(1, stats["steps"])
        pickup_target_steps = stats["pickup_target_steps"]
        alignment_count = stats["alignment_count"]
        return {
            "nav/pickup_dist_mean": (
                stats["pickup_dist_sum"] / pickup_target_steps if pickup_target_steps > 0 else 0.0
            ),
            "nav/pickup_alignment": (
                stats["alignment_sum"] / alignment_count if alignment_count > 0 else 0.0
            ),
            "state/no_target_rate_pickup": stats["pickup_no_target_steps"] / steps,
            "state/no_target_rate_enemy": stats["enemy_no_target_steps"] / steps,
            "state/no_target_rate_projectile": stats["projectile_no_target_steps"] / steps,
            "state/target_rate_pickup": pickup_target_steps / steps,
            "state/alignment_sample_rate": alignment_count / steps,
        }

    def _reset_nav_stats(self, env_idx: int) -> None:
        self._nav_stats_by_env[env_idx] = self._fresh_nav_stats()

    def _append_rolling(self, rewards, lengths, won, pickup, pickup_count, wall_collision_steps):
        w = self.ROLLING_WINDOW
        self._recent_rewards.append(rewards)
        self._recent_lengths.append(lengths)
        self._recent_wins.append(won)
        self._recent_pickups.append(pickup)
        self._recent_pickup_counts.append(pickup_count)
        self._recent_wall_collision_steps.append(wall_collision_steps)
        if len(self._recent_rewards) > w:
            self._recent_rewards = self._recent_rewards[-w:]
            self._recent_lengths = self._recent_lengths[-w:]
            self._recent_wins = self._recent_wins[-w:]
            self._recent_pickups = self._recent_pickups[-w:]
            self._recent_pickup_counts = self._recent_pickup_counts[-w:]
            self._recent_wall_collision_steps = self._recent_wall_collision_steps[-w:]

    def _rolling_metrics(self) -> dict:
        n = len(self._recent_rewards)
        if n == 0:
            return {}
        return {
            "rollout/win_rate": sum(self._recent_wins) / n,
            "rollout/ep_rew_mean": sum(self._recent_rewards) / n,
            "rollout/ep_len_mean": sum(self._recent_lengths) / n,
            "rollout/pickup_rate": sum(self._recent_pickups) / n,
            "rollout/pickups_collected_mean": sum(self._recent_pickup_counts) / n,
            "rollout/wall_collision_steps_mean": sum(self._recent_wall_collision_steps) / n,
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            self._update_nav_stats(env_idx, info.get("state"))

            if "episode" not in info:
                continue
            self._ep_count += 1

            state = info.get("state")
            ep_reward = info["episode"]["r"]
            ep_length = info["episode"]["l"]
            reason = state.terminal_reason if state else "unknown"
            kills = info.get("ep_kills", 0)
            dmg = info.get("ep_damage_taken", 0)
            tps = info.get("game_ticks_per_sec", 0)
            pickup_count = int(state.pickups_collected) if state else 0
            ep_reward_components = info.get("ep_reward_components", {})
            wall_collision_penalty_total = float(ep_reward_components.get("wall_collision", 0.0))
            wall_collision_steps = 0.0
            if self.wall_collision_penalty != 0.0:
                wall_collision_steps = wall_collision_penalty_total / self.wall_collision_penalty

            won = float(reason in ("room_cleared", "run_won"))
            pickup = float(pickup_count > 0)
            self._append_rolling(
                ep_reward,
                ep_length,
                won,
                pickup,
                float(pickup_count),
                wall_collision_steps,
            )
            nav_metrics = self._compute_nav_metrics(env_idx)

            # Console summary (visible in main process)
            log = logging.getLogger("train")
            if nav_metrics:
                log.info(
                    "EP %d (%s) | steps=%d reward=%.1f kills=%d dmg=%d ticks/s=%.1f "
                    "pickup_dist=%.3f align=%.3f no_target(p/e/pr)=%.2f/%.2f/%.2f [t=%d]",
                    self._ep_count, reason, ep_length, ep_reward, kills, dmg, tps,
                    nav_metrics["nav/pickup_dist_mean"],
                    nav_metrics["nav/pickup_alignment"],
                    nav_metrics["state/no_target_rate_pickup"],
                    nav_metrics["state/no_target_rate_enemy"],
                    nav_metrics["state/no_target_rate_projectile"],
                    self.num_timesteps,
                )
            else:
                log.info(
                    "EP %d (%s) | steps=%d reward=%.1f kills=%d dmg=%d ticks/s=%.1f [t=%d]",
                    self._ep_count, reason, ep_length, ep_reward, kills, dmg, tps,
                    self.num_timesteps,
                )

            if not self.use_wandb:
                self._reset_nav_stats(env_idx)
                continue
            import wandb

            # Gameplay metrics
            metrics = {
                "global_step": self.num_timesteps,
                "episode/reward": ep_reward,
                "episode/length": ep_length,
                "episode/won": won,
                "episode/kills": kills,
                "episode/damage_taken": dmg,
                "episode/pickups_collected": pickup_count,
                "episode/pickup_collected": pickup,
                "episode/wall_collision_penalty": wall_collision_penalty_total,
                "episode/wall_collision_steps": wall_collision_steps,
            }

            # Rolling averages
            metrics.update(self._rolling_metrics())
            metrics.update(nav_metrics)

            # Cumulative reward component breakdown
            for component, value in ep_reward_components.items():
                metrics[f"reward/{component}"] = value

            # Performance metrics
            if "frames_dropped" in info:
                metrics["perf/frames_dropped"] = info["frames_dropped"]
            if "avg_step_latency" in info:
                metrics["perf/avg_step_latency_ms"] = info["avg_step_latency"] * 1000
            if "instant_ratio" in info:
                metrics["perf/instant_ratio"] = info["instant_ratio"]
            if tps > 0:
                metrics["perf/game_ticks_per_sec"] = tps

            # Steps per second (wall-clock training speed)
            now = time.monotonic()
            if self._sps_last_time is not None:
                elapsed = now - self._sps_last_time
                if elapsed > 0:
                    sps = (self.num_timesteps - self._sps_last_step) / elapsed
                    metrics["perf/steps_per_second"] = sps
            self._sps_last_time = now
            self._sps_last_step = self.num_timesteps

            wandb.log(metrics, step=self.num_timesteps)
            self._reset_nav_stats(env_idx)

        # Forward PPO training metrics to wandb (logged by SB3 to its internal logger)
        if self.use_wandb and self.model is not None:
            self._log_train_metrics()

        return True

    def _log_train_metrics(self):
        """Forward SB3's internal training stats to wandb once per rollout."""
        logger = self.model.logger
        if logger is None:
            return
        name_to_value = getattr(logger, "name_to_value", {})
        if not name_to_value:
            return
        # Only log when SB3 updates (check if timestep advanced since last log)
        step = self.num_timesteps
        if step == self._last_train_log_step:
            return
        self._last_train_log_step = step

        import wandb
        train_metrics = {}
        key_map = {
            "train/entropy_loss": "train/entropy_loss",
            "train/policy_gradient_loss": "train/policy_loss",
            "train/value_loss": "train/value_loss",
            "train/approx_kl": "train/approx_kl",
            "train/clip_fraction": "train/clip_fraction",
            "train/explained_variance": "train/explained_variance",
            "train/learning_rate": "train/learning_rate",
        }
        for sb3_key, wandb_key in key_map.items():
            if sb3_key in name_to_value:
                train_metrics[wandb_key] = name_to_value[sb3_key]
        if train_metrics:
            train_metrics["global_step"] = step
            wandb.log(train_metrics, step=step)


def _make_env(config, port):
    """Factory that creates a single IsaacEnv with the given port."""
    def _init():
        worker_config = replace(config, env=replace(config.env, port=port))
        env = IsaacEnv(worker_config)
        return Monitor(env)
    return _init


def train(config_path: str | None = None, resume: str | None = None, config=None):
    log = logging.getLogger("train")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers
    for noisy in ("urllib3", "git", "git.cmd", "git.util", "asyncio", "wandb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if config is None:
        config = load_config(config_path)

    # Live mode: the game never pauses. A collector thread steps the env
    # continuously while PPO updates run on fresh buffered transitions.
    if config.train.live_mode:
        from live_trainer import LivePPOTrainer

        isaac_env = IsaacEnv(config)
        trainer = LivePPOTrainer(
            config,
            isaac_env,
            resume=resume,
            config_path=config_path,
        )

        interrupted = False

        def _on_sigint(_sig, _frame):
            nonlocal interrupted
            if interrupted:
                raise SystemExit(1)
            interrupted = True
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _on_sigint)
        try:
            trainer.learn()
        except KeyboardInterrupt:
            log.info("Interrupted; stopping collector and saving checkpoint...")
            trainer.stop_event.set()
            with trainer.buffer_cv:
                trainer.buffer_cv.notify_all()
            trainer.collector.join(timeout=10.0)
            checkpoint_path = trainer.save("interrupted")
            log.info("Interrupted checkpoint saved to %s", checkpoint_path)
        except Exception:
            trainer.stop_event.set()
            with trainer.buffer_cv:
                trainer.buffer_cv.notify_all()
            checkpoint_path = trainer.save("crashed")
            log.exception("Live training crashed; checkpoint saved to %s", checkpoint_path)
            raise
        finally:
            isaac_env.close()
        return

    n_workers = config.env.n_workers
    base_port = config.env.base_port
    host = config.env.host
    if n_workers > 1:
        ports = [base_port + i for i in range(n_workers)]
        log.info("Starting vectorized training with %d workers (ports %d-%d)",
                 n_workers, base_port, base_port + n_workers - 1)
        log.info("Expecting Isaac workers reachable at %s on ports: %s",
                 host, ", ".join(str(p) for p in ports))
        env_fns = [_make_env(config, base_port + i) for i in range(n_workers)]
        try:
            env = SubprocVecEnv(env_fns)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize one or more Isaac workers. "
                f"Expected {n_workers} active game instances at {host} on ports "
                f"{base_port}-{base_port + n_workers - 1}. "
                "If you only have one game window, set env.n_workers: 1. "
                "For parallel runs, launch workers first via python/launcher.py."
            ) from exc
    else:
        log.info("Starting single-worker training (expecting %s:%d)", host, base_port)
        isaac_env = IsaacEnv(config)
        env = Monitor(isaac_env)

    checkpoint_dir = get_checkpoint_dir()
    log_dir = get_log_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Wandb init
    use_wandb = config.wandb.enabled
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.run_name,
            tags=config.wandb.tags,
            config=asdict(config),
        )
        wandb.define_metric("global_step")
        for metric_pattern in (
            "episode/*",
            "reward/*",
            "rollout/*",
            "nav/*",
            "state/*",
            "perf/*",
            "train/*",
        ):
            wandb.define_metric(metric_pattern, step_metric="global_step")

    # Checkpoint manager — per-run folders, metadata, W&B artifacts
    ckpt_manager = CheckpointManager(
        base_dir=checkpoint_dir,
        config_path=config_path,
        config=config,
        wandb_run=wandb_run,
    )

    policy_kwargs = {
        "features_extractor_class": IsaacFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    }

    resume_path = resolve_resume_path(resume, checkpoint_dir, config_path, env)
    if resume_path:
        model = PPO.load(resume_path, env=env)
        log.info("Resumed from %s", resume_path)
        # Override hyperparameters from config (PPO.load restores saved values)
        old_n_steps = model.n_steps
        model.learning_rate = config.train.learning_rate
        model.n_steps = config.train.n_steps
        model.batch_size = config.train.batch_size
        model.n_epochs = config.train.n_epochs
        model.gamma = config.train.gamma
        model.gae_lambda = config.train.gae_lambda
        model.clip_range = constant_fn(config.train.clip_range)
        model.ent_coef = config.train.ent_coef
        model.vf_coef = config.train.vf_coef
        model.max_grad_norm = config.train.max_grad_norm
        if config.train.target_kl is not None:
            model.target_kl = config.train.target_kl
        # Recreate rollout buffer if n_steps changed (buffer size is fixed at load)
        if model.n_steps != old_n_steps:
            log.info("n_steps changed %d -> %d, recreating rollout buffer", old_n_steps, model.n_steps)
            model.rollout_buffer = model.rollout_buffer_class(
                model.n_steps,
                model.observation_space,
                model.action_space,
                model.device,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
                n_envs=model.n_envs,
            )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.train.learning_rate,
            n_steps=config.train.n_steps,
            batch_size=config.train.batch_size,
            n_epochs=config.train.n_epochs,
            gamma=config.train.gamma,
            gae_lambda=config.train.gae_lambda,
            clip_range=config.train.clip_range,
            target_kl=config.train.target_kl,
            ent_coef=config.train.ent_coef,
            vf_coef=config.train.vf_coef,
            max_grad_norm=config.train.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir),
        )

    checkpoint_callback = ManagedCheckpointCallback(
        manager=ckpt_manager,
        save_freq=config.train.save_interval,
    )

    callbacks = [
        checkpoint_callback,
        GamePauseCallback(env),
        IsaacMetricsCallback(
            use_wandb=use_wandb,
            wall_collision_penalty=config.reward.wall_collision_penalty,
        ),
    ]

    # Graceful SIGINT: save checkpoint before exiting
    interrupted = False
    interrupt_checkpoint_path: Path | None = None

    def _save_interrupt_checkpoint() -> Path:
        nonlocal interrupt_checkpoint_path
        if interrupt_checkpoint_path is None:
            step = model.num_timesteps
            interrupt_checkpoint_path = ckpt_manager.save(
                model, "interrupted_model", step, "interrupted"
            )
        return interrupt_checkpoint_path

    def _on_sigint(_sig, _frame):
        nonlocal interrupted
        if interrupted:
            log.warning("Second interrupt, forcing exit")
            raise SystemExit(1)
        interrupted = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        model.learn(
            total_timesteps=config.train.total_timesteps,
            callback=callbacks,
            log_interval=config.train.log_interval,
        )
        step = model.num_timesteps
        checkpoint_path = ckpt_manager.save(model, "final_model", step, "final")
        log.info("Training complete. Model saved to %s", checkpoint_path)
    except KeyboardInterrupt:
        log.info("Interrupt received, saving checkpoint...")
        checkpoint_path = _save_interrupt_checkpoint()
        log.info("Saved to %s", checkpoint_path)
        log.info("Training interrupted. Exiting cleanly.")
    except Exception:
        if interrupted:
            checkpoint_path = _save_interrupt_checkpoint()
            log.warning(
                "Interrupted during shutdown. Preserving interrupt checkpoint at %s",
                checkpoint_path,
            )
            return
        log.exception("Training crashed, saving emergency checkpoint...")
        step = model.num_timesteps
        checkpoint_path = ckpt_manager.save(model, "crashed_model", step, "crashed")
        log.info("Saved to %s", checkpoint_path)
        raise
    finally:
        env.close()
        if use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isaac RL agent")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path, run:<wandb_id>, or one of: latest, latest-any, latest-compatible",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (overrides config env.n_workers)",
    )
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)
    if args.workers is not None:
        config.env.n_workers = args.workers

    train(config_path, args.resume, config=config)
