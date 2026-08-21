"""Deterministic evaluation for checkpoints produced by live_trainer.py."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from config import config_from_dict, load_config
from isaac_env import IsaacEnv
from live_checkpoint import LiveCheckpointManager
from live_trainer import IsaacPolicy, _obs_to_tensors
from utils import get_repo_root


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _resolve_config(checkpoint: dict, explicit_path: str | None):
    if explicit_path:
        return load_config(explicit_path), str(Path(explicit_path).resolve())

    stored_path = checkpoint.get("config_path")
    if stored_path:
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = get_repo_root() / candidate
        if candidate.is_file():
            return load_config(str(candidate)), str(candidate.resolve())

    return config_from_dict(checkpoint.get("config")), "checkpoint snapshot"


def _normalized_entropy(values: list[int], choices: int = 9) -> float:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=choices)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / math.log(choices))


def evaluate(checkpoint_ref: str, config_path: str | None, episodes: int) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    checkpoint_path = LiveCheckpointManager.resolve_checkpoint(checkpoint_ref)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    config, resolved_config = _resolve_config(checkpoint, config_path)

    env = IsaacEnv(config)
    current_preprocessing = {
        "normalize_player_features": bool(
            config.env.normalize_player_features
        ),
        "frame_stack": int(config.env.frame_stack),
    }
    stored_preprocessing = checkpoint.get("preprocessing")
    if stored_preprocessing is None and config.env.normalize_player_features:
        raise ValueError(
            "Legacy checkpoint has no preprocessing metadata. Evaluate it with "
            "a config containing env.normalize_player_features: false."
        )
    if (
        stored_preprocessing is not None
        and stored_preprocessing != current_preprocessing
    ):
        raise ValueError(
            "Checkpoint preprocessing mismatch: "
            f"stored={stored_preprocessing}, current={current_preprocessing}."
        )
    policy = IsaacPolicy(
        env.observation_space,
        env.action_space.nvec.tolist(),
        recurrent=bool(config.train.recurrent),
        recurrent_hidden_size=int(config.train.recurrent_hidden_size),
        aux_coef=float(config.train.aux_coef),
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    rewards: list[float] = []
    lengths: list[int] = []
    kills: list[int] = []
    damage_taken: list[int] = []
    reasons: list[str] = []
    move_actions: list[int] = []
    shoot_actions: list[int] = []

    try:
        for episode in range(1, episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
            info = {}
            done = False
            hidden = None  # reset recurrent state each episode
            while not done:
                with torch.no_grad():
                    _, logits, hidden, _aux = policy(
                        _obs_to_tensors(obs, device), hidden=hidden
                    )
                action = [
                    int(torch.argmax(head_logits, dim=1).item())
                    for head_logits in logits
                ]
                if config.phase.mask_shoot:
                    action[1] = 0
                if not config.phase.enable_interactions:
                    action = action[:2] + [0, 0, 0]

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                move_actions.append(action[0])
                shoot_actions.append(action[1])
                done = terminated or truncated

            state = info.get("state")
            reason = state.terminal_reason if state is not None else "unknown"
            rewards.append(total_reward)
            lengths.append(steps)
            kills.append(int(info.get("ep_kills", 0)))
            damage_taken.append(int(info.get("ep_damage_taken", 0)))
            reasons.append(reason or "unknown")
            print(
                f"Eval episode {episode}/{episodes}: reason={reason} "
                f"reward={total_reward:.2f} steps={steps} kills={kills[-1]}"
            )
    finally:
        env.close()

    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_steps": int(checkpoint.get("global_steps", 0)),
        "config": resolved_config,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "win_rate": sum(
            1 for r in reasons if r in ("room_cleared", "run_won")
        ) / episodes,
        "death_rate": reasons.count("death") / episodes,
        "timeout_rate": reasons.count("timeout") / episodes,
        "mean_kills": float(np.mean(kills)),
        "mean_damage_taken": float(np.mean(damage_taken)),
        "move_action_entropy": _normalized_entropy(move_actions),
        "shoot_action_entropy": _normalized_entropy(shoot_actions),
        "terminal_reasons": {
            reason: reasons.count(reason) for reason in sorted(set(reasons))
        },
    }

    report_dir = checkpoint_path.parent / "evaluations"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        "eval_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report"] = str(report_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a live PPO checkpoint")
    parser.add_argument(
        "checkpoint",
        help="Checkpoint .pt path, run folder, or 'latest'",
    )
    parser.add_argument("--config", default=None, help="Optional YAML override")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, args.config, args.episodes)
    print("\nEvaluation summary")
    print(json.dumps(metrics, indent=2))
