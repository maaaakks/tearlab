"""Record AGENT gameplay as npz episodes for offline world-model training.

Same output format as record_human.py (grid/player/entities/action/reward/
done per step), but the actions come from a trained PPO checkpoint instead
of keyboard input — no human play required. This feeds train_rssm_offline.py
with real, on-policy transitions while live training keeps running.

Usage:
    .venv/Scripts/python.exe python/record_agent.py latest \
        --episodes 50 --out data/agent/session1
    # deterministic (argmax) rollouts:
    ... python/record_agent.py latest --deterministic --episodes 30
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from config import config_from_dict, load_config
from isaac_env import IsaacEnv
from live_checkpoint import LiveCheckpointManager
from live_trainer import IsaacPolicy, _obs_to_tensors
from utils import get_repo_root

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",
                        help="checkpoint .pt path, run folder, or 'latest'")
    parser.add_argument("--config", default=None, help="YAML override")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--stochastic", action="store_true",
        help="sample actions from the policy distribution instead of argmax "
             "(wider state coverage for world-model training)",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_config(checkpoint: dict, explicit_path: str | None):
    if explicit_path:
        return load_config(explicit_path)
    stored_path = checkpoint.get("config_path")
    if stored_path:
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = get_repo_root() / candidate
        if candidate.is_file():
            return load_config(str(candidate))
    return config_from_dict(checkpoint.get("config"))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = LiveCheckpointManager.resolve_checkpoint(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = resolve_config(checkpoint, args.config)

    env = IsaacEnv(config)
    stored_model = checkpoint.get("model", {})
    policy = IsaacPolicy(
        env.observation_space,
        env.action_space.nvec.tolist(),
        recurrent=bool(config.train.recurrent),
        recurrent_hidden_size=int(config.train.recurrent_hidden_size),
        aux_coef=float(stored_model.get("aux_coef", 0.0)),
        next_state_coef=float(stored_model.get("next_state_coef", 0.0)),
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "agent" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_checkpoint.txt").write_text(
        json.dumps({
            "checkpoint": str(checkpoint_path),
            "global_steps": int(checkpoint.get("global_steps", 0)),
            "mode": "stochastic" if args.stochastic else "deterministic",
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Recording agent episodes ({'stochastic' if args.stochastic else 'deterministic'}) "
          f"from {checkpoint_path.name} -> {out_dir}")
    print(f"Checkpoint step {checkpoint.get('global_steps', 0)}\n")

    buffers_template = ("grid", "player", "entities", "action", "reward", "done")
    started_at = time.monotonic()
    total_steps = 0
    reasons: dict[str, int] = {}
    try:
        for episode_idx in range(1, args.episodes + 1):
            buffers = {k: [] for k in buffers_template}
            obs, _ = env.reset()
            ep_reward = 0.0
            steps = 0
            hidden = None
            done = False
            reason = "unknown"
            while not done:
                tensors = _obs_to_tensors(obs, device)
                with torch.no_grad():
                    _value, logits, hidden_out, _aux = policy(tensors, hidden=hidden)
                if args.stochastic:
                    action = [int(torch.distributions.Categorical(
                        logits=head_logits).sample().item())
                        for head_logits in logits]
                else:
                    action = [int(head_logits.argmax(dim=1).item())
                              for head_logits in logits]
                if config.phase.mask_shoot:
                    action[1] = 0
                if not config.phase.enable_interactions:
                    action = action[:2] + [0, 0, 0]

                buffers["grid"].append(obs["grid"])
                buffers["player"].append(obs["player"])
                buffers["entities"].append(obs["entities"])
                buffers["action"].append(tuple(action))

                obs, reward, terminated, truncated, info = env.step(action)
                # Carry the GRU state only for recurrent policies.
                hidden = hidden_out if config.train.recurrent else None
                ep_reward += float(reward)
                steps += 1
                total_steps += 1
                buffers["reward"].append(float(reward))
                done = terminated or truncated
                buffers["done"].append(done)

            state = info.get("state")
            reason = str(state.terminal_reason) if state else "unknown"
            meta = json.dumps({
                "episode_reason": reason,
                "episode_reward": round(ep_reward, 3),
                "steps": steps,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            })
            path = out_dir / f"episode_{episode_idx:04d}.npz"
            np.savez_compressed(
                path,
                grid=np.asarray(buffers["grid"], dtype=np.float32),
                player=np.asarray(buffers["player"], dtype=np.float32),
                entities=np.asarray(buffers["entities"], dtype=np.float32),
                action=np.asarray(buffers["action"], dtype=np.int64),
                reward=np.asarray(buffers["reward"], dtype=np.float32),
                done=np.asarray(buffers["done"], dtype=bool),
                meta=meta,
            )
            reasons[reason] = reasons.get(reason, 0) + 1
            print(f"[EP {episode_idx:>3}] {reason:<16} steps={steps:>4} "
                  f"reward={ep_reward:+8.2f}")
    finally:
        env.close()

    minutes = (time.monotonic() - started_at) / 60.0
    print(f"\nDone: {args.episodes} episodes, {total_steps} steps "
          f"in {minutes:.1f} min -> {out_dir}")
    print(f"Outcomes: {reasons}")


if __name__ == "__main__":
    main()
