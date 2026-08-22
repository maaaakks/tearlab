"""Record human gameplay demonstrations for imitation warm-starting.

Launch Isaac with the mod loaded, run this script, then play. The mod's
HUMAN_MODE passes your keyboard through to Isaac untouched and attaches the
encoded input to every decision tick it streams, so this recorder stores
(observation, action, reward, done) tuples sharing the agent's exact
observation and action spaces.

Timing convention (matches env.step()): the state streamed at tick t carries
the input sampled at tick t; that input drives the game until tick t+1. Each
stored index i therefore holds (obs_i, action_i, reward_{i+1}, done_{i+1}),
where obs_i -> action_i is the behavior-cloning pair and reward/done describe
what followed.

Each episode is saved to <out>/episode_NNNN.npz:
    grid      (T, C, H, W)  float32 stacked observation channels
    player    (T, F)        float32 player feature vector
    entities  (T, E, D)     float32 padded entity table
    action    (T, 5)        int64 [move, shoot, bomb, active, pillcard]
    reward    (T,)          float32 shaped reward entering the NEXT tick
    done      (T,)          bool terminal flag of the NEXT tick
    meta      json string   {episode_reason, steps, saved_at}

Usage:
    python record_human.py --config configs/phase1a-human.yaml \
        --max-episodes 50 --out data/human/session1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import load_config
from isaac_env import IsaacEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "human"
ACTION_KEYS = ("move", "shoot", "bomb", "active", "pillcard")
ZERO_ACTION = (0, 0, 0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1a-human.yaml")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument(
        "--max-episodes", type=int, default=0,
        help="stop after N recorded episodes (0 = run until Ctrl+C)",
    )
    return parser.parse_args()


def new_buffers() -> dict[str, list]:
    return {
        "grid": [], "player": [], "entities": [],
        "action": [], "reward": [], "done": [],
    }


def save_episode(buffers: dict, path: Path, reason: str,
                 ep_reward: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({
        "episode_reason": reason,
        "episode_reward": round(float(ep_reward), 3),
        "steps": int(len(buffers["reward"])),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })
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
    return path


def main():
    args = parse_args()
    config = load_config(args.config)
    if not config.env.human_mode:
        raise SystemExit(
            f"{args.config}: env.human_mode must be true for recording."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else DEFAULT_OUT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    env = IsaacEnv(config)
    print(f"Recording demos to {out_dir}")
    print("Play! Episodes auto-restart on death / room clear / timeout.")
    print("Ctrl+C here to stop.\n")

    def action_of(state) -> tuple:
        raw = state.raw.get("action") if state.raw else None
        if isinstance(raw, dict):
            return tuple(int(raw.get(k, 0) or 0) for k in ACTION_KEYS)
        return ZERO_ACTION

    # reset() connects, sends configure (human_mode included), forces a fresh
    # episode and returns its observation.
    current_obs, _info = env.reset()
    current_episode_id = env._last_episode_id
    pending_action = action_of(env.last_state)

    buffers = new_buffers()
    ep_reward = 0.0
    episode_idx = 0
    total_steps = 0
    started_at = time.monotonic()
    reasons: dict[str, int] = {}

    def start_episode():
        """Reset per-episode machinery; the caller seeds the first obs."""
        nonlocal buffers, ep_reward
        buffers = new_buffers()
        ep_reward = 0.0
        env.reward_shaper.reset()
        env._grid_stack.clear()
        env.step_count = 0

    def finalize(reason: str):
        nonlocal episode_idx
        if len(buffers["reward"]) == 0:
            return
        path = save_episode(
            buffers, out_dir / f"episode_{episode_idx + 1:04d}.npz",
            reason, ep_reward,
        )
        print(
            f"[EP {episode_idx + 1}] {reason:<16} "
            f"steps={len(buffers['reward']):>4} reward={ep_reward:+7.2f} "
            f"-> {path.name}"
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        episode_idx += 1

    try:
        while True:
            if args.max_episodes and episode_idx >= args.max_episodes:
                break
            state = env._receive_state()

            if state.episode_id != current_episode_id:
                # First frame of a new episode (Lua auto-restarts). If we got
                # here without a terminal frame, a packet was dropped mid-
                # episode: salvage what was buffered.
                finalize("dropped_boundary" if buffers["reward"] else "skipped")
                current_episode_id = state.episode_id
                start_episode()
                current_obs = env._state_to_obs(state)
                pending_action = action_of(state)
                continue

            # Same episode: close the (prev_obs, prev_action) -> this state
            # transition, exactly like env.step() does.
            reward = env.reward_shaper.compute(state, pending_action)
            total_steps += 1
            ep_reward += reward
            terminated = bool(state.terminal)

            buffers["grid"].append(current_obs["grid"])
            buffers["player"].append(current_obs["player"])
            buffers["entities"].append(current_obs["entities"])
            buffers["action"].append(pending_action)
            buffers["reward"].append(reward)
            buffers["done"].append(terminated)

            pending_action = action_of(state)

            if terminated:
                reason = str(state.terminal_reason or "unknown")
                finalize(reason)
                start_episode()
                current_obs = None
            else:
                current_obs = env._state_to_obs(state)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        finalize("interrupted")
        env.close()

    minutes = (time.monotonic() - started_at) / 60.0
    print(f"\nDone: {episode_idx} episodes, {total_steps} steps "
          f"in {minutes:.1f} min -> {out_dir}")
    print(f"Outcomes: {reasons}")


if __name__ == "__main__":
    main()
