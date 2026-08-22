"""Train the RSSM world model offline from recorded npz episodes.

Consumes the episode format written by record_human.py (and, later, by the
live transition recorder): per-episode npz with grid/player/entities/action/
reward/done arrays. This runs fully off-line — no Isaac instance required —
so the stage-2 world model can be validated on demos while live training
keeps running.

Usage:
    .venv/Scripts/python.exe python/train_rssm_offline.py \
        --data data/human/<session> --train-steps 20000
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rssm import WorldModel

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True,
                        help="npz episode files and/or directories of them")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--free-nats", type=float, default=3.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def collect_episode_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("episode_*.npz")))
        elif path.suffix == ".npz":
            files.append(path)
        else:
            raise SystemExit(f"--data entry is not a dir or npz file: {p}")
    if not files:
        raise SystemExit(
            "No episode_*.npz found. Record demos first via "
            "record_human.py (see its docstring for usage)."
        )
    return files


class EpisodeStore:
    """In-memory episode table + window sampler that never crosses a
    terminal boundary inside a sequence."""

    def __init__(self, files: list[Path], seq_len: int):
        self.seq_len = seq_len
        self.episodes: list[dict] = []
        skipped = 0
        for f in files:
            data = np.load(f, allow_pickle=False)
            steps = len(data["reward"])
            if steps < seq_len:
                skipped += 1
                continue
            self.episodes.append({k: data[k] for k in (
                "grid", "player", "entities", "action", "reward", "done"
            )})
        if not self.episodes:
            raise SystemExit(
                f"All {len(files)} episodes are shorter than "
                f"seq_len={seq_len}."
            )
        if skipped:
            print(f"Skipped {skipped} episodes shorter than {seq_len} steps")
        total = sum(len(ep["reward"]) for ep in self.episodes)
        print(f"{len(self.episodes)} episodes, {total} transitions loaded")

    def sample_windows(self, indices) -> dict[str, np.ndarray]:
        B = len(indices)
        L = self.seq_len
        batch = {
            "grid": [], "player": [], "entities": [],
            "action": [], "reward": [], "done": [],
        }
        rng = np.random.default_rng()
        for _ in range(B):
            for _try in range(50):
                ep = self.episodes[rng.integers(len(self.episodes))]
                start = int(rng.integers(0, len(ep["reward"]) - L + 1))
                # done[t] terminates at t; require no interior terminal so
                # every consecutive pair in the window stays in-episode.
                if not ep["done"][start:start + L - 1].any():
                    break
            else:
                # Degenerate episode full of terminals; fall back to head.
                ep = self.episodes[0]
                start = 0
            sl = slice(start, start + L)
            batch["grid"].append(ep["grid"][sl])
            batch["player"].append(ep["player"][sl])
            batch["entities"].append(ep["entities"][sl])
            batch["action"].append(ep["action"][sl])
            batch["reward"].append(ep["reward"][sl])
            batch["done"].append(ep["done"][sl])
        return {
            k: np.stack(v).astype(
                np.bool_ if k == "done" else
                np.int64 if k == "action" else np.float32
            )
            for k, v in batch.items()
        }


def to_tensors(numpy_batch: dict, device) -> dict:
    tensors = {}
    for key, value in numpy_batch.items():
        t = torch.as_tensor(value, device=device)
        tensors[key] = t
    return tensors


@torch.no_grad()
def evaluate(model, store, device, batch_size: int, steps: int = 8) -> dict:
    model.eval()
    totals = {}
    for i in range(steps):
        idx = range(i * batch_size, (i + 1) * batch_size)
        losses = model.compute_losses(to_tensors(store.sample_windows(idx), device))
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
    model.train()
    return {k: v / steps for k, v in totals.items()}


def main():
    args = parse_args()
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "worldmodel" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_episode_files(args.data)
    store = EpisodeStore(files, args.seq_len)

    first = store.episodes[0]
    grid_shape = first["grid"].shape[1:]
    n_entities, entity_dim = first["entities"].shape[1], first["entities"].shape[2]
    action_nvec = [9, 9, 2, 2, 2]

    model = WorldModel(
        action_nvec=action_nvec,
        grid_channels=grid_shape[0],
        grid_hw=grid_shape[1:],
        player_features=first["player"].shape[1],
        max_entities=n_entities,
        entity_dim=entity_dim,
        free_nats=args.free_nats,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_count = max(1, int(len(store.episodes) * args.val_fraction))
    train_store = store
    print(f"device={device} seq={args.seq_len} batch={args.batch_size}")
    print(f"out_dir={out_dir}")

    started = time.monotonic()
    history = []
    for step in range(1, args.train_steps + 1):
        window = to_tensors(train_store.sample_windows(range(args.batch_size)), device)
        losses = model.compute_losses(window)
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()

        if step % args.log_every == 0 or step == args.train_steps:
            val = evaluate(model, store, device, args.batch_size)
            entry = {
                "type": "rssm_train",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": step,
                **{f"train_{k}": round(float(v.item() if hasattr(v, 'item') else v), 5)
                   for k, v in losses.items()},
                **{f"val_{k}": round(v, 5) for k, v in val.items()},
                "minutes": round((time.monotonic() - started) / 60, 2),
            }
            history.append(entry)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(
                f"[{step:>6}] total={float(losses['total']):7.4f} "
                f"kl={float(losses['kl']):6.3f} "
                f"grid={float(losses['grid_recon']):6.4f} "
                f"player={float(losses['player_recon']):7.5f} "
                f"| val_total={val['total']:7.4f} "
                f"val_grid={val['grid_recon']:6.4f}"
            )

        if step % args.save_every == 0 or step == args.train_steps:
            payload = {
                "format_version": 1,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "step": step,
                "state_dict": model.state_dict(),
                "config": {
                    "action_nvec": action_nvec,
                    "grid_channels": grid_shape[0],
                    "grid_hw": list(grid_shape[1:]),
                    "player_features": first["player"].shape[1],
                    "max_entities": n_entities,
                    "entity_dim": entity_dim,
                    "free_nats": args.free_nats,
                    "seq_len": args.seq_len,
                },
            }
            torch.save(payload, out_dir / f"worldmodel_{step:08d}.pt")

    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
