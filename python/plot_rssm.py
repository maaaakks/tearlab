"""Plot RSSM offline-training metrics from a run's metrics.jsonl.

Usage:
    .venv/Scripts/python.exe python/plot_rssm.py                # latest run
    .venv/Scripts/python.exe python\train_rssm_offline.py ... puis:
    .venv/Scripts/python.exe python/plot_rssm.py --run data/worldmodel/<stamp>
    .venv/Scripts/python.exe python/plot_rssm.py --show         # fenêtre live
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "worldmodel"

# Chance baselines measured on the session1 human demos (predict-the-mean).
BASELINES = {
    "val_grid_recon": 0.00480,
    "val_player_recon": 0.014524,
    "val_reward": 0.00066,
}

PANELS = [
    ("total", ["val_total", "train_total"], "loss totale"),
    ("reconstruction", ["val_grid_recon", "train_grid_recon"], "grid MSE"),
    ("reconstruction2", ["val_player_recon", "train_player_recon"],
     "player MSE"),
    ("kl", ["val_kl", "train_kl"], "KL (nats)"),
    ("reward", ["val_reward", "train_reward"], "reward MSE"),
    ("continue", ["val_continue", "train_continue"], "continue BCE"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None,
                        help="run dir containing metrics.jsonl "
                             "(default: most recent under data/worldmodel)")
    parser.add_argument("--out", default=None, help="PNG output path")
    parser.add_argument("--show", action="store_true",
                        help="also open an interactive window")
    return parser.parse_args()


def resolve_run(run: str | None) -> Path:
    if run:
        path = Path(run)
        if path.name == "metrics.jsonl":
            return path.parent
        return path
    candidates = sorted(
        (d for d in DEFAULT_ROOT.glob("*") if (d / "metrics.jsonl").exists()),
        key=lambda d: (d / "metrics.jsonl").stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(
            f"No metrics.jsonl under {DEFAULT_ROOT}. Train first via "
            "train_rssm_offline.py."
        )
    return candidates[-1]


def main():
    args = parse_args()
    if args.show:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = resolve_run(args.run)
    metrics_path = run_dir / "metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    steps = [r["step"] for r in rows]
    print(f"{len(rows)} log entries from {metrics_path}")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle(f"RSSM world model — {run_dir.name}")

    for ax, (_name, keys, label) in zip(axes.flat, PANELS):
        for key in keys:
            series = [(s, r[key]) for s, r in zip(steps, rows) if key in r]
            if not series:
                continue
            xs, ys = zip(*series)
            ax.plot(xs, ys, marker=".", markersize=3, label=key)
        # draw chance baselines when one of the plotted keys has one
        for key in keys:
            if key in BASELINES and BASELINES[key] > 0:
                ax.axhline(BASELINES[key], color="red", linestyle="--",
                           linewidth=1,
                           label=f"hasard ({key})")
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    out_path = Path(args.out) if args.out else run_dir / "curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
