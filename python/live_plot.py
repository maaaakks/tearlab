"""Live side-graph of training reward vs steps.

Reads checkpoints/live/<run_id>/metrics.jsonl (written by LiveCheckpointManager)
and plots X = global_steps, Y = rolling_reward / last_episode_reward in a
tkinter window. Pure stdlib: runs next to the trainer with zero new deps.

Usage:
    python live_plot.py                 # follow latest run (switches on resume)
    python live_plot.py --run <run_id>  # pin a specific run
    python live_plot.py --win-rate      # also plot rolling win rate (0-1, right axis)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import tkinter as tk
except ImportError:  # e.g. a minimal Linux python without tk
    tk = None

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_ROOT = REPO_ROOT / "checkpoints" / "live"

BG = "#14161a"
FG = "#d8dee9"
GRID = "#2e3440"
ROLLING_COLOR = "#88c0d0"
EPISODE_COLOR = "#4c566a"
WINRATE_COLOR = "#a3be8c"
ACCENT = "#ebcb8b"

MAX_POINTS = 4000  # downsample beyond this to keep redraws cheap


def resolve_run_dir(run: str | None) -> Path | None:
    """run_id directory, or the one pointed at by latest.json."""
    if run:
        candidate = LIVE_ROOT / run
        return candidate if candidate.is_dir() else None
    pointer = LIVE_ROOT / "latest.json"
    if not pointer.is_file():
        return None
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    candidate = LIVE_ROOT / str(data.get("path", "")).split("/")[0]
    return candidate if candidate.is_dir() else None


class MetricsTail:
    """Incremental JSONL reader that survives file truncation/rotation."""

    def __init__(self, path: Path):
        self.path = path
        self._offset = 0
        self._inode = None

    def poll(self) -> list[dict]:
        rows: list[dict] = []
        try:
            stat = self.path.stat()
        except OSError:
            return rows
        if self._inode is not None and (
            stat.st_ino != self._inode or stat.st_size < self._offset
        ):
            self._offset = 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # partially flushed line; picked up next poll
                self._offset = handle.tell()
        except OSError:
            return rows
        self._inode = stat.st_ino
        return rows


class LivePlot(tk.Tk):
    margin_left, margin_right = 64, 16
    margin_top, margin_bottom = 28, 40

    def __init__(self, run: str | None, poll_seconds: float, show_win_rate: bool):
        super().__init__()
        self.title("live training — binding-of-ai")
        self.geometry("900x520")
        self.configure(bg=BG)
        self.poll_seconds = poll_seconds
        self.show_win_rate = show_win_rate
        self.run_arg = run

        self.run_dir: Path | None = None
        self.tail: MetricsTail | None = None

        # series: parallel lists of (steps, value)
        self.steps: list[int] = []
        self.rolling: list[float] = []
        self.last_ep: list[float] = []
        self.win_rate: list[float] = []
        self.last_meta: dict = {}

        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

        self.after(500, self._poll_loop)

    def _poll_loop(self):
        try:
            self._ensure_tail()
            if self.tail is not None:
                for row in self.tail.poll():
                    self._absorb(row)
                self.redraw()
        except Exception:
            pass  # never let a transient read kill the graph mid-training
        self.after(int(self.poll_seconds * 1000), self._poll_loop)

    def _ensure_tail(self):
        target = resolve_run_dir(self.run_arg)
        if target is None:
            return
        if target == self.run_dir:
            return
        metrics_path = target / "metrics.jsonl"
        if not metrics_path.exists():
            return
        self.run_dir = target
        self.tail = MetricsTail(metrics_path)
        self.steps.clear()
        self.rolling.clear()
        self.last_ep.clear()
        self.win_rate.clear()

    def _absorb(self, row: dict):
        step = row.get("global_steps")
        if step is None:
            return
        rolling = row.get("rolling_reward")
        last_ep = row.get("last_episode_reward")
        win_rate = row.get("rolling_win_rate")
        self.steps.append(int(step))
        self.rolling.append(float(rolling) if rolling is not None else float("nan"))
        self.last_ep.append(float(last_ep) if last_ep is not None else float("nan"))
        self.win_rate.append(float(win_rate) if win_rate is not None else float("nan"))
        self.last_meta = {
            key: row[key]
            for key in (
                "steps_per_second",
                "last_episode_reason",
                "last_episode_length",
                "update_count",
                "explained_variance",
            )
            if key in row
        }
        if len(self.steps) > MAX_POINTS:
            stride = 2
            self.steps = self.steps[::stride]
            self.rolling = self.rolling[::stride]
            self.last_ep = self.last_ep[::stride]
            self.win_rate = self.win_rate[::stride]

    @property
    def _plot_rect(self):
        w = max(self.winfo_width(), 100)
        h = max(self.winfo_height(), 100)
        return (
            self.margin_left,
            self.margin_top,
            w - self.margin_right,
            h - self.margin_bottom,
        )

    @staticmethod
    def _nice_max(value: float) -> float:
        if value <= 0:
            return 1.0
        exponent = 10 ** int(f"{value:.0e}".split("e")[1])
        for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
            limit = factor * exponent
            if limit >= value:
                return limit
        return 10.0 * exponent

    @staticmethod
    def _clean(series: list[float]) -> tuple[list[int], list[float]]:
        idxs = [i for i, v in enumerate(series) if v == v]  # drop NaNs
        return idxs, [series[i] for i in idxs]

    def redraw(self):
        canvas = self.canvas
        canvas.delete("all")
        x0, y0, x1, y1 = self._plot_rect

        title_run = self.run_dir.name if self.run_dir else "waiting for a run…"
        n_eps = sum(1 for v in self.last_ep if v == v)
        canvas.create_text(
            x0, 12, anchor="w", fill=FG,
            font=("TkDefaultFont", 11, "bold"),
            text=f"{title_run}   ({len(self.steps)} updates, {n_eps} episodes)",
        )

        if len(self.steps) < 2:
            if self.tail is None:
                status = f"aucun run avec metrics.jsonl sous {LIVE_ROOT}"
            else:
                try:
                    size = self.tail.path.stat().st_size
                    status = f"{self.tail.path} ({size} octets)"
                except OSError:
                    status = f"introuvable : {self.tail.path}"
            canvas.create_text(
                (x0 + x1) // 2, (y0 + y1) // 2, fill="#616e88",
                text="en attente de metrics.jsonl…\n" + status,
                font=("TkDefaultFont", 12),
                justify="center",
            )
            return

        max_step = max(self.steps)
        x_scale = (x1 - x0) / max(1, max_step)

        _, clean_rolling = self._clean(self.rolling)
        _, clean_last = self._clean(self.last_ep)
        candidates = [abs(v) for v in clean_rolling + clean_last if v != 0]
        y_abs_max = self._nice_max(max(candidates) * 1.1 if candidates else 1.0)

        def px(step):
            return x0 + step * x_scale

        def py(value):
            span = y1 - y0
            return y1 - (value + y_abs_max) / (2 * y_abs_max) * span

        # horizontal gridlines every nice tick
        tick_count = 8
        for i in range(tick_count + 1):
            frac = i / tick_count
            gy = y0 + frac * (y1 - y0)
            value = -y_abs_max + 2 * y_abs_max * frac
            canvas.create_line(x0, gy, x1, gy, fill=GRID)
            canvas.create_text(
                x0 - 6, gy, anchor="e", fill=FG,
                text=f"{value:+.0f}" if abs(y_abs_max) >= 10 else f"{value:+.1f}",
                font=("TkDefaultFont", 9),
            )
        # vertical gridlines
        for i in range(6):
            frac = i / 5
            gx = x0 + frac * (x1 - x0)
            canvas.create_line(gx, y0, gx, y1, fill=GRID)
            label = int(max_step * frac)
            text = f"{label // 1000}k" if label >= 1000 else str(label)
            canvas.create_text(
                gx, y1 + 14, fill=FG, text=text, font=("TkDefaultFont", 9)
            )

        canvas.create_text(
            16, (y0 + y1) // 2, fill=FG, angle=90,
            text="reward", font=("TkDefaultFont", 9),
        )
        canvas.create_text(
            (x0 + x1) // 2, self.winfo_height() - 10, fill=FG,
            text="steps", font=("TkDefaultFont", 9),
        )

        def draw_series(values, color, width):
            pts = [
                (px(s), py(v))
                for s, v in zip(self.steps, values)
                if v == v
            ]
            if len(pts) >= 2:
                flat = [c for point in pts for c in point]
                canvas.create_line(*flat, fill=color, width=width)

        draw_series(self.last_ep, EPISODE_COLOR, 1)
        draw_series(self.rolling, ROLLING_COLOR, 2)

        if self.show_win_rate and any(v == v for v in self.win_rate):
            wr_pts = []
            for s, v in zip(self.steps, self.win_rate):
                if v == v:
                    wy = y1 - v * (y1 - y0)
                    wr_pts.append((px(s), wy))
            if len(wr_pts) >= 2:
                flat = [c for point in wr_pts for c in point]
                canvas.create_line(*flat, fill=WINRATE_COLOR, width=1, dash=(4, 3))
            current_wr = next((v for v in reversed(self.win_rate) if v == v), None)
            if current_wr is not None:
                canvas.create_text(
                    x1, y0 - 8, anchor="e", fill=WINRATE_COLOR,
                    text=f"win rate {current_wr:.0%}", font=("TkDefaultFont", 9),
                )

        legend_x = x0 + 8
        canvas.create_rectangle(
            legend_x, y0 + 8, legend_x + 18, y0 + 14,
            fill=ROLLING_COLOR, outline="",
        )
        canvas.create_text(
            legend_x + 24, y0 + 11, anchor="w", fill=FG,
            text="rolling reward (20 ep)", font=("TkDefaultFont", 9),
        )
        canvas.create_rectangle(
            legend_x, y0 + 24, legend_x + 18, y0 + 30,
            fill=EPISODE_COLOR, outline="",
        )
        canvas.create_text(
            legend_x + 24, y0 + 27, anchor="w", fill=FG,
            text="last episode reward", font=("TkDefaultFont", 9),
        )

        meta_parts = [f"step {self.steps[-1]:,}"]
        sps = self.last_meta.get("steps_per_second")
        if sps:
            meta_parts.append(f"{sps:.0f} steps/s")
        reason = self.last_meta.get("last_episode_reason")
        if reason:
            meta_parts.append(str(reason).replace("_", " "))
        ep_len = self.last_meta.get("last_episode_length")
        if ep_len:
            meta_parts.append(f"ep_len {ep_len}")
        canvas.create_text(
            x1, 12, anchor="e", fill=ACCENT, text="  ·  ".join(meta_parts),
            font=("TkDefaultFont", 10, "bold"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="run_id under checkpoints/live/")
    parser.add_argument("--interval", type=float, default=1.0, help="poll seconds")
    parser.add_argument("--win-rate", action="store_true", help="overlay win rate")
    args = parser.parse_args()

    if tk is None:
        print(
            "tkinter is unavailable: install the tk package for this python "
            "(on Windows it ships with the standard installer).",
            file=sys.stderr,
        )
        return 1

    app = LivePlot(args.run, args.interval, args.win_rate)
    app.mainloop()


if __name__ == "__main__":
    sys.exit(main())
