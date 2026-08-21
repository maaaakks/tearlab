"""Durable checkpoint paths and atomic saves for the live PPO trainer."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import torch

from utils import get_checkpoint_dir, get_repo_root


CHECKPOINT_FORMAT_VERSION = 2


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-._") or "training"


class LiveCheckpointManager:
    """Owns one precise per-run folder under ``checkpoints/live``."""

    def __init__(
        self,
        run_name: str,
        config_path: str | None = None,
        resume: str | None = None,
        root: Path | None = None,
    ):
        self.root = (root or (get_checkpoint_dir() / "live")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.resume_path = self.resolve_resume_path(resume) if resume else None

        if self.resume_path and self.resume_path.parent.parent == self.root:
            self.run_dir = self.resume_path.parent
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_resumed" if self.resume_path else ""
            self.run_dir = self.root / f"{timestamp}_{_safe_name(run_name)}{suffix}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = self.run_dir.name
        self.config_path = self._portable_config_path(config_path)

    def _portable_config_path(self, config_path: str | None) -> str | None:
        if not config_path:
            return None
        path = Path(config_path).expanduser().resolve()
        try:
            return path.relative_to(get_repo_root()).as_posix()
        except ValueError:
            return str(path)

    def resolve_resume_path(self, resume: str) -> Path:
        return self.resolve_checkpoint(resume, self.root)

    @classmethod
    def resolve_checkpoint(
        cls,
        resume: str,
        root: Path | None = None,
    ) -> Path:
        checkpoint_root = (root or (get_checkpoint_dir() / "live")).resolve()
        if resume in {"latest", "latest-any", "latest-compatible"}:
            latest = cls.find_latest(checkpoint_root)
            if latest is None:
                raise FileNotFoundError(
                    f"No live checkpoint found under {checkpoint_root}"
                )
            return latest

        raw = Path(resume).expanduser()
        candidates = [
            raw,
            get_repo_root() / raw,
            checkpoint_root / raw,
        ]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                latest = cls.find_latest(candidate)
                if latest is not None:
                    return latest

        raise FileNotFoundError(
            f"Live checkpoint not found: {resume}. Expected a .pt file, a run "
            "folder, or one of latest/latest-any/latest-compatible under "
            f"{checkpoint_root}."
        )

    @staticmethod
    def find_latest(root: Path) -> Path | None:
        pointer = root / "latest.json"
        if pointer.is_file():
            try:
                data = json.loads(pointer.read_text(encoding="utf-8"))
                candidate = (root / data["path"]).resolve()
                if candidate.is_file() and root.resolve() in candidate.parents:
                    return candidate
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                pass

        checkpoints = list(root.rglob("*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)

    def save(self, payload: dict, label: str, steps: int) -> Path:
        """Atomically write one checkpoint and update the root latest pointer."""
        safe_label = _safe_name(label)
        path = self.run_dir / f"{safe_label}_{steps:09d}.pt"
        temp_path = path.with_suffix(".pt.tmp")
        torch.save(payload, temp_path)
        temp_path.replace(path)

        pointer_data = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "path": path.relative_to(self.root).as_posix(),
            "run_id": self.run_id,
            "steps": int(steps),
            "saved_at": payload.get("saved_at"),
        }
        self._write_json_atomic(self.run_dir / "latest.json", pointer_data)
        self._write_json_atomic(self.root / "latest.json", pointer_data)
        return path

    def append_metrics(self, metrics: dict) -> None:
        """Persist one update/evaluation metric snapshot as JSON Lines."""
        path = self.run_dir / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    @staticmethod
    def _write_json_atomic(path: Path, data: dict) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)
