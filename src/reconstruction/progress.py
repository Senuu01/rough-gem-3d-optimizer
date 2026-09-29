"""Stage progress tracking persisted to ``status.json``.

The reconstruction runs in a separate process; the Streamlit UI polls this
file to show which stage is running, so the interface never appears frozen.
Writes are atomic (temp file + rename) so readers never see partial JSON.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Any

STAGES: list[tuple[str, str]] = [
    ("prepare", "Preparing images"),
    ("quality", "Checking image quality"),
    ("masks", "Creating masks"),
    ("features", "Extracting features"),
    ("matching", "Matching images"),
    ("sparse", "Sparse reconstruction"),
    ("dense", "Dense reconstruction"),
    ("fusion", "Fusing point cloud"),
    ("mesh", "Generating mesh"),
    ("cleaning", "Cleaning mesh"),
    ("complete", "Complete"),
]
STAGE_LABELS = dict(STAGES)


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_status(path: Path) -> dict | None:
    """Return the parsed status file, or ``None`` if absent/unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def process_alive(pid: int | None) -> bool:
    """True if a process with ``pid`` exists (POSIX & Windows)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ProgressTracker:
    """Records per-stage status and writes it to ``status.json`` on every change."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self._t0: dict[str, float] = {}
        self.data: dict[str, Any] = {
            "run_id": run_id,
            "state": "running",
            "pid": os.getpid(),
            "current_stage": None,
            "started_at": _now(),
            "updated_at": _now(),
            "warnings": [],
            "error": None,
            "stages": {
                key: {"label": label, "status": "pending", "message": "", "duration_s": None}
                for key, label in STAGES
            },
        }
        self._flush()

    def _flush(self) -> None:
        self.data["updated_at"] = _now()
        write_json_atomic(self.path, self.data)

    def _stage(self, key: str) -> dict:
        return self.data["stages"][key]

    def start(self, key: str, message: str = "") -> None:
        self._t0[key] = time.monotonic()
        self.data["current_stage"] = key
        self._stage(key).update(status="running", message=message)
        self._flush()

    def _elapsed(self, key: str) -> float | None:
        return round(time.monotonic() - self._t0[key], 2) if key in self._t0 else None

    def done(self, key: str, message: str = "") -> None:
        self._stage(key).update(status="done", message=message, duration_s=self._elapsed(key))
        self._flush()

    def skip(self, key: str, reason: str) -> None:
        self._stage(key).update(status="skipped", message=reason, duration_s=self._elapsed(key))
        self._flush()

    def fail(self, key: str, message: str, log_file: str | None = None, fatal: bool = True) -> None:
        """Mark a stage failed. Non-fatal failures (a fallback follows) do not set the run error."""
        self._stage(key).update(status="failed", message=message, duration_s=self._elapsed(key))
        if fatal:
            self.data["error"] = {"stage": STAGE_LABELS.get(key, key), "message": message, "log": log_file}
        self._flush()

    def warn(self, message: str) -> None:
        self.data["warnings"].append(message)
        self._flush()

    def finish(self, state: str) -> None:
        self.data["state"] = state
        self.data["finished_at"] = _now()
        if state == "completed":
            self.data["stages"]["complete"].update(status="done")
            self.data["current_stage"] = "complete"
        self._flush()

    def stage_durations(self) -> dict[str, float | None]:
        return {key: stage["duration_s"] for key, stage in self.data["stages"].items()}
