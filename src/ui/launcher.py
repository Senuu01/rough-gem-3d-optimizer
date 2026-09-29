"""Launch and monitor long-running jobs (reconstruction, masking) as detached processes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.preprocessing.masking import masking_status_path
from src.reconstruction.progress import process_alive, read_status, write_json_atomic
from src.utils.config import PROJECT_ROOT
from src.utils.paths import RunPaths

RECONSTRUCT = "reconstruct"
MASK = "mask"


@st.cache_resource
def _processes() -> dict[str, subprocess.Popen]:
    """Popen handles owned by this Streamlit server (lets us reap finished children)."""
    return {}


def _status_path(run: RunPaths, job: str) -> Path:
    return run.status_path if job == RECONSTRUCT else masking_status_path(run)


def is_running(run: RunPaths, job: str = RECONSTRUCT) -> bool:
    """True while the run's ``job`` process is alive."""
    proc = _processes().get(f"{run.run_id}:{job}")
    if proc is not None:
        return proc.poll() is None
    status = read_status(_status_path(run, job))
    return bool(status and status.get("state") in ("starting", "running") and process_alive(status.get("pid")))


def launch(run: RunPaths, job: str = RECONSTRUCT) -> None:
    """Start ``python -m src.cli <job> <run>`` in the background."""
    run.logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = _status_path(run, job)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log_name = "runner_stdout.log" if job == RECONSTRUCT else "masking_stdout.log"
    with open(run.logs_dir / log_name, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.cli", job, str(run.root)],
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
    _processes()[f"{run.run_id}:{job}"] = proc
    write_json_atomic(status_path, {"run_id": run.run_id, "state": "starting", "pid": proc.pid, "stages": {}})


def cancel(run: RunPaths, job: str = RECONSTRUCT) -> None:
    """Terminate the job and its child processes."""
    proc = _processes().get(f"{run.run_id}:{job}")
    status = read_status(_status_path(run, job)) or {}
    pid = proc.pid if proc else status.get("pid")
    if not pid:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
