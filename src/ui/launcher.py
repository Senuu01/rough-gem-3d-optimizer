"""Launch and monitor the reconstruction as a detached background process."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import streamlit as st

from src.reconstruction.progress import process_alive, read_status, write_json_atomic
from src.utils.config import PROJECT_ROOT
from src.utils.paths import RunPaths


@st.cache_resource
def _processes() -> dict[str, subprocess.Popen]:
    """Popen handles owned by this Streamlit server (lets us reap finished children)."""
    return {}


def is_running(run: RunPaths) -> bool:
    """True while the run's pipeline process is alive."""
    proc = _processes().get(run.run_id)
    if proc is not None:
        return proc.poll() is None
    status = read_status(run.status_path)
    return bool(status and status.get("state") in ("starting", "running") and process_alive(status.get("pid")))


def launch(run: RunPaths) -> None:
    """Start ``python -m src.cli reconstruct <run>`` in the background."""
    run.logs_dir.mkdir(parents=True, exist_ok=True)
    with open(run.logs_dir / "runner_stdout.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.cli", "reconstruct", str(run.root)],
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
    _processes()[run.run_id] = proc
    write_json_atomic(run.status_path, {"run_id": run.run_id, "state": "starting", "pid": proc.pid, "stages": {}})


def cancel(run: RunPaths) -> None:
    """Terminate the pipeline and its COLMAP child processes."""
    proc = _processes().get(run.run_id)
    status = read_status(run.status_path) or {}
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
