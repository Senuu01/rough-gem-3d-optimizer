"""Safe subprocess execution with streamed, logged output.

Commands are always passed as argument lists (never ``shell=True``), so paths
containing spaces or shell metacharacters are handled correctly.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger("process")


@dataclass
class ProcessResult:
    """Outcome of an external command."""

    args: list[str]
    returncode: int
    duration_s: float
    output_tail: list[str]
    log_file: Path | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def command_line(self) -> str:
        return shlex.join(self.args)


def run_command(
    args: Sequence[str | Path],
    log_file: Path | None = None,
    cwd: Path | None = None,
    on_line: Callable[[str], None] | None = None,
    tail_lines: int = 60,
    timeout_s: float | None = None,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    """Run ``args``, streaming merged stdout/stderr to ``log_file``.

    The exit code is returned rather than raised so callers can translate
    failures into domain-specific, user-friendly errors.

    Args:
        args: Executable followed by its arguments.
        log_file: File that receives the full command output (appended).
        cwd: Working directory for the child process.
        on_line: Optional callback invoked for every output line.
        tail_lines: Number of trailing output lines kept in memory.
        timeout_s: Kill the process after this many seconds.
        env: Extra environment variables layered over the current environment.
    """
    str_args = [str(a) for a in args]
    tail: deque[str] = deque(maxlen=tail_lines)
    start = time.monotonic()
    logger.info("Running: %s", shlex.join(str_args))

    log_handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "a", encoding="utf-8", errors="replace")
        log_handle.write(f"$ {shlex.join(str_args)}\n")
        log_handle.flush()

    try:
        proc = subprocess.Popen(
            str_args,
            cwd=cwd,
            env={**os.environ, **env} if env else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError:
        if log_handle:
            log_handle.write(f"ERROR: executable not found: {str_args[0]}\n")
            log_handle.close()
        return ProcessResult(str_args, 127, 0.0, [f"executable not found: {str_args[0]}"], log_file)

    timer: threading.Timer | None = None
    if timeout_s is not None:
        timer = threading.Timer(timeout_s, proc.kill)
        timer.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if log_handle:
                log_handle.write(line + "\n")
                log_handle.flush()
            if on_line:
                on_line(line)
        returncode = proc.wait()
    finally:
        if timer:
            timer.cancel()
        if log_handle:
            log_handle.write(f"[exit code {proc.returncode}]\n\n")
            log_handle.close()

    duration = time.monotonic() - start
    logger.info("Exit code %d after %.1fs: %s", returncode, duration, str_args[0])
    return ProcessResult(str_args, returncode, duration, list(tail), log_file)


def capture_output(
    args: Sequence[str | Path], timeout_s: float = 30.0, env: dict[str, str] | None = None
) -> tuple[int, str]:
    """Run a short command and return ``(returncode, combined_output)``."""
    try:
        completed = subprocess.run(
            [str(a) for a in args],
            env={**os.environ, **env} if env else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return 127, f"executable not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout_s}s: {args[0]}"
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")
