"""Per-run directory layout.

Every reconstruction attempt (successful or failed) gets its own directory
under ``runs/`` so experiments are reproducible and comparable::

    runs/2026-09-29_001/
        input/          original uploads, untouched
        frames/         images fed to COLMAP (converted photos + sampled video frames)
        masks/          COLMAP masks (Milestone 2)
        database.db     COLMAP feature/match database
        sparse/         COLMAP sparse models (0/, 1/, ...)
        dense/          undistorted images, depth maps, fused.ply, meshes
        output/         final deliverables (raw_mesh.ply, exports)
        logs/           pipeline.log and one log per COLMAP stage
        run_config.json parameters + software versions
        metrics.json    statistics and stage timings
        status.json     live progress, polled by the UI
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

_RUN_ID_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{3,})$")


@dataclass(frozen=True)
class RunPaths:
    """All well-known paths inside a single run directory."""

    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def database_path(self) -> Path:
        return self.root / "database.db"

    @property
    def sparse_dir(self) -> Path:
        return self.root / "sparse"

    @property
    def dense_dir(self) -> Path:
        return self.root / "dense"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def run_config_path(self) -> Path:
        return self.root / "run_config.json"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def ingestion_path(self) -> Path:
        return self.root / "ingestion.json"

    @property
    def fused_ply(self) -> Path:
        return self.dense_dir / "fused.ply"

    @property
    def sparse_points_ply(self) -> Path:
        return self.output_dir / "sparse_points.ply"

    @property
    def raw_mesh_ply(self) -> Path:
        return self.output_dir / "raw_mesh.ply"

    def create(self) -> "RunPaths":
        """Create all sub-directories (idempotent)."""
        for directory in (
            self.input_dir,
            self.frames_dir,
            self.masks_dir,
            self.sparse_dir,
            self.dense_dir,
            self.output_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def next_run_id(runs_dir: Path, today: _dt.date | None = None) -> str:
    """Return the next free run id for ``today``, e.g. ``2026-09-29_003``."""
    today = today or _dt.date.today()
    date_str = today.isoformat()
    highest = 0
    if runs_dir.is_dir():
        for child in runs_dir.iterdir():
            match = _RUN_ID_RE.match(child.name)
            if child.is_dir() and match and match.group(1) == date_str:
                highest = max(highest, int(match.group(2)))
    return f"{date_str}_{highest + 1:03d}"


def create_run(runs_dir: Path, today: _dt.date | None = None) -> RunPaths:
    """Create a new, uniquely numbered run directory and return its paths."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = next_run_id(runs_dir, today)
        root = runs_dir / run_id
        try:
            root.mkdir()
        except FileExistsError:  # created concurrently; try the next number
            continue
        return RunPaths(root).create()


def list_runs(runs_dir: Path) -> list[RunPaths]:
    """Return existing runs, newest first."""
    if not runs_dir.is_dir():
        return []
    runs = [RunPaths(p) for p in runs_dir.iterdir() if p.is_dir() and _RUN_ID_RE.match(p.name)]
    return sorted(runs, key=lambda r: r.run_id, reverse=True)
