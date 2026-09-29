"""Reconstruction statistics.

Sparse statistics are computed from COLMAP's TXT model export
(``cameras.txt``, ``images.txt``, ``points3D.txt``) rather than by scraping
log output, which changes between COLMAP versions. The ``ERROR`` column of
``points3D.txt`` is the mean reprojection error of each 3D point, so the
reported mean/median are over 3D points.
"""

from __future__ import annotations

import sqlite3
import statistics as _stats
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path


class StatisticsError(RuntimeError):
    """Raised when a model or PLY file cannot be parsed."""


@dataclass
class SparseStats:
    model_dir: str
    total_input_images: int
    registered_images: int
    cameras: int
    points3d: int
    observations: int
    mean_track_length: float | None
    mean_reprojection_error_px: float | None
    median_reprojection_error_px: float | None

    @property
    def registration_ratio(self) -> float:
        return self.registered_images / self.total_input_images if self.total_input_images else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["registration_percentage"] = round(100 * self.registration_ratio, 1)
        return data


def _data_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise StatisticsError(f"Missing model file: {path}")
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if not line.startswith("#")]


def parse_sparse_text_model(model_txt_dir: Path, total_input_images: int) -> SparseStats:
    """Compute statistics from a COLMAP model exported with ``--output_type TXT``."""
    cameras = [line for line in _data_lines(model_txt_dir / "cameras.txt") if line.strip()]

    # images.txt stores two lines per image; the second (2D points) may be empty.
    image_lines = _data_lines(model_txt_dir / "images.txt")
    while image_lines and not image_lines[-1].strip():
        image_lines.pop()
    registered = (len(image_lines) + 1) // 2

    errors: list[float] = []
    observations = 0
    for line in _data_lines(model_txt_dir / "points3D.txt"):
        parts = line.split()
        if len(parts) < 8:
            continue
        errors.append(float(parts[7]))
        observations += (len(parts) - 8) // 2

    points = len(errors)
    return SparseStats(
        model_dir=str(model_txt_dir),
        total_input_images=total_input_images,
        registered_images=registered,
        cameras=len(cameras),
        points3d=points,
        observations=observations,
        mean_track_length=round(observations / points, 3) if points else None,
        mean_reprojection_error_px=round(_stats.fmean(errors), 4) if errors else None,
        median_reprojection_error_px=round(_stats.median(errors), 4) if errors else None,
    )


@dataclass
class DatabaseStats:
    images: int
    images_with_features: int
    total_keypoints: int
    matched_pairs: int
    verified_pairs: int

    def to_dict(self) -> dict:
        return asdict(self)


def database_statistics(database: Path) -> DatabaseStats:
    """Count images, keypoints and geometrically verified pairs in a COLMAP database."""
    if not database.is_file():
        raise StatisticsError(f"COLMAP database not found: {database}")
    uri = f"file:{database.as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        def scalar(query: str) -> int:
            try:
                value = conn.execute(query).fetchone()[0]
            except sqlite3.OperationalError:  # table missing (stage not run yet)
                return 0
            return int(value or 0)

        return DatabaseStats(
            images=scalar("SELECT COUNT(*) FROM images"),
            images_with_features=scalar("SELECT COUNT(*) FROM keypoints WHERE rows > 0"),
            total_keypoints=scalar("SELECT SUM(rows) FROM keypoints"),
            matched_pairs=scalar("SELECT COUNT(*) FROM matches WHERE rows > 0"),
            verified_pairs=scalar("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"),
        )


@dataclass
class PlyHeader:
    vertex_count: int
    face_count: int
    properties: list[str]
    binary: bool


def read_ply_header(path: Path) -> PlyHeader:
    """Read element counts from a PLY header without loading the data."""
    if not path.is_file():
        raise StatisticsError(f"PLY file not found: {path}")
    vertex_count = face_count = 0
    properties: list[str] = []
    binary = False
    current = None
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise StatisticsError(f"Not a PLY file: {path}")
        for raw in handle:
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "format":
                binary = "binary" in tokens[1]
            elif tokens[0] == "element" and len(tokens) >= 3:
                current = tokens[1]
                if current == "vertex":
                    vertex_count = int(tokens[2])
                elif current == "face":
                    face_count = int(tokens[2])
            elif tokens[0] == "property" and current == "vertex":
                properties.append(tokens[-1])
        else:
            raise StatisticsError(f"PLY header has no end_header: {path}")
    return PlyHeader(vertex_count, face_count, properties, binary)
