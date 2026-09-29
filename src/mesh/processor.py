"""Mesh / point-cloud loading and geometric statistics.

Milestone 1 provides read-only analysis of the raw COLMAP mesh. Cleaning
operations (floating-component removal, hole detection, simplification, ...)
are planned for Milestone 2 and are intentionally not implemented yet.

All lengths are in *reconstruction units*: COLMAP's scale is arbitrary until
scale calibration (Milestone 2) is applied, so dimensions are not millimetres.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

from src.utils.logging import get_logger

logger = get_logger("mesh")

UNSCALED_UNITS = "reconstruction units (unscaled)"


class MeshLoadError(RuntimeError):
    """Raised when a mesh or point cloud cannot be loaded."""


@dataclass
class MeshStats:
    vertices: int
    faces: int
    bbox_min: list[float]
    bbox_max: list[float]
    bbox_extents: list[float]
    surface_area: float
    watertight: bool
    winding_consistent: bool
    connected_components: int
    volume: float | None  # only when watertight
    units: str = UNSCALED_UNITS

    def to_dict(self) -> dict:
        return asdict(self)


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a triangle mesh without modifying it (no merging / reordering)."""
    if not path.is_file():
        raise MeshLoadError(f"Mesh file not found: {path}")
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:  # trimesh raises many exception types for bad files
        raise MeshLoadError(f"Could not read mesh {path.name}: {exc}") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise MeshLoadError(f"{path.name} contains no triangles")
    return mesh


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a PLY point cloud; returns ``(points Nx3, colors Nx3 uint8 or None)``."""
    if not path.is_file():
        raise MeshLoadError(f"Point cloud not found: {path}")
    try:
        geom = trimesh.load(path, process=False)
    except Exception as exc:
        raise MeshLoadError(f"Could not read point cloud {path.name}: {exc}") from exc
    points = np.asarray(getattr(geom, "vertices", np.empty((0, 3))), dtype=np.float64)
    if len(points) == 0:
        raise MeshLoadError(f"{path.name} contains no points")
    colors = None
    raw_colors = getattr(geom, "colors", None)
    if raw_colors is None and hasattr(geom, "visual"):
        raw_colors = getattr(geom.visual, "vertex_colors", None)
    if raw_colors is not None and len(raw_colors) == len(points):
        colors = np.asarray(raw_colors)[:, :3].astype(np.uint8)
    return points, colors


def count_connected_components(mesh: trimesh.Trimesh) -> int:
    """Number of face-connected components."""
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1
    )
    return len(components)


def mesh_statistics(mesh: trimesh.Trimesh) -> MeshStats:
    """Compute geometric indicators of mesh completeness.

    These describe the mesh, not its accuracy: a watertight mesh can still be
    geometrically wrong, which must be validated against physical measurements.
    """
    bounds = np.asarray(mesh.bounds, dtype=float)
    watertight = bool(mesh.is_watertight)
    volume = float(abs(mesh.volume)) if watertight else None
    return MeshStats(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        bbox_min=bounds[0].round(6).tolist(),
        bbox_max=bounds[1].round(6).tolist(),
        bbox_extents=(bounds[1] - bounds[0]).round(6).tolist(),
        surface_area=float(mesh.area),
        watertight=watertight,
        winding_consistent=bool(mesh.is_winding_consistent),
        connected_components=count_connected_components(mesh),
        volume=volume,
    )


def decimate_for_preview(mesh: trimesh.Trimesh, max_faces: int) -> tuple[trimesh.Trimesh, bool]:
    """Return a reduced copy for interactive display (never used for analysis).

    Returns ``(mesh, was_decimated)``. Vertex colours are lost on decimation.
    """
    if len(mesh.faces) <= max_faces:
        return mesh, False
    try:
        reduced = mesh.simplify_quadric_decimation(face_count=max_faces)
        return reduced, True
    except Exception as exc:  # missing fast-simplification or degenerate input
        logger.warning("Quadric decimation failed (%s); sub-sampling faces for preview", exc)
        rng = np.random.default_rng(0)
        keep = rng.choice(len(mesh.faces), size=max_faces, replace=False)
        return mesh.submesh([keep], append=True), True
