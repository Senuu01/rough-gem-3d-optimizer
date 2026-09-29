"""Export meshes to common interchange formats.

The source PLY is never modified; exports are written as new files next to
it (``raw_mesh.obj``, ``raw_mesh.stl``, ``raw_mesh.glb``).
"""

from __future__ import annotations

from pathlib import Path

from src.mesh.processor import load_mesh
from src.utils.logging import get_logger

logger = get_logger("mesh.export")

EXPORT_FORMATS = ("obj", "stl", "glb")


def export_mesh(source_ply: Path, output_dir: Path, formats: tuple[str, ...] = EXPORT_FORMATS) -> dict[str, Path]:
    """Write ``source_ply`` in each requested format.

    Returns a mapping of format -> written path. Formats that fail are logged
    and omitted rather than aborting the other exports.
    """
    mesh = load_mesh(source_ply)
    written: dict[str, Path] = {}
    for fmt in formats:
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format: {fmt}")
        target = output_dir / f"{source_ply.stem}.{fmt}"
        try:
            mesh.export(target, file_type=fmt)
            written[fmt] = target
            logger.info("Exported %s", target.name)
        except Exception as exc:
            logger.error("Export to %s failed: %s", fmt.upper(), exc)
    return written
