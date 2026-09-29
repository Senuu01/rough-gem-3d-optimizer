"""Collect software/platform versions for reproducibility records."""

from __future__ import annotations

import platform
import sys
from importlib import metadata

_PACKAGES = (
    "numpy", "opencv-python-headless", "Pillow", "pillow-heif", "PyYAML",
    "trimesh", "fast-simplification", "streamlit", "plotly",
)


def software_versions() -> dict[str, str | None]:
    """Return Python, platform and key package versions."""
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions
