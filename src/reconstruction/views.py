"""Prepare the image tree COLMAP reads.

COLMAP's ``single_camera`` mode rejects every image whose width or height
differs from the first one (``CAMERA_SINGLE_DIM_ERROR``) and then matching
runs on whatever happened to be left. Photos and phone videos from the same
capture are often different sizes, so mixed runs are split into one folder
per resolution and reconstructed with ``single_camera_per_folder``.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ColmapViewSet:
    image_dir: Path
    mask_dir: Path | None
    image_list: Path | None
    single_camera: bool
    single_camera_per_folder: bool
    size_counts: dict[str, int]


def list_frame_names(frames_dir: Path, image_list: Path | None) -> list[str]:
    """Names relative to ``frames_dir``, from the image list or the directory."""
    if image_list is not None and image_list.is_file():
        return [line.strip() for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(
        path.name for path in frames_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES
    )


def layout_views(
    root: Path,
    frames_dir: Path,
    names: list[str],
    *,
    masks_dir: Path | None,
    image_list: Path | None,
    share_intrinsics: bool,
) -> ColmapViewSet:
    """Return the directories and camera flags COLMAP should use.

    A single resolution keeps the existing frame and mask folders. Several
    resolutions are symlinked into ``camera_groups/`` so each size shares one
    camera without moving the original files.
    """
    sizes = {name: _size(frames_dir / name) for name in names}
    counts = Counter(f"{width}x{height}" for width, height in sizes.values())
    if len(counts) <= 1:
        return ColmapViewSet(
            frames_dir, masks_dir, image_list, share_intrinsics, False, dict(counts)
        )

    group_root = root / "camera_groups"
    if group_root.exists():
        shutil.rmtree(group_root)
    image_dir = group_root / "images"
    mask_dir = group_root / "masks" if masks_dir is not None else None
    relative: list[str] = []
    for name, (width, height) in sizes.items():
        folder = f"{width}x{height}"
        relative.append(f"{folder}/{name}")
        _symlink(frames_dir / name, image_dir / folder / name)
        if mask_dir is not None and masks_dir is not None:
            source = masks_dir / f"{name}.png"
            if source.is_file():
                _symlink(source, mask_dir / folder / f"{name}.png")

    grouped_list = group_root / "image_list.txt"
    grouped_list.write_text("".join(f"{name}\n" for name in relative), encoding="utf-8")
    return ColmapViewSet(image_dir, mask_dir, grouped_list, False, True, dict(counts))


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _symlink(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(source.resolve())
