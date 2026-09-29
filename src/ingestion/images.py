"""Photo ingestion: validation, HEIC conversion and EXIF orientation handling.

COLMAP reads raw pixel data and ignores the EXIF orientation flag, while
smartphones usually store portrait photos as rotated landscape pixels plus an
orientation tag. Images with a non-trivial orientation are therefore rewritten
with the rotation applied. Other EXIF data (notably focal length, which COLMAP
uses as an intrinsics prior) is preserved.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from src.utils.logging import get_logger

logger = get_logger("ingestion.images")

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
_NATIVE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_EXIF_ORIENTATION_TAG = 0x0112

try:  # HEIC support is optional at import time; reported clearly when missing.
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover - depends on environment
    HEIC_SUPPORTED = False


class ImageIngestionError(RuntimeError):
    """Raised when a photo cannot be read or converted."""


@dataclass
class IngestedImage:
    """Record describing one image placed into the frames directory."""

    source: str
    output_name: str
    width: int
    height: int
    origin: str  # "photo" or "video"
    action: str  # "copied", "converted", "rotated", "extracted"
    timestamp_s: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def is_photo(path: Path) -> bool:
    return path.suffix.lower() in PHOTO_EXTENSIONS


def validate_image(path: Path) -> tuple[int, int]:
    """Fully decode ``path`` and return ``(width, height)``.

    Raises:
        ImageIngestionError: if the file is corrupt or not a supported image.
    """
    if path.suffix.lower() in {".heic", ".heif"} and not HEIC_SUPPORTED:
        raise ImageIngestionError(
            f"{path.name}: HEIC support requires the 'pillow-heif' package (pip install pillow-heif)"
        )
    try:
        with Image.open(path) as img:
            img.load()  # forces a full decode so truncated files are detected
            return img.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageIngestionError(f"{path.name}: unreadable or corrupt image ({exc})") from exc


def ingest_photo(source: Path, frames_dir: Path, output_stem: str, jpeg_quality: int = 95) -> IngestedImage:
    """Place ``source`` into ``frames_dir`` in a COLMAP-compatible form.

    JPEG/PNG files with normal orientation are copied byte-for-byte (no
    re-compression). HEIC files and EXIF-rotated images are re-encoded as
    high-quality JPEG with the rotation applied.
    """
    validate_image(source)
    suffix = source.suffix.lower()

    with Image.open(source) as img:
        orientation = img.getexif().get(_EXIF_ORIENTATION_TAG, 1)
        needs_rotation = orientation not in (0, 1)
        needs_conversion = suffix not in _NATIVE_EXTENSIONS

        if not needs_rotation and not needs_conversion:
            out_suffix = ".png" if suffix == ".png" else ".jpg"
            destination = frames_dir / f"{output_stem}{out_suffix}"
            shutil.copy2(source, destination)
            width, height = img.size
            action = "copied"
        else:
            upright = ImageOps.exif_transpose(img)
            exif = img.getexif()
            if _EXIF_ORIENTATION_TAG in exif:
                exif[_EXIF_ORIENTATION_TAG] = 1
            if upright.mode not in ("RGB", "L"):
                upright = upright.convert("RGB")
            destination = frames_dir / f"{output_stem}.jpg"
            upright.save(destination, "JPEG", quality=jpeg_quality, subsampling=0, exif=exif.tobytes())
            width, height = upright.size
            action = "converted" if needs_conversion else "rotated"

    logger.info("%s %s -> %s (%dx%d)", action, source.name, destination.name, width, height)
    return IngestedImage(
        source=source.name,
        output_name=destination.name,
        width=width,
        height=height,
        origin="photo",
        action=action,
    )
