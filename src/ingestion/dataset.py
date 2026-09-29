"""Build the COLMAP image dataset (``frames/``) from a run's ``input/`` folder.

Photos and sampled video frames are combined into a single flat directory.
Frames of each video keep consecutive, zero-padded names so COLMAP's
sequential matcher sees them in temporal order.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.ingestion.images import IngestedImage, ImageIngestionError, ingest_photo, is_photo
from src.ingestion.video import VideoIngestionError, extract_frames, is_video
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import RunPaths

logger = get_logger("ingestion.dataset")

DatasetKind = Literal["photos", "video", "mixed", "empty"]


@dataclass
class IngestionReport:
    """Summary of the prepared dataset, persisted as ``ingestion.json``."""

    images: list[IngestedImage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    photo_count: int = 0
    video_count: int = 0

    @property
    def kind(self) -> DatasetKind:
        if not self.images:
            return "empty"
        if self.photo_count and self.video_count:
            return "mixed"
        return "video" if self.video_count else "photos"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "image_count": len(self.images),
            "photo_count": self.photo_count,
            "video_count": self.video_count,
            "errors": self.errors,
            "skipped_files": self.skipped_files,
            "images": [img.to_dict() for img in self.images],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IngestionReport":
        return cls(
            images=[IngestedImage(**img) for img in data.get("images", [])],
            errors=list(data.get("errors", [])),
            skipped_files=list(data.get("skipped_files", [])),
            photo_count=int(data.get("photo_count", 0)),
            video_count=int(data.get("video_count", 0)),
        )


def _safe_stem(name: str) -> str:
    """Filesystem- and COLMAP-friendly version of a file stem."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")
    return stem[:40] or "file"


def prepare_dataset(run: RunPaths, config: AppConfig) -> IngestionReport:
    """Convert everything in ``run.input_dir`` into COLMAP-ready images.

    Existing contents of ``run.frames_dir`` are replaced. Individual corrupt
    files are reported in :attr:`IngestionReport.errors` instead of aborting
    the whole dataset.
    """
    if run.frames_dir.exists():
        shutil.rmtree(run.frames_dir)
    run.frames_dir.mkdir(parents=True)

    report = IngestionReport()
    log_file = run.logs_dir / "ingestion.log"
    inputs = sorted(p for p in run.input_dir.iterdir() if p.is_file() and not p.name.startswith("."))

    photos = [p for p in inputs if is_photo(p)]
    videos = [p for p in inputs if is_video(p)]
    report.skipped_files = [p.name for p in inputs if p not in photos and p not in videos]
    for name in report.skipped_files:
        logger.warning("Skipping unsupported file type: %s", name)

    for index, video in enumerate(videos):
        stem = f"v{index:02d}_{_safe_stem(video.name)}"
        try:
            report.images.extend(extract_frames(video, run.frames_dir, stem, config.video, config.tools, log_file))
            report.video_count += 1
        except VideoIngestionError as exc:
            logger.error("%s", exc)
            report.errors.append(str(exc))

    for index, photo in enumerate(photos):
        stem = f"p{index:04d}_{_safe_stem(photo.name)}"
        try:
            report.images.append(ingest_photo(photo, run.frames_dir, stem, config.images.jpeg_quality))
            report.photo_count += 1
        except (ImageIngestionError, OSError) as exc:
            logger.error("%s", exc)
            report.errors.append(str(exc))

    run.ingestion_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info(
        "Dataset ready: %d images (%d photos, %d videos), %d errors",
        len(report.images), report.photo_count, report.video_count, len(report.errors),
    )
    return report


def load_ingestion_report(run: RunPaths) -> IngestionReport | None:
    """Load a previously saved ``ingestion.json`` if it exists."""
    if not run.ingestion_path.is_file():
        return None
    return IngestionReport.from_dict(json.loads(run.ingestion_path.read_text(encoding="utf-8")))
