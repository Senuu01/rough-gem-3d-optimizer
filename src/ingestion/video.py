"""Video ingestion: probing and frame sampling.

Feeding every video frame to COLMAP is slow and hurts reconstruction, because
near-identical neighbouring frames add little baseline. Two strategies exist:

* ``interval`` - one frame every N seconds. Uses FFmpeg when available
  (robust decoding, applies phone rotation metadata) and OpenCV otherwise.
* ``auto`` - frames are probed every ``auto_probe_interval_seconds``. The
  median optical-flow displacement between consecutive probes (measured at
  640 px width) is accumulated, and a frame is kept once it reaches
  ``auto_min_motion_px`` or ``auto_max_interval_seconds`` has passed. Flow is
  measured between nearby probes because dense flow is unreliable for large
  displacements. This adapts the sampling rate to the capture speed.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.ingestion.images import IngestedImage
from src.utils.config import ToolsConfig, VideoConfig
from src.utils.logging import get_logger
from src.utils.process import capture_output, run_command

logger = get_logger("ingestion.video")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
_FLOW_WIDTH = 640


class VideoIngestionError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def find_ffmpeg(tools: ToolsConfig) -> str | None:
    """Return the FFmpeg executable path, or ``None`` if it is not installed."""
    return shutil.which(tools.ffmpeg_executable)


def ffmpeg_version(tools: ToolsConfig) -> str | None:
    exe = find_ffmpeg(tools)
    if not exe:
        return None
    code, out = capture_output([exe, "-version"])
    return out.splitlines()[0].strip() if code == 0 and out else None


def probe_video(path: Path) -> VideoInfo:
    """Read basic metadata and verify that at least one frame decodes."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise _corrupt_video_error(path, "OpenCV could not open the file")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise _corrupt_video_error(path, "no decodable frames")
        height, width = frame.shape[:2]
    finally:
        cap.release()
    if fps <= 0 or fps > 1000:
        logger.warning("%s reports implausible fps=%.2f; assuming 30", path.name, fps)
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoInfo(path, fps, frame_count, duration, width, height)


def _corrupt_video_error(path: Path, reason: str) -> VideoIngestionError:
    return VideoIngestionError(f"{path.name}: corrupted or unsupported video ({reason})")


def _ffmpeg_qscale(jpeg_quality: int) -> int:
    """Map a 1-100 JPEG quality to FFmpeg's ``-q:v`` scale (2 = best, 31 = worst)."""
    return int(round(2 + (100 - jpeg_quality) / 100 * 29))


def _extract_interval_ffmpeg(
    ffmpeg: str, info: VideoInfo, out_dir: Path, stem: str, cfg: VideoConfig, log_file: Path | None
) -> list[IngestedImage]:
    pattern = out_dir / f"{stem}_%05d.jpg"
    result = run_command(
        [
            ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-i", info.path,
            "-vf", f"fps=1/{cfg.interval_seconds}",
            "-frames:v", str(cfg.max_frames_per_video),
            "-q:v", str(_ffmpeg_qscale(cfg.jpeg_quality)),
            "-start_number", "0",
            pattern,
        ],
        log_file=log_file,
    )
    if not result.ok:
        raise VideoIngestionError(
            f"{info.path.name}: FFmpeg frame extraction failed (exit {result.returncode}): "
            + " | ".join(result.output_tail[-3:])
        )
    frames = sorted(out_dir.glob(f"{stem}_*.jpg"))
    records = []
    for index, frame_path in enumerate(frames):
        h, w = _image_size(frame_path)
        records.append(
            IngestedImage(
                source=info.path.name,
                output_name=frame_path.name,
                width=w,
                height=h,
                origin="video",
                action="extracted",
                timestamp_s=round(index * cfg.interval_seconds, 3),
            )
        )
    return records


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise VideoIngestionError(f"Extracted frame could not be read back: {path.name}")
    return img.shape[0], img.shape[1]


def _flow_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = _FLOW_WIDTH / w
    small = cv2.resize(frame, (_FLOW_WIDTH, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def median_flow_magnitude(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Median dense optical-flow magnitude (pixels) between two grey images."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.1, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    return float(np.median(magnitude))


def _extract_opencv(info: VideoInfo, out_dir: Path, stem: str, cfg: VideoConfig, auto: bool) -> list[IngestedImage]:
    """Decode sequentially with OpenCV, keeping frames by interval or motion."""
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise _corrupt_video_error(info.path, "OpenCV could not open the file")

    step_s = cfg.auto_probe_interval_seconds if auto else cfg.interval_seconds
    records: list[IngestedImage] = []
    next_probe_t = 0.0
    last_kept_t = -np.inf
    prev_probe_gray: np.ndarray | None = None
    accumulated_motion = 0.0
    frame_index = -1
    params = [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]

    try:
        while len(records) < cfg.max_frames_per_video:
            if not cap.grab():
                break
            frame_index += 1
            t = frame_index / info.fps
            if t + 1e-6 < next_probe_t:
                continue
            next_probe_t += step_s
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                logger.warning("%s: could not decode frame %d; skipping", info.path.name, frame_index)
                continue

            keep = True
            if auto:
                gray = _flow_gray(frame)
                if prev_probe_gray is not None:
                    accumulated_motion += median_flow_magnitude(prev_probe_gray, gray)
                    keep = (
                        accumulated_motion >= cfg.auto_min_motion_px
                        or (t - last_kept_t) >= cfg.auto_max_interval_seconds
                    )
                prev_probe_gray = gray
            if not keep:
                continue

            out_path = out_dir / f"{stem}_{len(records):05d}.jpg"
            if not cv2.imwrite(str(out_path), frame, params):
                raise VideoIngestionError(f"Could not write frame {out_path}")
            accumulated_motion = 0.0
            last_kept_t = t
            h, w = frame.shape[:2]
            records.append(
                IngestedImage(
                    source=info.path.name,
                    output_name=out_path.name,
                    width=w,
                    height=h,
                    origin="video",
                    action="extracted",
                    timestamp_s=round(t, 3),
                )
            )
    finally:
        cap.release()
    return records


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    output_stem: str,
    video_cfg: VideoConfig,
    tools: ToolsConfig,
    log_file: Path | None = None,
) -> list[IngestedImage]:
    """Sample frames from ``video_path`` into ``frames_dir``.

    Returns one :class:`IngestedImage` per written frame, in temporal order.

    Raises:
        VideoIngestionError: if the video is corrupt or yields no frames.
    """
    info = probe_video(video_path)
    logger.info(
        "Video %s: %dx%d, %.2f fps, %d frames, %.1fs; sampling=%s",
        video_path.name, info.width, info.height, info.fps, info.frame_count, info.duration_s, video_cfg.sampling,
    )
    ffmpeg = find_ffmpeg(tools)
    if video_cfg.sampling == "interval" and ffmpeg:
        records = _extract_interval_ffmpeg(ffmpeg, info, frames_dir, output_stem, video_cfg, log_file)
    else:
        if video_cfg.sampling == "interval":
            logger.warning("FFmpeg not found; falling back to OpenCV decoding for %s", video_path.name)
        records = _extract_opencv(info, frames_dir, output_stem, video_cfg, auto=video_cfg.sampling == "auto")

    if not records:
        raise VideoIngestionError(f"{video_path.name}: no frames could be extracted")
    logger.info("Extracted %d frames from %s", len(records), video_path.name)
    return records
