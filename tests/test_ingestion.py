from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.ingestion.dataset import prepare_dataset
from src.ingestion.images import ImageIngestionError, ingest_photo, validate_image
from src.ingestion.video import extract_frames
from src.utils.config import AppConfig, ToolsConfig, VideoConfig
from src.utils.paths import RunPaths
from src.utils.process import run_command


def _textured(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _write_video(path: Path, seconds: float = 3.0, fps: int = 20, speed_px: int = 4) -> None:
    base = _textured(1200, 480)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    assert writer.isOpened()
    for i in range(int(seconds * fps)):
        x = min(i * speed_px, base.shape[1] - 640)
        writer.write(np.ascontiguousarray(base[:, x:x + 640]))
    writer.release()


def test_copy_and_exif_rotation(tmp_path: Path):
    src_dir, out_dir = tmp_path / "in put", tmp_path / "out put"
    src_dir.mkdir()
    out_dir.mkdir()

    plain = src_dir / "plain photo.jpg"
    Image.fromarray(_textured(64, 32)).save(plain, quality=95)
    record = ingest_photo(plain, out_dir, "p0000")
    assert record.action == "copied"
    assert (out_dir / "p0000.jpg").read_bytes() == plain.read_bytes()

    rotated = src_dir / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    Image.fromarray(_textured(64, 32)).save(rotated, quality=95, exif=exif.tobytes())
    record = ingest_photo(rotated, out_dir, "p0001")
    assert record.action == "rotated"
    with Image.open(out_dir / "p0001.jpg") as img:
        assert img.size == (32, 64)
        assert img.getexif().get(0x0112, 1) == 1


def test_corrupt_image_rejected(tmp_path: Path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    with pytest.raises(ImageIngestionError):
        validate_image(bad)


def test_interval_sampling_opencv(tmp_path: Path):
    video = tmp_path / "clip with space.mp4"
    _write_video(video, seconds=3.0)
    out = tmp_path / "frames"
    out.mkdir()
    cfg = VideoConfig(sampling="interval", interval_seconds=0.5)
    # Force the OpenCV path by pointing FFmpeg at a non-existent executable.
    records = extract_frames(video, out, "v00", cfg, ToolsConfig(ffmpeg_executable="no-such-ffmpeg"))
    assert len(records) == 6
    assert [r.timestamp_s for r in records] == pytest.approx([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert all((out / r.output_name).is_file() for r in records)


def test_auto_sampling_skips_static_frames(tmp_path: Path):
    moving, static = tmp_path / "moving.mp4", tmp_path / "static.mp4"
    _write_video(moving, seconds=3.0, speed_px=8)
    _write_video(static, seconds=3.0, speed_px=0)
    cfg = VideoConfig(sampling="auto", auto_min_motion_px=20, auto_max_interval_seconds=10)
    out_m, out_s = tmp_path / "m", tmp_path / "s"
    out_m.mkdir()
    out_s.mkdir()
    n_moving = len(extract_frames(moving, out_m, "m", cfg, ToolsConfig()))
    n_static = len(extract_frames(static, out_s, "s", cfg, ToolsConfig()))
    assert n_static == 1
    assert n_moving > 3


def test_prepare_dataset_mixed(tmp_path: Path):
    run = RunPaths(tmp_path / "run 1").create()
    _write_video(run.input_dir / "orbit.mp4", seconds=2.0)
    Image.fromarray(_textured(64, 64)).save(run.input_dir / "IMG 1.jpg")
    (run.input_dir / "notes.txt").write_text("ignore me")
    (run.input_dir / "bad.mov").write_bytes(b"garbage")
    config = AppConfig()
    config.tools.ffmpeg_executable = "no-such-ffmpeg"
    report = prepare_dataset(run, config)
    assert report.kind == "mixed"
    assert report.photo_count == 1 and report.video_count == 1
    assert report.skipped_files == ["notes.txt"]
    assert len(report.errors) == 1 and "bad.mov" in report.errors[0]
    assert run.ingestion_path.is_file()


def test_run_command_handles_spaces(tmp_path: Path):
    target = tmp_path / "dir with spaces" / "file $name.txt"
    target.parent.mkdir()
    target.write_text("hello")
    result = run_command(["cat", target], log_file=tmp_path / "log.txt")
    assert result.ok and result.output_tail == ["hello"]
    assert run_command(["definitely-not-a-command-xyz"]).returncode == 127
