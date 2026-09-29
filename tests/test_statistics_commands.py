import sqlite3
from pathlib import Path

import pytest

from src.reconstruction.commands import ColmapCapabilities, CommandBuilder
from src.reconstruction.statistics import (
    StatisticsError,
    database_statistics,
    parse_sparse_text_model,
    read_ply_header,
)
from src.utils.config import ReconstructionConfig


def _write_model(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "cameras.txt").write_text("# Camera list\n1 SIMPLE_RADIAL 100 100 80 50 50 0.01\n")
    (directory / "images.txt").write_text(
        "# Image list\n# two lines per image\n"
        "1 1 0 0 0 0 0 0 1 a.jpg\n10 10 1 20 20 2\n"
        "2 1 0 0 0 0 0 0 1 b.jpg\n\n"  # image with no 2D points -> empty second line
        "3 1 0 0 0 0 0 0 1 c.jpg\n5 5 1\n"
    )
    (directory / "points3D.txt").write_text(
        "# 3D point list\n"
        "1 0 0 0 255 0 0 0.5 1 0 3 0\n"
        "2 1 1 1 0 255 0 1.5 1 1 2 0 3 1\n"
    )


def test_parse_sparse_text_model(tmp_path: Path):
    _write_model(tmp_path / "model 0")
    stats = parse_sparse_text_model(tmp_path / "model 0", total_input_images=4)
    assert stats.registered_images == 3
    assert stats.cameras == 1
    assert stats.points3d == 2
    assert stats.observations == 5
    assert stats.mean_reprojection_error_px == pytest.approx(1.0)
    assert stats.median_reprojection_error_px == pytest.approx(1.0)
    assert stats.to_dict()["registration_percentage"] == 75.0


def test_missing_model_file(tmp_path: Path):
    with pytest.raises(StatisticsError):
        parse_sparse_text_model(tmp_path, 1)


def test_read_ply_header(tmp_path: Path):
    ply = tmp_path / "mesh.ply"
    ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 8\nproperty float x\nproperty float y\n"
        b"property float z\nproperty uchar red\nelement face 12\nproperty list uchar int vertex_indices\n"
        b"end_header\n\x00\x01"
    )
    header = read_ply_header(ply)
    assert (header.vertex_count, header.face_count, header.binary) == (8, 12, True)
    assert header.properties == ["x", "y", "z", "red"]


def test_database_statistics(tmp_path: Path):
    db = tmp_path / "database.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE images (image_id INTEGER)")
        conn.execute("CREATE TABLE keypoints (image_id INTEGER, rows INTEGER)")
        conn.execute("CREATE TABLE matches (pair_id INTEGER, rows INTEGER)")
        conn.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
        conn.executemany("INSERT INTO images VALUES (?)", [(1,), (2,), (3,)])
        conn.executemany("INSERT INTO keypoints VALUES (?, ?)", [(1, 100), (2, 50), (3, 0)])
        conn.executemany("INSERT INTO matches VALUES (?, ?)", [(1, 30), (2, 0)])
        conn.executemany("INSERT INTO two_view_geometries VALUES (?, ?)", [(1, 20)])
    stats = database_statistics(db)
    assert (stats.images, stats.images_with_features, stats.total_keypoints) == (3, 2, 150)
    assert (stats.matched_pairs, stats.verified_pairs) == (1, 1)


class FakeCaps(ColmapCapabilities):
    def __init__(self, options: dict[str, set[str]]):
        super().__init__("colmap")
        self._cache = options


def test_option_names_follow_installed_version(tmp_path: Path):
    cfg = ReconstructionConfig()
    new = FakeCaps({"feature_extractor": {"FeatureExtraction.use_gpu", "FeatureExtraction.max_image_size",
                                          "ImageReader.single_camera", "ImageReader.camera_model",
                                          "SiftExtraction.max_num_features"}})
    old = FakeCaps({"feature_extractor": {"SiftExtraction.use_gpu", "SiftExtraction.max_image_size",
                                          "ImageReader.single_camera", "ImageReader.camera_model",
                                          "SiftExtraction.max_num_features"}})
    images = tmp_path / "my frames"
    new_args = CommandBuilder(new, cfg, use_gpu=False).feature_extractor(tmp_path / "db.db", images)
    old_args = CommandBuilder(old, cfg, use_gpu=False).feature_extractor(tmp_path / "db.db", images)
    assert "--FeatureExtraction.use_gpu" in new_args and "--SiftExtraction.use_gpu" not in new_args
    assert "--SiftExtraction.use_gpu" in old_args
    assert str(images) in new_args  # path with a space kept as one argument
    assert new_args[new_args.index("--FeatureExtraction.use_gpu") + 1] == "0"


def test_unsupported_option_is_skipped():
    caps = FakeCaps({"sequential_matcher": {"SequentialMatching.overlap"}})
    args = CommandBuilder(caps, ReconstructionConfig(), use_gpu=True).matcher("sequential", Path("db"))
    assert "--SequentialMatching.overlap" in args
    assert not any("use_gpu" in a for a in args)
    assert not any("loop_detection" in a for a in args)
