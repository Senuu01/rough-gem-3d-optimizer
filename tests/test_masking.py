from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingestion.dataset import IngestionReport, prepare_dataset
from src.ingestion.images import IngestedImage
from src.preprocessing.masking import (
    MaskingError,
    MaskPrompt,
    binarize_and_erode,
    build_sequences,
    frame_mask_info,
    load_prompts,
    mask_path,
    save_manual_mask,
    save_prompts,
    set_rejected,
    write_image_list,
    write_mask,
)
from src.reconstruction.commands import CommandBuilder
from src.utils.config import AppConfig, ConfigError, MaskingConfig, ObjectCaptureConfig, ReconstructionConfig, config_from_dict
from src.utils.paths import RunPaths
from tests.test_statistics_commands import FakeCaps


def _run_with_frames(tmp_path: Path, names: list[str], size=(80, 60)) -> tuple[RunPaths, IngestionReport]:
    run = RunPaths(tmp_path / "run with space").create()
    images = []
    for name in names:
        cv2.imwrite(str(run.frames_dir / name), np.full((size[1], size[0], 3), 128, np.uint8))
        origin = "photo" if name.startswith("p") else "video"
        source = "IMG_1.jpg" if origin == "photo" else name.split("_")[0] + ".mov"
        images.append(IngestedImage(source, name, size[0], size[1], origin, "extracted"))
    return run, IngestionReport(images=images, photo_count=1, video_count=1)


def _square_mask(size=(80, 60), side=20) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), np.uint8)
    mask[10:10 + side, 10:10 + side] = 255
    return mask


def test_sequences_group_by_video_and_photos(tmp_path: Path):
    _, report = _run_with_frames(tmp_path, ["vA_00000.jpg", "vA_00001.jpg", "vB_00000.jpg", "p0000_x.jpg"])
    seqs = build_sequences(report)
    assert [(s.key, len(s.images)) for s in seqs] == [("vA.mov", 2), ("vB.mov", 1), ("photos", 1)]


def test_colmap_mask_naming(tmp_path: Path):
    run, _ = _run_with_frames(tmp_path, ["vA_00000.jpg"])
    assert mask_path(run, "vA_00000.jpg").name == "vA_00000.jpg.png"


def test_binarize_and_erode():
    mask = _square_mask(side=21)
    eroded = binarize_and_erode(mask, 3)
    assert set(np.unique(eroded)) == {0, 255}
    assert (eroded > 0).sum() < (mask > 0).sum()
    assert (binarize_and_erode(mask.astype(bool), 0) > 0).sum() == (mask > 0).sum()


def test_frame_mask_info_statuses_and_image_list(tmp_path: Path):
    names = ["vA_00000.jpg", "vA_00001.jpg", "vA_00002.jpg", "vA_00003.jpg", "vA_00004.jpg"]
    run, report = _run_with_frames(tmp_path, names)
    write_mask(run, names[0], _square_mask())                          # ok (~8%)
    write_mask(run, names[1], np.zeros((60, 80), np.uint8))            # empty
    write_mask(run, names[2], np.full((60, 80), 255, np.uint8))        # too large
    write_mask(run, names[3], _square_mask())                          # rejected below
    set_rejected(run, names[3], True)
    # names[4] has no mask -> missing
    infos = {i.image: i for i in frame_mask_info(run, report, MaskingConfig())}
    assert [infos[n].status for n in names] == ["ok", "empty", "too_large", "rejected", "missing"]
    assert write_image_list(run, list(infos.values())).read_text() == f"{names[0]}\n"

    set_rejected(run, names[3], False)
    assert frame_mask_info(run, report, MaskingConfig())[3].status == "ok"


def test_manual_mask_is_resized_and_unrejects(tmp_path: Path):
    run, report = _run_with_frames(tmp_path, ["vA_00000.jpg"])
    set_rejected(run, "vA_00000.jpg", True)
    ok, png = cv2.imencode(".png", cv2.resize(_square_mask(), (160, 120), interpolation=cv2.INTER_NEAREST))
    save_manual_mask(run, "vA_00000.jpg", png.tobytes())
    info = frame_mask_info(run, report, MaskingConfig())[0]
    assert info.status == "ok" and info.manual
    assert cv2.imread(str(mask_path(run, "vA_00000.jpg")), cv2.IMREAD_GRAYSCALE).shape == (60, 80)
    with pytest.raises(MaskingError):
        save_manual_mask(run, "vA_00000.jpg", b"not an image")


def test_prompts_round_trip(tmp_path: Path):
    run, _ = _run_with_frames(tmp_path, ["vA_00000.jpg"])
    prompt = MaskPrompt()
    assert not prompt.valid
    prompt.add(10, 20, positive=False)
    assert not prompt.valid
    prompt.add(30, 40, positive=True)
    save_prompts(run, {"vA.mov": prompt})
    loaded = load_prompts(run)["vA.mov"]
    assert loaded.points == [[10.0, 20.0], [30.0, 40.0]] and loaded.labels == [0, 1] and loaded.valid


def test_reprepare_clears_stale_masks_but_keeps_prompts(tmp_path: Path):
    run, _ = _run_with_frames(tmp_path, ["p0000_a.jpg"])
    cv2.imwrite(str(run.input_dir / "a.jpg"), np.full((60, 80, 3), 90, np.uint8))
    write_mask(run, "p0000_a.jpg", _square_mask())
    save_prompts(run, {"photos": MaskPrompt([[1.0, 2.0]], [1])})
    prepare_dataset(run, AppConfig())
    assert not list(run.masks_dir.glob("*.png"))
    assert "photos" in load_prompts(run)


def test_handheld_mode_config():
    config = config_from_dict({"capture": {"mode": "handheld"}})
    assert config.capture.requires_masks
    assert not AppConfig().capture.requires_masks
    with pytest.raises(ConfigError):
        config_from_dict({"capture": {"mode": "spinning"}})


def test_object_capture_colmap_options(tmp_path: Path):
    caps = FakeCaps({
        "feature_extractor": {"ImageReader.mask_path", "SiftExtraction.use_gpu", "SiftExtraction.max_image_size",
                              "SiftExtraction.peak_threshold", "SiftExtraction.domain_size_pooling",
                              "SiftExtraction.estimate_affine_shape"},
        "sequential_matcher": {"SiftMatching.use_gpu", "SiftMatching.guided_matching", "SequentialMatching.overlap"},
        "mapper": {"Mapper.init_min_num_inliers", "Mapper.abs_pose_min_num_inliers", "Mapper.min_num_matches"},
        "stereo_fusion": {"StereoFusion.mask_path"},
    })
    obj = ObjectCaptureConfig()
    builder = CommandBuilder(caps, ReconstructionConfig(), use_gpu=False, object_cfg=obj)
    masks, image_list = tmp_path / "masks dir", tmp_path / "image_list.txt"
    fe = builder.feature_extractor(tmp_path / "db", tmp_path / "frames", masks, image_list)
    assert fe[fe.index("--ImageReader.mask_path") + 1] == str(masks)
    assert fe[fe.index("--image_list_path") + 1] == str(image_list)
    assert fe[fe.index("--SiftExtraction.max_image_size") + 1] == str(obj.max_image_size)
    assert "--SiftExtraction.domain_size_pooling" in fe
    match = builder.matcher("sequential", tmp_path / "db")
    assert match[match.index("--SequentialMatching.overlap") + 1] == str(obj.sequential_overlap)
    assert "--SiftMatching.guided_matching" in match
    mapper = builder.mapper(tmp_path / "db", tmp_path / "frames", tmp_path / "sparse")
    assert mapper[mapper.index("--Mapper.init_min_num_inliers") + 1] == str(obj.init_min_num_inliers)
    assert "--StereoFusion.mask_path" in builder.stereo_fusion(tmp_path / "dense", tmp_path / "f.ply", masks)

    plain = CommandBuilder(caps, ReconstructionConfig(), use_gpu=False)
    assert "--SiftExtraction.domain_size_pooling" not in plain.feature_extractor(tmp_path / "db", tmp_path / "f")
    assert not any(a.startswith("--Mapper.") for a in plain.mapper(tmp_path / "db", tmp_path / "f", tmp_path / "s"))
