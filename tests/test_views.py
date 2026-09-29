from PIL import Image

from src.reconstruction.commands import CommandBuilder
from src.reconstruction.views import layout_views
from src.utils.config import ReconstructionConfig
from tests.test_statistics_commands import FakeCaps


def _jpeg(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(path)


def test_mixed_sizes_use_one_camera_folder_each(tmp_path):
    frames, masks = tmp_path / "frames", tmp_path / "masks"
    masks.mkdir()
    _jpeg(frames / "a.jpg", (8, 8))
    _jpeg(frames / "b.jpg", (10, 12))
    Image.new("L", (8, 8), 255).save(masks / "a.jpg.png")
    Image.new("L", (10, 12), 255).save(masks / "b.jpg.png")
    image_list = tmp_path / "image_list.txt"
    image_list.write_text("a.jpg\nb.jpg\n", encoding="utf-8")

    layout = layout_views(
        tmp_path, frames, ["a.jpg", "b.jpg"], masks_dir=masks, image_list=image_list, share_intrinsics=True
    )

    assert layout.single_camera_per_folder and not layout.single_camera
    assert (layout.image_dir / "8x8" / "a.jpg").is_file()
    assert (layout.mask_dir / "10x12" / "b.jpg.png").is_file()
    assert set(layout.image_list.read_text().splitlines()) == {"8x8/a.jpg", "10x12/b.jpg"}
    assert layout.size_counts == {"8x8": 1, "10x12": 1}


def test_one_size_keeps_the_shared_camera(tmp_path):
    frames = tmp_path / "frames"
    _jpeg(frames / "a.jpg", (8, 8))
    _jpeg(frames / "b.jpg", (8, 8))
    image_list = tmp_path / "image_list.txt"

    layout = layout_views(
        tmp_path, frames, ["a.jpg", "b.jpg"], masks_dir=None, image_list=image_list, share_intrinsics=True
    )

    assert layout.image_dir == frames
    assert layout.image_list == image_list
    assert layout.single_camera and not layout.single_camera_per_folder


def test_per_folder_camera_flag_is_passed(tmp_path):
    caps = FakeCaps({
        "feature_extractor": {"ImageReader.single_camera", "ImageReader.single_camera_per_folder"},
    })
    args = CommandBuilder(caps, ReconstructionConfig(), use_gpu=False).feature_extractor(
        tmp_path / "db", tmp_path / "frames", single_camera=False, single_camera_per_folder=True
    )
    assert args[args.index("--ImageReader.single_camera") + 1] == "0"
    assert args[args.index("--ImageReader.single_camera_per_folder") + 1] == "1"
