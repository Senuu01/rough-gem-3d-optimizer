"""Gemstone segmentation masks for object-motion captures (turntable / hand-held).

When the camera is static and the stone moves, COLMAP must only see the stone;
otherwise it reconstructs the static background (a "successful" model of the
floor). Masks follow COLMAP's convention: for image ``frames/x.jpg`` the mask
is ``masks/x.jpg.png``, white (255) = use, black (0) = ignore.

Masks are produced by SAM 2 video segmentation: the user clicks the stone in
the first frame of each *sequence* (one video, or the set of photos) and SAM 2
tracks it through the remaining frames. Prompts are saved in
``masks/prompts.json`` so masks can be regenerated reproducibly. Individual
frames can be rejected, or their mask replaced with a manually made PNG.

SAM 2 / PyTorch are optional dependencies imported only by
:func:`generate_masks`.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.ingestion.dataset import IngestionReport
from src.reconstruction.progress import write_json_atomic
from src.utils.config import MaskingConfig
from src.utils.logging import get_logger
from src.utils.paths import RunPaths

logger = get_logger("masking")

SAM2_INSTALL_HINT = "pip install -r requirements-sam2.txt"
PHOTOS_SEQUENCE = "photos"


class MaskingError(RuntimeError):
    """Raised when masks cannot be generated or are unusable."""


# --------------------------------------------------------------------- data
@dataclass
class Sequence:
    """Temporally ordered frames tracked together by SAM 2."""

    key: str
    source: str
    images: list[str]


@dataclass
class MaskPrompt:
    """Clicks on the first frame of a sequence (label 1 = stone, 0 = not stone)."""

    points: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def add(self, x: float, y: float, positive: bool) -> None:
        self.points.append([float(x), float(y)])
        self.labels.append(1 if positive else 0)

    @property
    def valid(self) -> bool:
        return 1 in self.labels


@dataclass
class FrameMaskInfo:
    image: str
    status: str  # ok | missing | empty | too_large | rejected
    area_ratio: float | None
    manual: bool = False

    @property
    def usable(self) -> bool:
        return self.status == "ok"


def build_sequences(report: IngestionReport) -> list[Sequence]:
    """Group frames by video (in temporal order); all photos form one sequence."""
    sequences: dict[str, Sequence] = {}
    for img in report.images:
        key = img.source if img.origin == "video" else PHOTOS_SEQUENCE
        seq = sequences.setdefault(key, Sequence(key=key, source=img.source if img.origin == "video" else "photos", images=[]))
        seq.images.append(img.output_name)
    return list(sequences.values())


def mask_path(run: RunPaths, image_name: str) -> Path:
    """COLMAP mask location for ``image_name``."""
    return run.masks_dir / f"{image_name}.png"


def _prompts_path(run: RunPaths) -> Path:
    return run.masks_dir / "prompts.json"


def _selection_path(run: RunPaths) -> Path:
    return run.masks_dir / "selection.json"


def masking_status_path(run: RunPaths) -> Path:
    return run.masks_dir / "status.json"


def load_prompts(run: RunPaths) -> dict[str, MaskPrompt]:
    path = _prompts_path(run)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: MaskPrompt(**value) for key, value in data.items()}


def save_prompts(run: RunPaths, prompts: dict[str, MaskPrompt]) -> None:
    run.masks_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(_prompts_path(run), {k: asdict(v) for k, v in prompts.items()})


def load_selection(run: RunPaths) -> dict[str, list[str]]:
    path = _selection_path(run)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {"rejected": list(data.get("rejected", [])), "manual": list(data.get("manual", []))}


def save_selection(run: RunPaths, selection: dict[str, list[str]]) -> None:
    run.masks_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(_selection_path(run), {k: sorted(set(v)) for k, v in selection.items()})


def set_rejected(run: RunPaths, image_name: str, rejected: bool) -> None:
    selection = load_selection(run)
    names = set(selection["rejected"])
    if rejected:
        names.add(image_name)
    else:
        names.discard(image_name)
    selection["rejected"] = sorted(names)
    save_selection(run, selection)


# ------------------------------------------------------------ mask handling
def binarize_and_erode(mask: np.ndarray, erode_px: int) -> np.ndarray:
    """Return a uint8 0/255 mask, eroded by ``erode_px`` pixels."""
    binary = (mask > 127).astype(np.uint8) * 255 if mask.dtype == np.uint8 else (mask > 0).astype(np.uint8) * 255
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        binary = cv2.erode(binary, kernel)
    return binary


def write_mask(run: RunPaths, image_name: str, mask: np.ndarray) -> None:
    path = mask_path(run, image_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise MaskingError(f"Could not write mask {path}")


def _frame_size(run: RunPaths, image_name: str) -> tuple[int, int]:
    """(width, height) of a frame, read from the file header."""
    try:
        with Image.open(run.frames_dir / image_name) as img:
            return img.size
    except (OSError, UnidentifiedImageError) as exc:
        raise MaskingError(f"Frame not readable: {image_name}") from exc


def save_manual_mask(run: RunPaths, image_name: str, data: bytes) -> None:
    """Replace the mask of ``image_name`` with an uploaded image (any size, white = stone)."""
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise MaskingError("The uploaded mask is not a readable image")
    width, height = _frame_size(run, image_name)
    resized = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
    write_mask(run, image_name, binarize_and_erode(resized, 0))
    selection = load_selection(run)
    selection["manual"] = sorted(set(selection["manual"]) | {image_name})
    selection["rejected"] = [n for n in selection["rejected"] if n != image_name]
    save_selection(run, selection)


def frame_mask_info(run: RunPaths, report: IngestionReport, cfg: MaskingConfig) -> list[FrameMaskInfo]:
    """Status of every frame's mask (area thresholds are documented in config.yaml)."""
    selection = load_selection(run)
    rejected, manual = set(selection["rejected"]), set(selection["manual"])
    infos = []
    for img in report.images:
        path = mask_path(run, img.output_name)
        if img.output_name in rejected:
            infos.append(FrameMaskInfo(img.output_name, "rejected", None, img.output_name in manual))
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        if mask is None:
            infos.append(FrameMaskInfo(img.output_name, "missing", None))
            continue
        ratio = float((mask > 127).mean())
        if ratio < cfg.min_area_ratio:
            status = "empty"
        elif ratio > cfg.max_area_ratio:
            status = "too_large"
        else:
            status = "ok"
        infos.append(FrameMaskInfo(img.output_name, status, round(ratio, 5), img.output_name in manual))
    return infos


def write_image_list(run: RunPaths, infos: list[FrameMaskInfo]) -> Path:
    """Write COLMAP's ``--image_list_path`` file containing only usable frames."""
    path = run.root / "image_list.txt"
    path.write_text("".join(f"{i.image}\n" for i in infos if i.usable), encoding="utf-8")
    return path


def mask_overlay(run: RunPaths, image_name: str, max_side: int = 480) -> np.ndarray | None:
    """RGB preview: masked-out regions darkened and tinted red."""
    frame = cv2.imread(str(run.frames_dir / image_name), cv2.IMREAD_REDUCED_COLOR_4)
    if frame is None:
        return None
    scale = max_side / max(frame.shape[:2])
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    path = mask_path(run, image_name)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
    if mask is not None:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        dark = (frame * 0.3).astype(np.uint8)
        dark[..., 2] = np.clip(dark[..., 2].astype(int) + 60, 0, 255).astype(np.uint8)
        frame = np.where(mask[..., None], frame, dark)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ SAM 2
def sam2_available() -> bool:
    try:
        import sam2  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _select_device(preference: str):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prepare_chunk_dir(run: RunPaths, work: Path, names: list[str]) -> None:
    """SAM 2 reads a directory of JPEGs named ``00000.jpg``, ``00001.jpg``, ..."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for index, name in enumerate(names):
        source = (run.frames_dir / name).resolve()
        target = work / f"{index:05d}.jpg"
        if source.suffix.lower() in (".jpg", ".jpeg"):
            try:
                os.symlink(source, target)
                continue
            except OSError:  # e.g. Windows without symlink rights
                pass
        img = cv2.imread(str(source))
        if img is None or not cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise MaskingError(f"Could not prepare frame {name} for SAM 2")


def generate_masks(
    run: RunPaths,
    report: IngestionReport,
    cfg: MaskingConfig,
    prompts: dict[str, MaskPrompt],
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    """Track the stone with SAM 2 and write one COLMAP mask per frame.

    Long sequences are processed in chunks of ``cfg.chunk_size`` frames; each
    chunk is seeded with the final mask of the previous chunk.

    Returns:
        Number of masks written per sequence.

    Raises:
        MaskingError: if SAM 2 is missing, a sequence has no prompt, or the
            stone is lost (empty mask) at a chunk boundary.
    """
    if not sam2_available():
        raise MaskingError(f"SAM 2 is not installed. Install it with: {SAM2_INSTALL_HINT}")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    warnings.filterwarnings("ignore", message=".*cannot import name '_C'.*")
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    sequences = build_sequences(report)
    missing = [s.key for s in sequences if s.key not in prompts or not prompts[s.key].valid]
    if missing:
        raise MaskingError(f"Click the stone (positive point) in the first frame of: {', '.join(missing)}")

    device = _select_device(cfg.device)
    logger.info("Loading SAM 2 model %s on %s", cfg.sam2_model, device)
    predictor = SAM2VideoPredictor.from_pretrained(cfg.sam2_model, device=device)

    total = sum(len(s.images) for s in sequences)
    done = 0
    written: dict[str, int] = {}
    work_root = run.masks_dir / "_sam2_work"
    try:
        for seq in sequences:
            prompt = prompts[seq.key]
            seed_mask: np.ndarray | None = None
            written[seq.key] = 0
            for start in range(0, len(seq.images), cfg.chunk_size):
                names = seq.images[start:start + cfg.chunk_size]
                _prepare_chunk_dir(run, work_root, names)
                with torch.inference_mode():
                    state = predictor.init_state(video_path=str(work_root), offload_video_to_cpu=True)
                    if seed_mask is None:
                        predictor.add_new_points_or_box(
                            state, frame_idx=0, obj_id=1,
                            points=np.asarray(prompt.points, dtype=np.float32),
                            labels=np.asarray(prompt.labels, dtype=np.int32),
                        )
                    else:
                        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed_mask)
                    for frame_idx, _obj_ids, logits in predictor.propagate_in_video(state):
                        raw = (logits[0, 0] > 0).cpu().numpy()
                        write_mask(run, names[frame_idx], binarize_and_erode(raw, cfg.erode_px))
                        if frame_idx == len(names) - 1:
                            seed_mask = raw
                        written[seq.key] += 1
                        done += 1
                        if progress:
                            progress(done, total, f"{seq.source}: frame {start + frame_idx + 1}/{len(seq.images)}")
                    predictor.reset_state(state)
                if seed_mask is not None and not seed_mask.any():
                    raise MaskingError(
                        f"The stone was lost while tracking {seq.source} (empty mask at frame {start + len(names)}). "
                        "Add more clicks, or split the video."
                    )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    logger.info("Masks written: %s", written)
    return written
