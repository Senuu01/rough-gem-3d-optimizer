"""Typed configuration loaded from ``config.yaml``.

All tunable parameters live in the YAML file; the rest of the code base
receives an :class:`AppConfig` instance instead of reading constants. Unknown
keys are rejected so typos in the YAML fail loudly instead of being ignored.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOCAL_CONFIG_NAME = "config.local.yaml"

CaptureMode = Literal["orbit_camera", "turntable", "handheld"]
CAPTURE_MODES = ("orbit_camera", "turntable", "handheld")
OBJECT_MOTION_MODES = ("turntable", "handheld")  # camera static, stone moves: masks required
VideoSampling = Literal["interval", "auto"]
MatcherType = Literal["auto", "exhaustive", "sequential"]
MesherType = Literal["poisson", "delaunay"]


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass
class PathsConfig:
    runs_dir: str = "runs"

    def resolved_runs_dir(self) -> Path:
        """Return ``runs_dir`` as an absolute path (relative paths are project-relative)."""
        path = Path(self.runs_dir).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class ToolsConfig:
    colmap_executable: str = "colmap"
    ffmpeg_executable: str = "ffmpeg"
    ffprobe_executable: str = "ffprobe"


@dataclass
class CaptureConfig:
    mode: CaptureMode = "orbit_camera"

    @property
    def requires_masks(self) -> bool:
        return self.mode in OBJECT_MOTION_MODES


@dataclass
class MaskingConfig:
    sam2_model: str = "facebook/sam2.1-hiera-small"
    device: str = "auto"
    chunk_size: int = 60
    erode_px: int = 7
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.6


@dataclass
class ObjectCaptureConfig:
    max_image_size: int = 3840
    peak_threshold: float = 0.004
    domain_size_pooling: bool = True
    estimate_affine_shape: bool = True
    guided_matching: bool = True
    sequential_overlap: int = 15
    init_min_num_inliers: int = 30
    abs_pose_min_num_inliers: int = 12
    min_num_matches: int = 10


@dataclass
class VideoConfig:
    sampling: VideoSampling = "interval"
    interval_seconds: float = 0.5
    auto_probe_interval_seconds: float = 0.1
    auto_min_motion_px: float = 12.0
    auto_max_interval_seconds: float = 2.0
    max_frames_per_video: int = 300
    jpeg_quality: int = 95


@dataclass
class ImagesConfig:
    jpeg_quality: int = 95


@dataclass
class DenseConfig:
    enabled: bool = True
    geom_consistency: bool = True
    window_radius: int = 5


@dataclass
class ReconstructionConfig:
    matcher: MatcherType = "auto"
    auto_exhaustive_max_images: int = 200
    sequential_overlap: int = 10
    single_camera: bool = True
    camera_model: str = "SIMPLE_RADIAL"
    use_gpu: bool = True
    max_image_size: int = 2000
    max_num_features: int = 8192
    min_images: int = 8
    min_registered_ratio: float = 0.3
    dense: DenseConfig = field(default_factory=DenseConfig)
    mesher: MesherType = "poisson"
    poisson_trim: int = 10
    sparse_fallback: bool = True


@dataclass
class MeshConfig:
    preview_max_faces: int = 150_000
    preview_max_points: int = 150_000
    simplify: bool = True
    target_face_ratio: float = 0.5


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    object_capture: ObjectCaptureConfig = field(default_factory=ObjectCaptureConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable copy of the configuration."""
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Check value ranges that the type system cannot express."""
        errors: list[str] = []
        if self.capture.mode not in CAPTURE_MODES:
            errors.append(f"capture.mode must be one of {CAPTURE_MODES}, got {self.capture.mode!r}")
        if self.masking.device not in ("auto", "cuda", "mps", "cpu"):
            errors.append(f"masking.device must be auto, cuda, mps or cpu, got {self.masking.device!r}")
        if self.masking.chunk_size < 2:
            errors.append("masking.chunk_size must be >= 2")
        if self.masking.erode_px < 0:
            errors.append("masking.erode_px must be >= 0")
        if not 0 <= self.masking.min_area_ratio < self.masking.max_area_ratio <= 1:
            errors.append("masking area ratios must satisfy 0 <= min_area_ratio < max_area_ratio <= 1")
        if self.video.sampling not in ("interval", "auto"):
            errors.append(f"video.sampling must be interval or auto, got {self.video.sampling!r}")
        if self.video.interval_seconds <= 0:
            errors.append("video.interval_seconds must be > 0")
        if self.video.auto_probe_interval_seconds <= 0:
            errors.append("video.auto_probe_interval_seconds must be > 0")
        if self.video.max_frames_per_video < 1:
            errors.append("video.max_frames_per_video must be >= 1")
        if not 1 <= self.video.jpeg_quality <= 100 or not 1 <= self.images.jpeg_quality <= 100:
            errors.append("jpeg_quality values must be between 1 and 100")
        if self.reconstruction.matcher not in ("auto", "exhaustive", "sequential"):
            errors.append(f"reconstruction.matcher invalid: {self.reconstruction.matcher!r}")
        if self.reconstruction.mesher not in ("poisson", "delaunay"):
            errors.append(f"reconstruction.mesher invalid: {self.reconstruction.mesher!r}")
        if self.reconstruction.min_images < 2:
            errors.append("reconstruction.min_images must be >= 2")
        if not 0 <= self.reconstruction.min_registered_ratio <= 1:
            errors.append("reconstruction.min_registered_ratio must be within [0, 1]")
        if self.reconstruction.max_image_size < 100:
            errors.append("reconstruction.max_image_size must be >= 100")
        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


T = TypeVar("T")


def _build_dataclass(cls: type[T], data: dict[str, Any], prefix: str = "") -> T:
    """Recursively build dataclass ``cls`` from ``data``, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{prefix or 'root'}' must be a mapping")
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown config key(s) in '{prefix or 'root'}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_type = hints[name]
        if dataclasses.is_dataclass(field_type):
            kwargs[name] = _build_dataclass(field_type, value or {}, f"{prefix}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    """Build and validate an :class:`AppConfig` from a plain dictionary."""
    config = _build_dataclass(AppConfig, data)
    config.validate()
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc


def load_config(
    path: Path | None = None, overrides: dict[str, Any] | None = None
) -> AppConfig:
    """Load the YAML configuration and apply optional nested ``overrides``.

    If ``config.local.yaml`` exists next to the config file (git-ignored), it
    is merged on top; use it for machine-specific settings such as tool paths.

    Args:
        path: YAML file to read. Defaults to ``config.yaml`` in the project root.
        overrides: Nested dictionary merged on top of the file contents, e.g.
            ``{"reconstruction": {"matcher": "exhaustive"}}``.
    """
    path = path or DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    data = _read_yaml(path)
    local = path.with_name(LOCAL_CONFIG_NAME)
    if local.is_file():
        data = _deep_merge(data, _read_yaml(local))
    if overrides:
        data = _deep_merge(data, overrides)
    return config_from_dict(data)
