"""COLMAP command-line builders.

Each function returns the argument list for one explicit COLMAP stage. COLMAP
has renamed several options between releases (for example
``SiftExtraction.use_gpu`` became ``FeatureExtraction.use_gpu`` in 3.12), so
option names are resolved against the ``-h`` output of the installed binary
via :class:`ColmapCapabilities` instead of being hard-coded to one version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import ReconstructionConfig
from src.utils.logging import get_logger
from src.utils.process import capture_output

logger = get_logger("colmap.commands")

_OPTION_RE = re.compile(r"--([A-Za-z0-9_.]+)")


def _flag(value: bool) -> str:
    return "1" if value else "0"


@dataclass
class ColmapCapabilities:
    """Lazily queried option names supported by each COLMAP sub-command."""

    executable: str
    _cache: dict[str, set[str]] = field(default_factory=dict)

    def options(self, command: str) -> set[str]:
        """Return the set of ``--option`` names accepted by ``command``."""
        if command not in self._cache:
            code, out = capture_output([self.executable, command, "-h"], timeout_s=30)
            names = set(_OPTION_RE.findall(out)) if out else set()
            if not names:
                logger.warning("Could not read options for 'colmap %s -h' (exit %s)", command, code)
            self._cache[command] = names
        return self._cache[command]

    def supports_command(self, command: str) -> bool:
        return bool(self.options(command))

    def resolve(self, command: str, *candidates: str) -> str | None:
        """Return the first candidate option supported by ``command``.

        If the help text could not be parsed, the first candidate (newest
        name) is returned so COLMAP itself reports any incompatibility.
        """
        known = self.options(command)
        if not known:
            return candidates[0]
        for name in candidates:
            if name in known:
                return name
        return None


class CommandBuilder:
    """Builds argument lists for every pipeline stage."""

    def __init__(self, caps: ColmapCapabilities, cfg: ReconstructionConfig, use_gpu: bool) -> None:
        self.caps = caps
        self.cfg = cfg
        self.use_gpu = use_gpu

    def _add(self, args: list[str], command: str, value: str, *candidates: str) -> None:
        name = self.caps.resolve(command, *candidates)
        if name is None:
            logger.warning("colmap %s does not support any of %s; option skipped", command, candidates)
            return
        args += [f"--{name}", value]

    def _base(self, command: str) -> list[str]:
        return [self.caps.executable, command]

    def feature_extractor(self, database: Path, images: Path, masks: Path | None = None) -> list[str]:
        cmd = "feature_extractor"
        args = self._base(cmd) + ["--database_path", str(database), "--image_path", str(images)]
        self._add(args, cmd, _flag(self.cfg.single_camera), "ImageReader.single_camera")
        self._add(args, cmd, self.cfg.camera_model, "ImageReader.camera_model")
        if masks is not None:
            self._add(args, cmd, str(masks), "ImageReader.mask_path")
        self._add(args, cmd, _flag(self.use_gpu), "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu")
        self._add(
            args, cmd, str(self.cfg.max_image_size),
            "FeatureExtraction.max_image_size", "SiftExtraction.max_image_size",
        )
        self._add(args, cmd, str(self.cfg.max_num_features), "SiftExtraction.max_num_features")
        return args

    def matcher(self, matcher: str, database: Path) -> list[str]:
        if matcher not in ("exhaustive", "sequential"):
            raise ValueError(f"Unsupported matcher: {matcher}")
        cmd = f"{matcher}_matcher"
        args = self._base(cmd) + ["--database_path", str(database)]
        self._add(args, cmd, _flag(self.use_gpu), "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
        if matcher == "sequential":
            self._add(args, cmd, str(self.cfg.sequential_overlap), "SequentialMatching.overlap")
            # Loop detection needs a separately downloaded vocabulary tree.
            self._add(args, cmd, "0", "SequentialMatching.loop_detection")
        return args

    def mapper(self, database: Path, images: Path, output: Path) -> list[str]:
        return self._base("mapper") + [
            "--database_path", str(database),
            "--image_path", str(images),
            "--output_path", str(output),
        ]

    def model_converter(self, model: Path, output: Path, output_type: str) -> list[str]:
        return self._base("model_converter") + [
            "--input_path", str(model),
            "--output_path", str(output),
            "--output_type", output_type,
        ]

    def model_analyzer(self, model: Path) -> list[str]:
        return self._base("model_analyzer") + ["--path", str(model)]

    def image_undistorter(self, images: Path, model: Path, dense: Path) -> list[str]:
        return self._base("image_undistorter") + [
            "--image_path", str(images),
            "--input_path", str(model),
            "--output_path", str(dense),
            "--output_type", "COLMAP",
            "--max_image_size", str(self.cfg.max_image_size),
        ]

    def patch_match_stereo(self, dense: Path) -> list[str]:
        cmd = "patch_match_stereo"
        args = self._base(cmd) + ["--workspace_path", str(dense), "--workspace_format", "COLMAP"]
        self._add(args, cmd, _flag(self.cfg.dense.geom_consistency), "PatchMatchStereo.geom_consistency")
        self._add(args, cmd, str(self.cfg.dense.window_radius), "PatchMatchStereo.window_radius")
        return args

    def stereo_fusion(self, dense: Path, output: Path) -> list[str]:
        input_type = "geometric" if self.cfg.dense.geom_consistency else "photometric"
        return self._base("stereo_fusion") + [
            "--workspace_path", str(dense),
            "--workspace_format", "COLMAP",
            "--input_type", input_type,
            "--output_path", str(output),
        ]

    def poisson_mesher(self, fused_ply: Path, output: Path) -> list[str]:
        cmd = "poisson_mesher"
        args = self._base(cmd) + ["--input_path", str(fused_ply), "--output_path", str(output)]
        self._add(args, cmd, str(self.cfg.poisson_trim), "PoissonMeshing.trim")
        return args

    def delaunay_mesher(self, input_path: Path, output: Path, input_type: str = "dense") -> list[str]:
        args = self._base("delaunay_mesher") + ["--input_path", str(input_path), "--output_path", str(output)]
        if input_type != "dense":
            if "input_type" not in self.caps.options("delaunay_mesher"):
                raise ValueError("This COLMAP build's delaunay_mesher cannot mesh sparse models")
            args += ["--input_type", input_type]
        return args
