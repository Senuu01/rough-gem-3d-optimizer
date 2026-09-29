"""COLMAP installation detection and stage execution.

:func:`detect_colmap` validates the installation (``colmap -h``) and whether it
was built with CUDA, which dense stereo (``patch_match_stereo``) requires.
:class:`ColmapRunner` runs individual stages, writes one log file per stage and
converts non-zero exit codes into :class:`ColmapStageError` with an
explanation suitable for the UI.
"""

from __future__ import annotations

import platform
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from src.reconstruction.commands import ColmapCapabilities, CommandBuilder
from src.utils.config import ReconstructionConfig, ToolsConfig
from src.utils.logging import get_logger
from src.utils.process import ProcessResult, capture_output, run_command

logger = get_logger("colmap")

# COLMAP's GUI libraries are linked even for CLI commands; offscreen avoids
# display errors when the pipeline runs as a background process.
_COLMAP_ENV = {"QT_QPA_PLATFORM": "offscreen"}

_VERSION_RE = re.compile(r"COLMAP\s+([0-9][^\s]*)")

INSTALL_HELP = (
    "COLMAP was not found. Install it and make sure `colmap` is on your PATH "
    "(or set tools.colmap_executable in config.yaml):\n"
    "  macOS:   brew install colmap\n"
    "  Ubuntu:  sudo apt install colmap   (or build from source for CUDA support)\n"
    "  Windows: download a release from https://github.com/colmap/colmap/releases\n"
    "  conda:   conda install -c conda-forge colmap"
)


class ColmapNotFoundError(RuntimeError):
    """Raised when COLMAP is missing or does not run."""


class ColmapStageError(RuntimeError):
    """A COLMAP stage exited with an error.

    Attributes:
        stage: Human-readable stage name.
        explanation: Likely cause and suggested remedy, for display in the UI.
        log_file: Full stage output.
    """

    def __init__(self, stage: str, explanation: str, result: ProcessResult | None = None) -> None:
        self.stage = stage
        self.explanation = explanation
        self.result = result
        self.log_file = result.log_file if result else None
        detail = f" (exit code {result.returncode})" if result else ""
        super().__init__(f"{stage} failed{detail}: {explanation}")


@dataclass
class ColmapEnvironment:
    executable: str
    version: str
    banner: str
    has_cuda: bool | None  # None when the build banner does not say

    def to_dict(self) -> dict:
        return asdict(self)


def detect_colmap(tools: ToolsConfig) -> ColmapEnvironment:
    """Locate COLMAP, run ``colmap -h`` and parse version / CUDA support.

    Raises:
        ColmapNotFoundError: if the executable is missing or fails to start.
    """
    executable = shutil.which(tools.colmap_executable)
    if executable is None:
        raise ColmapNotFoundError(INSTALL_HELP)
    code, out = capture_output([executable, "-h"], timeout_s=60, env=_COLMAP_ENV)
    if code != 0 or "COLMAP" not in out:
        raise ColmapNotFoundError(
            f"`{executable} -h` failed (exit code {code}). The installation may be broken:\n{out[-800:]}"
        )
    banner = next((line.strip() for line in out.splitlines() if "COLMAP" in line), "")
    match = _VERSION_RE.search(banner)
    lowered = banner.lower()
    if "without cuda" in lowered:
        has_cuda: bool | None = False
    elif "with cuda" in lowered:
        has_cuda = True
    else:
        has_cuda = False if platform.system() == "Darwin" else None
    return ColmapEnvironment(executable, match.group(1) if match else "unknown", banner, has_cuda)


def _explain(stage_key: str, output: str) -> str:
    """Translate common COLMAP failure output into a user-facing explanation."""
    text = output.lower()
    if "cuda" in text and ("not" in text or "error" in text or "requires" in text):
        return (
            "This stage needs an NVIDIA GPU with CUDA and a CUDA-enabled COLMAP build. "
            "Disable GPU usage or dense reconstruction in the settings to use the CPU fallback."
        )
    if "opengl" in text or ("context" in text and "gpu" in text):
        return "GPU feature processing could not create an OpenGL/CUDA context. Disable 'Use GPU' and retry."
    if "unrecognised option" in text or "unrecognized option" in text:
        return "The installed COLMAP version does not accept one of the command-line options. See the stage log."
    if "cgal" in text:
        return "This COLMAP build lacks CGAL, which the Delaunay mesher requires. Use the Poisson mesher instead."
    hints = {
        "feature_extraction": "No usable images were read. Check that the frames are valid images.",
        "matching": "Feature matching failed. The images may have too little texture or overlap.",
        "mapper": "Structure-from-Motion failed to initialise. Images may lack overlap, texture or parallax.",
        "undistortion": "Image undistortion failed; the sparse model may be invalid.",
        "patch_match": "Dense stereo failed. It requires CUDA; the sparse fallback can be used instead.",
        "fusion": "Depth-map fusion failed or produced no points.",
        "meshing": "Surface meshing failed. The point cloud may be too sparse or noisy.",
    }
    return hints.get(stage_key, "See the stage log for details.")


class ColmapRunner:
    """Runs COLMAP stages for one run directory, logging each stage separately."""

    def __init__(
        self,
        env: ColmapEnvironment,
        cfg: ReconstructionConfig,
        logs_dir: Path,
        use_gpu: bool | None = None,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.logs_dir = logs_dir
        self.use_gpu = (cfg.use_gpu and bool(env.has_cuda)) if use_gpu is None else use_gpu
        self.caps = ColmapCapabilities(env.executable)
        self.commands = CommandBuilder(self.caps, cfg, self.use_gpu)
        self.executed: list[dict] = []

    def set_use_gpu(self, use_gpu: bool) -> None:
        """Switch GPU usage for subsequent stages (used for CPU fallback)."""
        self.use_gpu = use_gpu
        self.commands = CommandBuilder(self.caps, self.cfg, use_gpu)

    def run(self, stage_key: str, stage_label: str, args: list[str], log_name: str | None = None) -> ProcessResult:
        """Execute ``args``; raise :class:`ColmapStageError` on non-zero exit."""
        log_file = self.logs_dir / f"{log_name or stage_key}.log"
        result = run_command(args, log_file=log_file, env=_COLMAP_ENV)
        self.executed.append(
            {"stage": stage_key, "command": result.command_line, "returncode": result.returncode,
             "duration_s": round(result.duration_s, 2), "log": log_file.name}
        )
        if not result.ok:
            raise ColmapStageError(stage_label, _explain(stage_key, "\n".join(result.output_tail)), result)
        return result

    # ------------------------------------------------------------------ stages
    def extract_features(self, database: Path, images: Path, masks: Path | None = None) -> ProcessResult:
        return self.run("feature_extraction", "Feature extraction",
                        self.commands.feature_extractor(database, images, masks))

    def match(self, matcher: str, database: Path) -> ProcessResult:
        return self.run("matching", f"{matcher.capitalize()} matching", self.commands.matcher(matcher, database))

    def map(self, database: Path, images: Path, sparse_dir: Path) -> ProcessResult:
        sparse_dir.mkdir(parents=True, exist_ok=True)
        return self.run("mapper", "Sparse reconstruction (mapper)", self.commands.mapper(database, images, sparse_dir))

    def convert_model(self, model: Path, output: Path, output_type: str) -> ProcessResult:
        if output_type.upper() != "PLY":
            output.mkdir(parents=True, exist_ok=True)
        return self.run("model_converter", "Model conversion",
                        self.commands.model_converter(model, output, output_type), "model_converter")

    def analyze_model(self, model: Path) -> ProcessResult:
        return self.run("model_analyzer", "Model analysis", self.commands.model_analyzer(model))

    def undistort(self, images: Path, model: Path, dense: Path) -> ProcessResult:
        return self.run("undistortion", "Image undistortion", self.commands.image_undistorter(images, model, dense))

    def patch_match(self, dense: Path) -> ProcessResult:
        return self.run("patch_match", "Dense stereo (patch_match_stereo)", self.commands.patch_match_stereo(dense))

    def fuse(self, dense: Path, output: Path) -> ProcessResult:
        return self.run("fusion", "Stereo fusion", self.commands.stereo_fusion(dense, output))

    def poisson(self, fused_ply: Path, output: Path) -> ProcessResult:
        return self.run("meshing", "Poisson meshing", self.commands.poisson_mesher(fused_ply, output), "poisson_mesher")

    def delaunay(self, input_path: Path, output: Path, input_type: str = "dense") -> ProcessResult:
        try:
            args = self.commands.delaunay_mesher(input_path, output, input_type)
        except ValueError as exc:
            raise ColmapStageError("Delaunay meshing", str(exc)) from exc
        return self.run("meshing", "Delaunay meshing", args, "delaunay_mesher")
