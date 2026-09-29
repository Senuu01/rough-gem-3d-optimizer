"""End-to-end reconstruction pipeline for one run directory.

Runs the explicit COLMAP stages in order, records progress in ``status.json``
and writes ``run_config.json`` / ``metrics.json`` for every run, including
failed ones (failures are research data too).

Dense stereo requires a CUDA-enabled COLMAP. When it is unavailable (e.g. on
macOS) and ``reconstruction.sparse_fallback`` is enabled, a coarse surface is
built from the sparse model with COLMAP's Delaunay mesher. That mesh is much
less detailed than a dense reconstruction and is labelled as such.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import statistics
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.ingestion.dataset import IngestionReport, load_ingestion_report, prepare_dataset
from src.ingestion.video import ffmpeg_version
from src.mesh.export import export_mesh
from src.mesh.processor import MeshLoadError, load_mesh, mesh_statistics
from src.preprocessing.masking import frame_mask_info, load_prompts, write_image_list
from src.reconstruction.colmap_runner import (
    ColmapEnvironment,
    ColmapNotFoundError,
    ColmapRunner,
    ColmapStageError,
    detect_colmap,
)
from src.reconstruction.progress import ProgressTracker, write_json_atomic
from src.reconstruction.views import layout_views, list_frame_names
from src.reconstruction.statistics import (
    SparseStats,
    StatisticsError,
    database_statistics,
    parse_sparse_text_model,
    read_ply_header,
)
from src.utils.config import AppConfig, ReconstructionConfig, config_from_dict, load_config
from src.utils.environment import software_versions
from src.utils.logging import configure_run_logging, get_logger
from src.utils.paths import RunPaths

logger = get_logger("pipeline")

RUN_CONFIG_YAML = "config.yaml"
MIN_REGISTERED_IMAGES = 3
# Written before the pipeline starts (UI launcher, frame preparation); not part of an attempt.
_KEEP_LOGS = {"runner_stdout.log", "ingestion.log", "ingestion_ui.log", "masking.log", "masking_stdout.log"}


class PipelineError(RuntimeError):
    """A fatal, explained failure of one pipeline stage."""

    def __init__(self, stage: str, message: str, log_file: Path | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.log_file = log_file


def resolve_matcher(cfg: ReconstructionConfig, dataset_kind: str, image_count: int) -> str:
    """Choose the COLMAP matcher.

    ``auto`` uses sequential matching for video-only datasets (frames are
    temporally ordered) and exhaustive matching for unordered photos or mixed
    datasets, unless the dataset is too large for exhaustive matching.
    """
    if cfg.matcher != "auto":
        return cfg.matcher
    if dataset_kind == "video":
        return "sequential"
    if image_count <= cfg.auto_exhaustive_max_images:
        return "exhaustive"
    return "sequential"


def save_run_config(run: RunPaths, config: AppConfig) -> None:
    """Store the exact configuration a run should use (``<run>/config.yaml``)."""
    (run.root / RUN_CONFIG_YAML).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")


def load_run_config(run: RunPaths) -> AppConfig:
    """Load a run's own configuration, falling back to the project defaults.

    Tool locations always come from the current machine's configuration, so a
    run can be re-run on another machine or after reinstalling COLMAP.
    """
    current = load_config()
    path = run.root / RUN_CONFIG_YAML
    if not path.is_file():
        return current
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["tools"] = current.to_dict()["tools"]
    return config_from_dict(data)


class ReconstructionPipeline:
    """Orchestrates ingestion and COLMAP stages for a single run."""

    def __init__(self, run: RunPaths, config: AppConfig) -> None:
        self.run_paths = run
        self.config = config
        self.rcfg = config.reconstruction
        self.tracker: ProgressTracker | None = None
        self.env: ColmapEnvironment | None = None
        self.runner: ColmapRunner | None = None
        self.report: IngestionReport | None = None
        self.best_model: Path | None = None
        self.dense_ok = False
        self.masks_used: Path | None = None
        self.image_list: Path | None = None
        self.colmap_image_dir: Path | None = None
        self.metrics: dict[str, Any] = {}

    # ---------------------------------------------------------------- helpers
    @contextmanager
    def _stage(self, key: str, message: str = "") -> Iterator[None]:
        assert self.tracker is not None
        self.tracker.start(key, message)
        logger.info("=== Stage: %s ===", key)
        try:
            yield
        except ColmapStageError as exc:
            raise PipelineError(key, f"{exc.stage}: {exc.explanation}", exc.log_file) from exc
        if self.tracker.data["stages"][key]["status"] == "running":
            self.tracker.done(key, message)

    def _warn(self, message: str) -> None:
        logger.warning(message)
        assert self.tracker is not None
        self.tracker.warn(message)

    def _write_run_record(self) -> None:
        run = self.run_paths
        record = {
            "run_id": run.run_id,
            "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "capture_mode": self.config.capture.mode,
            "config": self.config.to_dict(),
            "software": software_versions(),
            "colmap": self.env.to_dict() if self.env else None,
            "ffmpeg": ffmpeg_version(self.config.tools),
            "gpu_used": self.runner.use_gpu if self.runner else None,
            "dataset": self.report.to_dict() if self.report else None,
            "colmap_commands": self.runner.executed if self.runner else [],
        }
        write_json_atomic(run.run_config_path, record)

    # ---------------------------------------------------------------- stages
    def _prepare(self) -> None:
        run = self.run_paths
        with self._stage("prepare", "Checking COLMAP and preparing images"):
            try:
                self.env = detect_colmap(self.config.tools)
            except ColmapNotFoundError as exc:
                raise PipelineError("prepare", str(exc)) from exc
            logger.info("COLMAP: %s", self.env.banner)
            object_cfg = self.config.object_capture if self.config.capture.requires_masks else None
            self.runner = ColmapRunner(self.env, self.rcfg, run.logs_dir, object_cfg=object_cfg)
            if self.rcfg.use_gpu and not self.runner.use_gpu:
                self._warn("COLMAP was built without CUDA (or CUDA was not detected): running on CPU.")

            report = load_ingestion_report(run)
            frames_present = run.frames_dir.is_dir() and any(run.frames_dir.iterdir())
            if report is None or not frames_present:
                report = prepare_dataset(run, self.config)
            self.report = report
            for error in report.errors:
                self._warn(f"Input skipped: {error}")

            count = len(report.images)
            if count < self.rcfg.min_images:
                raise PipelineError(
                    "prepare",
                    f"Insufficient images: {count} usable image(s), at least {self.rcfg.min_images} required. "
                    "Capture more overlapping views or use a shorter video sampling interval.",
                )
            self.metrics["dataset"] = {
                "kind": report.kind, "images": count,
                "photos": report.photo_count, "videos": report.video_count,
                "input_errors": len(report.errors),
            }
            self.tracker.done("prepare", f"{count} images ready ({report.kind})")

    def _quality_and_masks(self) -> None:
        run = self.run_paths
        assert self.tracker is not None and self.report is not None
        self.tracker.skip("quality", "Not implemented yet (Milestone 2): all frames are used unfiltered.")
        if not self.config.capture.requires_masks:
            self.tracker.skip("masks", "Not needed for orbit-camera captures (the static background helps).")
            return

        mode = self.config.capture.mode
        with self._stage("masks", "Checking gemstone masks"):
            infos = frame_mask_info(run, self.report, self.config.masking)
            counts = Counter(info.status for info in infos)
            usable = [i for i in infos if i.usable]
            areas = [i.area_ratio for i in usable if i.area_ratio is not None]
            self.metrics["masks"] = {
                "status_counts": dict(counts),
                "usable_frames": len(usable),
                "manual_masks": sum(1 for i in infos if i.manual),
                "median_area_ratio": round(statistics.median(areas), 5) if areas else None,
                "prompts": {k: asdict(v) for k, v in load_prompts(run).items()},
            }
            if counts.get("missing", 0) == len(infos):
                raise PipelineError(
                    "masks",
                    f"{mode} captures need gemstone masks, otherwise COLMAP reconstructs the static background. "
                    "Open the Preprocess page, click the stone in the first frame and generate masks "
                    "(or run `python -m src.cli mask <run> --point X,Y`).",
                )
            if len(usable) < self.rcfg.min_images:
                raise PipelineError(
                    "masks",
                    f"Only {len(usable)} frames have a usable mask (need {self.rcfg.min_images}). "
                    f"Status counts: {dict(counts)}. Review the masks on the Preprocess page.",
                )
            excluded = len(infos) - len(usable)
            if excluded:
                self._warn(f"{excluded} frame(s) excluded because their mask is missing, empty, too large or rejected.")
            self.image_list = write_image_list(run, infos)
            self.masks_used = run.masks_dir
            self.tracker.done("masks", f"{len(usable)}/{len(infos)} frames masked (median stone area "
                                       f"{100 * (self.metrics['masks']['median_area_ratio'] or 0):.1f}% of frame)")

    def _run_with_cpu_fallback(self, stage_key: str, action, reset=None) -> None:
        """Run ``action``; if it fails on GPU, retry once on CPU."""
        assert self.runner is not None
        try:
            action()
        except ColmapStageError as exc:
            if not self.runner.use_gpu:
                raise
            self._warn(f"{exc.stage} failed on GPU; retrying on CPU.")
            self.runner.set_use_gpu(False)
            if reset:
                reset()
            action()

    def _prepare_views(self) -> None:
        """Point COLMAP at one camera per image size when resolutions differ."""
        run = self.run_paths
        names = list_frame_names(run.frames_dir, self.image_list)
        layout = layout_views(
            run.root,
            run.frames_dir,
            names,
            masks_dir=self.masks_used,
            image_list=self.image_list,
            share_intrinsics=self.rcfg.single_camera,
        )
        self.colmap_image_dir = layout.image_dir
        self.masks_used = layout.mask_dir
        self.image_list = layout.image_list
        self._single_camera = layout.single_camera
        self._single_camera_per_folder = layout.single_camera_per_folder
        self.metrics["cameras"] = {
            "size_counts": layout.size_counts,
            "single_camera": layout.single_camera,
            "single_camera_per_folder": layout.single_camera_per_folder,
        }
        if layout.single_camera_per_folder:
            summary = ", ".join(f"{size} ({count})" for size, count in sorted(layout.size_counts.items()))
            self._warn(
                f"Images have {len(layout.size_counts)} different resolutions ({summary}). "
                "Each resolution uses its own camera so frames are not skipped."
            )

    def _features(self) -> None:
        run = self.run_paths
        assert self.runner is not None
        with self._stage("features", "SIFT feature extraction"):
            self._prepare_views()
            run.database_path.unlink(missing_ok=True)
            self._run_with_cpu_fallback(
                "features",
                lambda: self.runner.extract_features(
                    run.database_path,
                    self.colmap_image_dir or run.frames_dir,
                    self.masks_used,
                    self.image_list,
                    single_camera=self._single_camera,
                    single_camera_per_folder=self._single_camera_per_folder,
                ),
                reset=lambda: run.database_path.unlink(missing_ok=True),
            )
            log = run.logs_dir / "feature_extraction.log"
            skipped = log.read_text(encoding="utf-8", errors="replace").count("CAMERA_SINGLE_DIM_ERROR") if log.is_file() else 0
            if skipped:
                raise PipelineError(
                    "features",
                    f"{skipped} images were skipped because they do not share one image size while "
                    "shared-camera mode is on. Re-run reconstruction; mixed photo and video sizes are now "
                    "split into one camera per resolution.",
                    log,
                )
            stats = database_statistics(run.database_path)
            self.metrics["features"] = stats.to_dict()
            if stats.total_keypoints == 0:
                raise PipelineError(
                    "features",
                    "No features were detected. The surface may be too smooth, shiny, blurred or poorly lit. "
                    "Try diffuse lighting, a textured background (orbit mode) or a matte scanning spray.",
                    run.logs_dir / "feature_extraction.log",
                )
            without = stats.images - stats.images_with_features
            if without:
                self._warn(f"{without} image(s) produced no features.")
            self.tracker.done(
                "features", f"{stats.total_keypoints:,} keypoints in {stats.images_with_features} images"
            )

    def _matching(self) -> None:
        run = self.run_paths
        assert self.runner is not None and self.report is not None
        matcher = resolve_matcher(self.rcfg, self.report.kind, len(self.report.images))
        self.metrics["matcher"] = matcher
        if self.report.kind == "mixed" and matcher == "sequential":
            self._warn("Mixed photo/video dataset is being matched sequentially; photos may fail to register.")
        with self._stage("matching", f"{matcher} matching"):
            self._run_with_cpu_fallback("matching", lambda: self.runner.match(matcher, run.database_path))
            stats = database_statistics(run.database_path)
            self.metrics["matching"] = stats.to_dict()
            if stats.verified_pairs == 0:
                raise PipelineError(
                    "matching",
                    f"Poor feature matching: {stats.images_with_features} images have features "
                    f"({stats.total_keypoints:,} keypoints) but no pair could be geometrically verified. "
                    "The views may not overlap, or the stone is too clear or shiny for SIFT to match. "
                    "A matte coating such as developer spray usually gives a clear stone enough texture.",
                    run.logs_dir / "matching.log",
                )
            self.tracker.done("matching", f"{matcher}: {stats.verified_pairs} verified image pairs")

    def _sparse(self) -> None:
        run = self.run_paths
        assert self.runner is not None and self.report is not None
        total = self.metrics.get("masks", {}).get("usable_frames") or len(self.report.images)
        with self._stage("sparse", "Incremental Structure-from-Motion"):
            if run.sparse_dir.exists():
                shutil.rmtree(run.sparse_dir)
            self.runner.map(run.database_path, self.colmap_image_dir or run.frames_dir, run.sparse_dir)

            models = sorted(
                d for d in run.sparse_dir.iterdir()
                if d.is_dir() and ((d / "cameras.bin").is_file() or (d / "cameras.txt").is_file())
            )
            if not models:
                raise PipelineError(
                    "sparse",
                    "The mapper produced no reconstruction: no image pair had enough parallax/matches to "
                    "initialise. Capture views from more varied angles with more overlap.",
                    run.logs_dir / "mapper.log",
                )

            txt_root = run.root / "sparse_txt"
            if txt_root.exists():
                shutil.rmtree(txt_root)
            all_stats: list[tuple[Path, SparseStats]] = []
            for model in models:
                txt_dir = txt_root / model.name
                self.runner.convert_model(model, txt_dir, "TXT")
                all_stats.append((model, parse_sparse_text_model(txt_dir, total)))

            self.best_model, best = max(all_stats, key=lambda item: item[1].registered_images)
            self.metrics["sparse"] = best.to_dict()
            self.metrics["sparse"]["model_count"] = len(models)
            self.metrics["sparse"]["all_models"] = {m.name: s.to_dict() for m, s in all_stats}
            if len(models) > 1:
                self._warn(
                    f"The reconstruction split into {len(models)} disconnected models; using model "
                    f"{self.best_model.name} with {best.registered_images} images."
                )
            try:
                self.runner.analyze_model(self.best_model)
            except ColmapStageError as exc:
                self._warn(f"model_analyzer failed (non-fatal): {exc.explanation}")

            if best.registered_images < MIN_REGISTERED_IMAGES:
                raise PipelineError(
                    "sparse",
                    f"Too few images registered ({best.registered_images}/{total}). The capture does not "
                    "provide enough overlapping, textured views.",
                    run.logs_dir / "mapper.log",
                )
            if best.registration_ratio < self.rcfg.min_registered_ratio:
                self._warn(
                    f"Only {100 * best.registration_ratio:.0f}% of images registered "
                    f"(threshold {100 * self.rcfg.min_registered_ratio:.0f}%). Geometry is likely incomplete."
                )
            self.runner.convert_model(self.best_model, run.sparse_points_ply, "PLY")
            self.tracker.done(
                "sparse",
                f"{best.registered_images}/{total} images registered "
                f"({100 * best.registration_ratio:.0f}%), {best.points3d:,} points",
            )

    def _dense(self) -> None:
        run = self.run_paths
        assert self.runner is not None and self.env is not None and self.tracker is not None
        dense_cfg = self.rcfg.dense
        self.metrics["dense"] = {"enabled": dense_cfg.enabled, "ran": False, "fused_points": None}

        if not dense_cfg.enabled:
            reason = "Disabled in configuration."
        elif self.env.has_cuda is False:
            reason = ("This COLMAP build has no CUDA support (always the case on macOS); "
                      "patch_match_stereo requires an NVIDIA GPU.")
        else:
            reason = ""
        if reason:
            self.tracker.skip("dense", reason)
            self.tracker.skip("fusion", "Dense reconstruction was not run.")
            self.metrics["dense"]["skipped_reason"] = reason
            return

        try:
            with self._stage("dense", "Undistorting images and computing depth maps"):
                if run.dense_dir.exists():
                    shutil.rmtree(run.dense_dir)
                run.dense_dir.mkdir(parents=True)
                self.runner.undistort(self.colmap_image_dir or run.frames_dir, self.best_model, run.dense_dir)
                self.runner.patch_match(run.dense_dir)
            with self._stage("fusion", "Fusing depth maps into a point cloud"):
                self.runner.fuse(run.dense_dir, run.fused_ply, self.masks_used)
                points = read_ply_header(run.fused_ply).vertex_count
                self.metrics["dense"].update(ran=True, fused_points=points)
                if points == 0:
                    raise PipelineError("fusion", "Stereo fusion produced an empty point cloud.",
                                        run.logs_dir / "fusion.log")
                self.tracker.done("fusion", f"{points:,} dense points")
            self.dense_ok = True
        except (PipelineError, StatisticsError) as exc:
            stage = getattr(exc, "stage", "fusion")
            message = getattr(exc, "message", str(exc))
            if not self.rcfg.sparse_fallback:
                raise
            self.tracker.fail(stage, f"{message} Falling back to sparse meshing.", fatal=False)
            if stage == "dense":
                self.tracker.skip("fusion", "Dense reconstruction failed.")
            self.metrics["dense"]["error"] = message
            self._warn(f"Dense reconstruction failed; using the sparse fallback. ({message})")

    def _mesh(self) -> None:
        run = self.run_paths
        assert self.runner is not None and self.best_model is not None
        with self._stage("mesh", "Surface meshing"):
            if self.dense_ok:
                if self.rcfg.mesher == "poisson":
                    mesh_path, source = run.dense_dir / "meshed-poisson.ply", "dense_poisson"
                    self.runner.poisson(run.fused_ply, mesh_path)
                else:
                    mesh_path, source = run.dense_dir / "meshed-delaunay.ply", "dense_delaunay"
                    self.runner.delaunay(run.dense_dir, mesh_path, "dense")
            elif self.rcfg.sparse_fallback:
                mesh_path, source = run.sparse_dir / "meshed-delaunay-sparse.ply", "sparse_delaunay"
                self._warn("Mesh is built from SPARSE points only: coarse, low-detail surface.")
                self.runner.delaunay(self.best_model, mesh_path, "sparse")
            else:
                raise PipelineError("mesh", "No dense point cloud is available and sparse_fallback is disabled.")

            header = read_ply_header(mesh_path) if mesh_path.is_file() else None
            if header is None or header.face_count == 0:
                log_name = "poisson_mesher.log" if source == "dense_poisson" else "delaunay_mesher.log"
                raise PipelineError("mesh", "The mesher produced no triangles.", run.logs_dir / log_name)
            shutil.copy2(mesh_path, run.raw_mesh_ply)
            self.metrics["mesh"] = {"source": source, "colmap_output": str(mesh_path.relative_to(run.root)),
                                    "raw_mesh": str(run.raw_mesh_ply.relative_to(run.root))}
            self.tracker.done("mesh", f"{source}: {header.face_count:,} faces")

    def _analyse_mesh(self) -> None:
        run = self.run_paths
        assert self.tracker is not None
        self.tracker.start("cleaning", "Computing raw mesh statistics")
        try:
            stats = mesh_statistics(load_mesh(run.raw_mesh_ply))
        except MeshLoadError as exc:
            raise PipelineError("cleaning", str(exc)) from exc
        self.metrics["mesh"]["stats"] = stats.to_dict()
        exports = export_mesh(run.raw_mesh_ply, run.output_dir)
        self.metrics["mesh"]["exports"] = {fmt: p.name for fmt, p in exports.items()}
        self.tracker.skip(
            "cleaning",
            "Cleaning not implemented yet (Milestone 2); raw mesh left unmodified. Statistics and "
            f"{', '.join(f.upper() for f in exports)} exports written.",
        )

    def _archive_previous_attempt(self) -> None:
        """Move logs/metrics of an earlier attempt in this run to ``logs/archive/<timestamp>/``."""
        run = self.run_paths
        previous = [p for p in run.logs_dir.iterdir() if p.is_file() and p.name not in _KEEP_LOGS]
        previous += [p for p in (run.metrics_path, run.run_config_path) if p.is_file()]
        if not previous:
            return
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        archive = run.logs_dir / "archive" / stamp
        archive.mkdir(parents=True, exist_ok=True)
        for path in previous:
            shutil.move(str(path), archive / path.name)

    # ---------------------------------------------------------------- driver
    def run(self) -> dict[str, Any]:
        """Execute the pipeline. Always writes status, run_config and metrics files."""
        run = self.run_paths.create()
        self._archive_previous_attempt()
        configure_run_logging(run.logs_dir / "pipeline.log")
        self.tracker = ProgressTracker(run.status_path, run.run_id)
        started = time.monotonic()
        state = "failed"
        logger.info("Starting run %s (capture mode: %s)", run.run_id, self.config.capture.mode)
        try:
            self._write_run_record()
            self._prepare()
            self._write_run_record()
            self._quality_and_masks()
            self._features()
            self._matching()
            self._sparse()
            self._dense()
            self._mesh()
            self._analyse_mesh()
            state = "completed"
        except PipelineError as exc:
            logger.error("Stage '%s' failed: %s", exc.stage, exc.message)
            log = str(exc.log_file.relative_to(run.root)) if exc.log_file and exc.log_file.is_relative_to(run.root) else None
            self.tracker.fail(exc.stage, exc.message, log)
        except (KeyboardInterrupt, SystemExit):
            state = "cancelled"
            current = self.tracker.data.get("current_stage") or "prepare"
            self.tracker.fail(current, "Run cancelled by user.")
            raise
        except Exception as exc:  # unexpected bug: keep the traceback in the log, not the UI
            logger.exception("Unexpected error")
            current = self.tracker.data.get("current_stage") or "prepare"
            self.tracker.fail(current, f"Unexpected internal error ({exc}). See logs/pipeline.log.")
        finally:
            self.metrics.update(
                run_id=run.run_id,
                state=state,
                capture_mode=self.config.capture.mode,
                gpu_used=self.runner.use_gpu if self.runner else None,
                total_time_s=round(time.monotonic() - started, 2),
                stage_durations_s=self.tracker.stage_durations(),
                warnings=self.tracker.data["warnings"],
                error=self.tracker.data["error"],
            )
            write_json_atomic(run.metrics_path, self.metrics)
            self._write_run_record()
            self.tracker.finish(state)
            logger.info("Run %s finished: %s (%.1fs)", run.run_id, state, self.metrics["total_time_s"])
        return self.metrics
