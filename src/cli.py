"""Command-line interface.

Examples::

    python -m src.cli check
    python -m src.cli run "path/to/stone video.mov" --sampling interval --interval 0.5
    python -m src.cli run photos/*.jpg --mode orbit_camera --matcher exhaustive
    python -m src.cli reconstruct runs/2026-09-29_001      # (re)run an existing run directory
    python -m src.cli run stone.mov --mode handheld --point 1100,1740   # masked hand-held capture
    python -m src.cli mask runs/2026-09-29_001 --point 1100,1740        # (re)generate masks only

The Streamlit UI launches ``reconstruct`` as a background process.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
from pathlib import Path

from src.ingestion.dataset import load_ingestion_report, prepare_dataset
from src.ingestion.video import ffmpeg_version
from src.preprocessing.masking import (
    MaskingError,
    MaskPrompt,
    build_sequences,
    generate_masks,
    load_prompts,
    masking_status_path,
    save_prompts,
)
from src.reconstruction.colmap_runner import ColmapNotFoundError, detect_colmap
from src.reconstruction.pipeline import ReconstructionPipeline, load_run_config, save_run_config
from src.reconstruction.progress import write_json_atomic
from src.utils.config import CAPTURE_MODES, ConfigError, load_config
from src.utils.logging import configure_run_logging, get_logger
from src.utils.paths import RunPaths, create_run


def _cmd_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ok = True
    try:
        env = detect_colmap(config.tools)
        print(f"[ok]   COLMAP {env.version}: {env.executable}")
        print(f"       {env.banner}")
        print(f"       CUDA (dense stereo): {'yes' if env.has_cuda else 'no' if env.has_cuda is False else 'unknown'}")
    except ColmapNotFoundError as exc:
        ok = False
        print(f"[fail] {exc}")
    version = ffmpeg_version(config.tools)
    if version:
        print(f"[ok]   {version}")
    else:
        print("[warn] FFmpeg not found; video frames will be decoded with OpenCV instead.")
    return 0 if ok else 1


def _signal_to_exit(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _run_pipeline(run: RunPaths) -> int:
    signal.signal(signal.SIGTERM, _signal_to_exit)
    config = load_run_config(run)
    metrics = ReconstructionPipeline(run, config).run()
    print(json.dumps({k: metrics.get(k) for k in ("run_id", "state", "total_time_s", "error")}, indent=2))
    return 0 if metrics.get("state") == "completed" else 2


def _parse_points(values: list[str] | None) -> list[tuple[float, float]]:
    points = []
    for value in values or []:
        try:
            x, y = (float(v) for v in value.split(","))
        except ValueError as exc:
            raise SystemExit(f"Invalid point {value!r}; expected X,Y in pixels of the first frame") from exc
        points.append((x, y))
    return points


def _run_masking(run: RunPaths, point: list[str] | None = None, negative: list[str] | None = None) -> int:
    """Generate SAM 2 masks, writing progress to ``masks/status.json``."""
    config = load_run_config(run)
    configure_run_logging(run.logs_dir / "masking.log")
    status_path = masking_status_path(run)
    status = {"state": "running", "pid": os.getpid(), "done": 0, "total": 0, "message": "Loading SAM 2..."}
    write_json_atomic(status_path, status)

    try:
        report = load_ingestion_report(run)
        if report is None or not report.images:
            raise MaskingError("No prepared frames in this run; prepare the images first.")
        prompts = load_prompts(run)
        positives, negatives = _parse_points(point), _parse_points(negative)
        if positives:
            for seq in build_sequences(report):
                prompt = MaskPrompt()
                for x, y in positives:
                    prompt.add(x, y, positive=True)
                for x, y in negatives:
                    prompt.add(x, y, positive=False)
                prompts[seq.key] = prompt
            save_prompts(run, prompts)

        def progress(done: int, total: int, message: str) -> None:
            status.update(done=done, total=total, message=message)
            write_json_atomic(status_path, status)

        written = generate_masks(run, report, config.masking, prompts, progress)
        status.update(state="completed", message=f"Masks written: {sum(written.values())}")
    except MaskingError as exc:
        status.update(state="failed", message=str(exc))
    except Exception as exc:  # keep the traceback in the log, a short message in the UI
        get_logger("cli").exception("Mask generation failed")
        status.update(state="failed", message=f"Unexpected error ({exc}). See logs/masking.log.")
    write_json_atomic(status_path, status)
    print(status["message"])
    return 0 if status["state"] == "completed" else 2


def _cmd_mask(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGTERM, _signal_to_exit)
    run = RunPaths(Path(args.run_dir).resolve())
    if not run.root.is_dir():
        print(f"Run directory not found: {run.root}", file=sys.stderr)
        return 1
    return _run_masking(run, args.point, args.negative)


def _cmd_reconstruct(args: argparse.Namespace) -> int:
    run = RunPaths(Path(args.run_dir).resolve())
    if not run.root.is_dir():
        print(f"Run directory not found: {run.root}", file=sys.stderr)
        return 1
    return _run_pipeline(run)


def _cmd_run(args: argparse.Namespace) -> int:
    overrides: dict = {"capture": {"mode": args.mode}, "video": {}, "reconstruction": {}}
    if args.sampling:
        overrides["video"]["sampling"] = args.sampling
    if args.interval:
        overrides["video"]["interval_seconds"] = args.interval
    if args.matcher:
        overrides["reconstruction"]["matcher"] = args.matcher
    if args.cpu:
        overrides["reconstruction"]["use_gpu"] = False
    config = load_config(args.config, overrides)

    run = create_run(config.paths.resolved_runs_dir())
    for source in args.inputs:
        src = Path(source).expanduser()
        if not src.is_file():
            print(f"Input not found: {src}", file=sys.stderr)
            return 1
        shutil.copy2(src, run.input_dir / src.name)
    save_run_config(run, config)
    print(f"Created run {run.root}")
    if config.capture.requires_masks:
        if not args.point:
            print(f"{args.mode} mode needs masks: pass --point X,Y (stone position in the first frame).",
                  file=sys.stderr)
            return 1
        configure_run_logging(run.logs_dir / "ingestion.log")
        prepare_dataset(run, config)
        if _run_masking(run, args.point, args.negative) != 0:
            return 2
    return _run_pipeline(run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Rough gemstone 3D reconstruction")
    parser.add_argument("--config", type=Path, default=None, help="Alternative config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Check COLMAP / FFmpeg installation").set_defaults(func=_cmd_check)

    rec = sub.add_parser("reconstruct", help="Run the pipeline on an existing run directory")
    rec.add_argument("run_dir")
    rec.set_defaults(func=_cmd_reconstruct)

    mask = sub.add_parser("mask", help="Generate SAM 2 gemstone masks for an existing run")
    mask.add_argument("run_dir")
    mask.add_argument("--point", action="append", help="X,Y of the stone in the first frame (repeatable)")
    mask.add_argument("--negative", action="append", help="X,Y of something that is NOT the stone (repeatable)")
    mask.set_defaults(func=_cmd_mask)

    run = sub.add_parser("run", help="Create a new run from photos/videos and reconstruct")
    run.add_argument("inputs", nargs="+", help="Photo and/or video files")
    run.add_argument("--mode", choices=list(CAPTURE_MODES), default="orbit_camera")
    run.add_argument("--point", action="append", help="turntable/handheld: X,Y of the stone in the first frame")
    run.add_argument("--negative", action="append", help="turntable/handheld: X,Y of a non-stone point")
    run.add_argument("--sampling", choices=["interval", "auto"])
    run.add_argument("--interval", type=float, help="Seconds between sampled video frames")
    run.add_argument("--matcher", choices=["auto", "exhaustive", "sequential"])
    run.add_argument("--cpu", action="store_true", help="Force CPU for feature extraction/matching")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
