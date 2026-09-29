"""Command-line interface.

Examples::

    python -m src.cli check
    python -m src.cli run "path/to/stone video.mov" --sampling interval --interval 0.5
    python -m src.cli run photos/*.jpg --mode orbit_camera --matcher exhaustive
    python -m src.cli reconstruct runs/2026-09-29_001      # (re)run an existing run directory

The Streamlit UI launches ``reconstruct`` as a background process.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
from pathlib import Path

from src.ingestion.video import ffmpeg_version
from src.reconstruction.colmap_runner import ColmapNotFoundError, detect_colmap
from src.reconstruction.pipeline import ReconstructionPipeline, load_run_config, save_run_config
from src.utils.config import ConfigError, load_config
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
    return _run_pipeline(run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Rough gemstone 3D reconstruction")
    parser.add_argument("--config", type=Path, default=None, help="Alternative config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Check COLMAP / FFmpeg installation").set_defaults(func=_cmd_check)

    rec = sub.add_parser("reconstruct", help="Run the pipeline on an existing run directory")
    rec.add_argument("run_dir")
    rec.set_defaults(func=_cmd_reconstruct)

    run = sub.add_parser("run", help="Create a new run from photos/videos and reconstruct")
    run.add_argument("inputs", nargs="+", help="Photo and/or video files")
    run.add_argument("--mode", choices=["orbit_camera", "turntable"], default="orbit_camera")
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
