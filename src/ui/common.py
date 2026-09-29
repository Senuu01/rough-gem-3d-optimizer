"""Shared Streamlit helpers: configuration, environment checks, run selection."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.ingestion.video import ffmpeg_version
from src.reconstruction.colmap_runner import ColmapEnvironment, ColmapNotFoundError, detect_colmap
from src.utils.config import AppConfig, load_config
from src.utils.paths import RunPaths, list_runs

RUN_KEY = "current_run_dir"

LIMITATIONS_NOTE = (
    "This tool reconstructs only the **external visible surface**. It does not detect inclusions, "
    "internal cracks, gemstone identity, quality or value. Shiny, translucent or texture-poor stones "
    "often reconstruct poorly or not at all; results must be validated against physical measurements."
)


@dataclass
class EnvironmentStatus:
    colmap: ColmapEnvironment | None
    colmap_error: str | None
    ffmpeg: str | None


def app_config() -> AppConfig:
    """Project default configuration (re-read on every rerun so edits apply)."""
    return load_config()


@st.cache_data(ttl=60, show_spinner=False)
def _environment() -> tuple[dict | None, str | None, str | None]:
    config = load_config()
    try:
        env = detect_colmap(config.tools).to_dict()
        error = None
    except ColmapNotFoundError as exc:
        env, error = None, str(exc)
    return env, error, ffmpeg_version(config.tools)


def environment_status() -> EnvironmentStatus:
    env, error, ffmpeg = _environment()
    return EnvironmentStatus(ColmapEnvironment(**env) if env else None, error, ffmpeg)


def render_environment(status: EnvironmentStatus) -> None:
    """Show COLMAP / FFmpeg / CUDA availability."""
    if status.colmap:
        st.success(f"COLMAP {status.colmap.version} found at `{status.colmap.executable}`")
        if status.colmap.has_cuda:
            st.info("COLMAP has CUDA support: dense reconstruction is available.")
        else:
            st.warning(
                "COLMAP has no CUDA support on this machine (always the case on macOS). Dense stereo "
                "cannot run; the pipeline will build a **coarse mesh from the sparse model** instead. "
                "For detailed meshes, run on a machine with an NVIDIA GPU and CUDA-enabled COLMAP."
            )
    else:
        st.error(status.colmap_error or "COLMAP not found")
    if status.ffmpeg:
        st.caption(f"FFmpeg: {status.ffmpeg}")
    else:
        st.caption("FFmpeg not found - video frames will be decoded with OpenCV (install FFmpeg for robustness).")


def current_run() -> RunPaths | None:
    value = st.session_state.get(RUN_KEY)
    return RunPaths(value) if value and value.is_dir() else None


def set_current_run(run: RunPaths) -> None:
    st.session_state[RUN_KEY] = run.root


def sidebar_run_selector() -> RunPaths | None:
    """Sidebar select box listing all runs; returns the selected run."""
    runs = list_runs(app_config().paths.resolved_runs_dir())
    with st.sidebar:
        st.markdown("### Run")
        if not runs:
            st.caption("No runs yet - start on the Upload page.")
            return None
        roots = [r.root for r in runs]
        selected = current_run()
        index = roots.index(selected.root) if selected and selected.root in roots else 0
        choice = st.selectbox("Selected run", roots, index=index, format_func=lambda p: p.name)
        run = RunPaths(choice)
        set_current_run(run)
        st.caption(f"`{run.root}`")
        return run


def require_run() -> RunPaths:
    """Return the selected run or stop the page with a hint."""
    run = sidebar_run_selector()
    if run is None:
        st.info("No run selected. Create one on the **Upload** page first.")
        st.stop()
    return run
