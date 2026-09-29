"""Run COLMAP and display live stage progress, statistics and logs."""

from __future__ import annotations

import json

import streamlit as st

from src.ingestion.dataset import load_ingestion_report
from src.preprocessing.masking import frame_mask_info
from src.reconstruction.pipeline import load_run_config
from src.reconstruction.progress import STAGES, read_status
from src.ui.common import environment_status, render_environment, require_run
from src.ui.launcher import cancel, is_running, launch
from src.utils.paths import RunPaths

st.set_page_config(page_title="Reconstruct", page_icon="⚙️", layout="wide")
st.title("3. Reconstruct")
run = require_run()
config = load_run_config(run)
env = environment_status()

_ICONS = {"pending": "⚪", "running": "🔄", "done": "✅", "skipped": "⏭️", "failed": "❌"}


def _tail(path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])


def _render_status(run: RunPaths) -> None:
    status = read_status(run.status_path)
    running = is_running(run)
    if not status:
        st.caption("Not started yet.")
        return
    state = status.get("state", "unknown")
    if state in ("starting", "running") and not running:
        state = "interrupted"

    stages = status.get("stages", {})
    finished = sum(1 for s in stages.values() if s.get("status") in ("done", "skipped", "failed"))
    st.progress(finished / len(STAGES), text=f"State: **{state}**")

    for index, (key, label) in enumerate(STAGES, start=1):
        stage = stages.get(key, {})
        icon = _ICONS.get(stage.get("status", "pending"), "⚪")
        duration = f" ({stage['duration_s']:.1f}s)" if stage.get("duration_s") else ""
        message = f" - {stage['message']}" if stage.get("message") else ""
        st.markdown(f"{icon} **{index}. {label}**{duration}{message}")

    for warning in status.get("warnings", []):
        st.warning(warning)
    error = status.get("error")
    if error:
        st.error(f"**{error['stage']} failed:** {error['message']}")
        if error.get("log"):
            with st.expander(f"Log: {error['log']}"):
                st.code(_tail(run.root / error["log"], 60) or "(empty)")
    if state == "interrupted":
        st.error("The reconstruction process stopped unexpectedly. See logs/pipeline.log and runner_stdout.log.")

    with st.expander("Pipeline log (live)", expanded=running):
        st.code(_tail(run.logs_dir / "pipeline.log", 40) or "(no output yet)")


@st.fragment(run_every=2)
def live_status() -> None:
    _render_status(run)
    if not is_running(run) and st.session_state.get("was_running"):
        st.session_state["was_running"] = False
        st.rerun()  # refresh the whole page once to show final statistics


st.subheader("Environment")
render_environment(env)

report = load_ingestion_report(run)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Images", len(report.images) if report else 0)
col2.metric("Capture mode", config.capture.mode)
col3.metric("Matcher", config.reconstruction.matcher)
col4.metric("Dense", "on" if config.reconstruction.dense.enabled else "off")

masks_ready = True
if config.capture.requires_masks and report:
    usable = sum(1 for i in frame_mask_info(run, report, config.masking) if i.usable)
    masks_ready = usable >= config.reconstruction.min_images
    if usable == 0:
        st.warning(
            f"**{config.capture.mode} mode needs gemstone masks before reconstruction.** Go to Preprocess, "
            "click the stone in the first frame of each video, then press *Generate masks*."
        )
        st.page_link("pages/2_Preprocess.py", label="Go to Preprocess to create masks", icon="🎭")
    elif not masks_ready:
        st.warning(f"Only {usable} frames have usable masks (need {config.reconstruction.min_images}).")
    else:
        st.caption(f"{usable}/{len(report.images)} frames have usable masks and will be sent to COLMAP.")

running = is_running(run)
b1, b2, _ = st.columns([1, 1, 4])
start_label = "Re-run reconstruction" if run.status_path.is_file() else "Start reconstruction"
if b1.button(start_label, type="primary",
             disabled=running or env.colmap is None or not report or not masks_ready):
    launch(run)
    st.session_state["was_running"] = True
    st.rerun()
if b2.button("Cancel", disabled=not running):
    cancel(run)
    st.rerun()
if running:
    st.session_state["was_running"] = True
if run.status_path.is_file() and not running:
    st.caption("Re-running overwrites this run's COLMAP outputs. Create a new run to keep both for comparison.")

st.subheader("Progress")
live_status()

if not running and run.metrics_path.is_file():
    metrics = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    sparse = metrics.get("sparse")
    if sparse:
        st.subheader("Sparse reconstruction statistics")
        c = st.columns(6)
        c[0].metric("Input images", sparse["total_input_images"])
        c[1].metric("Registered", sparse["registered_images"])
        c[2].metric("Registration", f"{sparse['registration_percentage']}%")
        c[3].metric("Sparse 3D points", f"{sparse['points3d']:,}")
        c[4].metric("Observations", f"{sparse['observations']:,}")
        mean_err, median_err = sparse.get("mean_reprojection_error_px"), sparse.get("median_reprojection_error_px")
        c[5].metric("Reproj. error mean / median",
                    f"{mean_err:.3f} / {median_err:.3f} px" if mean_err is not None else "n/a")
    dense = metrics.get("dense")
    mesh = metrics.get("mesh", {})
    if dense or mesh.get("stats"):
        st.subheader("Dense / mesh statistics")
        stats = mesh.get("stats", {})
        c = st.columns(6)
        c[0].metric("Dense points", f"{dense['fused_points']:,}" if dense and dense.get("fused_points") else "n/a")
        c[1].metric("Mesh vertices", f"{stats['vertices']:,}" if stats else "n/a")
        c[2].metric("Mesh faces", f"{stats['faces']:,}" if stats else "n/a")
        c[3].metric("Watertight", str(stats.get("watertight", "n/a")))
        c[4].metric("Components", stats.get("connected_components", "n/a"))
        c[5].metric("Mesh source", mesh.get("source", "n/a"))
        if stats:
            ext = stats["bbox_extents"]
            st.caption(
                f"Bounding box: {ext[0]:.4g} x {ext[1]:.4g} x {ext[2]:.4g} {stats['units']}. "
                "These are completeness indicators only - a completed reconstruction is not necessarily "
                "accurate; validate against physical measurements."
            )
        if mesh.get("source") == "sparse_delaunay":
            st.warning("This mesh was built from sparse points only (no dense stereo) and is coarse.")
    st.caption(f"Total time: {metrics.get('total_time_s', 0):.1f}s. Full records: run_config.json, metrics.json, logs/.")
    if metrics.get("state") == "completed":
        st.page_link("pages/4_3D_Model.py", label="Open the 3D model", icon="➡️")
