"""Inspect the frames prepared for COLMAP."""

from __future__ import annotations

import streamlit as st

from src.ingestion.dataset import load_ingestion_report, prepare_dataset
from src.reconstruction.pipeline import load_run_config, save_run_config
from src.ui.common import require_run
from src.ui.launcher import is_running
from src.ui.masks_panel import render_masks_panel
from src.utils.logging import configure_run_logging

st.set_page_config(page_title="Preprocess", page_icon="🖼️", layout="wide")
st.title("2. Preprocess")
run = require_run()
config = load_run_config(run)
report = load_ingestion_report(run)

if report is None:
    st.info("This run has no prepared frames yet.")
    st.stop()

cols = st.columns(4)
cols[0].metric("Images for COLMAP", len(report.images))
cols[1].metric("Photos", report.photo_count)
cols[2].metric("Videos", report.video_count)
cols[3].metric("Input errors", len(report.errors))
for error in report.errors:
    st.error(error)

st.info(
    "Image-quality metrics (blur, exposure, duplicates) are not computed yet (Milestone 2); "
    "all frames are passed to COLMAP unless excluded by masking below."
)

with st.expander("Re-extract video frames with different sampling"):
    interval = st.select_slider("Seconds between frames", [0.1, 0.25, 0.5, 1.0, 2.0],
                                value=config.video.interval_seconds if config.video.interval_seconds in
                                (0.1, 0.25, 0.5, 1.0, 2.0) else 0.5)
    auto = st.checkbox("Automatic (motion-based) sampling", config.video.sampling == "auto")
    if st.button("Re-prepare frames", disabled=is_running(run)):
        config.video.interval_seconds = float(interval)
        config.video.sampling = "auto" if auto else "interval"
        save_run_config(run, config)
        configure_run_logging(run.logs_dir / "ingestion_ui.log")
        with st.spinner("Re-extracting..."):
            prepare_dataset(run, config)
        st.rerun()

st.subheader("Frames")
table = [
    {
        "filename": img.output_name,
        "resolution": f"{img.width}x{img.height}",
        "origin": img.origin,
        "source file": img.source,
        "time (s)": img.timestamp_s,
        "action": img.action,
        "status": "used",
    }
    for img in report.images
]
st.dataframe(table, width="stretch", hide_index=True)

per_page = 24
pages = max(1, -(-len(report.images) // per_page))
page = st.number_input("Gallery page", 1, pages, 1) if pages > 1 else 1
start = (page - 1) * per_page
grid = st.columns(6)
for i, img in enumerate(report.images[start:start + per_page]):
    path = run.frames_dir / img.output_name
    if path.is_file():
        grid[i % 6].image(str(path), caption=img.output_name, width="stretch")

render_masks_panel(run, config, report)

st.page_link("pages/3_Reconstruct.py", label="Continue to Reconstruct", icon="➡️")
