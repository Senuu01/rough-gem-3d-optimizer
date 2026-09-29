"""Streamlit entry point: ``streamlit run app.py``."""

from __future__ import annotations

import streamlit as st

from src.ui.common import LIMITATIONS_NOTE, environment_status, render_environment, sidebar_run_selector

st.set_page_config(page_title="Rough Gem 3D Reconstructor", page_icon="💎", layout="wide")

st.title("Rough Gem 3D Reconstructor")
st.markdown(
    "Reconstruct the external surface of a single rough gemstone from smartphone photos and/or videos "
    "using [COLMAP](https://colmap.github.io/) photogrammetry."
)
st.warning(LIMITATIONS_NOTE)

st.subheader("Environment")
render_environment(environment_status())

st.subheader("Workflow")
st.markdown(
    """
1. **Upload** - add photos/videos, choose capture mode and sampling, prepare frames.
2. **Preprocess** - inspect the frames that will be sent to COLMAP.
3. **Reconstruct** - run COLMAP stage by stage with live progress and logs.
4. **3D Model** - interactive mesh / point-cloud viewer and exports.
5. **Analysis** - geometric statistics (unscaled until calibration exists).

Milestone 1 status: frame-quality filtering, masking, mesh cleaning and scale calibration are **not yet
implemented**; the corresponding stages are reported as skipped.
"""
)
st.info("Verify the pipeline with an opaque, textured object (e.g. a rough rock) before testing a gemstone.")

sidebar_run_selector()
