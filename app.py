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
1. **Upload** - add photos/videos, choose capture mode (orbit / turntable / hand-held) and sampling.
2. **Preprocess** - inspect frames; for turntable / hand-held captures, click the stone and generate
   SAM 2 masks so COLMAP only sees the stone.
3. **Reconstruct** - run COLMAP stage by stage with live progress and logs.
4. **3D Model** - interactive mesh / point-cloud viewer and exports.
5. **Analysis** - geometric statistics (unscaled until calibration exists).

Not yet implemented (reported as skipped): frame-quality filtering, mesh cleaning, scale calibration.
"""
)
st.info("Verify the pipeline with an opaque, textured object (e.g. a rough rock) before testing a gemstone.")

sidebar_run_selector()
