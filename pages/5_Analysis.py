"""Geometric information about the reconstructed mesh (no gemstone assessment)."""

from __future__ import annotations

import json

import streamlit as st

from src.ui.common import require_run

st.set_page_config(page_title="Analysis", page_icon="📐", layout="wide")
st.title("5. Analysis")
run = require_run()

if not run.metrics_path.is_file():
    st.info("No metrics yet. Run the reconstruction first.")
    st.stop()

metrics = json.loads(run.metrics_path.read_text(encoding="utf-8"))
stats = metrics.get("mesh", {}).get("stats")

st.caption(
    "Only external surface geometry is analysed. No inclusion, crack, quality, identity or value "
    "assessment is performed. Cut optimisation is out of scope for this application."
)

if not stats:
    st.warning(f"This run has no mesh (state: {metrics.get('state')}).")
else:
    units = stats["units"]
    st.subheader("Dimensions")
    ext = stats["bbox_extents"]
    c = st.columns(3)
    c[0].metric("X", f"{ext[0]:.4g}")
    c[1].metric("Y", f"{ext[1]:.4g}")
    c[2].metric("Z", f"{ext[2]:.4g}")
    st.caption(
        f"Axis-aligned bounding box in {units}. COLMAP's scale is arbitrary: these are NOT millimetres "
        "until scale calibration (Milestone 2) is applied."
    )
    st.markdown(f"Bounding box min `{stats['bbox_min']}` · max `{stats['bbox_max']}`")

    st.subheader("Surface")
    c = st.columns(4)
    c[0].metric("Surface area", f"{stats['surface_area']:.4g}", help=f"{units} squared")
    c[1].metric("Watertight", str(stats["watertight"]))
    c[2].metric("Consistent winding", str(stats["winding_consistent"]))
    c[3].metric("Connected components", stats["connected_components"])

    st.subheader("Volume")
    st.info(
        "Volume is not reported: it requires a watertight mesh **and** real-world scale calibration, "
        "which is not implemented yet. "
        + ("The mesh is watertight, but unscaled." if stats["watertight"] else "The mesh is also not watertight.")
    )

    st.subheader("Mesh statistics")
    st.json({k: stats[k] for k in ("vertices", "faces", "watertight", "connected_components")})

st.subheader("Full run metrics")
st.json(metrics, expanded=False)
