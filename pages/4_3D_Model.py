"""Interactive 3D preview of the reconstructed mesh / point clouds, plus exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from src.mesh.processor import MeshLoadError, decimate_for_preview, load_mesh, load_point_cloud, mesh_statistics
from src.reconstruction.pipeline import load_run_config
from src.ui.common import require_run
from src.ui.viewer import MeshArrays, mesh_figure, points_figure, subsample_points

st.set_page_config(page_title="3D Model", page_icon="💎", layout="wide")
st.title("4. 3D Model")
run = require_run()
config = load_run_config(run)


@st.cache_data(show_spinner="Loading mesh...", max_entries=4)
def _mesh_arrays(path: str, mtime: float, max_faces: int) -> tuple[MeshArrays, dict]:
    mesh = load_mesh(Path(path))
    stats = mesh_statistics(mesh).to_dict()
    colors = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and getattr(visual, "kind", None) == "vertex":
        colors = np.asarray(visual.vertex_colors)[:, :3].astype(np.uint8)
    preview, decimated = decimate_for_preview(mesh, max_faces)
    arrays = MeshArrays(
        vertices=np.asarray(preview.vertices, dtype=np.float32),
        faces=np.asarray(preview.faces, dtype=np.int32),
        vertex_colors=None if decimated else colors,
        decimated=decimated,
    )
    return arrays, stats


@st.cache_data(show_spinner="Loading point cloud...", max_entries=4)
def _points(path: str, mtime: float, max_points: int) -> tuple[np.ndarray, np.ndarray | None, int]:
    points, colors = load_point_cloud(Path(path))
    total = len(points)
    points, colors = subsample_points(points, colors, max_points)
    return points.astype(np.float32), colors, total


layers = {
    "Mesh": run.raw_mesh_ply,
    "Dense point cloud": run.fused_ply,
    "Sparse point cloud": run.sparse_points_ply,
}
available = {name: path for name, path in layers.items() if path.is_file()}
if not available:
    st.info("No reconstruction output yet. Run the reconstruction first.")
    st.stop()

left, right = st.columns([3, 1])
with right:
    layer = st.radio("Display", list(available))
    wireframe = st.checkbox("Wireframe overlay", False, disabled=layer != "Mesh")
    colors_on = st.checkbox("Vertex colours", True)
    point_size = st.slider("Point size", 0.5, 5.0, 1.5, 0.5, disabled=layer == "Mesh")

path = available[layer]
try:
    if layer == "Mesh":
        arrays, stats = _mesh_arrays(str(path), path.stat().st_mtime, config.mesh.preview_max_faces)
        fig = mesh_figure(arrays, wireframe=wireframe, use_colors=colors_on)
    else:
        pts, cols, total = _points(str(path), path.stat().st_mtime, config.mesh.preview_max_points)
        fig = points_figure(pts, cols if colors_on else None, point_size=point_size)
except MeshLoadError as exc:
    st.error(str(exc))
    st.stop()

with left:
    st.plotly_chart(fig, width="stretch")
    st.caption("Drag to rotate, right-drag / shift-drag to pan, scroll to zoom.")

with right:
    if layer == "Mesh":
        st.metric("Vertices", f"{stats['vertices']:,}")
        st.metric("Triangles", f"{stats['faces']:,}")
        ext = stats["bbox_extents"]
        st.markdown(f"**Dimensions** ({stats['units']})  \nX {ext[0]:.4g} · Y {ext[1]:.4g} · Z {ext[2]:.4g}")
        st.markdown(f"Watertight: **{stats['watertight']}** · Components: **{stats['connected_components']}**")
        if arrays.decimated:
            st.caption(f"Preview decimated to {len(arrays.faces):,} faces (statistics use the full mesh).")
    else:
        st.metric("Points", f"{total:,}")
        if total > len(pts):
            st.caption(f"Showing a random subset of {len(pts):,} points.")

st.subheader("Export")
st.caption("Raw COLMAP mesh (unmodified, unscaled). Cleaned exports arrive with mesh cleaning in Milestone 2.")
export_cols = st.columns(5)
for col, (label, file) in zip(
    export_cols,
    [("PLY", run.raw_mesh_ply), ("OBJ", run.output_dir / "raw_mesh.obj"), ("STL", run.output_dir / "raw_mesh.stl"),
     ("GLB", run.output_dir / "raw_mesh.glb"), ("Point cloud PLY", run.fused_ply if run.fused_ply.is_file()
                                                else run.sparse_points_ply)],
):
    if file.is_file():
        col.download_button(f"Download {label}", file.read_bytes(), file_name=f"{run.run_id}_{file.name}")
    else:
        col.button(f"{label} unavailable", disabled=True)
