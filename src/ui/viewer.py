"""Plotly figures for interactive mesh and point-cloud preview.

Large geometry is reduced for display only (decimated mesh / sub-sampled
points); statistics are always computed on the full-resolution data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

_STONE_COLOR = "#9fb7c9"
_MAX_WIREFRAME_EDGES = 80_000


@dataclass
class MeshArrays:
    vertices: np.ndarray       # (N, 3) float
    faces: np.ndarray          # (M, 3) int
    vertex_colors: np.ndarray | None  # (N, 3) uint8
    decimated: bool = False


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        showlegend=False,
    )
    return fig


def _rgb_strings(colors: np.ndarray) -> list[str]:
    return [f"rgb({r},{g},{b})" for r, g, b in colors]


def mesh_figure(mesh: MeshArrays, wireframe: bool = False, use_colors: bool = True, height: int = 700) -> go.Figure:
    """Shaded triangle mesh, optionally with a wireframe overlay."""
    v, f = mesh.vertices, mesh.faces
    kwargs: dict = {}
    if use_colors and mesh.vertex_colors is not None:
        kwargs["vertexcolor"] = _rgb_strings(mesh.vertex_colors)
    else:
        kwargs["color"] = _STONE_COLOR
    fig = go.Figure(
        go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.2, roughness=0.6),
            hoverinfo="skip",
            **kwargs,
        )
    )
    if wireframe:
        edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
        if len(edges) > _MAX_WIREFRAME_EDGES:
            rng = np.random.default_rng(0)
            edges = edges[rng.choice(len(edges), _MAX_WIREFRAME_EDGES, replace=False)]
        segments = np.full((len(edges) * 3, 3), np.nan)
        segments[0::3] = v[edges[:, 0]]
        segments[1::3] = v[edges[:, 1]]
        fig.add_trace(
            go.Scatter3d(
                x=segments[:, 0], y=segments[:, 1], z=segments[:, 2], mode="lines",
                line=dict(color="rgba(20,20,20,0.6)", width=1), hoverinfo="skip",
            )
        )
    return _layout(fig, height)


def subsample_points(
    points: np.ndarray, colors: np.ndarray | None, max_points: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Deterministically reduce a point cloud to at most ``max_points``."""
    if len(points) <= max_points:
        return points, colors
    idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
    return points[idx], (colors[idx] if colors is not None else None)


def points_figure(
    points: np.ndarray, colors: np.ndarray | None, point_size: float = 1.5, height: int = 700
) -> go.Figure:
    """Coloured 3D scatter plot of a point cloud."""
    marker: dict = {"size": point_size}
    marker["color"] = _rgb_strings(colors) if colors is not None else _STONE_COLOR
    fig = go.Figure(
        go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", marker=marker, hoverinfo="skip")
    )
    return _layout(fig, height)
