"""Preprocess-page panel: click prompts, SAM 2 mask generation and mask review."""

from __future__ import annotations

import cv2
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from src.ingestion.dataset import IngestionReport
from src.preprocessing.masking import (
    SAM2_INSTALL_HINT,
    MaskingError,
    MaskPrompt,
    build_sequences,
    frame_mask_info,
    load_prompts,
    mask_overlay,
    masking_status_path,
    sam2_available,
    save_manual_mask,
    save_prompts,
    set_rejected,
)
from src.reconstruction.progress import read_status
from src.ui.launcher import MASK, cancel, is_running, launch
from src.utils.config import AppConfig
from src.utils.paths import RunPaths

_PROMPT_WIDTH = 420
_STATUS_ICONS = {"ok": "✅", "missing": "⚪", "empty": "⚠️", "too_large": "⚠️", "rejected": "🚫"}


def _prompt_canvas(run: RunPaths, key: str, first_frame: str, prompt: MaskPrompt) -> MaskPrompt:
    """Show the first frame; clicks add positive/negative points (in full-resolution pixels)."""
    frame = cv2.imread(str(run.frames_dir / first_frame))
    if frame is None:
        st.error(f"Cannot read {first_frame}")
        return prompt
    height, width = frame.shape[:2]
    scale = _PROMPT_WIDTH / width
    preview = cv2.resize(frame, (_PROMPT_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
    for (x, y), label in zip(prompt.points, prompt.labels):
        color = (0, 220, 0) if label == 1 else (0, 0, 255)
        cv2.circle(preview, (int(x * scale), int(y * scale)), 7, color, -1)
        cv2.circle(preview, (int(x * scale), int(y * scale)), 7, (255, 255, 255), 2)

    click = streamlit_image_coordinates(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), key=f"click_{key}")
    last_key = f"last_click_{key}"
    if click and click != st.session_state.get(last_key):
        st.session_state[last_key] = click
        positive = st.session_state.get(f"polarity_{key}", "Stone (+)") == "Stone (+)"
        prompt.add(click["x"] / scale, click["y"] / scale, positive)
        return prompt
    return prompt


def render_masks_panel(run: RunPaths, config: AppConfig, report: IngestionReport) -> None:
    st.subheader("Gemstone masks (SAM 2)")
    if not config.capture.requires_masks:
        st.caption("Orbit-camera captures do not use masks: the static background helps COLMAP register views.")
        return
    if not sam2_available():
        st.error(f"SAM 2 is not installed. In the project virtual environment run: `{SAM2_INSTALL_HINT}`")
        return

    running = is_running(run, MASK)
    sequences = build_sequences(report)
    prompts = load_prompts(run)
    st.markdown(
        "Click the **stone** in the first frame of each sequence (green). If SAM 2 also selects your "
        "fingers, switch to *Not stone (−)* and click them (red). Then generate masks."
    )

    changed = False
    for seq in sequences:
        prompt = prompts.get(seq.key, MaskPrompt())
        with st.container(border=True):
            left, right = st.columns([3, 2])
            with right:
                st.markdown(f"**{seq.source}** · {len(seq.images)} frames")
                st.radio("Click adds", ["Stone (+)", "Not stone (−)"], key=f"polarity_{seq.key}", horizontal=True)
                st.caption(f"{prompt.labels.count(1)} stone / {prompt.labels.count(0)} non-stone points")
                if st.button("Clear points", key=f"clear_{seq.key}", disabled=running):
                    prompts[seq.key] = MaskPrompt()
                    save_prompts(run, prompts)
                    st.rerun()
            with left:
                before = len(prompt.points)
                prompt = _prompt_canvas(run, seq.key, seq.images[0], prompt)
                if len(prompt.points) != before:
                    prompts[seq.key] = prompt
                    changed = True
    if changed:
        save_prompts(run, prompts)
        st.rerun()

    ready = all(prompts.get(s.key, MaskPrompt()).valid for s in sequences)
    c1, c2 = st.columns([1, 4])
    if c1.button("Generate masks", type="primary", disabled=running or not ready):
        launch(run, MASK)
        st.rerun()
    if running and c2.button("Cancel masking"):
        cancel(run, MASK)
        st.rerun()
    if not ready:
        st.caption("Add at least one stone click to every sequence to enable mask generation.")
    _masking_progress(run)
    _mask_review(run, config, report)


@st.fragment(run_every=2)
def _masking_progress(run: RunPaths) -> None:
    status = read_status(masking_status_path(run))
    if not status:
        return
    state = status.get("state")
    running = is_running(run, MASK)
    if state in ("starting", "running") and running:
        total = status.get("total") or 0
        done = status.get("done") or 0
        st.progress(done / total if total else 0.0, text=f"SAM 2: {status.get('message', 'starting...')}")
        st.session_state["mask_was_running"] = True
    elif state == "failed":
        st.error(f"Mask generation failed: {status.get('message')}")
    elif state == "completed":
        st.success(status.get("message", "Masks generated"))
    elif state in ("starting", "running"):
        st.error("Mask generation stopped unexpectedly. See logs/masking.log and logs/masking_stdout.log.")
    if not running and st.session_state.get("mask_was_running"):
        st.session_state["mask_was_running"] = False
        st.rerun()


def _mask_review(run: RunPaths, config: AppConfig, report: IngestionReport) -> None:
    infos = frame_mask_info(run, report, config.masking)
    if all(i.status == "missing" for i in infos):
        return
    counts: dict[str, int] = {}
    for info in infos:
        counts[info.status] = counts.get(info.status, 0) + 1
    st.markdown(
        "**Mask review** - " + " · ".join(f"{_STATUS_ICONS[k]} {k}: {v}" for k, v in sorted(counts.items()))
    )
    st.caption(
        f"Frames are excluded automatically when the mask covers < {100 * config.masking.min_area_ratio:.1f}% "
        f"or > {100 * config.masking.max_area_ratio:.0f}% of the frame. Reject frames where fingers hide most "
        "of the stone or the mask is wrong."
    )

    per_page = 18
    pages = max(1, -(-len(infos) // per_page))
    page = st.number_input("Mask page", 1, pages, 1, key="mask_page") if pages > 1 else 1
    grid = st.columns(6)
    for i, info in enumerate(infos[(page - 1) * per_page: page * per_page]):
        col = grid[i % 6]
        overlay = mask_overlay(run, info.image, 320)
        if overlay is not None:
            col.image(overlay, width="stretch")
        area = f"{100 * info.area_ratio:.1f}%" if info.area_ratio is not None else "-"
        col.caption(f"{_STATUS_ICONS[info.status]} {info.image[-9:]} · {area}{' · manual' if info.manual else ''}")
        rejected = col.checkbox("Reject", info.status == "rejected", key=f"rej_{info.image}")
        if rejected != (info.status == "rejected"):
            set_rejected(run, info.image, rejected)
            st.rerun()

    with st.expander("Replace a mask manually"):
        st.caption("Upload a black/white image (white = stone). It is resized to the frame and not eroded.")
        target = st.selectbox("Frame", [i.image for i in infos], key="manual_target")
        upload = st.file_uploader("Mask image", type=["png", "jpg", "jpeg"], key="manual_mask")
        if upload is not None and st.button("Use this mask"):
            try:
                save_manual_mask(run, target, upload.getvalue())
                st.success(f"Mask replaced for {target}")
            except MaskingError as exc:
                st.error(str(exc))
