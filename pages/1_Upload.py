"""Upload photos/videos, choose capture settings and prepare the frame dataset."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ingestion.dataset import prepare_dataset
from src.ingestion.images import PHOTO_EXTENSIONS
from src.ingestion.video import VIDEO_EXTENSIONS
from src.reconstruction.pipeline import save_run_config
from src.ui.common import app_config, set_current_run, sidebar_run_selector
from src.utils.config import ConfigError, config_from_dict
from src.utils.logging import configure_run_logging
from src.utils.paths import create_run

st.set_page_config(page_title="Upload", page_icon="📤", layout="wide")
st.title("1. Upload")
sidebar_run_selector()

base = app_config()

st.subheader("Capture mode")
mode_labels = {"orbit_camera": "Orbit Camera", "turntable": "Turntable", "handheld": "Hand-held (rotated in hand)"}
mode = st.radio(
    "How was the gemstone captured?",
    list(mode_labels),
    index=list(mode_labels).index(base.capture.mode),
    format_func=mode_labels.get,
    horizontal=True,
    help="Orbit: stone stationary, phone moves around it (recommended). "
    "Turntable: phone fixed, stone rotates on a turntable. "
    "Hand-held: phone fixed, stone turned in the fingers.",
)
if mode in ("turntable", "handheld"):
    st.info(
        "The camera is static and the stone moves, so COLMAP must only see the stone. After uploading, "
        "go to **Preprocess**, click the stone in the first frame and generate masks with SAM 2 "
        "(requires `pip install -r requirements-sam2.txt`)."
    )
if mode == "handheld":
    st.markdown(
        "**Hand-held capture tips:** hold the stone by its two ends with your fingertips; rotate it slowly "
        "through a full turn, then re-grip and turn about another axis; keep the phone still (tripod or "
        "propped); use soft, even light; a plain **black matte background** makes masking much more "
        "reliable. Transparent stones may still fail (refraction) - a temporary matte coating helps."
    )

st.subheader("Files")
uploads = st.file_uploader(
    "Photos (JPG/PNG/HEIC) and/or videos (MP4/MOV)",
    type=sorted(ext.lstrip(".") for ext in PHOTO_EXTENSIONS | VIDEO_EXTENSIONS),
    accept_multiple_files=True,
)

with st.expander("Video frame sampling", expanded=True):
    interval_options = {"1 frame every 0.25 s": 0.25, "1 frame every 0.5 s": 0.5, "1 frame every 1 s": 1.0}
    sampling_label = st.radio(
        "Sampling",
        [*interval_options, "Automatic (motion-based)"],
        index=1,
        help="Automatic keeps a frame once enough visual motion has accumulated since the last kept frame.",
    )
    max_frames = st.number_input("Max frames per video", 10, 2000, base.video.max_frames_per_video, step=10)

with st.expander("Reconstruction settings"):
    matcher = st.selectbox(
        "Matcher", ["auto", "exhaustive", "sequential"],
        index=["auto", "exhaustive", "sequential"].index(base.reconstruction.matcher),
        help="auto: sequential for video-only datasets, exhaustive for photos/mixed.",
    )
    single_camera = st.checkbox(
        "All images come from the same camera & zoom (shared intrinsics)",
        base.reconstruction.single_camera,
        help="Photos and videos with different resolutions each get their own camera automatically.",
    )
    use_gpu = st.checkbox("Use GPU when available", base.reconstruction.use_gpu)
    dense_enabled = st.checkbox("Dense reconstruction (requires CUDA)", base.reconstruction.dense.enabled)
    mesher = st.selectbox("Dense mesher", ["poisson", "delaunay"],
                          index=["poisson", "delaunay"].index(base.reconstruction.mesher))
    max_image_size = st.number_input("Max image size (px)", 500, 8000, base.reconstruction.max_image_size, step=100)

if st.button("Create run and prepare images", type="primary", disabled=not uploads):
    data = base.to_dict()
    data["capture"]["mode"] = mode
    if sampling_label in interval_options:
        data["video"].update(sampling="interval", interval_seconds=interval_options[sampling_label])
    else:
        data["video"]["sampling"] = "auto"
    data["video"]["max_frames_per_video"] = int(max_frames)
    data["reconstruction"].update(
        matcher=matcher, single_camera=single_camera, use_gpu=use_gpu,
        mesher=mesher, max_image_size=int(max_image_size),
    )
    data["reconstruction"]["dense"]["enabled"] = dense_enabled
    try:
        config = config_from_dict(data)
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()

    run = create_run(config.paths.resolved_runs_dir())
    configure_run_logging(run.logs_dir / "ingestion_ui.log")
    for upload in uploads:
        (run.input_dir / Path(upload.name).name).write_bytes(upload.getbuffer())
    save_run_config(run, config)
    set_current_run(run)

    with st.spinner("Extracting frames and converting images..."):
        report = prepare_dataset(run, config)

    if report.images:
        st.success(
            f"Run **{run.run_id}** created: {len(report.images)} images ready "
            f"({report.photo_count} photos, {report.video_count} videos)."
        )
    else:
        st.error("No usable images could be prepared from the uploaded files.")
    for error in report.errors:
        st.error(error)
    for name in report.skipped_files:
        st.warning(f"Unsupported file skipped: {name}")
    if len(report.images) < config.reconstruction.min_images:
        st.warning(
            f"Only {len(report.images)} images - at least {config.reconstruction.min_images} are required. "
            "Upload more views or use denser video sampling."
        )
    if config.capture.requires_masks:
        st.info("Next step: on **Preprocess**, click the stone in each video's first frame and generate masks.")
    st.page_link("pages/2_Preprocess.py", label="Continue to Preprocess", icon="➡️")
