# rough-gem-3d-reconstructor

Final-year Computer Science project: reconstruct the **external 3D surface** of a single rough
gemstone from smartphone photos and/or videos, using [COLMAP](https://colmap.github.io/) as the
photogrammetry engine.

> **Scope and honesty statement.** This software reconstructs only the externally visible surface.
> It does **not** detect inclusions or internal cracks, and does not assess gemstone identity,
> quality, value or optical properties. Rough gemstones are often shiny, translucent, transparent
> or texture-poor, all of which break conventional photogrammetry. **Many captures will fail or
> produce incomplete/incorrect geometry.** A completed reconstruction is not proof of accuracy;
> accuracy must be validated against physical measurements or reference scans.

## Status: Milestone 1

| Feature | Status |
|---|---|
| Photo / video / mixed upload, HEIC conversion, EXIF rotation | done |
| Video frame sampling (fixed interval or motion-based auto) | done |
| COLMAP detection (version, CUDA), explicit stage pipeline | done |
| Sparse SfM statistics (registration %, points, reprojection error) | done |
| Dense stereo, fusion, Poisson/Delaunay meshing (CUDA only) | done |
| Sparse-model Delaunay mesh fallback when CUDA is unavailable | done |
| Interactive mesh / point-cloud viewer, PLY/OBJ/STL/GLB export | done |
| Per-run reproducibility records (`run_config.json`, `metrics.json`, logs) | done |
| Hand-held / turntable capture modes with SAM 2 gemstone masks | done |
| Image quality filtering, duplicate removal | Milestone 2 |
| Mesh cleaning, scale calibration (mm), volume | Milestone 2 |
| Experimental comparison of capture conditions | Milestone 3 |

Stages that are not implemented yet are reported as **skipped** in the UI and in `metrics.json`;
nothing is simulated.

## Architecture

```text
 Streamlit UI (app.py, pages/)                 CLI (python -m src.cli)
   │ upload, settings, polling status.json        │
   └────────────── launches ──────────────────────┤
                                                  ▼
                        src/reconstruction/pipeline.py  (one run directory)
   ┌──────────────┬─────────────────┬───────────────────────────┬────────────────┐
   ▼              ▼                 ▼                           ▼                ▼
 ingestion/    preprocessing/    reconstruction/             mesh/            utils/
 images.py     (Milestone 2)     colmap_runner.py ──► colmap  processor.py     config.py
 video.py ──► ffmpeg/OpenCV      commands.py   (subprocess)   export.py        paths.py
 dataset.py                      statistics.py                calibration(M2)  logging.py
                                 progress.py ──► status.json                   process.py
```

Pipeline stages (each explicit COLMAP call is logged to `logs/<stage>.log`):

```text
1  Preparing images        ingestion, HEIC→JPEG, EXIF rotation, video frame sampling
2  Checking image quality  (Milestone 2 - skipped)
3  Creating masks          (Milestone 2 - skipped)
4  Extracting features     colmap feature_extractor
5  Matching images         colmap exhaustive_matcher | sequential_matcher
6  Sparse reconstruction   colmap mapper  (+ model_converter, model_analyzer)
7  Dense reconstruction    colmap image_undistorter + patch_match_stereo   [CUDA]
8  Fusing point cloud      colmap stereo_fusion → dense/fused.ply           [CUDA]
9  Generating mesh         colmap poisson_mesher | delaunay_mesher
10 Cleaning mesh           (Milestone 2 - raw statistics + exports only)
11 Complete
```

### CUDA and macOS

COLMAP's dense stereo (`patch_match_stereo`) **requires an NVIDIA GPU with CUDA**. macOS builds
never have CUDA. On such machines the pipeline:

1. runs feature extraction/matching on the CPU;
2. skips dense reconstruction and fusion (reported as skipped, with the reason);
3. if `reconstruction.sparse_fallback: true`, builds a **coarse mesh from the sparse model**
   with `colmap delaunay_mesher --input_type sparse`. This mesh is clearly labelled
   `sparse_delaunay` and is much less detailed than a dense mesh.

For detailed meshes, run the same code on a Linux/Windows machine with an NVIDIA GPU and a
CUDA-enabled COLMAP build. `python -m src.cli check` reports whether CUDA is available.

## Prerequisites

- Python **3.11 or 3.12** (3.13+ may lack wheels for some dependencies)
- COLMAP 3.9 or newer on `PATH` (option names are auto-detected, so 3.12+ renames are handled)
- FFmpeg (recommended; OpenCV decoding is used as a fallback)

### Installing COLMAP and FFmpeg

macOS (Homebrew):

```bash
sudo xcodebuild -license accept   # only if Homebrew complains about the Xcode licence
brew install colmap ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt install colmap ffmpeg     # distro build may lack CUDA; build from source for CUDA
```

Windows: download a CUDA release from <https://github.com/colmap/colmap/releases>, unzip it,
add the folder containing `colmap.bat`/`COLMAP.bat` to `PATH` (or set
`tools.colmap_executable` in `config.yaml`), and install FFmpeg from <https://ffmpeg.org>.

conda (any OS, CPU build): `conda install -c conda-forge colmap ffmpeg`

## Installation

```bash
cd rough-gem-3d-optimizer
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.cli check            # verifies COLMAP / FFmpeg / CUDA
python -m pytest -q                # unit tests
```

## Running

```bash
streamlit run app.py
```

Then work through the pages: **Upload → Preprocess → Reconstruct → 3D Model → Analysis**.

Command-line equivalent (useful for batch experiments):

```bash
python -m src.cli run "data/input/rock orbit.mov" --mode orbit_camera --sampling interval --interval 0.5
python -m src.cli run data/input/photos/*.jpg --matcher exhaustive
python -m src.cli reconstruct runs/2026-09-29_001      # re-run an existing run with its saved config
```

## Configuration

Machine-specific settings (e.g. absolute COLMAP/FFmpeg paths when they are not on `PATH`) go in
an optional, git-ignored `config.local.yaml`, which is merged on top of `config.yaml`:

```yaml
tools:
  colmap_executable: /Users/you/.gem3d/colmap-env/bin/colmap
  ffmpeg_executable: /Users/you/.gem3d/colmap-env/bin/ffmpeg
```

All parameters live in [`config.yaml`](config.yaml). **The defaults are starting values only and
have not been tuned for gemstones; tune them experimentally.** The UI's Upload page overrides the
most important ones per run. Each run stores its exact configuration in `runs/<id>/config.yaml`
(input to the pipeline) and `runs/<id>/run_config.json` (full record).

## Reproducibility: run directories

Every attempt, successful or failed, gets its own directory:

```text
runs/2026-09-29_001/
    input/            original uploads (untouched)
    frames/           images given to COLMAP
    masks/            COLMAP masks (<frame>.png), prompts.json (clicks), selection.json (rejections)
    database.db       COLMAP features + matches
    sparse/0/ ...     COLMAP sparse model(s); sparse_txt/ holds TXT exports
    dense/            undistorted images, depth maps, fused.ply, meshed-*.ply (CUDA only)
    output/           raw_mesh.ply (+ .obj/.stl/.glb), sparse_points.ply
    logs/             pipeline.log, one log per COLMAP stage
    logs/archive/     logs + metrics of earlier attempts when a run is re-run
    config.yaml       parameters used by this run
    ingestion.json    selected frames and ingestion errors
    run_config.json   parameters, software versions, COLMAP version/CUDA, exact COLMAP commands
    metrics.json      statistics, stage timings, warnings, errors, final state
    status.json       live progress (read by the UI)
```

The raw COLMAP mesh is never overwritten: `output/raw_mesh.ply` is a copy of COLMAP's output.

## Recommended capture procedure

Test the software with an **opaque, textured object** (for example a rough rock) first. If that
fails, the problem is the capture or the code; if it works but a gemstone fails, the failure is
likely optical and is itself a research result.

- **Orbit camera (recommended):** keep the stone still; walk the phone around it in 2-3 rings at
  different heights, with ~70-80% overlap between consecutive views.
- Place the stone on a **textured, non-repeating background** (newspaper, patterned cloth) so
  COLMAP has features to track in orbit mode.
- Use **diffuse, even lighting** (overcast daylight, light tent); avoid direct sunlight and flash,
  which cause moving specular highlights.
- Lock focus/exposure if the camera app allows it; keep the zoom fixed (shared intrinsics).
- Move slowly in video to avoid motion blur; 20-60 s of video sampled at 0.5 s is a good start.
- Fill a large part of the frame with the stone while keeping it fully in view.
- **Turntable / hand-held:** see the next section.

## Hand-held and turntable capture (object moves, camera static)

COLMAP assumes a static scene. If the stone is rotated (in the fingers or on a turntable) in front
of a static camera, COLMAP registers the *background* instead, and can even report a "successful"
reconstruction of the floor. These modes therefore mask everything except the stone:

1. Install SAM 2 once: `pip install -r requirements-sam2.txt` (PyTorch; uses CUDA, Apple MPS or CPU).
2. Upload with capture mode **Hand-held** or **Turntable**.
3. On **Preprocess**, click the stone in the first frame of each video (and on fingers with
   *Not stone* if SAM 2 includes them), then **Generate masks**. SAM 2 tracks the stone through the
   video; masks are eroded by `masking.erode_px` and saved as `masks/<frame>.png` (COLMAP layout).
4. Review the overlays, reject bad frames or upload a manual mask, then **Reconstruct**.

In these modes COLMAP receives `--ImageReader.mask_path`, an `--image_list_path` of usable frames,
and the `object_capture` settings from `config.yaml` (full-resolution, denser SIFT, guided matching,
relaxed mapper thresholds), because the stone typically covers only 2-5% of a phone frame.

CLI: `python -m src.cli run stone.mov --mode handheld --point X,Y` (X,Y = stone pixel in frame 1).

Hand-held tips: hold the stone by its two ends with fingertips, rotate slowly through a full turn,
re-grip and turn about another axis, keep the phone fixed, use soft light and a plain black matte
background.

**Observed result (clear quartz, hand-held, sunlit paving):** SAM 2 isolated the crystal in all 38
frames, but only 2/38 frames registered: features seen through a transparent crystal are not
consistent between views. Masking solves the "wrong scene" problem, not the optical one; expect
this mode to work for opaque/matte stones, or transparent stones with a temporary matte coating.

## Known gemstone limitations

- **Specular / shiny facets:** reflections move with the viewpoint, producing false matches.
- **Translucent / transparent material:** features seen *through* the stone are not on its
  surface; geometry becomes wrong or missing.
- **Texture-poor surfaces:** too few features to match or register images.
- **Small size:** needs close focus; shallow depth of field blurs parts of the stone.
- The mesh never contains internal geometry; do not infer inclusions or cracks from it.

A matte temporary scanning spray is the standard photogrammetry workaround for shiny objects, but
whether it is acceptable for your stones is a separate question.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `COLMAP was not found` | Install COLMAP; check `colmap -h` works in the same terminal; or set `tools.colmap_executable`. |
| Homebrew: *You have not agreed to the Xcode license* | `sudo xcodebuild -license accept`, then `brew install colmap ffmpeg`. |
| Dense reconstruction skipped | No CUDA (always on macOS). Coarse sparse mesh is used; use a CUDA machine for dense. |
| *Insufficient images* | Upload more views, or lower the video sampling interval. |
| *No features were detected* | Blurry, dark or featureless images; improve lighting/background texture. |
| *No image pair could be geometrically verified* | Too little overlap or reflective/transparent surface. |
| *Mapper produced no reconstruction* / low registration % | Views lack parallax; move the camera around the object rather than rotating in place. |
| Model split into several models | Gaps in coverage; capture continuous overlapping views. Largest model is used. |
| GPU / OpenGL errors in feature extraction | The pipeline retries on CPU automatically; or untick *Use GPU*. |
| Upload too large | Increase `maxUploadSize` in `.streamlit/config.toml`. |
| Anything else | Read `runs/<id>/logs/pipeline.log` and the failing stage's log. |
