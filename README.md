# video-to-splat

One command from a phone walk-around video to a photorealistic 3D model you can open in a browser.

```
phone video ──► frame selection ──► camera poses ──► beta splatting ──► web viewer
               sharpness + motion    COLMAP or VGGT    (6 GB GPU)        beta-splatting-showcase
```

<!-- TODO: demo GIF + live link once the first capture is trained -->

## Quick start

```bash
python run.py my_car.mp4                                   # frames -> COLMAP -> train
python run.py my_car.mp4 --poses vggt                      # poses from one VGGT forward pass
python run.py my_car.mp4 --showcase ../beta-splatting-showcase --title "My Car"   # publish to the gallery
```

Each stage writes to `runs/<name>/` and is skipped on re-runs (`--redo train` to force one).
`runs/<name>/report.json` collects timings and quality numbers for every stage.

## Stages

### 1. Frame selection ([`v2s/frames.py`](v2s/frames.py))

Sampling a hand-held video at a fixed rate gives blurry frames and uneven coverage, because people slow
down at corners. Instead, every frame is scored for:

- **sharpness**: variance of the Laplacian on a 320 px proxy;
- **exposure**: fraction of clipped pixels;
- **camera motion**: median Farneback optical-flow magnitude from the previous frame.

The timeline is split into windows of **equal accumulated motion** (equal viewpoint change, not equal time)
and the sharpest well-exposed frame of each window is kept. On a synthetic test with a pause, a 6× speed
change and every third frame blurred, it picks no blurred frames, nothing from the pause, and frames in
exact proportion to the motion. `frame_selection.png` shows the scores and the picks.

### 2. Camera poses

| Backend | How | Speed | Notes |
|---|---|---|---|
| `colmap` ([`v2s/colmap.py`](v2s/colmap.py)) | SIFT, exhaustive matching (closes the 360° loop), incremental SfM, undistortion | minutes | most accurate, uses every frame |
| `vggt` ([`v2s/vggt_poses.py`](v2s/vggt_poses.py)) | one forward pass of [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025) over all frames jointly | seconds | weights stored in bf16 so it fits 6 GB; cameras mapped from VGGT's 518 px padded square back to original pixels; point cloud lifted from confidence-filtered depth |

Both write the same undistorted COLMAP layout (`images/`, `sparse/0/`). VGGT output goes through a small
COLMAP binary writer ([`v2s/colmap_io.py`](v2s/colmap_io.py)) that COLMAP's own `model_analyzer` reads back,
so there's no `pycolmap` version conflict with the trainer.

### 3. Training ([`v2s/train.py`](v2s/train.py))

[Deformable Beta Splatting](https://github.com/RongLiu-Leo/beta-splatting) (Liu et al., SIGGRAPH 2025)
replaces Gaussian kernels with Beta kernels and view-dependent color with spherical Beta lobes, so it
needs fewer primitives for the same quality. Defaults (`--cap-max 300000`, 1600 px frames) fit a 6 GB laptop GPU.

### 4. Export ([`v2s/export.py`](v2s/export.py))

Prunes near-transparent splats and computes the orbit lock, camera ring and cinematic tour from the
training cameras, using the tools in
[beta-splatting-showcase](../beta-splatting-showcase). The scene then appears in the gallery.

## Results

<!-- TODO: table from report.json for the first real capture: frames kept, sharpness gain,
     COLMAP vs VGGT time and registered images, training time, PSNR, model size -->

## Setup (Windows, CUDA)

```bash
git clone --recursive https://github.com/neelgajera/video-to-splat
conda create -n splat python=3.10 -y && conda activate splat
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y      # nvcc inside the env, matches torch below
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# from a "x64 Native Tools" prompt (Visual Studio C++ build tools):
set CUDA_HOME=%CONDA_PREFIX%\Library && set TORCH_CUDA_ARCH_LIST=8.6 && set PIP_NO_BUILD_ISOLATION=0
cd third_party/beta-splatting/submodules && pip install --no-build-isolation . && cd .. && pip install --no-build-isolation .
```

COLMAP must be on `PATH`, or set `V2S_COLMAP` to `COLMAP.bat`.

## Credits and licenses

- This repository: MIT.
- [beta-splatting](https://github.com/RongLiu-Leo/beta-splatting) (Apache-2.0) and
  [VGGT](https://github.com/facebookresearch/vggt) (VGGT License) are git submodules under `third_party/`.
  The default `facebook/VGGT-1B` checkpoint is for research only; set `V2S_VGGT_WEIGHTS` to the
  `VGGT-1B-Commercial` checkpoint for commercial use.
- [COLMAP](https://colmap.github.io) (BSD).

## Background

I built the production version of this pipeline (video to 3D cars for dealerships) at Spyne. This is an
independent re-implementation written from scratch on public tools and my own captures. It contains no
Spyne code, models or data.
