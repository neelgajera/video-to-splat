"""Feed-forward camera poses with VGGT (Wang et al., CVPR 2025): one network pass instead of SfM.

VGGT predicts cameras and depth for all frames jointly in about a second. This module:
  1. runs VGGT on an evenly spaced subset of frames (as many as fit in GPU memory);
  2. maps the predicted intrinsics from VGGT's 518x518 padded square back to original pixels;
  3. lifts a confidence-filtered, colored point cloud from the depth maps (splat initialisation);
  4. writes a standard COLMAP model + the original images, the same layout as the COLMAP backend.

Weights: the facebook/VGGT-1B checkpoint is research-only. For commercial use request VGGT-1B-Commercial
and pass its path via $V2S_VGGT_WEIGHTS.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from v2s import colmap_io as cio

VGGT_REPO = Path(__file__).resolve().parent.parent / "third_party" / "vggt"
WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
RES = 518                  # VGGT's native resolution
MAX_POINTS = 150_000       # initial point cloud size for training


def _load_model(device: str, dtype):
    import torch
    sys.path.insert(0, str(VGGT_REPO))
    from vggt.models.vggt import VGGT

    model = VGGT()
    local = Path(__file__).resolve().parent.parent / "weights" / "vggt-1b.pt"
    weights = os.environ.get("V2S_VGGT_WEIGHTS") or (str(local) if local.exists() else None)
    state = torch.load(weights, map_location="cpu") if weights else \
        torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
    model.load_state_dict(state)
    # Storing weights in bf16 halves memory (~2.5 GB), which is what lets this run on a 6 GB GPU.
    return model.eval().to(device=device, dtype=dtype)


def reconstruct(images: Path, work: Path, out: Path, max_frames: int = 48, conf_percentile: float = 50.0) -> dict:
    import torch
    sys.path.insert(0, str(VGGT_REPO))
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = sorted(images.glob("*.jpg"))
    if len(paths) > max_frames:   # evenly spaced subset; frames are already ordered along the walk
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, max_frames).round().astype(int)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    t_load = time.time()
    model = _load_model(device, dtype)
    imgs, coords = load_and_preprocess_images_square([str(p) for p in paths], RES)
    imgs = imgs.to(device)
    t_load = time.time() - t_load

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    # bf16 weights + autocast: some heads build float32 tensors internally, autocast reconciles the two.
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        tokens, ps_idx = model.aggregator(imgs[None].to(dtype))
        pose_enc = model.camera_head(tokens)[-1]
        # The depth head upsamples to full resolution; 2 frames at a time keeps it inside 6 GB.
        depth, conf = model.depth_head(tokens, imgs[None].to(dtype), ps_idx, frames_chunk_size=2)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.float(), imgs.shape[-2:])
    if device == "cuda":
        torch.cuda.synchronize()
    t_infer = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    E = extrinsic[0].cpu().numpy()                 # (S, 3, 4) camera-from-world, OpenCV convention
    K = intrinsic[0].cpu().numpy()                 # (S, 3, 3) in 518x518 square pixels
    depth = depth[0].float().cpu().numpy()         # (S, 518, 518, 1)
    conf = conf[0].float().cpu().numpy()           # (S, 518, 518)
    coords = coords.numpy()                        # (S, 6): x1, y1, x2, y2, width, height
    del model, tokens
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- cameras: one shared phone camera, mapped back to original pixels -----------------------------
    # square pixel u_s = x1 + u_o * s with s = 518 / max(W, H), so f_o = f_s / s and c_o = (c_s - x1) / s.
    W, H = int(coords[0, 4]), int(coords[0, 5])
    s = RES / max(W, H)
    fx, fy = np.median(K[:, 0, 0]) / s, np.median(K[:, 1, 1]) / s
    cx, cy = (np.median(K[:, 0, 2]) - coords[0, 0]) / s, (np.median(K[:, 1, 2]) - coords[0, 1]) / s
    camera = cio.Camera(1, "PINHOLE", W, H, np.array([fx, fy, cx, cy]))
    poses = [cio.Image(i + 1, p.name, 1, E[i, :, :3].astype(np.float64), E[i, :, 3].astype(np.float64))
             for i, p in enumerate(paths)]

    # --- points: confident depth pixels inside the real (unpadded) image area --------------------------
    world = unproject_depth_map_to_point_map(depth, E, K)          # (S, 518, 518, 3)
    rgb = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    yy, xx = np.mgrid[0:RES, 0:RES]
    inside = (xx >= coords[0, 0]) & (xx < coords[0, 2]) & (yy >= coords[0, 1]) & (yy < coords[0, 3])
    mask = (conf > np.percentile(conf[:, inside], conf_percentile)) & inside[None]
    pts, cols = world[mask], rgb[mask]
    if len(pts) > MAX_POINTS:
        keep = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, cols = pts[keep], cols[keep]

    # --- write the dataset ------------------------------------------------------------------------------
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    for p in paths:
        shutil.copy2(p, out / "images" / p.name)
    cio.write_model(out / "sparse" / "0", [camera], poses, pts, cols)
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work / "vggt_raw.npz", extrinsic=E, intrinsic=K, conf=conf.astype(np.float16),
                        names=np.array([p.name for p in paths]))

    return {"input_images": len(list(images.glob("*.jpg"))), "registered_images": len(paths),
            "points": int(len(pts)), "inference_seconds": round(t_infer, 2),
            "model_load_seconds": round(t_load, 1), "peak_gpu_gb": round(peak_gb, 2),
            "focal_px": round(float(fx), 1)}
