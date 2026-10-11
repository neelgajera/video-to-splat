"""Pick the frames worth reconstructing from a walk-around video.

Two problems with sampling a phone video at a fixed rate:
  * motion blur: hand-held frames are often soft, and blurry frames hurt both SfM and splat training;
  * uneven coverage: people slow down at corners and speed up along the sides, so fixed-rate sampling
    over-samples some viewpoints and leaves gaps elsewhere.

So every frame is scored for sharpness (variance of the Laplacian) and exposure (clipped pixels), and camera
motion between frames is estimated with dense optical flow on a small proxy image. The timeline is then split
into windows of *equal accumulated motion* (equal viewpoint change rather than equal time) and the best
frame of each window is kept.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROXY_WIDTH = 320   # analysis resolution; scores are relative, so this only needs to be consistent


@dataclass
class FrameScore:
    index: int
    time: float
    sharpness: float   # variance of Laplacian on the proxy image (higher = sharper)
    clipped: float     # fraction of pixels at 0 or 255 in any channel
    motion: float      # median optical-flow magnitude from the previous frame, in proxy pixels
    selected: bool = False


def _proxy(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (PROXY_WIDTH, round(h * PROXY_WIDTH / w)), interpolation=cv2.INTER_AREA)


def score_video(video: Path, rotate: int | None = None) -> tuple[list[FrameScore], float]:
    """Score every frame. Returns the scores and the video frame rate."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scores: list[FrameScore] = []
    prev_gray = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        small = _proxy(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        clipped = float(np.mean((small.min(axis=2) >= 250) | (small.max(axis=2) <= 5)))
        motion = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion = float(np.median(np.linalg.norm(flow, axis=2)))
        scores.append(FrameScore(i, i / fps, sharp, clipped, motion))
        prev_gray = gray
        i += 1
    cap.release()
    if not scores:
        raise RuntimeError(f"no frames decoded from {video}")
    return scores, fps


def select(scores: list[FrameScore], n: int, max_clipped: float = 0.25, min_gap: int = 2) -> list[FrameScore]:
    """Mark the best frame in each of `n` equal-motion windows; returns the selected frames."""
    motion = np.array([s.motion for s in scores])
    cum = np.cumsum(motion)
    total = cum[-1]
    if total <= 0:   # static video: fall back to equal-time windows
        cum = np.arange(len(scores), dtype=float)
        total = cum[-1] or 1.0
    edges = np.linspace(0.0, total, n + 1)
    window = np.clip(np.searchsorted(edges, cum, side="right") - 1, 0, n - 1)

    chosen: list[FrameScore] = []
    last = -10**9
    for w in range(n):
        idx = np.flatnonzero(window == w)
        candidates = [scores[j] for j in idx if scores[j].clipped <= max_clipped and j - last >= min_gap]
        if not candidates:
            continue
        best = max(candidates, key=lambda s: s.sharpness)
        best.selected = True
        chosen.append(best)
        last = best.index
    return chosen


def export_frames(video: Path, chosen: list[FrameScore], out_dir: Path, max_side: int = 1600,
                  rotate: int | None = None) -> None:
    """Write the chosen frames as JPEGs (long side capped at `max_side`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {s.index: k for k, s in enumerate(chosen)}
    cap = cv2.VideoCapture(str(video))
    i = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            if rotate is not None:
                frame = cv2.rotate(frame, rotate)
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{wanted.pop(i):04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    cap.release()


def write_csv(scores: list[FrameScore], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "time", "sharpness", "clipped", "motion", "selected"])
        for s in scores:
            w.writerow([s.index, f"{s.time:.3f}", f"{s.sharpness:.2f}", f"{s.clipped:.4f}", f"{s.motion:.3f}", int(s.selected)])


def plot(scores: list[FrameScore], path: Path) -> None:
    """Sharpness and accumulated motion over time, with the kept frames marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([s.time for s in scores])
    sharp = np.array([s.sharpness for s in scores])
    sel = np.array([s.selected for s in scores])
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
    a.plot(t, sharp, lw=0.8, color="#8a94a3", label="sharpness (all frames)")
    a.scatter(t[sel], sharp[sel], s=10, color="#1f77b4", zorder=3, label="selected")
    a.set_ylabel("Laplacian variance")
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    b.plot(t, np.cumsum([s.motion for s in scores]), lw=1.2, color="#1f77b4")
    for x in t[sel]:
        b.axvline(x, color="#1f77b4", alpha=0.15, lw=0.8)
    b.set_ylabel("accumulated motion")
    b.set_xlabel("time (s)")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary(scores: list[FrameScore], chosen: list[FrameScore]) -> dict:
    sharp_all = np.array([s.sharpness for s in scores])
    sharp_sel = np.array([s.sharpness for s in chosen]) if chosen else np.array([math.nan])
    return {
        "frames_total": len(scores),
        "frames_selected": len(chosen),
        "median_sharpness_all": float(np.median(sharp_all)),
        "median_sharpness_selected": float(np.median(sharp_sel)),
        "sharpness_gain": float(np.median(sharp_sel) / max(np.median(sharp_all), 1e-9)),
    }
