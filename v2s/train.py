"""Train a Deformable Beta Splatting model on a COLMAP-format dataset.

Thin wrapper around third_party/beta-splatting/train.py: picks settings that fit a 6 GB GPU, logs to a file
and returns the paths of the final PLY and cameras.json.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

BETA_REPO = Path(__file__).resolve().parent.parent / "third_party" / "beta-splatting"


def latest_ply(model_dir: Path) -> Path | None:
    plys = list(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    return max(plys, key=lambda p: int(p.parent.name.split("_")[1])) if plys else None


def train(dataset: Path, model_dir: Path, iterations: int = 30_000, cap_max: int = 300_000,
          sb_number: int = 2, extra: list[str] | None = None) -> dict:
    """Run training; returns {'ply', 'cameras', 'seconds', 'final_psnr'}.

    cap_max bounds the number of primitives. 300k fits a 6 GB GPU at ~1600 px images; raise it on bigger GPUs.
    """
    if not (BETA_REPO / "train.py").exists():
        raise FileNotFoundError(f"beta-splatting not found at {BETA_REPO}; run: git submodule update --init --recursive")
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "train.py", "-s", str(dataset.resolve()), "-m", str(model_dir.resolve()),
           "--iterations", str(iterations), "--cap_max", str(cap_max), "--sb_number", str(sb_number),
           "--disable_viewer", "--no-compress", *(extra or [])]
    log_path = model_dir / "train.log"
    t0 = time.time()
    psnr = None
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=BETA_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", env=env)
        for line in proc.stdout:
            log.write(line)
            m = re.search(r"PSNR\s*[:=]?\s*([\d.]+)", line)
            if m:
                psnr = float(m.group(1))
            if "ITER" in line or "PSNR" in line or "Error" in line:
                print("   ", line.rstrip()[-120:], flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"training failed (exit {proc.returncode}); see {log_path}")
    ply = latest_ply(model_dir)
    if ply is None:
        raise RuntimeError(f"training finished but no point_cloud.ply in {model_dir}")
    return {"ply": str(ply), "cameras": str(model_dir / "cameras.json"),
            "seconds": round(time.time() - t0), "final_psnr": psnr}
