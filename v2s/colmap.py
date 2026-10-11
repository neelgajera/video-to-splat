"""Camera poses with COLMAP: features, matching, incremental SfM, undistortion.

The output directory has the layout splatting trainers expect:
    <out>/images/          undistorted images
    <out>/sparse/0/        cameras.bin, images.bin, points3D.bin (PINHOLE cameras)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

EXHAUSTIVE_MAX = 300   # above this many images, use sequential matching instead of all pairs


def find_colmap() -> str:
    """COLMAP executable: $V2S_COLMAP, then PATH, then the default Windows install."""
    for cand in (os.environ.get("V2S_COLMAP"), shutil.which("colmap"), r"C:\COLMAP\COLMAP.bat"):
        if cand and Path(cand).exists():
            # The Windows binary release needs COLMAP.bat, which puts its lib/ folder on PATH;
            # running bin/colmap.exe directly fails with a missing-DLL error (0xC0000135).
            bat = Path(cand).parent.parent / "COLMAP.bat"
            if Path(cand).suffix.lower() == ".exe" and bat.exists():
                return str(bat)
            return cand
    raise FileNotFoundError("COLMAP not found: install it or set V2S_COLMAP to the executable / COLMAP.bat")


def _run(colmap: str, *args: str, log) -> str:
    cmd = [colmap, *args]
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    log.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"COLMAP {args[0]} failed (exit {proc.returncode}); see {log.name}")
    return proc.stdout


def _analyze(colmap: str, model: Path, log) -> dict:
    out = _run(colmap, "model_analyzer", "--path", str(model), log=log)
    stats = {}
    for key, pat in {
        "registered_images": r"Registered images:\s*(\d+)",
        "points": r"Points:\s*(\d+)",
        "mean_track_length": r"Mean track length:\s*([\d.]+)",
        "mean_reprojection_error_px": r"Mean reprojection error:\s*([\d.]+)",
    }.items():
        m = re.search(pat, out)
        if m:
            stats[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    return stats


def reconstruct(images: Path, work: Path, out: Path, use_gpu: bool = True) -> dict:
    """Run SfM on `images` and write an undistorted dataset to `out`. Returns reconstruction stats."""
    colmap = find_colmap()
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    sparse = work / "sparse"
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()
    n_images = len(list(images.glob("*.jpg")))
    gpu = "1" if use_gpu else "0"

    with open(work / "colmap.log", "w", encoding="utf-8") as log:
        _run(colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(images),
             "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV",
             "--SiftExtraction.use_gpu", gpu, log=log)
        if n_images <= EXHAUSTIVE_MAX:
            # A walk-around closes a loop; matching all pairs lets the last frames connect to the first.
            _run(colmap, "exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", gpu, log=log)
        else:
            _run(colmap, "sequential_matcher", "--database_path", str(db), "--SequentialMatching.overlap", "20",
                 "--SiftMatching.use_gpu", gpu, log=log)
        _run(colmap, "mapper", "--database_path", str(db), "--image_path", str(images),
             "--output_path", str(sparse), log=log)

        models = [p for p in sparse.iterdir() if (p / "images.bin").exists()]
        if not models:
            raise RuntimeError("COLMAP did not produce a reconstruction; try more overlap or sharper frames")
        # The mapper can split a capture into several models; keep the one with the most registered images.
        analyzed = {m: _analyze(colmap, m, log) for m in models}
        best = max(models, key=lambda m: analyzed[m].get("registered_images", 0))

        shutil.rmtree(out, ignore_errors=True)
        _run(colmap, "image_undistorter", "--image_path", str(images), "--input_path", str(best),
             "--output_path", str(out), "--output_type", "COLMAP", log=log)

    # image_undistorter writes sparse/*.bin; trainers expect sparse/0/*.bin.
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for f in (out / "sparse").glob("*.bin"):
        f.replace(out / "sparse" / "0" / f.name)
    for extra in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
        p = out / extra
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    stats = analyzed[best]
    stats["input_images"] = n_images
    stats["submodels"] = len(models)
    return stats
