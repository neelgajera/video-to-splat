"""video-to-splat: one command from a phone walk-around video to a browser-ready 3D model.

    python run.py my_car.mp4                          # frames -> COLMAP poses -> beta splatting
    python run.py my_car.mp4 --poses vggt             # feed-forward poses with VGGT (seconds, not minutes)
    python run.py my_car.mp4 --showcase ../beta-splatting-showcase --title "My Car"

Each stage writes into runs/<name>/ and is skipped on re-runs when its output exists (use --redo <stage>).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from v2s import frames as F

STAGES = ["frames", "poses", "train", "export"]
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--name", help="run name (default: video file name)")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--frames", type=int, default=180, help="frames to keep (default 180)")
    ap.add_argument("--max-side", type=int, default=1600, help="long side of exported frames in px")
    ap.add_argument("--rotate", type=int, choices=sorted(ROTATIONS), default=0, help="rotate frames clockwise")
    ap.add_argument("--poses", choices=["colmap", "vggt"], default="colmap")
    ap.add_argument("--iterations", type=int, default=30_000)
    ap.add_argument("--cap-max", type=int, default=300_000, help="max splats (300k fits a 6 GB GPU)")
    ap.add_argument("--showcase", type=Path, help="beta-splatting-showcase checkout to publish into")
    ap.add_argument("--title", help="scene title in the showcase gallery")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--until", choices=STAGES, default="export", help="stop after this stage")
    ap.add_argument("--redo", choices=STAGES, nargs="*", default=[], help="re-run these stages")
    args = ap.parse_args()

    name = args.name or args.video.stem
    run = args.runs / name
    run.mkdir(parents=True, exist_ok=True)
    report_path = run / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    rotate = ROTATIONS[args.rotate]

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    def stage(key: str, done: bool) -> bool:
        if done and key not in args.redo:
            print(f"[{key}] already done, skipping")
            return False
        print(f"[{key}] ...", flush=True)
        return True

    # 1. frames -------------------------------------------------------------------------------------------
    images = run / "frames"
    if stage("frames", images.exists() and any(images.glob("*.jpg"))):
        t0 = time.time()
        scores, fps = F.score_video(args.video, rotate)
        chosen = F.select(scores, args.frames)
        for old in images.glob("*.jpg"):
            old.unlink()
        F.export_frames(args.video, chosen, images, args.max_side, rotate)
        F.write_csv(scores, run / "frame_scores.csv")
        F.plot(scores, run / "frame_selection.png")
        report["frames"] = {**F.summary(scores, chosen), "fps": fps, "seconds": round(time.time() - t0, 1)}
        save()
        print(f"   kept {len(chosen)}/{len(scores)} frames, "
              f"median sharpness x{report['frames']['sharpness_gain']:.2f} vs all frames")
    if args.until == "frames":
        return

    # 2. poses --------------------------------------------------------------------------------------------
    dataset = run / "dataset"
    if stage("poses", (dataset / "sparse" / "0" / "images.bin").exists()):
        t0 = time.time()
        if args.poses == "colmap":
            from v2s import colmap
            stats = colmap.reconstruct(images, run / "colmap", dataset)
        else:
            from v2s import vggt_poses
            stats = vggt_poses.reconstruct(images, run / "vggt", dataset)
        report["poses"] = {"backend": args.poses, **stats, "seconds": round(time.time() - t0, 1)}
        save()
        print(f"   {args.poses}: {stats.get('registered_images')}/{stats.get('input_images')} images registered "
              f"in {report['poses']['seconds']} s")
    if args.until == "poses":
        return

    # 3. train --------------------------------------------------------------------------------------------
    model_dir = run / "model"
    from v2s import train as T
    if stage("train", T.latest_ply(model_dir) is not None):
        report["train"] = T.train(dataset, model_dir, args.iterations, args.cap_max)
        save()
    if args.until == "train" or not args.showcase:
        print(f"done: {T.latest_ply(model_dir)}")
        return

    # 4. export -------------------------------------------------------------------------------------------
    from v2s import export as E
    ply = T.latest_ply(model_dir)
    report["export"] = E.export(ply, model_dir / "cameras.json", args.showcase, name,
                                args.title or name.replace("-", " ").title(), args.subtitle)
    save()
    print(f"done: open the showcase and pick '{report['export']['model']}'")


if __name__ == "__main__":
    main()
