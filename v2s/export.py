"""Publish a trained model into a beta-splatting-showcase checkout.

Uses the showcase's own tools so the scene gets the same pruning, orbit config and tour path as every
other scene in the gallery:
    tools/prune_ply.py             drop near-transparent splats
    tools/compute_scene_config.py  up vector, orbit lock, camera ring and tour from the training cameras
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def export(ply: Path, cameras: Path, showcase: Path, scene_id: str, title: str, subtitle: str = "",
           min_opacity: float = 0.05) -> dict:
    tools = showcase / "tools"
    if not (tools / "compute_scene_config.py").exists():
        raise FileNotFoundError(f"{showcase} doesn't look like a beta-splatting-showcase checkout")
    model = showcase / "models" / f"{scene_id}.ply"
    config = showcase / "configs" / f"{scene_id}.json"
    model.parent.mkdir(exist_ok=True)
    config.parent.mkdir(exist_ok=True)

    subprocess.run([sys.executable, str(tools / "prune_ply.py"), str(ply), str(model),
                    "--min-opacity", str(min_opacity)], check=True)
    subprocess.run([sys.executable, str(tools / "compute_scene_config.py"), "--ply", str(model),
                    "--cameras", str(cameras), "--out", str(config)], check=True)

    manifest_path = showcase / "scenes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = {"id": scene_id, "title": title, "subtitle": subtitle,
             "model": f"models/{scene_id}.ply", "config": f"configs/{scene_id}.json",
             "thumbnail": f"thumbs/{scene_id}.jpg"}
    manifest["scenes"] = [s for s in manifest["scenes"] if s["id"] != scene_id] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"model": str(model), "config": str(config), "size_mb": round(model.stat().st_size / 1e6, 1)}
