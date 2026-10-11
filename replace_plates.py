"""Replace licence plates in a posed dataset, consistently across all views.

    python replace_plates.py runs/my-car/dataset                    # edits images/, keeps images_original/
    python replace_plates.py runs/my-car/dataset --text "MY CAR" --sheet plates.jpg

Run it after the poses stage and before training (python run.py ... --until poses), then
`python run.py ... --redo train` so the splat is trained on the edited images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2s import plates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="folder with images/ and sparse/0/ (COLMAP backend: needs point tracks)")
    ap.add_argument("--text", default="VIDEO2SPLAT", help="main text on the replacement plate")
    ap.add_argument("--sub", default="github.com/neelgajera", help="small text on the replacement plate")
    ap.add_argument("--weights", type=Path, default=plates.DEFAULT_WEIGHTS, help="YOLO plate detector weights")
    ap.add_argument("--sheet", type=Path, help="write a before/after contact sheet here")
    args = ap.parse_args()

    stats = plates.replace_plates(args.dataset, args.text, args.sub, args.weights)
    print(json.dumps(stats, indent=2))
    if args.sheet:
        plates.contact_sheet(args.dataset, args.sheet)
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    main()
