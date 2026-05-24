import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from miniproc.dataset.sunrgbd import extract_layout_priors


SUNRGBD_ROOT = Path(os.environ.get("SUNRGBD_ROOT", r"C:\Users\V4lde\Downloads\SUNRGBD-1\SUNRGBD"))
OUTPUT_PATH = Path("configs/sunrgbd_layout_priors.yaml")
MAX_SCENES = None


def main() -> None:
    priors = extract_layout_priors(SUNRGBD_ROOT, max_scenes=MAX_SCENES)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(priors, file, sort_keys=False)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Scenes with layout: {priors['scenes_with_layout']} / {priors['scenes_seen']}")
    print(f"Scenes used after room filtering: {priors['scenes_used']}")
    print(f"Categories: {', '.join(priors['categories'])}")


if __name__ == "__main__":
    main()
