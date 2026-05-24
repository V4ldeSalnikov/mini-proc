import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image


SUNRGBD_ROOT = Path(os.environ.get("SUNRGBD_ROOT", r"C:\Users\V4lde\Downloads\SUNRGBD-1\SUNRGBD"))
NYU_ROOT = SUNRGBD_ROOT / "kv1" / "NYUdata"
OUTPUT_ROOT = Path("outputs/datasets/nyu_depth_test")
MAX_SAMPLES = 100
ROOM_TYPES: set[str] | None = None
DEPTH_VIS_MIN = 0.5
DEPTH_VIS_MAX = 6.0


def main() -> None:
    rgb_dir = OUTPUT_ROOT / "rgb"
    depth_dir = OUTPUT_ROOT / "depth"
    depth_vis_dir = OUTPUT_ROOT / "depth_vis"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for scene_dir in sorted(NYU_ROOT.glob("NYU*")):
        if count >= MAX_SAMPLES:
            break

        scene_type = _read_scene_type(scene_dir)
        if ROOM_TYPES is not None and scene_type not in ROOM_TYPES:
            continue

        rgb_path = scene_dir / "image" / f"{scene_dir.name}.jpg"
        depth_path = scene_dir / "depth_bfx" / f"{scene_dir.name}.png"
        if not rgb_path.exists() or not depth_path.exists():
            continue

        stem = f"{scene_dir.name}_{scene_type}"
        rgb_output = rgb_dir / f"{stem}.png"
        depth_output = depth_dir / f"{stem}.npy"
        depth_vis_output = depth_vis_dir / f"{stem}.png"

        _copy_rgb(rgb_path, rgb_output)
        depth = decode_sunrgbd_depth(depth_path)
        np.save(depth_output, depth)
        _save_depth_vis(depth, depth_vis_output)

        count += 1
        print(f"wrote {stem}")

    print(f"Wrote {count} NYU/SUNRGBD test samples to {OUTPUT_ROOT}")


def decode_sunrgbd_depth(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path), dtype=np.uint16)
    decoded = (raw >> 3) | ((raw << 13) & 0xFFFF)
    depth_meters = decoded.astype(np.float32) / 1000.0
    depth_meters[raw == 0] = 0.0
    return depth_meters


def _copy_rgb(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_path).convert("RGB")
    image.save(output_path)


def _save_depth_vis(depth: np.ndarray, output_path: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = (depth[valid] - DEPTH_VIS_MIN) / (DEPTH_VIS_MAX - DEPTH_VIS_MIN)
    normalized = np.clip(normalized, 0.0, 1.0)

    near = np.array([255, 235, 13], dtype=np.float32)
    far = np.array([64, 5, 217], dtype=np.float32)
    image = near * (1.0 - normalized[..., None]) + far * normalized[..., None]
    image[~valid] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(output_path)


def _read_scene_type(scene_dir: Path) -> str:
    scene_path = scene_dir / "scene.txt"
    if not scene_path.exists():
        return "unknown"

    scene_type = scene_path.read_text(encoding="utf-8", errors="ignore").strip()
    return scene_type or "unknown"


if __name__ == "__main__":
    main()
