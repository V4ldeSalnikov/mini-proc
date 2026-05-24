import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import miniproc as mp
from miniproc.blender.assets import load_object, sample_pix3d_asset
from miniproc.blender.camera import add_camera
from miniproc.blender.lighting import add_light
from miniproc.blender.render import render_depth, render_rgb
from miniproc.blender.room import create_room
from miniproc.core.config import load_config
from miniproc.core.types import LightSpec, RoomSpec
from miniproc.layout.camera import sample_camera
from miniproc.layout.rule_pgm import create_rule_layout


OUTPUT_ROOT = Path("outputs/datasets/depth")
NUM_IMAGES = 100
NUM_VAL = 20
RESOLUTION = (320, 240)
RENDER_SAMPLES = 16
DEPTH_VIS_MIN = 1.0
DEPTH_VIS_MAX = 6.0


def main() -> None:
    config = load_config("configs/debug.yaml")
    room = _make_room(config["scene"]["room"])
    lights = [_make_light(light_config) for light_config in config["frame"]["lights"]]
    categories = [object_config["category"] for object_config in config["scene"]["objects"]]

    for index in range(NUM_IMAGES):
        split = "val" if index >= NUM_IMAGES - NUM_VAL else "train"
        stem = f"sample_{index:05d}"
        seed = index

        objects, camera = _sample_valid_scene(room, categories, seed)

        mp.init(samples=RENDER_SAMPLES)
        create_room(room)
        for obj in objects:
            load_object(obj)
        for light in lights:
            add_light(light)
        add_camera(camera)

        rgb_path = OUTPUT_ROOT / split / "rgb" / f"{stem}.png"
        depth_path = OUTPUT_ROOT / split / "depth" / f"{stem}.npy"
        depth_exr_path = OUTPUT_ROOT / split / "depth" / f"{stem}.exr"
        depth_vis_path = OUTPUT_ROOT / split / "depth_vis" / f"{stem}.png"

        _remove_previous_outputs(rgb_path, depth_path, depth_exr_path, depth_vis_path)
        render_rgb(str(rgb_path), resolution=RESOLUTION)
        render_depth(
            depth_path=str(depth_exr_path),
            visualization_path=str(depth_vis_path),
            resolution=RESOLUTION,
            depth_min=DEPTH_VIS_MIN,
            depth_max=DEPTH_VIS_MAX,
        )

        _convert_depth_exr_to_npy(depth_exr_path, depth_path)
        _rename_compositor_png(depth_vis_path)
        print(f"wrote {split}/{stem}")


def _sample_valid_scene(room: RoomSpec, categories: list[str], seed: int):
    for attempt in range(100):
        sample_seed = seed + attempt * 1000
        assets = [
            sample_pix3d_asset(category=category, seed=sample_seed + object_index)
            for object_index, category in enumerate(categories)
        ]

        try:
            objects = create_rule_layout(room, assets, seed=sample_seed, max_attempts=100)
            camera = sample_camera(room, objects, seed=sample_seed)
            return objects, camera
        except ValueError:
            continue

    raise ValueError(f"Could not sample a valid rule scene for seed {seed}.")


def _remove_previous_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        for suffixed_path in path.parent.glob(f"{path.stem}_*{path.suffix}"):
            suffixed_path.unlink(missing_ok=True)


def _convert_depth_exr_to_npy(requested_exr_path: Path, depth_path: Path) -> None:
    import bpy
    import numpy as np

    exr_path = _find_compositor_output(requested_exr_path)
    image = bpy.data.images.load(str(exr_path.resolve()))
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape((height, width, image.channels))
    depth = pixels[:, :, 0]

    depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_path, depth)

    bpy.data.images.remove(image)
    exr_path.unlink(missing_ok=True)


def _rename_compositor_png(requested_png_path: Path) -> None:
    actual_path = _find_compositor_output(requested_png_path)
    requested_png_path.parent.mkdir(parents=True, exist_ok=True)
    if actual_path != requested_png_path:
        shutil.move(str(actual_path), str(requested_png_path))


def _find_compositor_output(requested_path: Path) -> Path:
    if requested_path.exists():
        return requested_path

    matches = sorted(requested_path.parent.glob(f"{requested_path.stem}_*{requested_path.suffix}"))
    if not matches:
        raise FileNotFoundError(f"Could not find compositor output for {requested_path}")

    return matches[-1]


def _make_room(room_config: dict) -> RoomSpec:
    return RoomSpec(
        room_type=room_config["room_type"],
        width=room_config["width"],
        length=room_config["length"],
        height=room_config["height"],
    )


def _make_light(light_config: dict) -> LightSpec:
    return LightSpec(
        light_type=light_config["light_type"],
        location=tuple(light_config["location"]),
        rotation=tuple(light_config["rotation"]),
        energy=light_config["energy"],
        size=light_config.get("size"),
    )


if __name__ == "__main__":
    main()
