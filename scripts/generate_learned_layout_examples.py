import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import miniproc as mp
from miniproc.blender.assets import load_object, sample_pix3d_asset
from miniproc.blender.camera import add_camera
from miniproc.blender.lighting import add_light
from miniproc.blender.render import render_rgb
from miniproc.blender.room import create_room
from miniproc.core.config import load_config
from miniproc.core.types import LightSpec
from miniproc.layout.camera import sample_camera
from miniproc.layout.learned_pgm import create_learned_layout, sample_asset_categories, sample_room


def main() -> None:
    config = load_config("configs/debug.yaml")
    lights = [_make_light(light_config) for light_config in config["frame"]["lights"]]

    for seed in range(10):
        room, categories, assets, objects, camera = sample_valid_scene(seed)

        mp.init()
        create_room(room)
        for obj in objects:
            load_object(obj)
        for light in lights:
            add_light(light)
        add_camera(camera)

        render_rgb(f"outputs/renders/learned_layout_examples/layout_{seed:03d}.png")
        save_blend(f"outputs/scenes/learned_layout_examples/layout_{seed:03d}.blend")


def sample_valid_scene(seed: int):
    for attempt in range(100):
        sample_seed = seed + attempt * 1000
        room = sample_room(seed=sample_seed)
        categories = sample_asset_categories(
            seed=sample_seed,
            room_type=room.room_type,
            min_objects=3,
            max_objects=4,
        )
        assets = [
            sample_pix3d_asset(category=category, seed=sample_seed + object_index)
            for object_index, category in enumerate(categories)
        ]
        try:
            objects = create_learned_layout(room, assets, seed=sample_seed, max_attempts=500)
            camera = sample_camera(room, objects, seed=sample_seed)
            return room, categories, assets, objects, camera
        except ValueError:
            continue

    raise ValueError("Could not sample a valid learned scene.")


def _make_light(light_config: dict) -> LightSpec:
    return LightSpec(
        light_type=light_config["light_type"],
        location=tuple(light_config["location"]),
        rotation=tuple(light_config["rotation"]),
        energy=light_config["energy"],
        size=light_config.get("size"),
    )


def save_blend(output_path: str) -> None:
    import bpy

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


if __name__ == "__main__":
    main()
