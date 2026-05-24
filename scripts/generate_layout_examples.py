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
from miniproc.core.types import CameraSpec, LightSpec, RoomSpec
from miniproc.layout.random_layout import create_random_layout


def main() -> None:
    config = load_config("configs/debug.yaml")
    room = _make_room(config["scene"]["room"])
    camera = _make_camera(config["frame"]["camera"])
    lights = [_make_light(light_config) for light_config in config["frame"]["lights"]]
    categories = [object_config["category"] for object_config in config["scene"]["objects"]]

    for seed in range(10):
        assets = [
            sample_pix3d_asset(category=category, seed=seed + object_index)
            for object_index, category in enumerate(categories)
        ]
        objects = create_random_layout(room, assets, seed=seed)

        mp.init()
        create_room(room)
        for obj in objects:
            load_object(obj)
        for light in lights:
            add_light(light)
        add_camera(camera)

        render_rgb(f"outputs/renders/layout_examples/layout_{seed:03d}.png")
        save_blend(f"outputs/scenes/layout_examples/layout_{seed:03d}.blend")


def _make_room(room_config: dict) -> RoomSpec:
    return RoomSpec(
        room_type=room_config["room_type"],
        width=room_config["width"],
        length=room_config["length"],
        height=room_config["height"],
    )


def _make_camera(camera_config: dict) -> CameraSpec:
    return CameraSpec(
        location=tuple(camera_config["location"]),
        rotation=tuple(camera_config["rotation"]),
        focal_length=camera_config["focal_length"],
    )


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
