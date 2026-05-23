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
from miniproc.core.types import CameraSpec, LightSpec, RoomSpec
from miniproc.layout.rule_pgm import create_rule_layout


def main() -> None:
    config = load_config("configs/debug.yaml")

    room_config = config["scene"]["room"]
    scene_seed = config["scene"].get("seed", 0)
    frame_config = config["frame"]
    camera_config = frame_config["camera"]

    room = RoomSpec(
        room_type=room_config["room_type"],
        width=room_config["width"],
        length=room_config["length"],
        height=room_config["height"],
    )
    assets = []
    for object_index, object_config in enumerate(config["scene"]["objects"]):
        assets.append(
            sample_pix3d_asset(
                category=object_config["category"],
                seed=scene_seed + object_index,
            )
        )
    objects = create_rule_layout(room, assets)
    camera = CameraSpec(
        location=tuple(camera_config["location"]),
        rotation=tuple(camera_config["rotation"]),
        focal_length=camera_config["focal_length"],
    )
    lights = [
        LightSpec(
            light_type=light_config["light_type"],
            location=tuple(light_config["location"]),
            rotation=tuple(light_config["rotation"]),
            energy=light_config["energy"],
            size=light_config.get("size"),
        )
        for light_config in frame_config["lights"]
    ]

    mp.init()
    create_room(room)
    for obj in objects:
        load_object(obj)
    for light in lights:
        add_light(light)
    add_camera(camera)
    render_rgb("outputs/renders/debug.png")
    render_depth(
        depth_path="outputs/renders/debug_depth.exr",
        visualization_path="outputs/renders/debug_depth_vis.png",
    )
    save_blend("outputs/scenes/debug_scene.blend")


def save_blend(output_path: str) -> None:
    import bpy

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


if __name__ == "__main__":
    main()
