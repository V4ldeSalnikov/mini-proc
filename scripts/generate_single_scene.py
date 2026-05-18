import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import miniproc as mp
from miniproc.blender.camera import add_camera
from miniproc.blender.lighting import add_light
from miniproc.blender.render import render_rgb
from miniproc.blender.room import create_room
from miniproc.core.config import load_config
from miniproc.core.types import CameraSpec, LightSpec, RoomSpec


def main() -> None:
    config = load_config("configs/debug.yaml")

    room_config = config["scene"]["room"]
    frame_config = config["frame"]
    camera_config = frame_config["camera"]

    room = RoomSpec(
        room_type=room_config["room_type"],
        width=room_config["width"],
        length=room_config["length"],
        height=room_config["height"],
    )
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
    for light in lights:
        add_light(light)
    add_camera(camera)
    render_rgb("outputs/renders/debug.png")


if __name__ == "__main__":
    main()
