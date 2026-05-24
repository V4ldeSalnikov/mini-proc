from math import atan2, cos, sin, sqrt
from random import Random

from miniproc.core.types import CameraSpec, ObjectSpec, RoomSpec
from miniproc.layout.validators import object_footprint


def sample_camera(
    room: RoomSpec,
    objects: list[ObjectSpec],
    seed: int = 0,
    max_attempts: int = 100,
    focal_length: float = 28.0,
) -> CameraSpec:
    rng = Random(seed)

    for _ in range(max_attempts):
        location = _sample_camera_location(room, rng)
        target = _sample_camera_target(room, objects, rng)
        camera = CameraSpec(
            location=location,
            rotation=_look_at_rotation(location, target),
            focal_length=focal_length,
        )

        if validate_camera(camera, room, objects):
            return camera

    raise ValueError("Could not sample a valid camera for this scene.")


def validate_camera(
    camera: CameraSpec,
    room: RoomSpec,
    objects: list[ObjectSpec],
    room_margin: float = 0.15,
    object_margin: float = 0.25,
    min_wall_hit_distance: float = 1.5,
    min_object_hit_distance: float = 0.8,
) -> bool:
    if not _camera_inside_room(camera, room, room_margin):
        return False

    if _camera_inside_object(camera, objects, object_margin):
        return False

    if not _reasonable_pitch(camera):
        return False

    view_direction = _camera_forward_xy(camera)
    wall_hit_distance = _room_exit_distance(camera.location, view_direction, room)
    if wall_hit_distance < min_wall_hit_distance:
        return False

    object_hit_distance = _nearest_object_hit_distance(camera.location, view_direction, objects, object_margin)
    return object_hit_distance is None or object_hit_distance >= min_object_hit_distance


def _sample_camera_location(room: RoomSpec, rng: Random) -> tuple[float, float, float]:
    wall = rng.choice(["front", "back", "left", "right"])
    wall_offset = rng.uniform(0.35, 0.7)
    max_camera_height = max(1.2, min(1.7, room.height - 0.3))
    z = rng.uniform(1.2, max_camera_height)

    if wall == "front":
        return (
            rng.uniform(-0.35, 0.35) * room.width,
            -room.length / 2.0 + wall_offset,
            z,
        )
    if wall == "back":
        return (
            rng.uniform(-0.35, 0.35) * room.width,
            room.length / 2.0 - wall_offset,
            z,
        )
    if wall == "left":
        return (
            -room.width / 2.0 + wall_offset,
            rng.uniform(-0.35, 0.35) * room.length,
            z,
        )

    return (
        room.width / 2.0 - wall_offset,
        rng.uniform(-0.35, 0.35) * room.length,
        z,
    )


def _sample_camera_target(
    room: RoomSpec,
    objects: list[ObjectSpec],
    rng: Random,
) -> tuple[float, float, float]:
    if objects and rng.random() < 0.75:
        obj = rng.choice(objects)
        target_height = min(1.1, obj.asset.size[2] * obj.scale[2] * 0.6)
        return (obj.location[0], obj.location[1], max(0.5, target_height))

    return (
        rng.uniform(-0.15, 0.15) * room.width,
        rng.uniform(-0.15, 0.15) * room.length,
        rng.uniform(0.7, 1.1),
    )


def _look_at_rotation(
    location: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    dx = target[0] - location[0]
    dy = target[1] - location[1]
    dz = target[2] - location[2]
    horizontal_distance = sqrt(dx * dx + dy * dy)

    pitch = atan2(horizontal_distance, -dz)
    yaw = atan2(-dx, dy)
    return (pitch, 0.0, yaw)


def _camera_inside_room(camera: CameraSpec, room: RoomSpec, margin: float) -> bool:
    x, y, z = camera.location
    return (
        -room.width / 2.0 + margin <= x <= room.width / 2.0 - margin
        and -room.length / 2.0 + margin <= y <= room.length / 2.0 - margin
        and margin <= z <= room.height - margin
    )


def _camera_inside_object(camera: CameraSpec, objects: list[ObjectSpec], margin: float) -> bool:
    x, y, _ = camera.location
    for obj in objects:
        min_x, max_x, min_y, max_y = object_footprint(obj, margin)
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return True
    return False


def _reasonable_pitch(camera: CameraSpec) -> bool:
    pitch = camera.rotation[0]
    return 0.85 <= pitch <= 1.5


def _camera_forward_xy(camera: CameraSpec) -> tuple[float, float]:
    pitch, _, yaw = camera.rotation
    dx = -sin(yaw) * sin(pitch)
    dy = cos(yaw) * sin(pitch)
    length = sqrt(dx * dx + dy * dy)

    if length == 0.0:
        return (0.0, 0.0)

    return (dx / length, dy / length)


def _room_exit_distance(
    location: tuple[float, float, float],
    direction: tuple[float, float],
    room: RoomSpec,
) -> float:
    x, y, _ = location
    dx, dy = direction
    distances = []

    if dx > 0.0:
        distances.append((room.width / 2.0 - x) / dx)
    elif dx < 0.0:
        distances.append((-room.width / 2.0 - x) / dx)

    if dy > 0.0:
        distances.append((room.length / 2.0 - y) / dy)
    elif dy < 0.0:
        distances.append((-room.length / 2.0 - y) / dy)

    positive_distances = [distance for distance in distances if distance > 0.0]
    return min(positive_distances, default=0.0)


def _nearest_object_hit_distance(
    location: tuple[float, float, float],
    direction: tuple[float, float],
    objects: list[ObjectSpec],
    margin: float,
) -> float | None:
    distances = [
        distance
        for obj in objects
        if (distance := _ray_box_hit_distance(location, direction, object_footprint(obj, margin))) is not None
    ]
    return min(distances, default=None)


def _ray_box_hit_distance(
    location: tuple[float, float, float],
    direction: tuple[float, float],
    box: tuple[float, float, float, float],
) -> float | None:
    x, y, _ = location
    dx, dy = direction
    min_x, max_x, min_y, max_y = box
    t_min = -float("inf")
    t_max = float("inf")

    for origin, ray_direction, box_min, box_max in (
        (x, dx, min_x, max_x),
        (y, dy, min_y, max_y),
    ):
        if abs(ray_direction) < 1e-6:
            if origin < box_min or origin > box_max:
                return None
            continue

        t1 = (box_min - origin) / ray_direction
        t2 = (box_max - origin) / ray_direction
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))

    if t_max < 0.0 or t_min > t_max:
        return None

    return max(t_min, 0.0)
