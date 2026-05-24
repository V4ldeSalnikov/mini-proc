from math import cos, sin

from miniproc.core.types import ObjectSpec, RoomSpec


Footprint = tuple[float, float, float, float]


def object_footprint(obj: ObjectSpec, margin: float = 0.0) -> Footprint:
    width = obj.asset.size[0] * obj.scale[0]
    depth = obj.asset.size[1] * obj.scale[1]
    yaw = obj.rotation[2]

    footprint_width = abs(cos(yaw)) * width + abs(sin(yaw)) * depth
    footprint_depth = abs(sin(yaw)) * width + abs(cos(yaw)) * depth

    x, y, _ = obj.location
    return (
        x - footprint_width / 2.0 - margin,
        x + footprint_width / 2.0 + margin,
        y - footprint_depth / 2.0 - margin,
        y + footprint_depth / 2.0 + margin,
    )


def is_inside_room(obj: ObjectSpec, room: RoomSpec, margin: float = 0.0) -> bool:
    min_x, max_x, min_y, max_y = object_footprint(obj, margin)
    return (
        min_x >= -room.width / 2.0
        and max_x <= room.width / 2.0
        and min_y >= -room.length / 2.0
        and max_y <= room.length / 2.0
    )


def objects_overlap(obj_a: ObjectSpec, obj_b: ObjectSpec, margin: float = 0.0) -> bool:
    a_min_x, a_max_x, a_min_y, a_max_y = object_footprint(obj_a, margin)
    b_min_x, b_max_x, b_min_y, b_max_y = object_footprint(obj_b, margin)

    separated = (
        a_max_x <= b_min_x
        or b_max_x <= a_min_x
        or a_max_y <= b_min_y
        or b_max_y <= a_min_y
    )
    return not separated


def validate_layout(objects: list[ObjectSpec], room: RoomSpec, margin: float = 0.0) -> bool:
    for obj in objects:
        if not is_inside_room(obj, room, margin):
            return False

    for index, obj in enumerate(objects):
        for other in objects[index + 1 :]:
            if objects_overlap(obj, other, margin):
                return False

    return True
