from math import atan2, cos, pi, sin
from random import Random

from miniproc.core.types import AssetSpec, ObjectSpec, RoomSpec
from miniproc.layout.validators import validate_layout


def create_rule_layout(
    room: RoomSpec,
    assets: list[AssetSpec],
    seed: int = 0,
    max_attempts: int = 50,
    margin: float = 0.05,
) -> list[ObjectSpec]:
    rng = Random(seed)

    for _ in range(max_attempts):
        table_location = (rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35))
        objects = [_sample_object(asset, room, table_location, rng, margin) for asset in assets]

        if validate_layout(objects, room, margin):
            return objects

    raise ValueError("Rule PGM could not sample a valid layout. Try a larger room or smaller/fewer assets.")


def _sample_object(
    asset: AssetSpec,
    room: RoomSpec,
    table_location: tuple[float, float],
    rng: Random,
    margin: float,
) -> ObjectSpec:
    if asset.category == "table":
        location = (*table_location, 0.0)
        yaw = rng.uniform(-0.25, 0.25)
    elif asset.category == "sofa":
        location, yaw = _sample_wall_object(asset, room, rng, margin)
    elif asset.category == "chair":
        location, yaw = _sample_chair(table_location, rng)
    elif asset.category in ("bookcase", "wardrobe"):
        location, yaw = _sample_wall_object(asset, room, rng, margin)
    else:
        location = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.0)
        yaw = rng.uniform(-pi, pi)

    return ObjectSpec(
        asset=asset,
        location=location,
        rotation=(0.0, 0.0, yaw),
        scale=(1.0, 1.0, 1.0),
        bbox_2d=(0.0, 0.0, 0.0, 0.0),
    )


def _sample_chair(table_location: tuple[float, float], rng: Random) -> tuple[tuple[float, float, float], float]:
    angle = rng.uniform(-pi, pi)
    distance = rng.uniform(0.9, 1.4)
    location = (
        table_location[0] + cos(angle) * distance,
        table_location[1] + sin(angle) * distance,
        0.0,
    )
    return location, _yaw_facing(location, (*table_location, 0.0))


def _sample_wall_object(
    asset: AssetSpec,
    room: RoomSpec,
    rng: Random,
    margin: float,
) -> tuple[tuple[float, float, float], float]:
    wall = rng.choice(["front", "back", "left", "right"])

    if wall == "front":
        yaw = 0.0
    elif wall == "back":
        yaw = pi
    elif wall == "left":
        yaw = -pi / 2.0
    else:
        yaw = pi / 2.0

    width, depth = _rotated_size(asset.size, yaw)
    min_x = -room.width / 2.0 + width / 2.0 + margin
    max_x = room.width / 2.0 - width / 2.0 - margin
    min_y = -room.length / 2.0 + depth / 2.0 + margin
    max_y = room.length / 2.0 - depth / 2.0 - margin

    if wall == "front":
        location = (rng.uniform(min_x, max_x), min_y, 0.0)
    elif wall == "back":
        location = (rng.uniform(min_x, max_x), max_y, 0.0)
    elif wall == "left":
        location = (min_x, rng.uniform(min_y, max_y), 0.0)
    else:
        location = (max_x, rng.uniform(min_y, max_y), 0.0)

    return location, yaw


def _rotated_size(size: tuple[float, float, float], yaw: float) -> tuple[float, float]:
    width, depth, _ = size
    return (
        abs(cos(yaw)) * width + abs(sin(yaw)) * depth,
        abs(sin(yaw)) * width + abs(cos(yaw)) * depth,
    )


def _yaw_facing(location: tuple[float, float, float], target: tuple[float, float, float]) -> float:
    dx = target[0] - location[0]
    dy = target[1] - location[1]
    return atan2(-dx, dy)
