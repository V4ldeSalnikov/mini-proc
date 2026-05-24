from math import cos, pi, sin
from random import Random

from miniproc.core.types import AssetSpec, ObjectSpec, RoomSpec
from miniproc.layout.validators import validate_layout


def create_random_layout(
    room: RoomSpec,
    assets: list[AssetSpec],
    seed: int,
    max_attempts: int = 100,
    margin: float = 0.05,
) -> list[ObjectSpec]:
    rng = Random(seed)

    for _ in range(max_attempts):
        objects = [_sample_object(asset, room, rng, margin) for asset in assets]

        if validate_layout(objects, room, margin):
            return objects

    raise ValueError("Random layout could not sample a valid layout. Try a larger room or smaller/fewer assets.")


def _sample_object(asset: AssetSpec, room: RoomSpec, rng: Random, margin: float) -> ObjectSpec:
    yaw = rng.uniform(-pi, pi)
    width, depth = _rotated_size(asset.size, yaw)

    min_x = -room.width / 2.0 + width / 2.0 + margin
    max_x = room.width / 2.0 - width / 2.0 - margin
    min_y = -room.length / 2.0 + depth / 2.0 + margin
    max_y = room.length / 2.0 - depth / 2.0 - margin

    if min_x > max_x or min_y > max_y:
        raise ValueError(f"Asset {asset.asset_id} is too large for this room.")

    return ObjectSpec(
        asset=asset,
        location=(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y), 0.0),
        rotation=(0.0, 0.0, yaw),
        scale=(1.0, 1.0, 1.0),
        bbox_2d=(0.0, 0.0, 0.0, 0.0),
    )


def _rotated_size(size: tuple[float, float, float], yaw: float) -> tuple[float, float]:
    width, depth, _ = size
    return (
        abs(cos(yaw)) * width + abs(sin(yaw)) * depth,
        abs(sin(yaw)) * width + abs(cos(yaw)) * depth,
    )
