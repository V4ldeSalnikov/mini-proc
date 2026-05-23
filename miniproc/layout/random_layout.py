from math import sqrt, tau
from random import Random

from miniproc.core.types import AssetSpec, ObjectSpec, RoomSpec


CATEGORY_RADIUS = {
    "bed": 1.0,
    "bookcase": 0.6,
    "chair": 0.45,
    "desk": 0.8,
    "sofa": 0.9,
    "table": 0.75,
    "wardrobe": 0.7,
}


def create_random_layout(room: RoomSpec, assets: list[AssetSpec], seed: int) -> list[ObjectSpec]:
    rng = Random(seed)
    placed = []
    objects = []

    for asset in assets:
        radius = CATEGORY_RADIUS.get(asset.category, 0.6)
        x, y = _sample_position(room, placed, radius, rng)
        placed.append((x, y, radius))

        objects.append(
            ObjectSpec(
                asset=asset,
                location=(x, y, 0.0),
                rotation=(0.0, 0.0, rng.uniform(0.0, tau)),
                scale=(1.0, 1.0, 1.0),
                bbox_2d=(0.0, 0.0, 0.0, 0.0),
            )
        )

    return objects


def _sample_position(
    room: RoomSpec,
    placed: list[tuple[float, float, float]],
    radius: float,
    rng: Random,
) -> tuple[float, float]:
    margin = radius + 0.2
    min_x = -room.width / 2.0 + margin
    max_x = room.width / 2.0 - margin
    min_y = -room.length / 2.0 + margin
    max_y = room.length / 2.0 - margin

    for _ in range(1000):
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        if _does_not_overlap(x, y, radius, placed):
            return x, y

    return x, y


def _does_not_overlap(x: float, y: float, radius: float, placed: list[tuple[float, float, float]]) -> bool:
    for other_x, other_y, other_radius in placed:
        distance = sqrt((x - other_x) ** 2 + (y - other_y) ** 2)
        if distance < radius + other_radius + 0.25:
            return False
    return True
