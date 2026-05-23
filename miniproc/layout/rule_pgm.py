from math import atan2, pi

from miniproc.core.types import AssetSpec, ObjectSpec, RoomSpec


def create_rule_layout(room: RoomSpec, assets: list[AssetSpec]) -> list[ObjectSpec]:
    table_location = (0.0, 0.0)
    objects = []

    for asset in assets:
        if asset.category == "table":
            location = (*table_location, 0.0)
            yaw = 0.0
        elif asset.category == "sofa":
            location = (0.0, room.length / 2.0 - 0.75, 0.0)
            yaw = _yaw_facing(location, (0.0, 0.0, 0.0))
        elif asset.category == "chair":
            location = (-0.9, -0.8, 0.0)
            yaw = _yaw_facing(location, (*table_location, 0.0))
        elif asset.category in ("bookcase", "wardrobe"):
            location = (-room.width / 2.0 + 0.45, 0.2, 0.0)
            yaw = pi / 2.0
        else:
            location = (0.0, 0.0, 0.0)
            yaw = 0.0

        objects.append(
            ObjectSpec(
                asset=asset,
                location=location,
                rotation=(0.0, 0.0, yaw),
                scale=(1.0, 1.0, 1.0),
                bbox_2d=(0.0, 0.0, 0.0, 0.0),
            )
        )

    return objects


def _yaw_facing(location: tuple[float, float, float], target: tuple[float, float, float]) -> float:
    dx = target[0] - location[0]
    dy = target[1] - location[1]
    return atan2(-dx, dy)
