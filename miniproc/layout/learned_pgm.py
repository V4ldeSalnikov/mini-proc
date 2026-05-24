from math import atan2, cos, pi, sin
from pathlib import Path
from random import Random
from typing import Any

import yaml

from miniproc.core.types import AssetSpec, ObjectSpec, RoomSpec
from miniproc.layout.validators import validate_layout


DEFAULT_PRIORS_PATH = Path("configs/sunrgbd_layout_priors.yaml")


def create_learned_layout(
    room: RoomSpec,
    assets: list[AssetSpec],
    seed: int = 0,
    priors_path: str | Path = DEFAULT_PRIORS_PATH,
    max_attempts: int = 100,
    margin: float = 0.05,
) -> list[ObjectSpec]:
    priors = load_priors(priors_path)
    rng = Random(seed)

    for _ in range(max_attempts):
        objects = []
        anchors: dict[str, list[ObjectSpec]] = {}

        for asset in sorted(assets, key=_asset_priority):
            obj = _sample_object(asset, room, priors, anchors, rng, margin)
            objects.append(obj)
            anchors.setdefault(asset.category, []).append(obj)

        if validate_layout(objects, room, margin):
            return objects

    raise ValueError("Learned PGM could not sample a valid layout. Try a larger room or smaller/fewer assets.")


def load_priors(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def sample_room_type(
    seed: int = 0,
    priors_path: str | Path = DEFAULT_PRIORS_PATH,
) -> str:
    priors = load_priors(priors_path)
    rng = Random(seed)
    return str(_sample_weighted(priors.get("scene_types", {}), rng))


def sample_room(
    seed: int = 0,
    priors_path: str | Path = DEFAULT_PRIORS_PATH,
    room_type: str | None = None,
    height: float | None = None,
    min_width: float = 3.0,
    max_width: float = 10.0,
    min_length: float = 3.0,
    max_length: float = 8.0,
    min_height: float = 2.2,
    max_height: float = 3.5,
) -> RoomSpec:
    priors = load_priors(priors_path)
    rng = Random(seed)

    if room_type is None:
        room_type = str(_sample_weighted(priors.get("scene_types", {}), rng))

    room_priors = priors.get("room_by_type", {}).get(room_type) or priors.get("room", {})
    width = _sample_summary(room_priors.get("width", {}), rng, low=min_width, high=max_width)
    length = _sample_summary(room_priors.get("length", {}), rng, low=min_length, high=max_length)
    if height is None:
        height_prior = room_priors.get("height")
        height = _sample_summary(height_prior, rng, low=min_height, high=max_height) if height_prior else 2.8

    return RoomSpec(
        room_type=room_type,
        width=width,
        length=length,
        height=height,
    )


def sample_asset_categories(
    seed: int = 0,
    priors_path: str | Path = DEFAULT_PRIORS_PATH,
    room_type: str | None = None,
    min_objects: int = 1,
    max_objects: int = 5,
    max_per_category: int = 3,
) -> list[str]:
    priors = load_priors(priors_path)
    rng = Random(seed)

    if room_type is None:
        room_type = str(_sample_weighted(priors.get("scene_types", {}), rng))

    count_priors = priors.get("object_counts", {}).get(room_type, {})
    if not count_priors:
        return ["chair", "table", "sofa"]

    categories = []
    for category, count_distribution in sorted(count_priors.items()):
        count = int(_sample_weighted(count_distribution, rng))
        count = min(count, max_per_category)
        categories.extend([category] * count)

    rng.shuffle(categories)
    categories = categories[:max_objects]

    attempts = 0
    while len(categories) < min_objects and len(categories) < max_objects and attempts < 100:
        category = _sample_present_category(count_priors, rng)
        if categories.count(category) < max_per_category:
            categories.append(category)
        attempts += 1

    if not categories:
        categories.append(_most_likely_present_category(count_priors))

    return categories


def _sample_object(
    asset: AssetSpec,
    room: RoomSpec,
    priors: dict[str, Any],
    anchors: dict[str, list[ObjectSpec]],
    rng: Random,
    margin: float,
) -> ObjectSpec:
    scale = _category_scale(asset, priors)

    if asset.category == "chair" and "table" in anchors:
        location, yaw = _sample_relative(asset, room, priors, "chair_to_nearest_table", anchors["table"], rng)
    elif asset.category == "chair" and "desk" in anchors:
        location, yaw = _sample_relative(asset, room, priors, "desk_to_nearest_chair", anchors["desk"], rng, invert=True)
    elif asset.category == "sofa" and "table" in anchors:
        location, yaw = _sample_relative(asset, room, priors, "sofa_to_nearest_table", anchors["table"], rng)
    elif asset.category == "wardrobe" and "bed" in anchors:
        location, yaw = _sample_relative(asset, room, priors, "bed_to_nearest_wardrobe", anchors["bed"], rng, invert=True)
    elif asset.category in ("sofa", "bookcase", "wardrobe"):
        location, yaw = _sample_wall_object(asset, room, priors, rng, margin, scale)
    else:
        location = _sample_category_location(asset.category, room, priors, rng)
        yaw = rng.uniform(-pi, pi)

    location = _clamp_location(location, asset, yaw, scale, room, margin)

    return ObjectSpec(
        asset=asset,
        location=location,
        rotation=(0.0, 0.0, yaw),
        scale=scale,
        bbox_2d=(0.0, 0.0, 0.0, 0.0),
    )


def _sample_category_location(
    category: str,
    room: RoomSpec,
    priors: dict[str, Any],
    rng: Random,
) -> tuple[float, float, float]:
    category_prior = priors.get("categories", {}).get(category)
    if category_prior is None:
        return (rng.uniform(-0.4, 0.4) * room.width, rng.uniform(-0.4, 0.4) * room.length, 0.0)

    x_norm = _sample_summary(category_prior["x_norm"], rng, low=-0.45, high=0.45)
    y_norm = _sample_summary(category_prior["y_norm"], rng, low=-0.45, high=0.45)
    return (x_norm * room.width, y_norm * room.length, 0.0)


def _sample_relative(
    asset: AssetSpec,
    room: RoomSpec,
    priors: dict[str, Any],
    relation_name: str,
    targets: list[ObjectSpec],
    rng: Random,
    invert: bool = False,
) -> tuple[tuple[float, float, float], float]:
    relation = priors.get("relations", {}).get(relation_name)
    if relation is None:
        return _sample_category_location(asset.category, room, priors, rng), rng.uniform(-pi, pi)

    target = rng.choice(targets)
    dx_norm = _sample_summary(relation["dx_norm"], rng, low=-0.35, high=0.35)
    dy_norm = _sample_summary(relation["dy_norm"], rng, low=-0.35, high=0.35)

    if invert:
        dx_norm = -dx_norm
        dy_norm = -dy_norm

    location = (
        target.location[0] + dx_norm * room.width,
        target.location[1] + dy_norm * room.length,
        0.0,
    )
    return location, _yaw_facing(location, target.location)


def _sample_wall_object(
    asset: AssetSpec,
    room: RoomSpec,
    priors: dict[str, Any],
    rng: Random,
    margin: float,
    scale: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float]:
    category_prior = priors.get("categories", {}).get(asset.category, {})
    wall = _sample_wall(category_prior.get("wall_probs", {}), rng)

    if wall == "front":
        yaw = 0.0
    elif wall == "back":
        yaw = pi
    elif wall == "left":
        yaw = -pi / 2.0
    else:
        yaw = pi / 2.0

    width, depth = _rotated_size(_scaled_size(asset.size, scale), yaw)
    wall_distance = _sample_summary(category_prior.get("wall_distance", {}), rng, low=0.0, high=0.8)

    min_x = -room.width / 2.0 + width / 2.0 + margin
    max_x = room.width / 2.0 - width / 2.0 - margin
    min_y = -room.length / 2.0 + depth / 2.0 + margin
    max_y = room.length / 2.0 - depth / 2.0 - margin

    if wall == "front":
        location = (rng.uniform(min_x, max_x), min(min_y + wall_distance, max_y), 0.0)
    elif wall == "back":
        location = (rng.uniform(min_x, max_x), max(max_y - wall_distance, min_y), 0.0)
    elif wall == "left":
        location = (min(min_x + wall_distance, max_x), rng.uniform(min_y, max_y), 0.0)
    else:
        location = (max(max_x - wall_distance, min_x), rng.uniform(min_y, max_y), 0.0)

    return location, yaw


def _category_scale(asset: AssetSpec, priors: dict[str, Any]) -> tuple[float, float, float]:
    category_prior = priors.get("categories", {}).get(asset.category)
    if category_prior is None:
        return (1.0, 1.0, 1.0)

    target_size = (
        float(category_prior["width"]["mean"]),
        float(category_prior["depth"]["mean"]),
        float(category_prior["height"]["mean"]),
    )
    ratios = [
        target / raw
        for target, raw in zip(target_size, asset.size)
        if target > 0.0 and raw > 0.0
    ]
    if not ratios:
        return (1.0, 1.0, 1.0)

    scale = sorted(ratios)[len(ratios) // 2]
    scale = max(0.5, min(4.0, scale))
    return (scale, scale, scale)


def _clamp_location(
    location: tuple[float, float, float],
    asset: AssetSpec,
    yaw: float,
    scale: tuple[float, float, float],
    room: RoomSpec,
    margin: float,
) -> tuple[float, float, float]:
    width, depth = _rotated_size(_scaled_size(asset.size, scale), yaw)
    min_x = -room.width / 2.0 + width / 2.0 + margin
    max_x = room.width / 2.0 - width / 2.0 - margin
    min_y = -room.length / 2.0 + depth / 2.0 + margin
    max_y = room.length / 2.0 - depth / 2.0 - margin

    if min_x > max_x or min_y > max_y:
        return location

    x, y, z = location
    return (
        max(min_x, min(max_x, x)),
        max(min_y, min(max_y, y)),
        z,
    )


def _sample_weighted(distribution: dict[Any, int | float], rng: Random) -> Any:
    if not distribution:
        return None

    total_weight = sum(float(weight) for weight in distribution.values())
    value = rng.random() * total_weight
    running_weight = 0.0

    for item, weight in distribution.items():
        running_weight += float(weight)
        if value <= running_weight:
            return item

    return next(reversed(distribution))


def _most_likely_present_category(count_priors: dict[str, dict[Any, int | float]]) -> str:
    best_category = "table"
    best_count = 0.0

    for category, count_distribution in count_priors.items():
        present_count = sum(
            float(frequency)
            for count, frequency in count_distribution.items()
            if int(count) > 0
        )
        if present_count > best_count:
            best_category = category
            best_count = present_count

    return best_category


def _sample_present_category(count_priors: dict[str, dict[Any, int | float]], rng: Random) -> str:
    present_weights = {}

    for category, count_distribution in count_priors.items():
        present_weights[category] = sum(
            float(frequency)
            for count, frequency in count_distribution.items()
            if int(count) > 0
        )

    return str(_sample_weighted(present_weights, rng))


def _sample_wall(wall_probs: dict[str, float], rng: Random) -> str:
    value = rng.random()
    total = 0.0
    for wall in ("front", "back", "left", "right"):
        total += float(wall_probs.get(wall, 0.25))
        if value <= total:
            return wall
    return rng.choice(["front", "back", "left", "right"])


def _sample_summary(
    summary: dict[str, float | int],
    rng: Random,
    low: float | None = None,
    high: float | None = None,
) -> float:
    mean = float(summary.get("mean", 0.0))
    std = float(summary.get("std", 0.0))
    value = rng.gauss(mean, std)

    if low is None:
        low = float(summary.get("min", value))
    if high is None:
        high = float(summary.get("max", value))

    return max(low, min(high, value))


def _asset_priority(asset: AssetSpec) -> int:
    if asset.category in ("table", "desk", "bed"):
        return 0
    if asset.category in ("chair", "sofa", "wardrobe"):
        return 1
    return 2


def _rotated_size(size: tuple[float, float, float], yaw: float) -> tuple[float, float]:
    width, depth, _ = size
    return (
        abs(cos(yaw)) * width + abs(sin(yaw)) * depth,
        abs(sin(yaw)) * width + abs(cos(yaw)) * depth,
    )


def _scaled_size(
    size: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        size[0] * scale[0],
        size[1] * scale[1],
        size[2] * scale[2],
    )


def _yaw_facing(location: tuple[float, float, float], target: tuple[float, float, float]) -> float:
    dx = target[0] - location[0]
    dy = target[1] - location[1]
    return atan2(-dx, dy)
