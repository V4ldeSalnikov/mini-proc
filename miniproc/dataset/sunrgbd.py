import json
from collections import Counter, defaultdict
from math import atan2, sqrt
from pathlib import Path
from typing import Any


SUPPORTED_ROOM_TYPES = {
    "bedroom",
    "bookstore",
    "classroom",
    "computer_room",
    "conference_room",
    "dining_area",
    "dining_room",
    "discussion_area",
    "furniture_store",
    "home",
    "home_office",
    "library",
    "living_room",
    "office",
    "office_dining",
    "playroom",
    "recreation_room",
    "rest_space",
    "study",
    "study_space",
}

NEAR_WALL_DISTANCE = 0.3

TARGET_CATEGORIES = {
    "bed": "bed",
    "bookcase": "bookcase",
    "bookshelf": "bookcase",
    "cabinet": "wardrobe",
    "chair": "chair",
    "couch": "sofa",
    "desk": "desk",
    "shelf": "bookcase",
    "sofa": "sofa",
    "sofa_chair": "chair",
    "table": "table",
    "wardrobe": "wardrobe",
}

SUPPORTED_CATEGORIES = sorted(set(TARGET_CATEGORIES.values()))


def extract_layout_priors(sunrgbd_root: str | Path, max_scenes: int | None = None) -> dict[str, Any]:
    root = Path(sunrgbd_root)
    samples_by_category: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    relation_samples: dict[str, list[dict[str, float]]] = defaultdict(list)
    object_counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    scene_types = Counter()
    room_widths = []
    room_lengths = []
    room_heights = []
    room_widths_by_type: dict[str, list[float]] = defaultdict(list)
    room_lengths_by_type: dict[str, list[float]] = defaultdict(list)
    room_heights_by_type: dict[str, list[float]] = defaultdict(list)
    scenes_seen = 0
    scenes_with_layout = 0
    scenes_used = 0
    skipped_unsupported_room = 0

    for annotation_path in root.rglob("annotation3Dfinal/index.json"):
        if max_scenes is not None and scenes_seen >= max_scenes:
            break

        scene_dir = annotation_path.parents[1]
        layout_path = scene_dir / "annotation3Dlayout" / "index.json"
        room_bounds = _read_room_bounds(layout_path)
        scenes_seen += 1

        if room_bounds is None:
            continue

        scenes_with_layout += 1
        scene_type = _read_scene_type(scene_dir)
        if scene_type not in SUPPORTED_ROOM_TYPES:
            skipped_unsupported_room += 1
            continue

        room_min_x, room_max_x, room_min_z, room_max_z, room_min_y, room_max_y = room_bounds
        room_width = room_max_x - room_min_x
        room_length = room_max_z - room_min_z
        room_height = room_max_y - room_min_y
        room_center_x = (room_min_x + room_max_x) / 2.0
        room_center_z = (room_min_z + room_max_z) / 2.0

        if room_width <= 0.0 or room_length <= 0.0 or room_height <= 0.0:
            continue

        scenes_used += 1
        room_widths.append(room_width)
        room_lengths.append(room_length)
        room_heights.append(room_height)
        room_widths_by_type[scene_type].append(room_width)
        room_lengths_by_type[scene_type].append(room_length)
        room_heights_by_type[scene_type].append(room_height)
        scene_types[scene_type] += 1
        scene_objects = []

        for obj in _read_objects(annotation_path):
            category = _map_category(obj.get("name", ""))
            if category is None:
                continue

            polygon = _first_polygon(obj)
            if polygon is None:
                continue

            object_bounds = _polygon_bounds(polygon)
            if object_bounds is None:
                continue

            obj_min_x, obj_max_x, obj_min_z, obj_max_z, obj_min_y, obj_max_y = object_bounds
            center_x = (obj_min_x + obj_max_x) / 2.0
            center_z = (obj_min_z + obj_max_z) / 2.0
            x_norm = (center_x - room_center_x) / room_width
            y_norm = (center_z - room_center_z) / room_length

            scene_objects.append(
                {
                    "category": category,
                    "center_x": center_x,
                    "center_z": center_z,
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                }
            )

            samples_by_category[category].append(
                {
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "width": obj_max_x - obj_min_x,
                    "depth": obj_max_z - obj_min_z,
                    "height": obj_max_y - obj_min_y,
                    "wall_distance": _wall_distance(
                        obj_min_x,
                        obj_max_x,
                        obj_min_z,
                        obj_max_z,
                        room_min_x,
                        room_max_x,
                        room_min_z,
                        room_max_z,
                    ),
                    "nearest_wall": _nearest_wall(
                        obj_min_x,
                        obj_max_x,
                        obj_min_z,
                        obj_max_z,
                        room_min_x,
                        room_max_x,
                        room_min_z,
                        room_max_z,
                    ),
                }
            )

        relation_samples["chair_to_nearest_table"].extend(
            _nearest_relation_samples(scene_objects, "chair", "table", room_width, room_length)
        )
        relation_samples["sofa_to_nearest_table"].extend(
            _nearest_relation_samples(scene_objects, "sofa", "table", room_width, room_length)
        )
        relation_samples["desk_to_nearest_chair"].extend(
            _nearest_relation_samples(scene_objects, "desk", "chair", room_width, room_length)
        )
        relation_samples["bed_to_nearest_wardrobe"].extend(
            _nearest_relation_samples(scene_objects, "bed", "wardrobe", room_width, room_length)
        )
        _add_object_counts(object_counts, scene_type, scene_objects)

    return {
        "source": str(root),
        "supported_room_types": sorted(SUPPORTED_ROOM_TYPES),
        "near_wall_distance": NEAR_WALL_DISTANCE,
        "scenes_seen": scenes_seen,
        "scenes_with_layout": scenes_with_layout,
        "scenes_used": scenes_used,
        "skipped_unsupported_room": skipped_unsupported_room,
        "room": {
            "width": _summary(room_widths),
            "length": _summary(room_lengths),
            "height": _summary(room_heights),
        },
        "room_by_type": {
            scene_type: {
                "width": _summary(room_widths_by_type[scene_type]),
                "length": _summary(room_lengths_by_type[scene_type]),
                "height": _summary(room_heights_by_type[scene_type]),
            }
            for scene_type in sorted(room_widths_by_type)
        },
        "scene_types": dict(scene_types.most_common()),
        "categories": {
            category: _category_summary(samples)
            for category, samples in sorted(samples_by_category.items())
        },
        "object_counts": _object_count_summary(object_counts),
        "relations": {
            relation: _relation_summary(samples)
            for relation, samples in sorted(relation_samples.items())
        },
    }


def _read_objects(annotation_path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    return [obj for obj in data.get("objects", []) if isinstance(obj, dict)]


def _read_room_bounds(layout_path: Path) -> tuple[float, float, float, float, float, float] | None:
    objects = _read_objects(layout_path)
    for obj in objects:
        if obj.get("name") != "room":
            continue

        polygon = _first_polygon(obj)
        if polygon is None:
            continue

        bounds = _polygon_bounds(polygon)
        if bounds is not None:
            return bounds

    return None


def _read_scene_type(scene_dir: Path) -> str | None:
    scene_path = scene_dir / "scene.txt"
    if not scene_path.exists():
        return None

    scene_type = scene_path.read_text(encoding="utf-8", errors="ignore").strip()
    return scene_type or None


def _map_category(name: str) -> str | None:
    base_name = name.strip().lower().split(":")[0]
    return TARGET_CATEGORIES.get(base_name)


def _first_polygon(obj: dict[str, Any]) -> dict[str, Any] | None:
    polygons = obj.get("polygon", [])
    if not polygons:
        return None
    polygon = polygons[0]
    return polygon if isinstance(polygon, dict) else None


def _polygon_bounds(polygon: dict[str, Any]) -> tuple[float, float, float, float, float, float] | None:
    xs = polygon.get("X", [])
    zs = polygon.get("Z", [])
    if len(xs) == 0 or len(zs) == 0:
        return None

    y_min = float(polygon.get("Ymin", 0.0))
    y_max = float(polygon.get("Ymax", 0.0))

    return (
        min(xs),
        max(xs),
        min(zs),
        max(zs),
        min(y_min, y_max),
        max(y_min, y_max),
    )


def _wall_distance(
    obj_min_x: float,
    obj_max_x: float,
    obj_min_z: float,
    obj_max_z: float,
    room_min_x: float,
    room_max_x: float,
    room_min_z: float,
    room_max_z: float,
) -> float:
    return max(
        0.0,
        min(
            obj_min_x - room_min_x,
            room_max_x - obj_max_x,
            obj_min_z - room_min_z,
            room_max_z - obj_max_z,
        ),
    )


def _nearest_wall(
    obj_min_x: float,
    obj_max_x: float,
    obj_min_z: float,
    obj_max_z: float,
    room_min_x: float,
    room_max_x: float,
    room_min_z: float,
    room_max_z: float,
) -> str:
    distances = {
        "left": obj_min_x - room_min_x,
        "right": room_max_x - obj_max_x,
        "front": obj_min_z - room_min_z,
        "back": room_max_z - obj_max_z,
    }
    return min(distances, key=distances.get)


def _nearest_relation_samples(
    scene_objects: list[dict[str, float | str]],
    source_category: str,
    target_category: str,
    room_width: float,
    room_length: float,
) -> list[dict[str, float]]:
    sources = [obj for obj in scene_objects if obj["category"] == source_category]
    targets = [obj for obj in scene_objects if obj["category"] == target_category]
    samples = []

    for source in sources:
        target = min(targets, key=lambda candidate: _center_distance(source, candidate), default=None)
        if target is None:
            continue

        dx = float(source["center_x"]) - float(target["center_x"])
        dy = float(source["center_z"]) - float(target["center_z"])
        samples.append(
            {
                "distance": sqrt(dx * dx + dy * dy),
                "dx": dx,
                "dy": dy,
                "dx_norm": dx / room_width,
                "dy_norm": dy / room_length,
                "angle": atan2(dy, dx),
            }
        )

    return samples


def _center_distance(obj_a: dict[str, float | str], obj_b: dict[str, float | str]) -> float:
    dx = float(obj_a["center_x"]) - float(obj_b["center_x"])
    dy = float(obj_a["center_z"]) - float(obj_b["center_z"])
    return sqrt(dx * dx + dy * dy)


def _add_object_counts(
    object_counts: dict[str, dict[str, Counter]],
    scene_type: str,
    scene_objects: list[dict[str, float | str]],
) -> None:
    counts = Counter(str(obj["category"]) for obj in scene_objects)
    for category in SUPPORTED_CATEGORIES:
        object_counts[scene_type][category][counts[category]] += 1


def _category_summary(samples: list[dict[str, float | str]]) -> dict[str, Any]:
    wall_counts = Counter(str(sample["nearest_wall"]) for sample in samples)
    near_wall_count = sum(float(sample["wall_distance"]) <= NEAR_WALL_DISTANCE for sample in samples)
    return {
        "count": len(samples),
        "x_norm": _summary([float(sample["x_norm"]) for sample in samples]),
        "y_norm": _summary([float(sample["y_norm"]) for sample in samples]),
        "width": _summary([float(sample["width"]) for sample in samples]),
        "depth": _summary([float(sample["depth"]) for sample in samples]),
        "height": _summary([float(sample["height"]) for sample in samples]),
        "wall_distance": _summary([float(sample["wall_distance"]) for sample in samples]),
        "near_wall_probability": near_wall_count / len(samples),
        "wall_probs": {
            wall: wall_counts[wall] / len(samples)
            for wall in ("front", "back", "left", "right")
        },
    }


def _object_count_summary(object_counts: dict[str, dict[str, Counter]]) -> dict[str, Any]:
    return {
        scene_type: {
            category: {
                int(count): frequency
                for count, frequency in sorted(counts.items())
            }
            for category, counts in sorted(category_counts.items())
        }
        for scene_type, category_counts in sorted(object_counts.items())
    }


def _relation_summary(samples: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "count": len(samples),
        "distance": _summary([sample["distance"] for sample in samples]),
        "dx": _summary([sample["dx"] for sample in samples]),
        "dy": _summary([sample["dy"] for sample in samples]),
        "dx_norm": _summary([sample["dx_norm"] for sample in samples]),
        "dy_norm": _summary([sample["dy_norm"] for sample in samples]),
        "angle": _summary([sample["angle"] for sample in samples]),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": sqrt(variance),
        "min": min(values),
        "max": max(values),
    }
