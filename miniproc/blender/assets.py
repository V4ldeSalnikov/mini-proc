from pathlib import Path
from random import Random

from miniproc.core.types import AssetSpec, ObjectSpec


def list_pix3d_assets(category: str, pix3d_root: str = "assets/pix3d") -> list[AssetSpec]:
    category_dir = Path(pix3d_root).resolve() / "model" / category

    assets = []
    for obj_path in sorted(category_dir.rglob("*.obj")):
        asset_id = obj_path.parent.name
        if obj_path.stem != "model":
            asset_id = f"{asset_id}_{obj_path.stem}"
        assets.append(
            AssetSpec(
                category=category,
                asset_id=asset_id,
                asset_path=str(obj_path),
            )
        )

    return assets


def get_pix3d_asset(category: str, asset_index: int = 0, pix3d_root: str = "assets/pix3d") -> AssetSpec:
    assets = list_pix3d_assets(category, pix3d_root)
    return measure_asset(assets[asset_index])


def sample_pix3d_asset(category: str, seed: int, pix3d_root: str = "assets/pix3d") -> AssetSpec:
    assets = list_pix3d_assets(category, pix3d_root)
    return measure_asset(Random(seed).choice(assets))


def measure_asset(asset: AssetSpec) -> AssetSpec:
    return AssetSpec(
        category=asset.category,
        asset_id=asset.asset_id,
        asset_path=asset.asset_path,
        size=measure_obj_size(asset.asset_path),
    )


def measure_obj_size(obj_path: str | Path) -> tuple[float, float, float]:
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    with Path(obj_path).open("r", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                x = float(x)
                y = float(y)
                z = float(z)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                min_z = min(min_z, z)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                max_z = max(max_z, z)

    width = max_x - min_x
    depth = max_z - min_z
    height = max_y - min_y
    return (width, depth, height)


def load_object(spec: ObjectSpec) -> list:
    import bpy
    from mathutils import Vector

    asset_path = Path(spec.asset.asset_path).resolve()

    before_import = set(bpy.context.scene.objects)
    bpy.ops.wm.obj_import(filepath=str(asset_path))

    imported_objects = [obj for obj in bpy.context.scene.objects if obj not in before_import]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_objects[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    for obj in imported_objects:
        obj.name = f"{spec.asset.category}_{spec.asset.asset_id}_{obj.name}"
        obj.location = spec.location
        obj.rotation_euler = spec.rotation
        obj.scale = spec.scale

    bpy.context.view_layer.update()
    min_z = min(
        (obj.matrix_world @ Vector(corner)).z
        for obj in imported_objects
        for corner in obj.bound_box
    )
    lift = spec.location[2] - min_z
    for obj in imported_objects:
        obj.location.z += lift

    return imported_objects
