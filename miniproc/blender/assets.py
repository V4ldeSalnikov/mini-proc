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
    return assets[asset_index]


def sample_pix3d_asset(category: str, seed: int, pix3d_root: str = "assets/pix3d") -> AssetSpec:
    assets = list_pix3d_assets(category, pix3d_root)
    return Random(seed).choice(assets)


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
