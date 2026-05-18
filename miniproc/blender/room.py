from miniproc.blender.materials import create_material
from miniproc.core.types import MaterialSpec, RoomSpec


def create_room(room: RoomSpec) -> None:
    floor_material = create_material(
        MaterialSpec(
            name="floor_mat",
            color=(0.30, 0.30, 0.28, 1.0),
            roughness=0.7,
        )
    )
    wall_material = create_material(
        MaterialSpec(
            name="wall_mat",
            color=(0.72, 0.70, 0.64, 1.0),
            roughness=0.65,
        )
    )

    half_width = room.width / 2.0
    half_length = room.length / 2.0
    wall_height = room.height / 2.0
    wall_thickness = 0.08

    _add_cube(
        name="floor",
        location=(0.0, 0.0, -0.05),
        scale=(half_width, half_length, 0.05),
        material=floor_material,
    )
    _add_cube(
        name="wall_front",
        location=(0.0, -half_length, wall_height),
        scale=(half_width, wall_thickness / 2.0, wall_height),
        material=wall_material,
    )
    _add_cube(
        name="wall_back",
        location=(0.0, half_length, wall_height),
        scale=(half_width, wall_thickness / 2.0, wall_height),
        material=wall_material,
    )
    _add_cube(
        name="wall_left",
        location=(-half_width, 0.0, wall_height),
        scale=(wall_thickness / 2.0, half_length, wall_height),
        material=wall_material,
    )
    _add_cube(
        name="wall_right",
        location=(half_width, 0.0, wall_height),
        scale=(wall_thickness / 2.0, half_length, wall_height),
        material=wall_material,
    )


def _add_cube(name: str, location: tuple[float, float, float], scale: tuple[float, float, float], material) -> None:
    import bpy

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
