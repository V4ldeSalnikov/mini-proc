from miniproc.core.types import LightSpec


def add_light(light: LightSpec) -> None:
    import bpy

    bpy.ops.object.light_add(
        type=light.light_type,
        location=light.location,
        rotation=light.rotation,
    )

    obj = bpy.context.object
    obj.name = "light"
    obj.data.energy = light.energy

    if light.size is not None and hasattr(obj.data, "size"):
        obj.data.size = light.size
