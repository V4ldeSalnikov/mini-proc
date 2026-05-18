from miniproc.core.types import MaterialSpec


def create_material(spec: MaterialSpec):
    import bpy

    material = bpy.data.materials.new(spec.name)
    material.use_nodes = True

    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = spec.color
    bsdf.inputs["Roughness"].default_value = spec.roughness

    return material
