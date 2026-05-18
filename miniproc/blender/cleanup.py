
def clear_scene() -> None:
    # to be ablte to import outside of blender
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
