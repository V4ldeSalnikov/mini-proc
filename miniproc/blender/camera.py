from miniproc.core.types import CameraSpec


def add_camera(camera: CameraSpec) -> None:
    import bpy

    bpy.ops.object.camera_add(
        location=camera.location,
        rotation=camera.rotation,
    )

    obj = bpy.context.object
    obj.name = "camera"
    obj.data.lens = camera.focal_length
    bpy.context.scene.camera = obj
