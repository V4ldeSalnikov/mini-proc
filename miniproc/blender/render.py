from pathlib import Path


def configure_render_engine(samples: int = 32) -> None:
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples


def render_rgb(output_path: str, resolution: tuple[int, int] = (640, 480)) -> None:
    import bpy

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path)

    bpy.ops.render.render(write_still=True)
