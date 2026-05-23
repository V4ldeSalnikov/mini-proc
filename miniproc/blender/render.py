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


def render_depth(
    depth_path: str,
    visualization_path: str,
    resolution: tuple[int, int] = (640, 480),
    depth_min: float = 1.5,
    depth_max: float = 5.5,
) -> None:
    import bpy

    depth_path = Path(depth_path).resolve()
    visualization_path = Path(visualization_path).resolve()
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_path.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.use_nodes = True

    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new("CompositorNodeRLayers")

    depth_file = tree.nodes.new("CompositorNodeOutputFile")
    depth_file.base_path = str(depth_path.parent)
    depth_file.file_slots[0].path = f"{depth_path.stem}_"
    depth_file.format.file_format = "OPEN_EXR"
    depth_file.format.color_depth = "32"
    tree.links.new(render_layers.outputs["Depth"], depth_file.inputs[0])

    map_range = tree.nodes.new("CompositorNodeMapRange")
    map_range.inputs["From Min"].default_value = depth_min
    map_range.inputs["From Max"].default_value = depth_max
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.use_clamp = True
    tree.links.new(render_layers.outputs["Depth"], map_range.inputs["Value"])

    color_ramp = tree.nodes.new("CompositorNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (1.0, 0.92, 0.05, 1.0)
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (0.25, 0.02, 0.85, 1.0)
    color_ramp.color_ramp.elements.new(0.5).color = (0.95, 0.12, 0.65, 1.0)
    tree.links.new(map_range.outputs["Value"], color_ramp.inputs["Fac"])

    visualization_file = tree.nodes.new("CompositorNodeOutputFile")
    visualization_file.base_path = str(visualization_path.parent)
    visualization_file.file_slots[0].path = f"{visualization_path.stem}_"
    visualization_file.format.file_format = "PNG"
    tree.links.new(color_ramp.outputs["Image"], visualization_file.inputs[0])

    bpy.ops.render.render(write_still=False)
