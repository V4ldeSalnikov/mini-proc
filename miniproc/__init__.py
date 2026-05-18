from miniproc.blender.cleanup import clear_scene
from miniproc.blender.render import configure_render_engine


def init(samples: int = 32) -> None:
    clear_scene()
    configure_render_engine(samples=samples)
