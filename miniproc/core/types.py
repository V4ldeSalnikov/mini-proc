from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RoomSpec:
    room_type: str
    width: float
    length: float
    height: float


@dataclass
class AssetSpec:
    category: str
    asset_id: str
    asset_path: str
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ObjectSpec:
    asset: AssetSpec
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    bbox_2d: Tuple[float, float, float, float]


@dataclass
class CameraSpec:
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    focal_length: float


@dataclass
class LightSpec:
    light_type: str
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    energy: float
    size: Optional[float] = None


@dataclass
class MaterialSpec:
    name: str
    color: Tuple[float, float, float, float]
    roughness: float = 0.6


@dataclass
class SceneSpec:
    scene_id: str
    room: RoomSpec
    objects: List[ObjectSpec]


@dataclass
class FrameSpec:
    frame_id: str
    scene_id: str
    camera: CameraSpec
    lights: List[LightSpec]
