import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEPTH_EXTENSIONS = (".npy", ".npz", ".exr", ".png", ".tif", ".tiff")
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DepthDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (240, 320),
        max_depth: float = 20.0,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.max_depth = max_depth

        image_dir = _first_existing_dir(self.root, ["rgb", "images", "image"])
        depth_dir = _first_existing_dir(self.root, ["depth", "depths"])
        self.samples = _find_samples(image_dir, depth_dir)

        if not self.samples:
            raise ValueError(
                f"No RGB/depth pairs found in {self.root}. Expected folders like rgb/*.png and depth/*.npy."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, depth_path = self.samples[index]

        image = _load_rgb(image_path, self.image_size)
        depth, mask = _load_depth(depth_path, self.image_size, self.max_depth)

        return {
            "image": image,
            "depth": depth,
            "mask": mask,
            "stem": image_path.stem,
        }


def _first_existing_dir(root: Path, names: list[str]) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return root / names[0]


def _find_samples(image_dir: Path, depth_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = sorted(
        path
        for extension in IMAGE_EXTENSIONS
        for path in image_dir.glob(f"*{extension}")
    )

    samples = []
    for image_path in image_paths:
        depth_path = _matching_depth_path(depth_dir, image_path.stem)
        if depth_path is not None:
            samples.append((image_path, depth_path))

    return samples


def _matching_depth_path(depth_dir: Path, stem: str) -> Path | None:
    for extension in DEPTH_EXTENSIONS:
        path = depth_dir / f"{stem}{extension}"
        if path.exists():
            return path

    for extension in DEPTH_EXTENSIONS:
        matches = sorted(depth_dir.glob(f"{stem}_*{extension}"))
        if matches:
            return matches[0]

    return None


def _load_rgb(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    image = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def _load_depth(path: Path, image_size: tuple[int, int], max_depth: float) -> tuple[torch.Tensor, torch.Tensor]:
    depth = torch.from_numpy(_read_depth_array(path).astype(np.float32)).unsqueeze(0)
    mask = torch.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    depth = torch.where(mask, depth, torch.zeros_like(depth)).clamp(0.0, max_depth)

    depth = F.interpolate(
        depth.unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask = F.interpolate(
        mask.float().unsqueeze(0),
        size=image_size,
        mode="nearest",
    ).squeeze(0).bool()

    return depth, mask


def _read_depth_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        return data[key]

    if path.suffix == ".exr":
        return _read_exr(path)

    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 1000.0
    return array.astype(np.float32)


def _read_exr(path: Path) -> np.ndarray:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"Could not read depth EXR: {path}")
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.float32)
