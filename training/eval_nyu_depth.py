import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from training.datasets import DepthDataset
from training.metrics import MetricAverager, compute_depth_metrics
from training.models import ResNet18UNet


DATASET_ROOT = Path("outputs/datasets/nyu_depth_test")
CHECKPOINT_PATH = Path("outputs/checkpoints/depth/best_resnet18_unet.pt")
IMAGE_SIZE = (240, 320)
BATCH_SIZE = 4
NUM_WORKERS = 0
DEFAULT_MAX_DEPTH = 8.0


def main() -> None:
    device = get_device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    max_depth = float(checkpoint.get("max_depth", DEFAULT_MAX_DEPTH))

    dataset = DepthDataset(DATASET_ROOT, image_size=IMAGE_SIZE, max_depth=max_depth)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ResNet18UNet(max_depth=max_depth).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    raw_metrics = MetricAverager()
    aligned_metrics = MetricAverager()

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            prediction = model(image)

            raw_metrics.update(compute_depth_metrics(prediction, depth, mask))
            aligned_metrics.update(compute_depth_metrics(prediction, depth, mask, median_align=True))

    print("NYU/SUNRGBD real-depth test")
    print(f"samples: {len(dataset)}")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"max_depth: {max_depth}")
    _print_metrics("raw metric", raw_metrics.mean())
    _print_metrics("median aligned", aligned_metrics.mean())


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
