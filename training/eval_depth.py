import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from training.datasets import DepthDataset
from training.metrics import MetricAverager, compute_depth_metrics
from training.models import ResNet18UNet


DATASET_ROOT = Path("outputs/datasets/depth/val")
CHECKPOINT_PATH = Path("outputs/checkpoints/depth/best_resnet18_unet.pt")
IMAGE_SIZE = (240, 320)
MAX_DEPTH = 8.0
BATCH_SIZE = 4
NUM_WORKERS = 0


def main() -> None:
    device = get_device()
    dataset = DepthDataset(DATASET_ROOT, image_size=IMAGE_SIZE, max_depth=MAX_DEPTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ResNet18UNet(max_depth=MAX_DEPTH).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = MetricAverager()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device)
            prediction = model(image)
            metrics.update(compute_depth_metrics(prediction, depth, mask))

    for name, value in metrics.mean().items():
        print(f"{name}: {value:.4f}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
