import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from training.datasets import DepthDataset
from training.losses import masked_l1_loss
from training.metrics import MetricAverager, compute_depth_metrics
from training.models import ResNet18UNet


DATASET_ROOT = Path("outputs/datasets/depth")
TRAIN_ROOT = DATASET_ROOT / "train"
VAL_ROOT = DATASET_ROOT / "val"
CHECKPOINT_DIR = Path("outputs/checkpoints/depth")

IMAGE_SIZE = (240, 320)
MAX_DEPTH = 8.0
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
PRETRAINED_ENCODER = False


def main() -> None:
    device = get_device()
    train_loader, val_loader = make_dataloaders()

    model = ResNet18UNet(
        max_depth=MAX_DEPTH,
        pretrained_encoder=PRETRAINED_ENCODER,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_abs_rel = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, device)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"{format_metrics(val_metrics)}"
        )

        save_checkpoint(CHECKPOINT_DIR / "last_resnet18_unet.pt", model, optimizer, epoch, val_metrics)
        if val_metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = val_metrics["abs_rel"]
            save_checkpoint(CHECKPOINT_DIR / "best_resnet18_unet.pt", model, optimizer, epoch, val_metrics)


def make_dataloaders() -> tuple[DataLoader, DataLoader]:
    if TRAIN_ROOT.exists():
        train_dataset = DepthDataset(TRAIN_ROOT, image_size=IMAGE_SIZE, max_depth=MAX_DEPTH)
        if VAL_ROOT.exists():
            val_dataset = DepthDataset(VAL_ROOT, image_size=IMAGE_SIZE, max_depth=MAX_DEPTH)
        else:
            train_dataset, val_dataset = split_dataset(train_dataset)
    elif DATASET_ROOT.exists():
        train_dataset, val_dataset = split_dataset(
            DepthDataset(DATASET_ROOT, image_size=IMAGE_SIZE, max_depth=MAX_DEPTH)
        )
    else:
        raise SystemExit(
            f"Dataset not found at {DATASET_ROOT}. Expected rgb/depth folders, "
            "for example outputs/datasets/depth/train/rgb/*.png and depth/*.npy."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def split_dataset(dataset: Dataset) -> tuple[Dataset, Dataset]:
    if len(dataset) < 2:
        return dataset, dataset

    val_count = max(1, int(0.1 * len(dataset)))
    train_count = len(dataset) - val_count
    return random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(0),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        mask = batch["mask"].to(device)

        prediction = model(image)
        loss = masked_l1_loss(prediction, depth, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics = MetricAverager()

    for batch in loader:
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        mask = batch["mask"].to(device)

        prediction = model(image)
        loss = masked_l1_loss(prediction, depth, mask)

        total_loss += loss.item()
        metrics.update(compute_depth_metrics(prediction, depth, mask))

    return total_loss / max(len(loader), 1), metrics.mean()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "max_depth": MAX_DEPTH,
            "image_size": IMAGE_SIZE,
        },
        path,
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{name}={value:.4f}" for name, value in metrics.items())


if __name__ == "__main__":
    main()
