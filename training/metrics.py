import torch


@torch.no_grad()
def compute_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    median_align: bool = False,
    eps: float = 1e-6,
) -> dict[str, float]:
    if median_align:
        prediction = median_align_depth(prediction, target, mask, eps)

    prediction = prediction[mask].clamp_min(eps)
    target = target[mask].clamp_min(eps)

    if prediction.numel() == 0:
        return {
            "abs_rel": 0.0,
            "rmse": 0.0,
            "rmse_log": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0,
        }

    ratio = torch.maximum(prediction / target, target / prediction)
    error = prediction - target
    log_error = torch.log(prediction) - torch.log(target)

    return {
        "abs_rel": torch.mean(torch.abs(error) / target).item(),
        "rmse": torch.sqrt(torch.mean(error * error)).item(),
        "rmse_log": torch.sqrt(torch.mean(log_error * log_error)).item(),
        "delta1": torch.mean((ratio < 1.25).float()).item(),
        "delta2": torch.mean((ratio < 1.25**2).float()).item(),
        "delta3": torch.mean((ratio < 1.25**3).float()).item(),
    }


@torch.no_grad()
def median_align_depth(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    aligned = prediction.clone()

    for batch_index in range(prediction.shape[0]):
        valid = mask[batch_index]
        if valid.sum() == 0:
            continue

        prediction_median = prediction[batch_index][valid].median().clamp_min(eps)
        target_median = target[batch_index][valid].median().clamp_min(eps)
        aligned[batch_index] = prediction[batch_index] * (target_median / prediction_median)

    return aligned


class MetricAverager:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float]) -> None:
        self.count += 1
        for name, value in metrics.items():
            self.totals[name] = self.totals.get(name, 0.0) + value

    def mean(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {name: value / self.count for name, value in self.totals.items()}
