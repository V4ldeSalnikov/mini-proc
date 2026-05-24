import torch


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid_prediction = prediction[mask]
    valid_target = target[mask]

    if valid_prediction.numel() == 0:
        return prediction.sum() * 0.0

    return torch.abs(valid_prediction - valid_target).mean()
