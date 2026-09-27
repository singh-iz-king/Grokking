from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Nanda reference's float64 log-softmax loss, including MPS fallback."""
    if logits.device.type == "mps":
        logits = logits.to(device="cpu").to(dtype=torch.float64)
        labels = labels.to(device="cpu")
    else:
        logits = logits.to(dtype=torch.float64)
        labels = labels.to(device=logits.device)
    return F.cross_entropy(logits, labels)


def split_metrics(logits: Tensor, labels: Tensor, indices: Tensor) -> dict[str, float]:
    selected_logits = logits[indices, : labels.max().item() + 1]
    selected_labels = labels[indices]
    loss = cross_entropy(selected_logits, selected_labels)
    accuracy = (selected_logits.argmax(dim=-1) == selected_labels).float().mean()
    return {"loss": float(loss.item()), "accuracy": float(accuracy.item())}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    all_inputs: Tensor,
    all_labels: Tensor,
    train_indices: Tensor,
    test_indices: Tensor,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    logits = model(all_inputs.to(device))  # type: ignore[operator]
    logits = logits[:, -1, : int(all_labels.max().item()) + 1].detach().cpu()
    labels = all_labels.cpu()
    result: dict[str, float] = {}
    for split, indices in (("train", train_indices), ("test", test_indices)):
        values = split_metrics(logits, labels, indices.cpu())
        result[f"{split}_loss"] = values["loss"]
        result[f"{split}_accuracy"] = values["accuracy"]
    if was_training:
        model.train()
    return result
