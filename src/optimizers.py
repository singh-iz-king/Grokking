from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch
from torch import nn


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    name = str(config["name"]).lower()
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(float(config["beta1"]), float(config["beta2"])),
            eps=float(config["epsilon"]),
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {config['name']}")


def optimizer_momentum_l2_norm(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
) -> float:
    state_key = (
        "exp_avg"
        if isinstance(optimizer, torch.optim.AdamW)
        else "momentum_buffer"
    )
    norms = [
        float(torch.linalg.vector_norm(optimizer.state[parameter][state_key]).item())
        for parameter in parameters
        if state_key in optimizer.state[parameter]
    ]
    return math.hypot(*norms)
