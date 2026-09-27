from __future__ import annotations

import torch


def select_device(selection: str = "auto", allow_cpu_fallback: bool = True) -> torch.device:
    selection = selection.lower()
    if selection == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    if selection == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
    elif selection == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    elif selection == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device selection: {selection}")
    if allow_cpu_fallback:
        return torch.device("cpu")
    raise RuntimeError(f"Requested device {selection!r} is unavailable")

