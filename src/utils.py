from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def environment_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
    }
    for package in ("PyYAML", "matplotlib"):
        try:
            info[package] = version(package)
        except PackageNotFoundError:
            info[package] = None
    if device.type == "cuda":
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        info["mps_built"] = torch.backends.mps.is_built()
        info["mps_available"] = torch.backends.mps.is_available()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        info["git_commit"] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
