from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModularAdditionData:
    all_inputs: Tensor
    all_labels: Tensor
    train_indices: Tensor
    test_indices: Tensor
    train_inputs: Tensor
    train_labels: Tensor
    test_inputs: Tensor
    test_labels: Tensor


def make_modular_addition_data(
    modulus: int,
    train_fraction: float,
    seed: int,
) -> ModularAdditionData:
    if modulus < 3:
        raise ValueError("modulus must be at least 3")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")

    pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    shuffled_pairs = pairs.copy()
    random.Random(seed).shuffle(shuffled_pairs)
    split_at = int(train_fraction * len(pairs))
    pair_to_index = {pair: index for index, pair in enumerate(pairs)}
    train_indices = torch.tensor(
        [pair_to_index[pair] for pair in shuffled_pairs[:split_at]],
        dtype=torch.long,
    )
    test_indices = torch.tensor(
        [pair_to_index[pair] for pair in shuffled_pairs[split_at:]],
        dtype=torch.long,
    )
    all_inputs = torch.tensor(
        [(a, b, modulus) for a, b in pairs],
        dtype=torch.long,
    )
    all_labels = torch.tensor(
        [(a + b) % modulus for a, b in pairs],
        dtype=torch.long,
    )
    return ModularAdditionData(
        all_inputs=all_inputs,
        all_labels=all_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        train_inputs=all_inputs[train_indices],
        train_labels=all_labels[train_indices],
        test_inputs=all_inputs[test_indices],
        test_labels=all_labels[test_indices],
    )
