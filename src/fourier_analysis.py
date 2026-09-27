from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FrequencyDiscovery:
    key_frequencies: list[int]
    neuron_frequencies: list[int]
    explained_fractions: list[float]


def real_fourier_basis(modulus: int, *, device: torch.device | None = None,
                       dtype: torch.dtype = torch.float64) -> tuple[Tensor, list[str]]:
    """Orthonormal real DFT basis ordered as DC, cos(1), sin(1), ... ."""
    if modulus < 3:
        raise ValueError("modulus must be at least 3")
    x = torch.arange(modulus, device=device, dtype=dtype)
    rows = [torch.ones_like(x) / modulus**0.5]
    names = ["dc"]
    for frequency in range(1, modulus // 2 + 1):
        angle = 2 * torch.pi * frequency * x / modulus
        rows.extend((torch.cos(angle) * (2 / modulus) ** 0.5,
                     torch.sin(angle) * (2 / modulus) ** 0.5))
        names.extend((f"cos_{frequency}", f"sin_{frequency}"))
    return torch.stack(rows), names


def fourier_1d(values: Tensor, basis: Tensor, axis: int = 0) -> Tensor:
    """Project a length-p axis into the real orthonormal Fourier basis."""
    moved = values.movedim(axis, 0)
    if moved.shape[0] != basis.shape[1]:
        raise ValueError("Fourier basis and input axis have different lengths")
    transformed = torch.tensordot(basis.to(values), moved, dims=([1], [0]))
    return transformed.movedim(0, axis)


def fourier_2d(values: Tensor, basis: Tensor) -> Tensor:
    """Transform a [p*p, ...] array on the ordered (a,b) modular grid."""
    p = basis.shape[1]
    if values.shape[0] != p * p:
        raise ValueError(f"expected {p*p} rows for a {p} by {p} Fourier grid")
    grid = values.reshape(p, p, *values.shape[1:])
    basis = basis.to(values)
    return torch.einsum("fx,Fy,xy...->fF...", basis, basis, grid).reshape_as(values)


def inverse_fourier_2d(coefficients: Tensor, basis: Tensor) -> Tensor:
    """Reconstruct [p*p, ...] values from a real two-dimensional DFT."""
    p = basis.shape[1]
    if coefficients.shape[0] != p * p:
        raise ValueError(f"expected {p*p} Fourier rows")
    grid = coefficients.reshape(p, p, *coefficients.shape[1:])
    basis = basis.to(coefficients)
    return torch.einsum("fx,Fy,fF...->xy...", basis, basis, grid).reshape_as(coefficients)


def frequency_indices(frequency: int) -> tuple[int, int]:
    """Return (cosine, sine) row indices in the real Fourier basis."""
    if frequency < 1:
        raise ValueError("frequency must be positive")
    return 2 * frequency - 1, 2 * frequency


def sum_frequency_vectors(modulus: int, frequency: int,
                          *, device: torch.device | None = None,
                          dtype: torch.dtype = torch.float64) -> tuple[Tensor, Tensor]:
    """Unit vectors for cos(w(a+b)) and sin(w(a+b)) over all ordered pairs."""
    basis, _ = real_fourier_basis(modulus, device=device, dtype=dtype)
    cos_index, sin_index = frequency_indices(frequency)
    cos_basis, sin_basis = basis[cos_index], basis[sin_index]
    cos_vector = (
        torch.outer(cos_basis, cos_basis) - torch.outer(sin_basis, sin_basis)
    ).reshape(-1) / 2**0.5
    sin_vector = (
        torch.outer(sin_basis, cos_basis) + torch.outer(cos_basis, sin_basis)
    ).reshape(-1) / 2**0.5
    return cos_vector, sin_vector


def project_frequency_components(logits: Tensor, modulus: int, frequency: int) -> Tensor:
    """Project [p*p, classes] logits onto cos(w(a+b)) and sin(w(a+b))."""
    cos_vector, sin_vector = sum_frequency_vectors(
        modulus, frequency, device=logits.device, dtype=logits.dtype
    )
    return cos_vector[:, None] * (cos_vector @ logits)[None, :] + (
        sin_vector[:, None] * (sin_vector @ logits)[None, :]
    )


def restrict_logits(logits: Tensor, modulus: int, frequencies: list[int],
                    *, preserve_mean: bool = True) -> Tensor:
    """Keep only the constant and cos/sin(a+b) directions for given frequencies."""
    if logits.shape[0] != modulus * modulus:
        raise ValueError("logits must contain every ordered modular-addition pair")
    mean = logits.mean(dim=0, keepdim=True) if preserve_mean else torch.zeros_like(logits[:1])
    result = mean.expand_as(logits).clone()
    for frequency in frequencies:
        result += project_frequency_components(logits, modulus, frequency)
    return result


def exclude_frequency(logits: Tensor, modulus: int, frequency: int) -> Tensor:
    """Remove one frequency's two input-logit directions, retaining other logits."""
    return logits - project_frequency_components(logits, modulus, frequency)


def gini_coefficient(values: Tensor) -> float:
    """Gini coefficient of a nonnegative vector using the standard sorted formula."""
    x = torch.as_tensor(values, dtype=torch.float64).detach().cpu().flatten().abs()
    if x.numel() == 0 or x.sum().item() == 0:
        return 0.0
    x = torch.sort(x).values
    ranks = torch.arange(1, x.numel() + 1, dtype=x.dtype)
    result = (2 * torch.sum(ranks * x) / (x.numel() * x.sum())
              - (x.numel() + 1) / x.numel())
    return float(result.clamp(0.0, 1.0).item())


def matrix_fourier_norms(matrix: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """Return L2 and squared-L2 norm per output-vocabulary Fourier component."""
    if matrix.ndim != 2 or matrix.shape[0] != basis.shape[1]:
        raise ValueError("matrix must have shape [modulus, features]")
    coefficients = basis.to(matrix) @ matrix
    squared_norms = coefficients.square().sum(dim=1)
    return squared_norms.sqrt(), squared_norms


def discover_key_frequencies(
    neuron_activations: Tensor,
    modulus: int,
    explained_threshold: float = 0.85,
) -> FrequencyDiscovery:
    """Apply the notebook's max-energy neuron frequency and 0.85 cluster rule.

    Activations are centered over all p^2 input pairs. For each neuron and each
    k=1,...,floor(p/2)-1, the energy in the constant, linear-k and quadratic-k
    3x3 Fourier block is divided by total centered 2D Fourier energy. A neuron's
    best-scoring k is retained; key clusters require score >= the documented
    0.85 threshold.
    """
    if neuron_activations.ndim != 2 or neuron_activations.shape[0] != modulus**2:
        raise ValueError("neuron_activations must have shape [p*p, neurons]")
    if not 0 < explained_threshold <= 1:
        raise ValueError("explained_threshold must be in (0, 1]")
    acts = neuron_activations.to(dtype=torch.float64)
    acts = acts - acts.mean(dim=0, keepdim=True)
    basis, _ = real_fourier_basis(modulus, device=acts.device, dtype=acts.dtype)
    spectrum = fourier_2d(acts, basis).reshape(modulus, modulus, -1)
    total = spectrum.square().sum(dim=(0, 1))
    scores: list[Tensor] = []
    frequencies = list(range(1, modulus // 2))
    for frequency in frequencies:
        indices = [0, *frequency_indices(frequency)]
        block = spectrum[indices][:, indices, :]
        scores.append(block.square().sum(dim=(0, 1)) / total.clamp_min(1e-30))
    if not scores:
        raise ValueError("no frequencies are available for this modulus")
    score_matrix = torch.stack(scores)
    best_fraction, best_index = score_matrix.max(dim=0)
    nonzero = total > 1e-30
    best_frequency = torch.tensor(frequencies, device=acts.device)[best_index]
    best_frequency = torch.where(nonzero, best_frequency, -1)
    best_fraction = torch.where(nonzero, best_fraction, 0.0)
    key_frequencies = sorted({
        int(best_frequency[i].item())
        for i in range(best_frequency.numel())
        if nonzero[i] and best_fraction[i].item() >= explained_threshold
    })
    return FrequencyDiscovery(
        key_frequencies=key_frequencies,
        neuron_frequencies=[int(value) for value in best_frequency.cpu().tolist()],
        explained_fractions=[float(value) for value in best_fraction.cpu().tolist()],
    )


def key_logit_coefficients(logits: Tensor, modulus: int,
                           frequencies: list[int]) -> dict[str, float]:
    """Project logits onto cos(2*pi*k*(a+b-c)/p), one scalar per k."""
    logits = logits[:, :modulus].reshape(modulus, modulus, modulus).to(torch.float64)
    a = torch.arange(modulus, device=logits.device, dtype=torch.float64)[:, None, None]
    b = torch.arange(modulus, device=logits.device, dtype=torch.float64)[None, :, None]
    c = torch.arange(modulus, device=logits.device, dtype=torch.float64)[None, None, :]
    output: dict[str, float] = {}
    for frequency in frequencies:
        angle = 2 * torch.pi * frequency * (a + b - c) / modulus
        template = torch.cos(angle)
        output[str(frequency)] = float(
            ((logits * template).sum() / template.square().sum()).item()
        )
    return output
