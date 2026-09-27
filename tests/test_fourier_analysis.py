import torch

from src.fourier_analysis import (
    discover_key_frequencies,
    exclude_frequency,
    fourier_2d,
    gini_coefficient,
    inverse_fourier_2d,
    real_fourier_basis,
    restrict_logits,
    sum_frequency_vectors,
)


def test_real_fourier_basis_is_orthonormal_and_round_trips() -> None:
    p = 11
    basis, _ = real_fourier_basis(p)
    assert torch.allclose(basis @ basis.T, torch.eye(p, dtype=basis.dtype), atol=1e-12)
    values = torch.randn(p * p, 3, dtype=torch.float64)
    restored = inverse_fourier_2d(fourier_2d(values, basis), basis)
    assert torch.allclose(restored, values, atol=1e-12)


def test_restricted_and_excluded_logits_project_expected_frequency() -> None:
    p, k = 11, 2
    cos_vector, sin_vector = sum_frequency_vectors(p, k)
    coefficients = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    target_component = cos_vector[:, None] * coefficients[:, :1]
    target_component += sin_vector[:, None] * coefficients[:, 1:]
    other_component = torch.randn(p * p, 3, dtype=torch.float64)
    other_component = exclude_frequency(other_component, p, k)
    logits = target_component + other_component
    restricted = restrict_logits(logits, p, [k], preserve_mean=False)
    excluded = exclude_frequency(logits, p, k)
    assert torch.allclose(restricted, target_component, atol=1e-12)
    assert torch.allclose(excluded + restricted, logits, atol=1e-12)


def test_gini_and_synthetic_frequency_discovery() -> None:
    assert abs(gini_coefficient(torch.tensor([0.0, 0.0, 1.0])) - 2 / 3) < 1e-12
    assert gini_coefficient(torch.ones(5)) == 0.0
    p, k = 11, 3
    a = torch.arange(p, dtype=torch.float64)[:, None]
    b = torch.arange(p, dtype=torch.float64)[None, :]
    activation = torch.cos(2 * torch.pi * k * (a + b) / p).reshape(p * p, 1)
    discovered = discover_key_frequencies(activation, p, explained_threshold=0.85)
    assert discovered.key_frequencies == [k]
    assert discovered.neuron_frequencies == [k]
    assert discovered.explained_fractions[0] > 0.99
