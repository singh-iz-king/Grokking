from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.dataset import ModularAdditionData
from src.evaluation import cross_entropy
from src.fourier_analysis import (
    discover_key_frequencies,
    exclude_frequency,
    gini_coefficient,
    key_logit_coefficients,
    matrix_fourier_norms,
    real_fourier_basis,
    restrict_logits,
)
from src.model import NandaOneLayerTransformer, neuron_to_logit_map


def _split_loss_accuracy(logits: Tensor, labels: Tensor, indices: Tensor) -> tuple[float, float]:
    selected = logits[indices]
    selected_labels = labels[indices]
    return (
        float(cross_entropy(selected, selected_labels).item()),
        float((selected.argmax(dim=-1) == selected_labels).float().mean().item()),
    )


@torch.no_grad()
def calculate_progress_measures(
    model: NandaOneLayerTransformer,
    data: ModularAdditionData,
    fixed_frequencies: list[int],
    explained_threshold: float,
    include_ablation_metrics: bool = True,
    frequency_mode: str = "fixed",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute logits ablations, Fourier spectra, Ginis, and per-kernel summaries.

    Fourier and ablation calculations are deliberately run on CPU, including
    when the model itself is on MPS. Model inference remains on its training
    device and the resulting tensors are copied once for analysis.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    logits, activations = model(
        data.all_inputs.to(device),
        return_activations=True,
    )
    logits = logits[:, -1, : data.all_labels.max().item() + 1]
    logits = logits.detach().cpu().to(torch.float64)
    labels = data.all_labels.cpu()
    post_acts = activations["mlp_post"][:, -1].detach().cpu().to(torch.float64)
    p = model.modulus

    detection = discover_key_frequencies(post_acts, p, explained_threshold)
    fixed = sorted(set(int(k) for k in fixed_frequencies))
    detected = detection.key_frequencies
    active = fixed if frequency_mode == "fixed" else detected
    basis, basis_names = real_fourier_basis(p, dtype=torch.float64)

    def restricted_summary(frequencies: list[int]) -> dict[str, float]:
        restricted = restrict_logits(logits, p, frequencies, preserve_mean=True)
        train_loss, train_accuracy = _split_loss_accuracy(
            restricted, labels, data.train_indices
        )
        test_loss, test_accuracy = _split_loss_accuracy(
            restricted, labels, data.test_indices
        )
        return {
            "all_loss": float(cross_entropy(restricted, labels).item()),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }

    restricted_fixed = restricted_summary(fixed)
    restricted_discovered = restricted_summary(detected)
    restricted_active = restricted_fixed if frequency_mode == "fixed" else restricted_discovered

    per_frequency: dict[str, dict[str, float]] = {}
    if include_ablation_metrics:
        all_excluded = logits.clone()
        for frequency in fixed:
            ablated = exclude_frequency(logits, p, frequency)
            train_loss, train_accuracy = _split_loss_accuracy(
                ablated, labels, data.train_indices
            )
            test_loss, test_accuracy = _split_loss_accuracy(
                ablated, labels, data.test_indices
            )
            per_frequency[str(frequency)] = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
            all_excluded -= logits - ablated
        excluded_train_loss, excluded_train_accuracy = _split_loss_accuracy(
            all_excluded, labels, data.train_indices
        )
        excluded_test_loss, excluded_test_accuracy = _split_loss_accuracy(
            all_excluded, labels, data.test_indices
        )

    embedding = model.W_E[:, :p].detach().cpu().T.to(torch.float64)
    logit_map = neuron_to_logit_map(model).detach().cpu().to(torch.float64)
    embedding_norms, embedding_squared_norms = matrix_fourier_norms(embedding, basis)
    logit_norms, logit_squared_norms = matrix_fourier_norms(logit_map, basis)
    total_squared_weight = sum(
        float(parameter.detach().cpu().double().square().sum().item())
        for parameter in model.parameters()
    )
    metrics: dict[str, Any] = {
        "restricted_loss": restricted_active["all_loss"],
        "restricted_train_loss": restricted_active["train_loss"],
        "restricted_test_loss": restricted_active["test_loss"],
        "restricted_train_accuracy": restricted_active["train_accuracy"],
        "restricted_test_accuracy": restricted_active["test_accuracy"],
        "restricted_loss_fixed": restricted_fixed["all_loss"],
        "restricted_train_loss_fixed": restricted_fixed["train_loss"],
        "restricted_test_loss_fixed": restricted_fixed["test_loss"],
        "restricted_loss_discovered": restricted_discovered["all_loss"],
        "restricted_train_loss_discovered": restricted_discovered["train_loss"],
        "restricted_test_loss_discovered": restricted_discovered["test_loss"],
        "frequency_mode": frequency_mode,
        "fixed_key_frequencies": fixed,
        "detected_key_frequencies": detected,
        "neuron_frequency_assignments": detection.neuron_frequencies,
        "neuron_frequency_explained_fractions": detection.explained_fractions,
        "embedding_fourier_norms": embedding_norms.tolist(),
        "embedding_fourier_squared_norms": embedding_squared_norms.tolist(),
        "logit_map_fourier_norms": logit_norms.tolist(),
        "logit_map_fourier_squared_norms": logit_squared_norms.tolist(),
        "fourier_component_names": basis_names,
        "embedding_fourier_gini": gini_coefficient(embedding_norms),
        "logit_map_fourier_gini": gini_coefficient(logit_norms),
        "weight_l2_norm": total_squared_weight**0.5,
        "weight_squared_norm": total_squared_weight,
        "key_logit_coefficients": key_logit_coefficients(logits, p, fixed),
        "discovered_key_logit_coefficients": key_logit_coefficients(logits, p, detected),
    }
    spectrum_rows: list[dict[str, Any]] = []
    for matrix_name, norms, squared_norms in (
        ("embedding", embedding_norms, embedding_squared_norms),
        ("logit_map", logit_norms, logit_squared_norms),
    ):
        for index, (name, norm, squared) in enumerate(
            zip(basis_names, norms.tolist(), squared_norms.tolist())
        ):
            spectrum_rows.append({
                "matrix": matrix_name,
                "component_index": index,
                "component": name,
                "norm": norm,
                "squared_norm": squared,
            })
    if include_ablation_metrics:
        metrics.update({
            "excluded_loss": excluded_train_loss,
            "excluded_train_loss": excluded_train_loss,
            "excluded_test_loss": excluded_test_loss,
            "excluded_train_accuracy": excluded_train_accuracy,
            "excluded_test_accuracy": excluded_test_accuracy,
            "excluded_loss_by_frequency": per_frequency,
        })
    if was_training:
        model.train()
    return metrics, spectrum_rows
