import torch

from src.dataset import make_modular_addition_data
from src.model import NandaOneLayerTransformer
from src.progress_measures import calculate_progress_measures


def test_progress_measures_are_deterministic_for_fixed_checkpoint() -> None:
    torch.manual_seed(17)
    model = NandaOneLayerTransformer(
        modulus=7, d_model=4, n_heads=1, d_head=4, d_mlp=8
    )
    data = make_modular_addition_data(7, 0.4, 9)
    first, spectra = calculate_progress_measures(
        model, data, [2], 0.85, include_ablation_metrics=True
    )
    second, _ = calculate_progress_measures(
        model, data, [2], 0.85, include_ablation_metrics=True
    )
    assert first["restricted_loss"] == second["restricted_loss"]
    assert first["excluded_loss"] == second["excluded_loss"]
    assert first["embedding_fourier_gini"] == second["embedding_fourier_gini"]
    assert first["fixed_key_frequencies"] == [2]
    assert first["excluded_loss_by_frequency"].keys() == {"2"}
    assert len(spectra) == 2 * 7

