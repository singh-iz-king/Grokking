from __future__ import annotations

import csv
import json

import torch

from scripts.run_primary_experiment import (
    _adamw_step_with_second_moment,
    run_primary_experiment,
)


def _config(tmp_path, *, epochs=3):
    return {
        "experiment": {
            "name": "primary_test",
            "seed": 11,
            "modulus": 7,
            "train_fraction": 0.4,
            "epochs": epochs,
        },
        "data": {
            "token_equals": 7,
            "input_length": 3,
            "split_method": "shuffled_all_pairs_python_random",
        },
        "model": {
            "n_layers": 1,
            "n_heads": 1,
            "d_model": 4,
            "d_head": 4,
            "d_mlp": 8,
            "activation": "relu",
            "normalization": None,
            "positional_embeddings": "learned",
            "causal_attention": True,
            "dropout": 0.0,
            "tie_embeddings": False,
            "initialization": "nanda_manual_normal",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 1.0,
            "beta1": 0.9,
            "beta2": 0.98,
            "epsilon": 1e-8,
            "warmup_steps": 2,
            "weight_decay_all_parameters": True,
        },
        "training": {
            "batch_size": "full",
            "loss": "cross_entropy",
            "loss_precision": "float64",
            "early_stopping_test_loss_below": None,
        },
        "logging": {
            "train_metrics_interval": 1,
            "validation_interval": 1,
            "progress_interval": 100,
            "mechanistic_metrics_interval": 10,
            "checkpoint_interval": 10,
            "ablation_interval": 10,
        },
        "mechanistic": {
            "key_frequencies": {
                "mode": "fixed",
                "paper_mainline": [1],
                "neuron_explained_fraction": 0.85,
            },
            "frequency_ablations": True,
            "save_fourier_spectra": True,
        },
        "device": {"selection": "cpu", "allow_cpu_fallback": True},
        "output": {
            "root": str(tmp_path / "results"),
            "checkpoint_root": str(tmp_path / "checkpoints"),
            "save_plots": True,
        },
    }


def test_custom_second_moment_step_matches_adamw_when_inputs_are_gradient_squares():
    parameters_a = [torch.nn.Parameter(torch.tensor([0.3, -0.5]))]
    parameters_b = [torch.nn.Parameter(parameters_a[0].detach().clone())]
    optimizer_a = torch.optim.AdamW(
        parameters_a, lr=0.01, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.2
    )
    optimizer_b = torch.optim.AdamW(
        parameters_b, lr=0.01, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.2
    )
    for gradient in (torch.tensor([0.4, -0.2]), torch.tensor([-0.1, 0.3])):
        parameters_a[0].grad = gradient.clone()
        optimizer_a.step()
        optimizer_a.zero_grad(set_to_none=True)
        parameters_b[0].grad = gradient.clone()
        _adamw_step_with_second_moment(
            optimizer_b, parameters_b, [gradient.square()]
        )
        optimizer_b.zero_grad(set_to_none=True)
    assert torch.allclose(parameters_a[0], parameters_b[0], atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        optimizer_a.state[parameters_a[0]]["exp_avg_sq"],
        optimizer_b.state[parameters_b[0]]["exp_avg_sq"],
        atol=1e-8,
    )


def test_primary_experiment_branches_three_arms_and_writes_diagnostics(tmp_path):
    output = run_primary_experiment(
        _config(tmp_path),
        seed=11,
        run_dir=tmp_path / "primary",
        nodes=2,
        delayed_nodes=1,
        tau=1,
        selection_train_accuracy=0.0,
        selection_test_max=1.0,
        selection_streak=1,
        diagnostic_interval=1,
        window_steps=1,
    )
    selection = json.loads((output / "t0_selection.json").read_text())
    assert selection["step"] == 1
    assert (output / "checkpoint_t0.pt").is_file()
    status = json.loads((output / "primary_experiment_status.json").read_text())
    assert status["status"] == "completed"
    assert len(list(csv.DictReader((output / "common_path_replay.csv").open()))) > 0
    assert len(list(csv.DictReader((output / "persistence_windows.csv").open()))) > 0
    for arm in (
        "A_fresh_adamw",
        "B_delayed_adamw",
        "C_no_second_moment_cancellation",
    ):
        with (output / arm / "metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["step"]) for row in rows] == [1, 2, 3]
        assert rows[0]["restricted_test_loss_paper_keys"] != ""
        assert rows[0]["excluded_train_loss_all_addition"] != ""
        assert rows[1]["adaptive_update_l2_norm"] != ""
        assert (output / arm / "final.pt").is_file()
    assert (output / "figures" / "test_accuracy.pdf").is_file()
    assert (output / "figures" / "restricted_test_loss_paper_keys.png").is_file()
    assert (output / "figures" / "persistence_windows.pdf").is_file()
    assert (output / "figures" / "common_path_replay.pdf").is_file()
