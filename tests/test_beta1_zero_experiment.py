from __future__ import annotations

import csv
import json

import torch

from scripts.run_beta1_zero_experiment import (
    ARMS,
    run_beta1_zero_experiment,
)


def _config(tmp_path, *, epochs=3):
    return {
        "experiment": {
            "name": "beta1_zero_test",
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


def test_beta1_zero_experiment_branches_three_gradient_inputs(tmp_path):
    output = run_beta1_zero_experiment(
        _config(tmp_path),
        seed=0,
        run_dir=tmp_path / "beta1_zero",
        nodes=3,
        delayed_nodes=2,
        tau=1,
        selection_train_accuracy=0.0,
        selection_test_max=1.0,
        selection_streak=1,
        diagnostic_interval=1,
    )
    selected = json.loads((output / "t0_selection.json").read_text())
    assert selected["step"] == 1
    status = json.loads((output / "beta1_zero_experiment_status.json").read_text())
    assert status["status"] == "completed"
    for arm in ARMS:
        with (output / arm / "metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["step"]) for row in rows] == [1, 2, 3]
        assert [row["beta1"] for row in rows] == ["0.0"] * 3
        assert {row["first_moment_input"] for row in rows} == {arm}
        assert all(
            group["betas"][0] == 0.0
            for group in torch.load(
                output / arm / "final.pt",
                map_location="cpu",
                weights_only=False,
            )["optimizer_state_dict"]["param_groups"]
        )
        assert (output / arm / "final.pt").is_file()
    assert len(list(csv.DictReader((output / "arm_outcomes.csv").open()))) == 3
    assert (output / "figures" / "test_accuracy.pdf").is_file()
    assert (
        output / "figures" / "first_moment_exp_avg_l2_norm.png"
    ).is_file()
