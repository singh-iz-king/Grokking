from __future__ import annotations

import csv
import json

import torch

from scripts.run_first_moment_experiment import (
    MAX_EXPERIMENT_STEPS,
    _effective_step_limit,
    _mix,
    run_first_moment_experiment,
)


def _config(tmp_path, *, epochs=3):
    return {
        "experiment": {
            "name": "first_moment_test",
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


def test_gradient_mixture_and_common_second_moment_input():
    fresh = [torch.tensor([2.0, -1.0])]
    stale = [torch.tensor([-1.0, 3.0])]
    rho = 4 / 6

    mixed, squared = _mix(fresh, stale, rho)

    assert torch.allclose(mixed[0], torch.tensor([0.0, 5 / 3]))
    assert torch.allclose(squared[0], mixed[0].square())


def test_experiment_step_limit_is_capped_at_twenty_thousand():
    assert _effective_step_limit(40_000) == MAX_EXPERIMENT_STEPS == 20_000
    assert _effective_step_limit(12_000) == 12_000


def test_three_first_moment_arms_share_branch_and_write_metrics_and_plots(tmp_path):
    output = run_first_moment_experiment(
        _config(tmp_path),
        seed=0,
        run_dir=tmp_path / "first_moment",
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
    experiment_config = __import__("yaml").safe_load(
        (output / "config.yaml").read_text()
    )["first_moment_experiment"]
    assert experiment_config["seed"] == 0
    assert experiment_config["rho"] == 2 / 3
    assert "same" in experiment_config["second_moment_input"]
    status = json.loads((output / "first_moment_experiment_status.json").read_text())
    assert status["status"] == "completed"
    for arm in (
        "fresh_first_moment",
        "mixed_first_moment",
        "delayed_only_first_moment",
    ):
        with (output / arm / "metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["step"]) for row in rows] == [1, 2, 3]
        assert rows[0]["first_moment_exp_avg_l2_norm"] != ""
        assert rows[0]["second_moment_sqrt_l2_norm"] != ""
        assert rows[0]["restricted_test_loss_paper_keys"] != ""
        assert (output / arm / "final.pt").is_file()
    assert (output / "figures" / "test_accuracy.pdf").is_file()
    assert (output / "figures" / "first_moment_exp_avg_l2_norm.png").is_file()
    outcomes = list(csv.DictReader((output / "arm_outcomes.csv").open()))
    assert len(outcomes) == 3
