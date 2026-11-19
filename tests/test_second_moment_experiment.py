from __future__ import annotations

import csv
import json

import torch

from scripts.run_second_moment_experiment import (
    ARMS,
    _second_moment_inputs,
    run_second_moment_experiment,
)


def _config(tmp_path, *, epochs=3):
    return {
        "experiment": {
            "name": "second_moment_test",
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


def test_second_moment_inputs_are_four_distinct_formulas():
    fresh = [torch.tensor([2.0, -1.0])]
    stale = [torch.tensor([-1.0, 3.0])]
    rho = 2 / 3

    inputs = _second_moment_inputs(fresh, stale, rho)

    assert set(inputs) == set(ARMS)
    assert torch.allclose(inputs["fresh_square"][0], torch.tensor([4.0, 1.0]))
    assert torch.allclose(inputs["stale_square"][0], torch.tensor([1.0, 9.0]))
    assert torch.allclose(inputs["mixed_square"][0], torch.tensor([0.0, 25 / 9]))
    assert torch.allclose(
        inputs["average_squares"][0], torch.tensor([2.0, 19 / 3])
    )


def test_four_arms_share_branch_first_moment_and_generate_figures(tmp_path):
    output = run_second_moment_experiment(
        _config(tmp_path),
        seed=0,
        run_dir=tmp_path / "second_moment",
        nodes=3,
        delayed_nodes=2,
        tau=1,
        selection_train_accuracy=0.0,
        selection_test_max=1.0,
        selection_streak=1,
        diagnostic_interval=1,
    )

    selection = json.loads((output / "t0_selection.json").read_text())
    assert selection["step"] == 1
    status = json.loads(
        (output / "second_moment_experiment_status.json").read_text()
    )
    assert status["status"] == "completed"
    first_moment_norms = []
    for arm in ARMS:
        with (output / arm / "metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["step"]) for row in rows] == [1, 2, 3]
        assert rows[0]["first_moment_input"] == "mixed_gradient"
        assert rows[0]["second_moment_input"] == arm
        assert rows[0]["second_moment_sqrt_l2_norm"] != ""
        first_moment_norms.append(float(rows[0]["first_moment_exp_avg_l2_norm"]))
        assert (output / arm / "final.pt").is_file()
    assert max(first_moment_norms) - min(first_moment_norms) < 1e-7
    assert len(list(csv.DictReader((output / "arm_outcomes.csv").open()))) == 4
    assert (output / "figures" / "test_accuracy.pdf").is_file()
    assert (
        output / "figures" / "second_moment_sqrt_l2_norm.png"
    ).is_file()
