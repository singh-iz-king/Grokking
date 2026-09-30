import json

import torch

from src.distributed_trainer import train_distributed
from src.trainer import train


def _config(tmp_path, *, epochs=2, mechanistic_interval=1):
    return {
        "experiment": {
            "name": "distributed_test", "seed": 3, "modulus": 7,
            "train_fraction": 0.4, "epochs": epochs,
        },
        "data": {
            "token_equals": 7, "input_length": 3,
            "split_method": "shuffled_all_pairs_python_random",
        },
        "model": {
            "n_layers": 1, "n_heads": 1, "d_model": 4, "d_head": 4,
            "d_mlp": 8, "activation": "relu", "normalization": None,
            "positional_embeddings": "learned", "causal_attention": True,
            "dropout": 0.0, "tie_embeddings": False,
            "initialization": "nanda_manual_normal",
        },
        "optimizer": {
            "name": "AdamW", "learning_rate": 0.001, "weight_decay": 1.0,
            "beta1": 0.9, "beta2": 0.98, "epsilon": 1e-8,
            "warmup_steps": 2, "weight_decay_all_parameters": True,
        },
        "training": {
            "batch_size": "full", "loss": "cross_entropy",
            "loss_precision": "float64",
            "early_stopping_test_loss_below": None,
        },
        "logging": {
            "train_metrics_interval": 1, "validation_interval": 1,
            "progress_interval": 1,
            "mechanistic_metrics_interval": mechanistic_interval,
            "checkpoint_interval": 1, "ablation_interval": mechanistic_interval,
        },
        "mechanistic": {
            "key_frequencies": {
                "mode": "fixed", "paper_mainline": [2],
                "neuron_explained_fraction": 0.85,
            },
            "frequency_ablations": True,
            "save_fourier_spectra": True,
        },
        "device": {"selection": "cpu", "allow_cpu_fallback": True},
        "output": {
            "root": str(tmp_path / "results"),
            "checkpoint_root": str(tmp_path / "checkpoints"),
            "save_plots": False,
        },
    }


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tau_zero_matches_canonical_adamw_updates(tmp_path, capsys) -> None:
    config = _config(tmp_path, epochs=2)
    canonical_dir = train(config, tmp_path / "canonical")
    distributed_dir = train_distributed(
        config, tau=0, run_dir=tmp_path / "tau_zero"
    )
    capsys.readouterr()
    canonical = torch.load(canonical_dir / "final.pt", weights_only=False)
    distributed = torch.load(distributed_dir / "final.pt", weights_only=False)
    for name, value in canonical["model_state_dict"].items():
        assert torch.allclose(
            value, distributed["model_state_dict"][name], atol=1e-7, rtol=1e-6
        ), name

    rows = _read_jsonl(distributed_dir / "metrics.jsonl")
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert all(row["staleness"] == [0] * 5 for row in rows)
    node_rows = _read_jsonl(distributed_dir / "node_metrics.jsonl")
    assert len(node_rows) == 5 * 3
    assert all(row["staleness"] == 0 for row in node_rows)
    assert all("train_accuracy" in row for row in node_rows)


def test_positive_tau_warms_up_then_samples_valid_staleness(tmp_path, capsys) -> None:
    config = _config(tmp_path, epochs=6, mechanistic_interval=6)
    run_dir = train_distributed(
        config, tau=2, run_dir=tmp_path / "tau_two"
    )
    output = capsys.readouterr().out
    assert "synchronous through global step 2" in output
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    assert rows[1]["staleness"] == [0] * 5
    assert rows[2]["staleness"] == [0] * 5
    assert all(1 <= lag <= 2 for lag in rows[3]["staleness"])
    assert all(1 <= lag <= 2 for lag in rows[-1]["staleness"])
    assert "restricted_loss" in rows[-1]

    node_rows = _read_jsonl(run_dir / "node_metrics.jsonl")
    by_step = [row for row in node_rows if row["step"] == 6]
    assert len(by_step) == 5
    assert all("progress_measures" in row for row in by_step)
    checkpoint = torch.load(run_dir / "final.pt", weights_only=False)
    state = checkpoint["distributed_state"]
    assert state["tau"] == 2
    assert state["nodes"] == 5
    assert len(state["history"]) == 3
