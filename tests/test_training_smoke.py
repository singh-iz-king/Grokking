import json

from src.trainer import train


def test_short_training_smoke(tmp_path, capsys) -> None:
    config = {
        "experiment": {
            "name": "smoke", "seed": 3, "modulus": 7,
            "train_fraction": 0.4, "epochs": 2,
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
            "mechanistic_metrics_interval": 1, "checkpoint_interval": 1,
            "ablation_interval": 1,
        },
        "mechanistic": {
            "key_frequencies": {
                "mode": "fixed", "paper_mainline": [2],
                "neuron_explained_fraction": 0.85,
            },
        },
        "device": {"selection": "cpu", "allow_cpu_fallback": True},
        "output": {
            "root": str(tmp_path / "results"),
            "checkpoint_root": str(tmp_path / "checkpoints"),
            "save_plots": False,
        },
    }
    run_dir = train(config, tmp_path / "run")
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3
    assert records[-1]["epoch"] == 2
    assert "restricted_loss" in records[0]
    output = capsys.readouterr().out
    assert "epoch      0/2" in output
    assert "train acc" in output
    assert "val acc" in output
    assert "elapsed" in output
    assert "Starting run: smoke" in output
    assert "Device: cpu" in output
    assert "modular addition mod 7" in output
    assert "Optimizer: AdamW" in output
    assert "Run output:" in output
    assert (run_dir / "final.pt").is_file()
    assert (tmp_path / "checkpoints" / "run" / "final.pt").is_file()
