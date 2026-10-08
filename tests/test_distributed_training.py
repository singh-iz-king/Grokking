import json

import torch

from src.distributed_trainer import train_distributed
from src.plotting import plot_node_metrics, plot_results
from scripts.train_distributed import (
    run_fixed_delay_composition_sweep,
    run_learning_rate_sweep,
)
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
    assert all(row["has_delay"] is False for row in node_rows)
    for start in range(5, len(node_rows), 5):
        assert len({
            row["gradient_l2_norm"]
            for row in node_rows[start:start + 5]
        }) == 1
    assert all(
        isinstance(row["gradient_l2_norm"], (int, float))
        for row in node_rows
        if row["step"] > 0
    )
    assert all("train_accuracy" in row for row in node_rows)
    assert rows[0]["aggregated_momentum_l2_norm"] == 0.0
    assert all(
        row["aggregated_momentum_l2_norm"] > 0.0 for row in rows[1:]
    )
    plot_results(distributed_dir)
    figures = plot_node_metrics(distributed_dir)
    assert (distributed_dir / "figures" / "aggregated_momentum_l2_norm.png").is_file()
    assert (distributed_dir / "figures" / "gradient_norms_by_node.png").is_file()
    assert (
        distributed_dir / "figures" / "gradient_norms_by_delay_status.png"
    ).is_file()
    assert (figures / "node_0" / "gradient_l2_norm.png").is_file()


def test_distributed_node_count_configures_metrics_and_checkpoint(tmp_path, capsys) -> None:
    config = _config(tmp_path, epochs=2, mechanistic_interval=2)
    run_dir = train_distributed(
        config, tau=0, nodes=3, run_dir=tmp_path / "three_nodes"
    )
    capsys.readouterr()

    rows = _read_jsonl(run_dir / "metrics.jsonl")
    assert all(len(row["staleness"]) == 3 for row in rows)
    node_rows = _read_jsonl(run_dir / "node_metrics.jsonl")
    assert len(node_rows) == 3 * 3
    assert {row["node"] for row in node_rows} == {0, 1, 2}
    saved_config = __import__("yaml").safe_load(
        (run_dir / "config.yaml").read_text()
    )
    assert saved_config["distributed"]["nodes"] == 3
    checkpoint = torch.load(run_dir / "final.pt", weights_only=False)
    assert checkpoint["distributed_state"]["nodes"] == 3


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
    figures_dir = plot_node_metrics(run_dir)
    for node in range(5):
        node_figures = figures_dir / f"node_{node}"
        assert (node_figures / "train_test_loss.png").is_file()
        assert (node_figures / "restricted_loss_keys.png").is_file()
        assert (node_figures / "excluded_loss_keys.png").is_file()
        assert (node_figures / "restricted_accuracy_keys.png").is_file()
        assert (node_figures / "excluded_accuracy_keys.png").is_file()
        assert (node_figures / "weight_l2_norm.png").is_file()
        assert (node_figures / "fourier_gini.png").is_file()
        assert (node_figures / "embedding_fourier_norms.png").is_file()
        assert (node_figures / "logit_map_fourier_norms.png").is_file()
        assert (node_figures / "key_logit_coefficients.png").is_file()
        assert (node_figures / "excluded_by_frequency_train_loss.png").is_file()
        assert (node_figures / "staleness.png").is_file()
    checkpoint = torch.load(run_dir / "final.pt", weights_only=False)
    state = checkpoint["distributed_state"]
    assert state["tau"] == 2
    assert state["nodes"] == 5
    assert len(state["history"]) == 3


def test_mixed_fixed_delay_keeps_first_n_nodes_fresh(tmp_path, capsys) -> None:
    config = _config(tmp_path, epochs=5, mechanistic_interval=5)
    run_dir = train_distributed(
        config,
        tau=2,
        nodes=4,
        staleness_mode="mixed_fixed",
        zero_delay_nodes=2,
        run_dir=tmp_path / "mixed_delay",
    )
    output = capsys.readouterr().out
    assert "zero-delay nodes [0, 1] use lag 0" in output
    assert "delayed nodes [2, 3] use fixed lag 2" in output

    rows = _read_jsonl(run_dir / "metrics.jsonl")
    assert rows[1]["staleness"] == [0] * 4
    assert rows[2]["staleness"] == [0] * 4
    assert rows[3]["staleness"] == [0, 0, 2, 2]
    assert rows[3]["node_source_versions"] == [2, 2, 0, 0]
    assert rows[4]["staleness"] == [0, 0, 2, 2]

    node_rows = _read_jsonl(run_dir / "node_metrics.jsonl")
    final_step = [row for row in node_rows if row["step"] == 5]
    assert [row["staleness"] for row in final_step] == [0, 0, 2, 2]
    assert [row["has_delay"] for row in final_step] == [False, False, True, True]
    assert all(isinstance(row["gradient_l2_norm"], float) for row in final_step)
    assert [row["source_model_version"] for row in final_step] == [4, 4, 2, 2]
    checkpoint = torch.load(run_dir / "final.pt", weights_only=False)
    state = checkpoint["distributed_state"]
    assert state["tau"] == 2
    assert state["staleness_mode"] == "mixed_fixed"
    assert state["zero_delay_nodes"] == 2
    assert state["zero_delay_node_ids"] == [0, 1]
    assert state["delayed_node_ids"] == [2, 3]


def test_mixed_fixed_delay_all_zero_delay_and_invalid_count(tmp_path, capsys) -> None:
    config = _config(tmp_path, epochs=4, mechanistic_interval=5)
    run_dir = train_distributed(
        config,
        tau=1,
        staleness_mode="mixed_fixed",
        nodes=3,
        zero_delay_nodes=3,
        run_dir=tmp_path / "all_fresh",
    )
    capsys.readouterr()
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    assert all(row["staleness"] == [0] * 3 for row in rows)

    try:
        train_distributed(
            config,
            tau=1,
            nodes=5,
            staleness_mode="mixed_fixed",
            zero_delay_nodes=6,
            run_dir=tmp_path / "invalid_count",
        )
    except ValueError as error:
        assert "zero_delay_nodes must be between 0 and 5" in str(error)
    else:
        raise AssertionError("out-of-range zero_delay_nodes should fail")


def test_learning_rate_sweep_runs_three_tagged_experiments(tmp_path, capsys) -> None:
    config = _config(tmp_path, epochs=1)
    config["logging"]["mechanistic_metrics_interval"] = 2
    config["logging"]["ablation_interval"] = 2
    config["output"]["save_plots"] = False
    results = run_learning_rate_sweep(
        config,
        tau=0,
        run_dir=tmp_path / "sweep",
    )
    capsys.readouterr()

    assert [factor for factor, _, _ in results] == [1.0, 0.1, 0.01]
    assert [learning_rate for _, learning_rate, _ in results] == [
        0.001,
        0.0001,
        0.00001,
    ]
    assert len({path for _, _, path in results}) == 3
    for factor, learning_rate, run_dir in results:
        tag = run_dir.name
        assert f"lr_factor_{factor:g}".replace(".", "p") in tag
        assert run_dir.joinpath("final.pt").is_file()
        saved_config = __import__("yaml").safe_load(
            run_dir.joinpath("config.yaml").read_text()
        )
        assert saved_config["optimizer"]["learning_rate"] == learning_rate
        checkpoint_dir = tmp_path / "checkpoints" / tag
        assert (checkpoint_dir / "final.pt").is_file()


def test_learning_rate_sweep_supports_mixed_fixed_delay(tmp_path, capsys) -> None:
    config = _config(tmp_path, epochs=1, mechanistic_interval=2)
    config["logging"]["ablation_interval"] = 2
    config["output"]["save_plots"] = False
    results = run_learning_rate_sweep(
        config,
        tau=2,
        staleness_mode="mixed_fixed",
        zero_delay_nodes=1,
        nodes=3,
        run_dir=tmp_path / "mixed_sweep",
    )
    capsys.readouterr()

    for _, _, run_dir in results:
        assert "mixed_fixed_tau_2_zero_delay_nodes_1_nodes_3" in run_dir.name
        assert "lr_factor_" in run_dir.name
        saved_config = __import__("yaml").safe_load(
            run_dir.joinpath("config.yaml").read_text()
        )
        assert saved_config["distributed"]["staleness_mode"] == "mixed_fixed"
        assert saved_config["distributed"]["zero_delay_nodes"] == 1
        checkpoint = torch.load(run_dir / "final.pt", weights_only=False)
        assert checkpoint["distributed_state"]["staleness_mode"] == "mixed_fixed"


def test_fixed_delay_composition_sweep_runs_all_node_assignments(
    tmp_path, capsys
) -> None:
    config = _config(tmp_path, epochs=1, mechanistic_interval=2)
    config["logging"]["ablation_interval"] = 2
    config["output"]["save_plots"] = False
    results = run_fixed_delay_composition_sweep(
        config,
        tau=2,
        nodes=3,
        run_dir=tmp_path / "composition_sweep",
    )
    capsys.readouterr()

    assert [zero_delay_nodes for zero_delay_nodes, _ in results] == [0, 1, 2, 3]
    assert len({run_dir for _, run_dir in results}) == 4
    for zero_delay_nodes, run_dir in results:
        assert (
            f"mixed_fixed_tau_2_zero_delay_nodes_{zero_delay_nodes}_nodes_3"
            in run_dir.name
        )
        saved_config = __import__("yaml").safe_load(
            (run_dir / "config.yaml").read_text()
        )
        assert saved_config["distributed"]["nodes"] == 3
        assert saved_config["distributed"]["tau"] == 2
        assert saved_config["distributed"]["zero_delay_nodes"] == zero_delay_nodes
        checkpoint = torch.load(run_dir / "final.pt", weights_only=False)
        assert (
            checkpoint["distributed_state"]["zero_delay_nodes"]
            == zero_delay_nodes
        )
