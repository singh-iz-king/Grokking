import csv
import json

from scripts.summarize_experiments import (
    _stable_window_start,
    summarize_experiments,
)
from scripts.audit_diagnostics import audit_diagnostics


def _create_run(
    run_dir,
    *,
    seed=1,
    test_accuracies=(0.5, 0.99, 0.995, 0.85, 0.99, 0.995, 0.99),
    distributed=True,
    train_accuracy_after=0.99,
):
    run_dir.mkdir(parents=True)
    config = {
        "experiment": {
            "name": f"summary_test_seed{seed}",
            "seed": seed,
            "modulus": 3,
            "train_fraction": 0.5,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 1.0,
            "beta1": 0.9,
            "beta2": 0.98,
        },
        "logging": {"validation_interval": 1},
        "runtime": {"run_dir": f"/runs/{run_dir.name}"},
    }
    if distributed:
        config["distributed"] = {
            "enabled": True,
            "nodes": 4,
            "tau": 5,
            "staleness_mode": "mixed_fixed",
            "zero_delay_nodes": 1,
        }
    (run_dir / "config.yaml").write_text(
        __import__("yaml").safe_dump(config), encoding="utf-8"
    )
    rows = []
    for step, test_accuracy in enumerate(test_accuracies):
        rows.append({
            "step": step,
            "train_accuracy": train_accuracy_after if step >= 1 else 0.8,
            "test_accuracy": test_accuracy,
        })
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_experiment_summary_fields_and_split_comparison(tmp_path):
    runs_root = tmp_path / "runs"
    first_run = runs_root / "run_a"
    second_run = runs_root / "run_b"
    _create_run(first_run)
    _create_run(second_run)

    output = summarize_experiments(
        [runs_root],
        tmp_path / "experiment_summary.csv",
        stable_window_steps=2,
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    first = rows[0]
    assert first["run_id"] == "run_a"
    assert first["experiment_name"] == "summary_test_seed1"
    assert first["seed"] == "1"
    assert first["m"] == "3"
    assert first["d"] == "5"
    assert first["staleness_mode"] == "mixed_fixed"
    assert first["configured_tau"] == "5"
    assert first["nodes"] == "4"
    assert first["zero_delay_nodes"] == "1"
    assert first["optimizer"] == "AdamW"
    assert first["lr"] == "0.001"
    assert first["wd"] == "1.0"
    assert first["beta1"] == "0.9"
    assert first["beta2"] == "0.98"
    assert first["train99"] == "1"
    assert first["test99"] == "1"
    assert first["stable99"] == "4"
    assert first["stable_window_steps"] == "2"
    assert first["stable_window_sampling_cadence"] == "1 steps between measured test evaluations"
    assert float(first["mean_test_post"]) > 0.9
    assert first["min_test_post"] == "0.85"
    assert first["collapse_count"] == "1"
    assert first["final_train"] == "0.99"
    assert first["final_test"] == "0.99"
    assert first["time_unit"] == "global_optimizer_step"
    assert first["evaluation_interval_steps"] == "1"
    assert first["train_examples"] == "4"
    assert first["test_examples"] == "5"
    assert first["total_examples"] == "9"
    assert first["identical_train_test_splits"] == "True"


def test_missing_thresholds_are_reported_as_na(tmp_path):
    run_dir = tmp_path / "never_grokked"
    _create_run(
        run_dir,
        test_accuracies=(0.2, 0.3, 0.5),
        distributed=False,
        train_accuracy_after=0.8,
    )
    output = summarize_experiments(
        [run_dir],
        tmp_path / "summary.csv",
        stable_window_steps=2,
    )
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["train99"] == "NA"
    assert row["test99"] == "NA"
    assert row["stable99"] == "NA"
    assert row["mean_test_post"] == "NA"
    assert row["min_test_post"] == "NA"
    assert row["collapse_count"] == "NA"
    assert row["m"] == "0"
    assert row["d"] == "0"
    assert row["time_unit"] == "epoch"


def test_stable_window_does_not_bridge_unmeasured_intervals():
    evaluations = [(10, 0.99), (20, 0.995), (40, 0.999)]
    assert _stable_window_start(
        evaluations,
        window_steps=20,
        threshold=0.99,
        max_sampling_gap=10,
    ) == "NA"


def test_diagnostic_audit_reports_logged_and_checkpoint_derived_data(tmp_path):
    run_dir = tmp_path / "audited"
    _create_run(run_dir)
    rows = []
    for step in range(3):
        rows.append({
            "step": step,
            "train_loss": 1.0,
            "test_loss": 1.2,
            "train_accuracy": 0.9,
            "test_accuracy": 0.8,
            "model_version": step,
            "node_source_versions": [step - 1],
            "detected_key_frequencies": [1, 2],
            "fixed_key_frequencies": [1],
            "embedding_fourier_norms": [0.1, 0.2],
            "excluded_loss_by_frequency": {"1": {"test_loss": 0.5}},
            "aggregated_momentum_l2_norm": 0.7,
        })
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_dir / "node_metrics.jsonl").write_text(
        "".join(json.dumps({
            "step": step,
            "node": 0,
            "gradient_l2_norm": 0.4,
            "source_model_version": step - 1,
        }) + "\n" for step in range(1, 4)),
        encoding="utf-8",
    )
    (run_dir / "final.pt").write_bytes(b"checkpoint exists")
    (run_dir / "fourier_spectra.csv").write_text(
        "epoch,step,matrix,component_index,component,norm,squared_norm\n"
        "2,2,embedding,0,DC,0.2,0.04\n",
        encoding="utf-8",
    )

    output = audit_diagnostics([run_dir], tmp_path / "available_diagnostics.csv")
    with output.open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))

    assert len(audit_rows) == 16
    by_name = {row["diagnostic"]: row for row in audit_rows}
    assert by_name["train_loss"]["status"] == "logged"
    assert by_name["train_loss"]["source_file"] == "metrics.jsonl"
    assert by_name["per_node_gradient_norms"]["status"] == "logged"
    assert by_name["per_node_gradient_norms"]["source_file"] == "node_metrics.jsonl"
    assert by_name["per_node_gradient_vectors"]["status"] == "derivable offline"
    assert by_name["stale_fresh_gradient_alignment"]["status"] == "requires new computation or run"
    assert by_name["adam_first_moment"]["status"] == "logged"
    assert by_name["adam_second_moment"]["status"] == "derivable offline"
    assert by_name["actual_update_norm"]["status"] == "derivable offline"
    assert by_name["fourier_amplitudes_norms"]["status"] == "logged"
    assert by_name["fourier_complex_coefficients"]["status"] == "derivable offline"
    assert by_name["discovered_key_frequencies"]["status"] == "logged"
    assert by_name["frequency_ablation_metrics"]["status"] == "logged"
    assert by_name["logits_and_margins"]["status"] == "derivable offline"
    assert by_name["model_and_optimizer_checkpoints"]["status"] == "logged"
