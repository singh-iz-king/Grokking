from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.summarize_experiments import (
    NA,
    _discover_run_dirs,
    _load_metric_rows,
    _load_run_config,
    _run_identifier,
)

STATUS_VALUES = {"logged", "derivable offline", "requires new computation or run"}
CSV_FIELDS = ("run_id", "diagnostic", "status", "source_file", "sampling_interval", "notes")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def _step(row: dict[str, Any]) -> int | None:
    value = row.get("step", row.get("epoch"))
    return int(value) if value is not None else None


def _interval(rows: list[dict[str, Any]], predicate) -> str:
    steps = sorted({
        step
        for row in rows
        if predicate(row)
        if (step := _step(row)) is not None
    })
    if not steps:
        return NA
    gaps = {later - earlier for earlier, later in zip(steps, steps[1:])}
    if len(gaps) == 1:
        return f"every {gaps.pop()} steps"
    if len(gaps) > 1:
        return "irregular; observed step gaps vary"
    return "single recorded step"


def _present_sources(run_dir: Path, names: list[str]) -> list[str]:
    return [name for name in names if (run_dir / name).is_file()]


def _audit_run(run_dir: Path) -> list[dict[str, str]]:
    config = _load_run_config(run_dir)
    metrics = _load_metric_rows(run_dir)
    node_rows = _read_jsonl(run_dir / "node_metrics.jsonl")
    all_rows = metrics
    run_id = _run_identifier(config)
    optimizer = config.get("optimizer", {})
    logging = config.get("logging", {})
    mechanistic = config.get("mechanistic", {})
    distributed = config.get("distributed", {})
    checkpoint_exists = (run_dir / "final.pt").is_file()
    checkpoint_interval = logging.get("checkpoint_interval", NA)

    def add(
        diagnostic: str,
        status: str,
        sources: list[str],
        sampling: str,
        notes: str,
    ) -> dict[str, str]:
        if status not in STATUS_VALUES:
            raise ValueError(f"Invalid diagnostic status: {status}")
        return {
            "run_id": run_id,
            "diagnostic": diagnostic,
            "status": status,
            "source_file": "; ".join(sources) if sources else NA,
            "sampling_interval": sampling,
            "notes": notes,
        }

    results: list[dict[str, str]] = []
    for field, label in (("train_loss", "train_loss"), ("test_loss", "test_loss")):
        logged = any(isinstance(row.get(field), (int, float)) for row in metrics)
        results.append(add(
            label,
            "logged" if logged else "requires new computation or run",
            ["metrics.jsonl"] if logged else [],
            _interval(metrics, lambda row: isinstance(row.get(field), (int, float))),
            "Scalar loss values are recorded only at measured evaluation steps; gaps are not interpolated."
            if logged else "No scalar values for this loss were found in the run metrics.",
        ))

    gradient_logged = any(
        isinstance(row.get("gradient_l2_norm"), (int, float))
        for row in node_rows
    )
    results.append(add(
        "per_node_gradient_norms",
        "logged" if gradient_logged else (
            "derivable offline" if checkpoint_exists else "requires new computation or run"
        ),
        ["node_metrics.jsonl"] if gradient_logged else (
            ["final.pt"] if checkpoint_exists else []
        ),
        _interval(
            node_rows,
            lambda row: isinstance(row.get("gradient_l2_norm"), (int, float)),
        ) if gradient_logged else ("final checkpoint only" if checkpoint_exists else NA),
        "Norms are computed before aggregation from each node's full-batch gradient; this is a scalar, not the gradient tensor."
        if gradient_logged else "Gradient norms can be recomputed at saved model states only; no per-step norm series is logged.",
    ))
    results.append(add(
        "per_node_gradient_vectors",
        "derivable offline" if checkpoint_exists else "requires new computation or run",
        ["final.pt"] if checkpoint_exists else [],
        "saved checkpoint/history versions only" if checkpoint_exists else NA,
        "Gradient tensors are not saved. They can be recomputed from saved model states and split indices, but the complete historical per-step vector series is unavailable.",
    ))
    alignment_logged = any(
        isinstance(row.get("delayed_gradient_cosine_mean"), (int, float))
        and isinstance(
            row.get("delayed_gradient_relative_discrepancy_mean"), (int, float)
        )
        for row in metrics
    ) or any(
        isinstance(row.get("gradient_cosine_similarity"), (int, float))
        and isinstance(row.get("gradient_relative_discrepancy"), (int, float))
        for row in node_rows
    )
    results.append(add(
        "stale_fresh_gradient_alignment",
        "logged" if alignment_logged else "requires new computation or run",
        (
            (["metrics.jsonl"] if any(
                isinstance(row.get("delayed_gradient_cosine_mean"), (int, float))
                for row in metrics
            ) else []) +
            (["node_metrics.jsonl"] if any(
                isinstance(row.get("gradient_cosine_similarity"), (int, float))
                for row in node_rows
            ) else [])
        ),
        _interval(
            metrics,
            lambda row: isinstance(
                row.get("delayed_gradient_cosine_mean"), (int, float)
            ),
        ) if alignment_logged else NA,
        "Per-delayed-worker cosine similarity and relative discrepancy are logged, together with their delayed-worker means."
        if alignment_logged else
        "Gradient vectors and stale/fresh inner products are absent; existing gradient norms cannot recover alignment. A targeted run with fresh/stale gradient comparisons is required.",
    ))

    first_moment_logged = any(
        isinstance(row.get("aggregated_momentum_l2_norm"), (int, float))
        for row in metrics
    )
    moment_sources = _present_sources(run_dir, ["final.pt"])
    results.append(add(
        "adam_first_moment",
        "logged" if first_moment_logged else (
            "derivable offline" if checkpoint_exists else "requires new computation or run"
        ),
        (["metrics.jsonl"] if first_moment_logged else []) + moment_sources,
        _interval(
            metrics,
            lambda row: isinstance(row.get("aggregated_momentum_l2_norm"), (int, float)),
        ) if first_moment_logged else ("final checkpoint only" if checkpoint_exists else NA),
        "The norm of centralized AdamW exp_avg is logged; exact exp_avg tensors are in optimizer_state_dict at the final checkpoint."
        if first_moment_logged and checkpoint_exists else
        "Only the logged scalar norm is present; full per-parameter moment tensors are unavailable."
        if first_moment_logged else
        "Exact first-moment tensors are available only in the saved final optimizer state."
        if checkpoint_exists else
        "No first-moment scalar series or optimizer checkpoint was found.",
    ))
    second_moment_logged = False
    results.append(add(
        "adam_second_moment",
        "derivable offline" if checkpoint_exists else "requires new computation or run",
        ["final.pt"] if checkpoint_exists else [],
        "final checkpoint only" if checkpoint_exists else NA,
        "AdamW exp_avg_sq tensors are stored in the final optimizer_state_dict, but no scalar or trajectory is logged."
        if checkpoint_exists else "No optimizer checkpoint containing exp_avg_sq was found.",
    ))

    history_available = False
    if checkpoint_exists and distributed.get("enabled"):
        history_available = True
    results.append(add(
        "actual_update_norm",
        "derivable offline" if history_available else (
            "requires new computation or run"
        ),
        ["final.pt"] if history_available else [],
        "adjacent versions in retained final history only" if history_available else NA,
        "No update-norm time series is logged. Distributed final checkpoints retain a bounded model history, allowing recent adjacent-version update norms to be calculated; a complete trajectory requires saving adjacent checkpoints or logging each update norm."
        if history_available else
        "The run directory has no adjacent model-state snapshots to difference; log update norms or retain checkpoints at every step for a full trajectory.",
    ))

    parameter_ids_logged = any(
        "model_version" in row or "node_source_versions" in row
        for row in metrics
    ) or any("source_model_version" in row for row in node_rows)
    results.append(add(
        "parameter_version_ids",
        "logged" if parameter_ids_logged else (
            "derivable offline" if checkpoint_exists else "requires new computation or run"
        ),
        (["metrics.jsonl"] if parameter_ids_logged else []) +
        (["node_metrics.jsonl"] if node_rows and parameter_ids_logged else []),
        "every global step" if parameter_ids_logged and distributed.get("enabled") else (
            "every logged epoch" if parameter_ids_logged else NA
        ),
        "Version numbers/source-version IDs are logged; parameter tensors themselves are checkpointed only at saved checkpoint versions."
        if parameter_ids_logged else "No explicit parameter version identifiers were found.",
    ))

    spectra_path = run_dir / "fourier_spectra.csv"
    spectrum_rows = []
    if spectra_path.is_file():
        with spectra_path.open(newline="", encoding="utf-8") as handle:
            spectrum_rows = list(csv.DictReader(handle))
    amplitude_logged = bool(spectrum_rows) or any(
        isinstance(row.get("embedding_fourier_norms"), list)
        or isinstance(row.get("logit_map_fourier_norms"), list)
        for row in metrics
    )
    results.append(add(
        "fourier_amplitudes_norms",
        "logged" if amplitude_logged else (
            "derivable offline" if checkpoint_exists else "requires new computation or run"
        ),
        (["fourier_spectra.csv"] if spectrum_rows else []) +
        (["metrics.jsonl"] if any(
            isinstance(row.get("embedding_fourier_norms"), list)
            or isinstance(row.get("logit_map_fourier_norms"), list)
            for row in metrics
        ) else []) +
        (["final.pt"] if not amplitude_logged and checkpoint_exists else []),
        (
            _interval(
                [{"step": row.get("step", row.get("epoch"))} for row in spectrum_rows],
                lambda row: True,
            ) if spectrum_rows else
            _interval(metrics, lambda row: isinstance(row.get("embedding_fourier_norms"), list))
        ) if amplitude_logged else ("final checkpoint only" if checkpoint_exists else NA),
        "Fourier component norms (amplitudes) are recorded; these are not complex Fourier coefficients."
        if amplitude_logged else "Norm spectra can be recomputed from saved weights only at checkpoint states.",
    ))
    complex_logged = any(
        "fourier_complex_coefficients" in row for row in metrics
    )
    results.append(add(
        "fourier_complex_coefficients",
        "logged" if complex_logged else (
            "derivable offline" if checkpoint_exists else "requires new computation or run"
        ),
        ["metrics.jsonl"] if complex_logged else (["final.pt"] if checkpoint_exists else []),
        _interval(metrics, lambda row: "fourier_complex_coefficients" in row)
        if complex_logged else ("final checkpoint only" if checkpoint_exists else NA),
        "Complex-valued Fourier coefficient vectors are not logged. They can be recomputed from saved weights at checkpoint states."
        if checkpoint_exists and not complex_logged else
        "No complex Fourier coefficient field was found in saved metrics.",
    ))

    fixed_logged = any(
        isinstance(row.get("fixed_key_frequencies"), list) for row in metrics
    ) or bool(mechanistic.get("key_frequencies", {}).get("paper_mainline"))
    results.append(add(
        "fixed_key_frequencies",
        "logged" if fixed_logged else "requires new computation or run",
        (["metrics.jsonl"] if any(
            isinstance(row.get("fixed_key_frequencies"), list) for row in metrics
        ) else []) + (["config.yaml"] if mechanistic.get("key_frequencies", {}).get("paper_mainline") else []),
        _interval(metrics, lambda row: isinstance(row.get("fixed_key_frequencies"), list))
        if any(isinstance(row.get("fixed_key_frequencies"), list) for row in metrics)
        else "fixed configuration value",
        "Configured fixed keys are in config.yaml and are repeated in mechanistic metric records."
        if fixed_logged else "No fixed key set is saved.",
    ))
    discovered_logged = any(
        isinstance(row.get("detected_key_frequencies"), list) for row in metrics
    ) or any(
        isinstance((row.get("progress_measures") or {}).get("detected_key_frequencies"), list)
        for row in node_rows
    )
    discovered_sources = (
        (["metrics.jsonl"] if any(
            isinstance(row.get("detected_key_frequencies"), list) for row in metrics
        ) else []) +
        (["node_metrics.jsonl"] if any(
            isinstance((row.get("progress_measures") or {}).get("detected_key_frequencies"), list)
            for row in node_rows
        ) else [])
    )
    results.append(add(
        "discovered_key_frequencies",
        "logged" if discovered_logged else (
            "derivable offline" if checkpoint_exists else "requires new computation or run"
        ),
        discovered_sources or (["final.pt"] if checkpoint_exists else []),
        _interval(metrics, lambda row: isinstance(row.get("detected_key_frequencies"), list))
        if discovered_logged else ("final checkpoint only" if checkpoint_exists else NA),
        "Detected key frequencies are stored at mechanistic-analysis cadence; older intervals cannot be reconstructed without saved model states."
        if discovered_logged else "Key discovery can be rerun on saved checkpoint weights only.",
    ))

    ablation_logged = any(
        isinstance(row.get("excluded_loss_by_frequency"), dict) for row in metrics
    ) or any(
        isinstance((row.get("progress_measures") or {}).get("excluded_loss_by_frequency"), dict)
        for row in node_rows
    )
    ablation_sources = (
        (["metrics.jsonl"] if any(
            isinstance(row.get("excluded_loss_by_frequency"), dict) for row in metrics
        ) else []) +
        (["node_metrics.jsonl"] if any(
            isinstance((row.get("progress_measures") or {}).get("excluded_loss_by_frequency"), dict)
            for row in node_rows
        ) else [])
    )
    results.append(add(
        "frequency_ablation_metrics",
        "logged" if ablation_logged else (
            "derivable offline" if checkpoint_exists and mechanistic.get("frequency_ablations", True)
            else "requires new computation or run"
        ),
        ablation_sources or (["final.pt"] if checkpoint_exists else []),
        _interval(metrics, lambda row: isinstance(row.get("excluded_loss_by_frequency"), dict))
        if ablation_logged else ("final checkpoint only" if checkpoint_exists else NA),
        "Per-frequency excluded loss/accuracy is saved where ablation data is present."
        if ablation_logged else
        "No historical ablation values were logged; final-checkpoint ablations can be recomputed."
        if checkpoint_exists and mechanistic.get("frequency_ablations", True) else
        "Ablations are disabled or cannot be computed from the files in this run directory.",
    ))

    logits_logged = any("logits" in row for row in metrics)
    results.append(add(
        "logits_and_margins",
        "logged" if logits_logged else (
            "derivable offline" if checkpoint_exists else "requires new computation or run"
        ),
        ["metrics.jsonl"] if logits_logged else (["final.pt"] if checkpoint_exists else []),
        _interval(metrics, lambda row: "logits" in row)
        if logits_logged else ("final checkpoint only" if checkpoint_exists else NA),
        "Raw logits and per-example margins are not logged; they can be computed from saved checkpoint weights and the saved/reconstructed dataset at checkpoint states."
        if checkpoint_exists and not logits_logged else
        "Raw logits and margins are not present in this run's files.",
    ))

    checkpoint_files = ["final.pt"] if checkpoint_exists else []
    checkpoint_files.extend(
        str(path.relative_to(run_dir))
        for path in sorted((run_dir / "checkpoints").glob("*.pt"))
    ) if (run_dir / "checkpoints").is_dir() else None
    results.append(add(
        "model_and_optimizer_checkpoints",
        "logged" if checkpoint_files else "requires new computation or run",
        checkpoint_files,
        f"configured every {checkpoint_interval} steps plus final" if checkpoint_files else NA,
        "Run-directory checkpoint artifacts found: " + ", ".join(checkpoint_files)
        if checkpoint_files else
        "No checkpoint files are present inside the run directory. Configured checkpoint files may be stored outside it.",
    ))
    return results


def audit_diagnostics(
    inputs: list[str | Path],
    output: str | Path,
) -> Path:
    run_dirs = _discover_run_dirs([Path(path) for path in inputs])
    records: list[dict[str, str]] = []
    for run_dir in run_dirs:
        records.extend(_audit_run(run_dir))
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit diagnostic data saved in Grokking run directories."
    )
    parser.add_argument("runs", nargs="+", help="Run directories or parent directories.")
    parser.add_argument(
        "--output",
        default="available_diagnostics.csv",
        help="Output CSV path (default: available_diagnostics.csv).",
    )
    args = parser.parse_args()
    try:
        output = audit_diagnostics(args.runs, args.output)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Diagnostic audit saved to {output}")


if __name__ == "__main__":
    main()
