from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from src.dataset import make_modular_addition_data

NA = "NA"
CSV_FIELDS = (
    "run_id",
    "experiment_name",
    "seed",
    "m",
    "d",
    "staleness_mode",
    "configured_tau",
    "nodes",
    "zero_delay_nodes",
    "optimizer",
    "lr",
    "wd",
    "beta1",
    "beta2",
    "train99",
    "test99",
    "stable99",
    "stable_window_steps",
    "stable_window_sampling_cadence",
    "mean_test_post",
    "min_test_post",
    "collapse_count",
    "collapse_event_rule",
    "final_train",
    "final_test",
    "time_unit",
    "evaluation_interval_steps",
    "configured_validation_interval",
    "train_examples",
    "test_examples",
    "total_examples",
    "identical_train_test_splits",
)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config.yaml in {run_dir}")
    return config


def _discover_run_dirs(inputs: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for path in inputs:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_file():
            raise ValueError(f"Expected a run directory or results root, got file: {path}")
        if (path / "config.yaml").is_file() and (path / "metrics.jsonl").is_file():
            discovered.add(path)
            continue
        discovered.update(
            candidate.parent
            for candidate in path.rglob("config.yaml")
            if (candidate.parent / "metrics.jsonl").is_file()
        )
    if not discovered:
        raise ValueError("No run directories containing config.yaml and metrics.jsonl found")
    return sorted(discovered)


def _load_metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {run_dir / 'metrics.jsonl'}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {run_dir / 'metrics.jsonl'}:{line_number}"
                )
            rows.append(record)
    return rows


def _step(row: dict[str, Any]) -> int:
    value = row.get("step", row.get("epoch"))
    if value is None:
        raise ValueError("metric row has neither step nor epoch")
    return int(value)


def _evaluations(rows: list[dict[str, Any]], field: str) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result.append((_step(row), float(value)))
    result.sort(key=lambda point: point[0])
    return result


def _first_threshold(
    evaluations: list[tuple[int, float]],
    threshold: float,
) -> int | str:
    return next(
        (step for step, accuracy in evaluations if accuracy >= threshold),
        NA,
    )


def _stable_window_start(
    evaluations: list[tuple[int, float]],
    window_steps: int,
    threshold: float,
    max_sampling_gap: int,
) -> int | str:
    streak_start: int | None = None
    previous_step: int | None = None
    for step, accuracy in evaluations:
        if (
            accuracy < threshold
            or (
                previous_step is not None
                and step - previous_step > max_sampling_gap
            )
        ):
            streak_start = None
        if accuracy >= threshold:
            if streak_start is None:
                streak_start = step
            if step - streak_start >= window_steps:
                return streak_start
        previous_step = step
    return NA


def _collapse_count(
    evaluations: list[tuple[int, float]],
    first_test99: int | str,
) -> int | str:
    if first_test99 == NA:
        return NA
    below = False
    episodes = 0
    for step, accuracy in evaluations:
        if step < first_test99:
            continue
        if accuracy < 0.90:
            if not below:
                episodes += 1
            below = True
        else:
            below = False
    return episodes


def _observed_cadence(
    evaluations: list[tuple[int, float]],
    config: dict[str, Any],
) -> int | str:
    intervals = {
        later[0] - earlier[0]
        for earlier, later in zip(evaluations, evaluations[1:])
    }
    if len(intervals) == 1:
        return intervals.pop()
    if len(intervals) > 1:
        return "irregular"
    distributed = config.get("distributed", {}).get("enabled", False)
    if distributed:
        return 1
    return config.get("logging", {}).get("validation_interval", NA)


def _split_signature(config: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    try:
        experiment = config["experiment"]
        data = make_modular_addition_data(
            modulus=int(experiment["modulus"]),
            train_fraction=float(experiment["train_fraction"]),
            seed=int(experiment["seed"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return (
        tuple(sorted(int(index) for index in data.train_indices.tolist())),
        tuple(sorted(int(index) for index in data.test_indices.tolist())),
    )


def _example_counts(run_dir: Path, config: dict[str, Any]) -> tuple[int | str, int | str, int | str]:
    environment_path = run_dir / "environment.json"
    if environment_path.is_file():
        with environment_path.open(encoding="utf-8") as handle:
            environment = json.load(handle)
        if all(
            key in environment
            for key in ("train_examples", "test_examples", "total_examples")
        ):
            return (
                int(environment["train_examples"]),
                int(environment["test_examples"]),
                int(environment["total_examples"]),
            )

    try:
        experiment = config["experiment"]
        data = make_modular_addition_data(
            modulus=int(experiment["modulus"]),
            train_fraction=float(experiment["train_fraction"]),
            seed=int(experiment["seed"]),
        )
        return (
            int(data.train_indices.numel()),
            int(data.test_indices.numel()),
            int(data.all_inputs.shape[0]),
        )
    except (KeyError, TypeError, ValueError):
        return NA, NA, NA


def _run_identifier(config: dict[str, Any]) -> str:
    run_dir = config.get("runtime", {}).get("run_dir")
    if isinstance(run_dir, str) and run_dir:
        return Path(run_dir).name
    name = config.get("experiment", {}).get("name")
    seed = config.get("experiment", {}).get("seed")
    if name is not None and seed is not None:
        return f"{name}_seed{seed}"
    if name is not None:
        return str(name)
    return NA


def _summarize_run(
    run_dir: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    stable_window_steps: int,
) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    optimizer = config.get("optimizer", {})
    distributed = config.get("distributed", {})
    test_evaluations = _evaluations(rows, "test_accuracy")
    train_evaluations = _evaluations(rows, "train_accuracy")
    first_test99 = _first_threshold(test_evaluations, 0.99)
    post_99 = (
        [accuracy for step, accuracy in test_evaluations if step >= first_test99]
        if first_test99 != NA
        else []
    )

    if distributed.get("staleness_mode") == "mixed_fixed":
        nodes = distributed.get("nodes")
        zero_delay_nodes = distributed.get("zero_delay_nodes")
        delayed_workers = (
            int(nodes) - int(zero_delay_nodes)
            if nodes is not None and zero_delay_nodes is not None
            else NA
        )
        delay_steps = distributed.get("tau", NA)
    elif distributed.get("enabled"):
        delayed_workers = NA
        delay_steps = NA
    else:
        delayed_workers = 0
        delay_steps = 0

    train_examples, test_examples, total_examples = _example_counts(run_dir, config)
    cadence = _observed_cadence(test_evaluations, config)
    cadence_description = (
        f"{cadence} steps between measured test evaluations"
        if isinstance(cadence, int)
        else str(cadence)
    )
    time_unit = (
        "global_optimizer_step"
        if distributed.get("enabled")
        else "epoch"
    )
    configured_interval = (
        1 if distributed.get("enabled")
        else config.get("logging", {}).get("validation_interval", NA)
    )
    try:
        max_sampling_gap = max(1, int(configured_interval))
    except (TypeError, ValueError):
        max_sampling_gap = 1
    final_train = train_evaluations[-1][1] if train_evaluations else NA
    final_test = test_evaluations[-1][1] if test_evaluations else NA
    stable99 = _stable_window_start(
        test_evaluations,
        stable_window_steps,
        0.99,
        max_sampling_gap,
    )
    mean_test_post = sum(post_99) / len(post_99) if post_99 else NA
    min_test_post = min(post_99) if post_99 else NA

    return {
        "run_id": _run_identifier(config),
        "experiment_name": experiment.get("name", NA),
        "seed": experiment.get("seed", NA),
        "m": delayed_workers,
        "d": delay_steps,
        "staleness_mode": distributed.get("staleness_mode", "canonical"),
        "configured_tau": distributed.get("tau", 0),
        "nodes": distributed.get("nodes", 1),
        "zero_delay_nodes": distributed.get("zero_delay_nodes", 0),
        "optimizer": optimizer.get("name", NA),
        "lr": optimizer.get("learning_rate", NA),
        "wd": optimizer.get("weight_decay", NA),
        "beta1": optimizer.get("beta1", NA),
        "beta2": optimizer.get("beta2", NA),
        "train99": _first_threshold(train_evaluations, 0.99),
        "test99": first_test99,
        "stable99": stable99,
        "stable_window_steps": stable_window_steps,
        "stable_window_sampling_cadence": cadence_description,
        "mean_test_post": mean_test_post,
        "min_test_post": min_test_post,
        "collapse_count": _collapse_count(test_evaluations, first_test99),
        "collapse_event_rule": "contiguous logged test evaluations below 0.90 after first test99",
        "final_train": final_train,
        "final_test": final_test,
        "time_unit": time_unit,
        "evaluation_interval_steps": cadence,
        "configured_validation_interval": configured_interval,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "total_examples": total_examples,
        "identical_train_test_splits": NA,
    }


def summarize_experiments(
    inputs: list[str | Path],
    output: str | Path,
    stable_window_steps: int = 1000,
) -> Path:
    if stable_window_steps < 1:
        raise ValueError("stable_window_steps must be positive")
    run_dirs = _discover_run_dirs([Path(path) for path in inputs])
    records: list[dict[str, Any]] = []
    signatures: list[tuple[tuple[int, ...], tuple[int, ...]] | None] = []
    for run_dir in run_dirs:
        config = _load_run_config(run_dir)
        metric_rows = _load_metric_rows(run_dir)
        records.append(
            _summarize_run(run_dir, config, metric_rows, stable_window_steps)
        )
        signatures.append(_split_signature(config))

    valid_signatures = [signature for signature in signatures if signature is not None]
    all_splits_identical = (
        len(valid_signatures) == len(signatures)
        and bool(valid_signatures)
        and all(signature == valid_signatures[0] for signature in valid_signatures)
    )
    for record, signature in zip(records, signatures):
        record["identical_train_test_splits"] = (
            all_splits_identical
            if signature is not None and len(valid_signatures) == len(signatures)
            else NA
        )

    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize saved Grokking runs into one experiment_summary.csv."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run directories or parent directories (searched recursively).",
    )
    parser.add_argument(
        "--output",
        default="experiment_summary.csv",
        help="Output CSV path (default: experiment_summary.csv).",
    )
    parser.add_argument(
        "--stable-window-steps",
        type=int,
        default=1000,
        help="Required observed high-accuracy span for stable99, in training steps (default: 1000).",
    )
    args = parser.parse_args()
    try:
        destination = summarize_experiments(
            args.runs,
            args.output,
            stable_window_steps=args.stable_window_steps,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Experiment summary saved to {destination}")


if __name__ == "__main__":
    main()
