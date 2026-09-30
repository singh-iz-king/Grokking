from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_step(row: dict[str, Any]) -> int:
    value = row.get("epoch", row.get("step"))
    if value is None:
        raise ValueError("metric record has neither epoch nor step")
    return int(value)


def _plot_series(
    rows: list[dict[str, Any]],
    keys: list[str],
    title: str,
    ylabel: str,
    path: Path,
    *,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    has_values = False
    for key in keys:
        points = [
            (_row_step(row), row[key])
            for row in rows
            if isinstance(row.get(key), (float, int))
        ]
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], label=key)
            has_values = True
    if has_values:
        if log_y:
            ax.set_yscale("log")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_spectra(run_dir: Path, figures_dir: Path, matrix: str, title: str) -> None:
    path = run_dir / "fourier_spectra.csv"
    if not path.exists():
        return
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["matrix"] == matrix:
                grouped[row["component"]].append(
                    (int(row["epoch"]), float(row["norm"]))
                )
    fig, ax = plt.subplots(figsize=(10, 6))
    for component, points in grouped.items():
        ax.plot(
            [epoch for epoch, _ in points],
            [norm for _, norm in points],
            label=component,
            linewidth=0.8,
        )
    if grouped:
        ax.set(title=title, xlabel="Epoch", ylabel="Fourier component L2 norm")
        ax.legend(ncol=4, fontsize=6)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{matrix}_fourier_norms.png", dpi=160)
    plt.close(fig)


def _plot_frequency_ablations(
    rows: list[dict[str, Any]],
    figures_dir: Path,
    metric: str,
    split: str,
    *,
    log_y: bool = False,
) -> None:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        ablations = row.get("excluded_loss_by_frequency")
        if not isinstance(ablations, dict):
            continue
        for frequency, values in ablations.items():
            if isinstance(values, dict) and metric in values:
                series[str(frequency)].append(
                    (_row_step(row), float(values[metric]))
                )
    if not series:
        return
    metric_label = metric.removeprefix(f"{split}_")
    fig, ax = plt.subplots(figsize=(9, 5))
    for frequency, points in series.items():
        ax.plot(
            [epoch for epoch, _ in points],
            [value for _, value in points],
            label=f"k={frequency}",
        )
    if log_y:
        ax.set_yscale("log")
    ax.set(
        title=f"Individual-frequency excluded {metric_label.replace('_', ' ')}",
        xlabel="Epoch",
        ylabel=f"{split.title()} {metric_label.replace('_', ' ')}",
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"excluded_by_frequency_{split}_{metric_label}.png",
        dpi=160,
    )
    plt.close(fig)


def plot_results(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    rows = _read_metrics(run_dir)
    if not rows:
        raise ValueError(f"No metric records found in {run_dir}")
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    _plot_series(rows, ["train_loss", "test_loss"], "Training and test loss",
                 "Cross-entropy loss", figures_dir / "loss.png", log_y=True)
    _plot_series(rows, ["train_accuracy", "test_accuracy"], "Training and test accuracy",
                 "Accuracy", figures_dir / "accuracy.png")
    _plot_series(
        rows,
        ["restricted_loss", "excluded_loss", "train_loss", "test_loss"],
        "Restricted and excluded loss",
        "Cross-entropy loss",
        figures_dir / "restricted_excluded_loss.png",
        log_y=True,
    )
    _plot_series(
        rows,
        ["restricted_train_accuracy", "restricted_test_accuracy",
         "excluded_train_accuracy", "excluded_test_accuracy"],
        "Restricted and excluded accuracy",
        "Accuracy",
        figures_dir / "restricted_excluded_accuracy.png",
    )
    _plot_frequency_ablations(
        rows, figures_dir, "train_loss", "train", log_y=True
    )
    _plot_frequency_ablations(
        rows, figures_dir, "train_accuracy", "train"
    )
    _plot_series(rows, ["weight_l2_norm"], "L2 norm of all trainable weights",
                 "L2 norm", figures_dir / "weight_l2_norm.png")
    _plot_series(
        rows,
        ["embedding_fourier_gini", "logit_map_fourier_gini"],
        "Fourier-component Gini coefficients",
        "Gini coefficient",
        figures_dir / "fourier_gini.png",
    )
    _plot_spectra(run_dir, figures_dir, "embedding", "Embedding Fourier spectrum")
    _plot_spectra(run_dir, figures_dir, "logit_map", "Neuron-to-logit map Fourier spectrum")

    coefficients: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        values = row.get("key_logit_coefficients")
        if isinstance(values, dict):
            for frequency, value in values.items():
                coefficients[str(frequency)].append((_row_step(row), float(value)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for frequency, points in coefficients.items():
        ax.plot([x for x, _ in points], [y for _, y in points], label=f"k={frequency}")
    if coefficients:
        ax.set(title="Key-frequency logit coefficients", xlabel="Epoch", ylabel="Projection coefficient")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "key_logit_coefficients.png", dpi=160)
    plt.close(fig)
    return figures_dir


def _plot_node_scalar_series(
    rows: list[dict[str, Any]],
    series: dict[str, list[tuple[int, float]]],
    title: str,
    ylabel: str,
    path: Path,
    *,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, points in series.items():
        if points:
            ax.plot(
                [step for step, _ in points],
                [value for _, value in points],
                marker=".",
                markersize=3,
                label=label,
            )
    if log_y:
        ax.set_yscale("log")
    ax.set(
        title=title,
        xlabel="Global step (node's gradient-source model)",
        ylabel=ylabel,
    )
    ax.grid(True, alpha=0.25)
    if any(series.values()):
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_node_spectrum(
    rows: list[dict[str, Any]],
    figures_dir: Path,
    metric_name: str,
    title: str,
) -> None:
    available = [
        (int(row["step"]), row.get("progress_measures"))
        for row in rows
        if isinstance(row.get("progress_measures"), dict)
        and isinstance(row["progress_measures"].get(metric_name), list)
    ]
    if not available:
        return
    spectra = [values[metric_name] for _, values in available]
    if not spectra or not spectra[0]:
        return
    names = available[-1][1].get("fourier_component_names", [])
    matrix = [[float(value) for value in spectrum] for spectrum in spectra]
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax.set(
        title=title,
        xlabel="Real Fourier component",
        ylabel="Global step (node's gradient-source model)",
    )
    if names:
        tick_indices = list(range(0, len(names), max(1, len(names) // 12)))
        ax.set_xticks(tick_indices, [names[index] for index in tick_indices], rotation=45)
    ytick_indices = list(range(0, len(available), max(1, len(available) // 10)))
    ax.set_yticks(
        ytick_indices,
        [available[index][0] for index in ytick_indices],
    )
    fig.colorbar(image, ax=ax, label="Fourier component L2 norm")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{metric_name}.png", dpi=160)
    plt.close(fig)


def _plot_node_frequency_ablations(
    rows: list[dict[str, Any]],
    figures_dir: Path,
    metric: str,
    *,
    log_y: bool = False,
) -> None:
    by_frequency: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        measures = row.get("progress_measures")
        if not isinstance(measures, dict):
            continue
        ablations = measures.get("excluded_loss_by_frequency")
        if not isinstance(ablations, dict):
            continue
        for frequency, values in ablations.items():
            value = values.get(metric) if isinstance(values, dict) else None
            if isinstance(value, (int, float)):
                by_frequency[str(frequency)].append((int(row["step"]), float(value)))
    if not by_frequency:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for frequency, points in by_frequency.items():
        ax.plot(
            [step for step, _ in points],
            [value for _, value in points],
            marker=".",
            markersize=3,
            label=f"k={frequency}",
        )
    if log_y:
        ax.set_yscale("log")
    ax.set(
        title=f"Node-source individual-frequency excluded {metric.replace('_', ' ')}",
        xlabel="Global step (node's gradient-source model)",
        ylabel=metric.replace("_", " ").title(),
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"excluded_by_frequency_{metric}.png", dpi=160)
    plt.close(fig)


def plot_node_metrics(run_dir: str | Path) -> Path:
    """Plot global-step and mechanistic histories separately for each node."""
    run_dir = Path(run_dir)
    node_log = run_dir / "node_metrics.jsonl"
    if not node_log.exists():
        raise FileNotFoundError(f"No distributed node log found at {node_log}")
    with node_log.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No node metric records found in {node_log}")

    output_root = run_dir / "figures" / "nodes"
    output_root.mkdir(parents=True, exist_ok=True)
    nodes = sorted({int(row["node"]) for row in rows})
    for node in nodes:
        node_rows = sorted(
            (row for row in rows if int(row["node"]) == node),
            key=lambda row: int(row["step"]),
        )
        figures_dir = output_root / f"node_{node}"
        figures_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"Node {node}: selected gradient-source model"

        for filename, fields, title, ylabel, log_y in (
            (
                "train_test_loss.png",
                ("train_loss", "test_loss"),
                "Training and test loss",
                "Cross-entropy loss",
                True,
            ),
            (
                "train_test_accuracy.png",
                ("train_accuracy", "test_accuracy"),
                "Training and test accuracy",
                "Accuracy",
                False,
            ),
        ):
            series: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for row in node_rows:
                for field in fields:
                    value = row.get(field)
                    if isinstance(value, (int, float)):
                        series[field].append((int(row["step"]), float(value)))
            _plot_node_scalar_series(
                node_rows, series, f"{prefix}: {title}", ylabel,
                figures_dir / filename, log_y=log_y,
            )

        scalar_specs = (
            (
                "restricted_excluded_loss.png",
                ("restricted_loss", "excluded_loss", "restricted_train_loss",
                 "restricted_test_loss"),
                "Restricted and excluded loss",
                "Cross-entropy loss",
                True,
            ),
            (
                "restricted_excluded_accuracy.png",
                ("restricted_train_accuracy", "restricted_test_accuracy",
                 "excluded_train_accuracy", "excluded_test_accuracy"),
                "Restricted and excluded accuracy",
                "Accuracy",
                False,
            ),
            (
                "weight_l2_norm.png",
                ("weight_l2_norm",),
                "L2 weight norm",
                "L2 norm",
                False,
            ),
            (
                "fourier_gini.png",
                ("embedding_fourier_gini", "logit_map_fourier_gini"),
                "Fourier-component Gini",
                "Gini coefficient",
                False,
            ),
        )
        for filename, fields, title, ylabel, log_y in scalar_specs:
            series = defaultdict(list)
            for row in node_rows:
                measures = row.get("progress_measures")
                if not isinstance(measures, dict):
                    continue
                for field in fields:
                    value = measures.get(field)
                    if isinstance(value, (int, float)):
                        series[field].append((int(row["step"]), float(value)))
            _plot_node_scalar_series(
                node_rows, series, f"{prefix}: {title}", ylabel,
                figures_dir / filename, log_y=log_y,
            )

        _plot_node_frequency_ablations(
            node_rows, figures_dir, "train_loss", log_y=True
        )
        _plot_node_frequency_ablations(
            node_rows, figures_dir, "train_accuracy"
        )
        _plot_node_spectrum(
            node_rows,
            figures_dir,
            "embedding_fourier_norms",
            f"{prefix}: embedding Fourier spectrum",
        )
        _plot_node_spectrum(
            node_rows,
            figures_dir,
            "logit_map_fourier_norms",
            f"{prefix}: neuron-to-logit map Fourier spectrum",
        )

        coefficients: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in node_rows:
            measures = row.get("progress_measures")
            values = measures.get("key_logit_coefficients") if isinstance(measures, dict) else None
            if isinstance(values, dict):
                for frequency, value in values.items():
                    if isinstance(value, (int, float)):
                        coefficients[str(frequency)].append(
                            (int(row["step"]), float(value))
                        )
        if coefficients:
            _plot_node_scalar_series(
                node_rows,
                coefficients,
                f"{prefix}: key-frequency logit coefficients",
                "Projection coefficient",
                figures_dir / "key_logit_coefficients.png",
            )

        lag_series = {
            "staleness": [
                (int(row["step"]), float(row["staleness"]))
                for row in node_rows
            ]
        }
        _plot_node_scalar_series(
            node_rows,
            lag_series,
            f"Node {node}: sampled source-model lag",
            "Staleness (global updates)",
            figures_dir / "staleness.png",
        )
    return output_root
