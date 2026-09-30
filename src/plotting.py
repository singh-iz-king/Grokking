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
