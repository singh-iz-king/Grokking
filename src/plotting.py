from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

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


def _add_legacy_fixed_frequency_aliases(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    frequency_mode = normalized.get("frequency_mode", "fixed")
    if frequency_mode == "fixed":
        for field in (
            "restricted_loss",
            "restricted_train_loss",
            "restricted_test_loss",
            "restricted_train_accuracy",
            "restricted_test_accuracy",
        ):
            normalized.setdefault(f"{field}_fixed", normalized.get(field))
    for field in (
        "excluded_loss",
        "excluded_train_loss",
        "excluded_test_loss",
        "excluded_train_accuracy",
        "excluded_test_accuracy",
    ):
        normalized.setdefault(f"{field}_fixed", normalized.get(field))
    return normalized


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


def plot_gradient_alignment(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Plot delayed-worker mean stale-vs-fresh gradient alignment."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    rows = _read_metrics(run_dir)
    destination = (
        Path(output_path)
        if output_path is not None
        else run_dir / "figures" / "gradient_alignment.pdf"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    cosine_points = [
        (_row_step(row), float(row["delayed_gradient_cosine_mean"]))
        for row in rows
        if isinstance(row.get("delayed_gradient_cosine_mean"), (int, float))
    ]
    discrepancy_points = [
        (_row_step(row), float(row["delayed_gradient_relative_discrepancy_mean"]))
        for row in rows
        if isinstance(
            row.get("delayed_gradient_relative_discrepancy_mean"),
            (int, float),
        )
    ]
    run_config: dict[str, Any] = {}
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        import yaml

        with config_path.open(encoding="utf-8") as handle:
            run_config = yaml.safe_load(handle) or {}
    distributed = run_config.get("distributed", {})
    epsilon = distributed.get("gradient_alignment", {}).get("epsilon", 1e-12)
    run_id = run_config.get("runtime", {}).get("run_dir", run_dir.name)
    first_99 = next(
        (
            _row_step(row)
            for row in rows
            if isinstance(row.get("test_accuracy"), (int, float))
            and row["test_accuracy"] >= 0.99
        ),
        None,
    )
    collapse_intervals: list[tuple[int, int]] = []
    collapse_start: int | None = None
    collapse_end: int | None = None
    if first_99 is not None:
        for row in rows:
            step = _row_step(row)
            accuracy = row.get("test_accuracy")
            if step < first_99 or not isinstance(accuracy, (int, float)):
                continue
            if accuracy < 0.90:
                if collapse_start is None:
                    collapse_start = step
                collapse_end = step
            elif collapse_start is not None and collapse_end is not None:
                collapse_intervals.append((collapse_start, collapse_end))
                collapse_start = None
                collapse_end = None
        if collapse_start is not None and collapse_end is not None:
            collapse_intervals.append((collapse_start, collapse_end))

    with PdfPages(destination) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        if cosine_points and discrepancy_points:
            axes[0].plot(
                [step for step, _ in cosine_points],
                [value for _, value in cosine_points],
                marker=".",
                markersize=2.5,
                linewidth=0.9,
                label="Mean cosine over delayed workers",
            )
            axes[0].set_ylabel("Cosine similarity")
            axes[0].set_ylim(-1.05, 1.05)
            axes[0].legend(loc="best")
            axes[0].grid(alpha=0.25)

            axes[1].plot(
                [step for step, _ in discrepancy_points],
                [value for _, value in discrepancy_points],
                marker=".",
                markersize=2.5,
                linewidth=0.9,
                color="#dd8452",
                label="Mean relative discrepancy over delayed workers",
            )
            axes[1].set_ylabel("Relative discrepancy")
            axes[1].legend(loc="best")
            axes[1].grid(alpha=0.25)
            for ax in axes:
                for index, (start, end) in enumerate(collapse_intervals):
                    ax.axvspan(
                        start, end,
                        color="#c44e52",
                        alpha=0.15,
                        label="Test accuracy <0.90 episode" if index == 0 else None,
                    )
                if collapse_intervals:
                    ax.legend(loc="best")
            axes[1].set_xlabel("Global optimizer step")
            fig.suptitle(
                f"Are stale gradients misdirected?\n{Path(run_id).name}",
                fontsize=13,
            )
            fig.text(
                0.01, 0.01,
                "Per delayed worker, compare its same-data full-batch gradient at the stale source "
                "parameters with a fresh gradient at the current global parameters. Curves show the "
                "arithmetic mean across workers with positive lag. Cosine = <g_stale,g_fresh> / "
                "(||g_stale|| ||g_fresh|| + epsilon); relative discrepancy = "
                "||g_stale-g_fresh|| / (||g_fresh|| + epsilon). "
                f"epsilon={epsilon:g}. Shading marks contiguous measured test-accuracy <0.90 "
                "episodes after first measured >=0.99. Points are per-step records; no interpolation.",
                fontsize=7.5,
            )
        else:
            for ax in axes:
                ax.axis("off")
            reason = (
                "The run has no delayed-worker gradient-alignment values in metrics.jsonl. "
                "Existing gradient-norm records cannot recover cosine similarity or relative "
                "discrepancy. This run predates alignment logging or contains no delayed updates; "
                "a targeted run with fresh/stale gradient comparisons is required."
                if metrics_path.is_file()
                else f"Missing metrics file: {metrics_path}"
            )
            axes[0].text(
                0.5, 0.55, "NOT YET AVAILABLE",
                ha="center", va="center", transform=axes[0].transAxes,
                fontsize=18, color="#9b3d3d", weight="bold",
            )
            axes[1].text(
                0.5, 0.55, reason,
                ha="center", va="center", transform=axes[1].transAxes,
                wrap=True, fontsize=11,
            )
            fig.suptitle(f"Are stale gradients misdirected?\n{Path(run_id).name}")
        fig.tight_layout(rect=(0, 0.07, 1, 0.91))
        pdf.savefig(fig)
        plt.close(fig)
    return destination


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
    rows = [_add_legacy_fixed_frequency_aliases(row) for row in _read_metrics(run_dir)]
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
        [
            "restricted_loss_fixed", "restricted_loss_discovered",
            "train_loss", "test_loss",
        ],
        "Restricted loss: fixed vs. discovered keys",
        "Cross-entropy loss",
        figures_dir / "restricted_loss_keys.png",
        log_y=True,
    )
    _plot_series(
        rows,
        ["excluded_loss_fixed", "excluded_loss_discovered"],
        "Excluded loss: fixed vs. discovered keys",
        "Cross-entropy loss",
        figures_dir / "excluded_loss_keys.png",
        log_y=True,
    )
    _plot_series(
        rows,
        [
            "restricted_train_accuracy_fixed", "restricted_test_accuracy_fixed",
            "restricted_train_accuracy_discovered",
            "restricted_test_accuracy_discovered",
        ],
        "Restricted accuracy: fixed vs. discovered keys",
        "Accuracy",
        figures_dir / "restricted_accuracy_keys.png",
    )
    _plot_series(
        rows,
        [
            "excluded_train_accuracy_fixed", "excluded_test_accuracy_fixed",
            "excluded_train_accuracy_discovered",
            "excluded_test_accuracy_discovered",
        ],
        "Excluded accuracy: fixed vs. discovered keys",
        "Accuracy",
        figures_dir / "excluded_accuracy_keys.png",
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
        ["aggregated_momentum_l2_norm"],
        "L2 norm of aggregated AdamW first moment",
        "L2 norm of exp_avg",
        figures_dir / "aggregated_momentum_l2_norm.png",
    )
    if (run_dir / "node_metrics.jsonl").is_file():
        plot_gradient_alignment(run_dir)
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


def _plot_distributed_gradient_norms(
    rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    per_node: dict[int, list[tuple[int, float]]] = defaultdict(list)
    by_delay_status: dict[bool, dict[int, list[float]]] = {
        False: defaultdict(list),
        True: defaultdict(list),
    }
    for row in rows:
        value = row.get("gradient_l2_norm")
        if not isinstance(value, (int, float)):
            continue
        step = int(row["step"])
        node = int(row["node"])
        per_node[node].append((step, float(value)))
        has_delay = bool(row.get("has_delay", int(row.get("staleness", 0)) > 0))
        by_delay_status[has_delay][step].append(float(value))

    if per_node:
        fig, ax = plt.subplots(figsize=(10, 6))
        for node, points in sorted(per_node.items()):
            ax.plot(
                [step for step, _ in points],
                [value for _, value in points],
                linewidth=0.9,
                label=f"Node {node}",
            )
        ax.set(
            title="Per-node full-batch gradient norms",
            xlabel="Global step",
            ylabel="Gradient L2 norm",
        )
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "gradient_norms_by_node.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for has_delay, label in ((False, "No delay"), (True, "Has delay")):
            points = [
                (step, sum(values) / len(values))
                for step, values in sorted(by_delay_status[has_delay].items())
            ]
            if points:
                ax.plot(
                    [step for step, _ in points],
                    [value for _, value in points],
                    marker=".",
                    markersize=3,
                    label=label,
                )
        ax.set(
            title="Mean node gradient norm by delay status",
            xlabel="Global step",
            ylabel="Mean gradient L2 norm",
        )
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "gradient_norms_by_delay_status.png", dpi=160)
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
    _plot_distributed_gradient_norms(rows, run_dir / "figures")
    nodes = sorted({int(row["node"]) for row in rows})
    for node in nodes:
        node_rows = []
        for row in rows:
            if int(row["node"]) != node:
                continue
            normalized = dict(row)
            measures = normalized.get("progress_measures")
            if isinstance(measures, dict):
                normalized["progress_measures"] = _add_legacy_fixed_frequency_aliases(
                    measures
                )
            node_rows.append(normalized)
        node_rows.sort(key=lambda row: int(row["step"]))
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
            (
                "gradient_l2_norm.png",
                ("gradient_l2_norm",),
                "Full-batch gradient L2 norm",
                "Gradient L2 norm",
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
                "restricted_loss_keys.png",
                (
                    "restricted_loss_fixed", "restricted_loss_discovered",
                    "restricted_train_loss_fixed",
                    "restricted_train_loss_discovered",
                ),
                "Restricted loss: fixed vs. discovered keys",
                "Cross-entropy loss",
                True,
            ),
            (
                "excluded_loss_keys.png",
                ("excluded_loss_fixed", "excluded_loss_discovered"),
                "Excluded loss: fixed vs. discovered keys",
                "Cross-entropy loss",
                True,
            ),
            (
                "restricted_accuracy_keys.png",
                (
                    "restricted_train_accuracy_fixed",
                    "restricted_test_accuracy_fixed",
                    "restricted_train_accuracy_discovered",
                    "restricted_test_accuracy_discovered",
                ),
                "Restricted accuracy: fixed vs. discovered keys",
                "Accuracy",
                False,
            ),
            (
                "excluded_accuracy_keys.png",
                (
                    "excluded_train_accuracy_fixed",
                    "excluded_test_accuracy_fixed",
                    "excluded_train_accuracy_discovered",
                    "excluded_test_accuracy_discovered",
                ),
                "Excluded accuracy: fixed vs. discovered keys",
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
