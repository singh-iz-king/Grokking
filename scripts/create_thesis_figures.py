from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from src.plotting import plot_gradient_alignment
from matplotlib.colors import ListedColormap

NA = "NA"
STABLE_WINDOW_STEPS = 1000
FIXED_KEYS = (14, 35, 41, 42, 52)
CATEGORY_COLORS = {
    "sustained": "#4c9f70",
    "collapse": "#d97941",
    "never": "#657b91",
}
CATEGORY_LABELS = {
    "sustained": "Sustained >=99%",
    "collapse": "Reaches 99%, then falls below",
    "never": "Never reaches 99%",
}


@dataclass
class Run:
    path: Path
    config: dict[str, Any]
    metrics: list[dict[str, Any]]
    spectra: list[dict[str, str]]
    run_id: str
    seed: Any
    nodes: int
    delayed_nodes: int
    delay: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_runs(results_root: Path) -> list[Run]:
    runs: list[Run] = []
    for config_path in sorted(results_root.rglob("config.yaml")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        distributed = config.get("distributed", {})
        if not distributed.get("enabled") or distributed.get("staleness_mode") != "mixed_fixed":
            continue
        nodes = int(distributed["nodes"])
        zero_delay_nodes = int(distributed["zero_delay_nodes"])
        spectra_path = run_dir / "fourier_spectra.csv"
        spectra: list[dict[str, str]] = []
        if spectra_path.is_file():
            with spectra_path.open(newline="", encoding="utf-8") as handle:
                spectra = list(csv.DictReader(handle))
        runtime_run_dir = config.get("runtime", {}).get("run_dir")
        run_id = Path(runtime_run_dir).name if runtime_run_dir else run_dir.name
        runs.append(Run(
            path=run_dir,
            config=config,
            metrics=_read_jsonl(metrics_path),
            spectra=spectra,
            run_id=run_id,
            seed=config.get("experiment", {}).get("seed", NA),
            nodes=nodes,
            delayed_nodes=nodes - zero_delay_nodes,
            delay=int(distributed["tau"]),
        ))
    return runs


def _eval_series(run: Run, key: str) -> list[tuple[int, float]]:
    points = []
    for row in run.metrics:
        value = row.get(key)
        step = row.get("step", row.get("epoch"))
        if isinstance(value, (int, float)) and step is not None:
            points.append((int(step), float(value)))
    return sorted(points)


def _short_run_label(run: Run) -> str:
    relative = "/".join(run.path.parts[-2:])
    return f"{relative} (seed={run.seed})"


def _first_at_least(points: list[tuple[int, float]], threshold: float) -> int | None:
    return next((step for step, value in points if value >= threshold), None)


def _stable_start(points: list[tuple[int, float]], window_steps: int) -> int | None:
    start: int | None = None
    previous: int | None = None
    for step, value in points:
        if value < 0.99 or (previous is not None and step - previous > 1):
            start = None
        if value >= 0.99:
            if start is None:
                start = step
            if step - start >= window_steps:
                return start
        previous = step
    return None


def _category(run: Run) -> str:
    test = _eval_series(run, "test_accuracy")
    stable = _stable_start(test, STABLE_WINDOW_STEPS)
    if stable is not None:
        return "sustained"
    first = _first_at_least(test, 0.99)
    if first is None:
        return "never"
    expected = int(run.config.get("experiment", {}).get("epochs", -1))
    if not test or test[-1][0] < expected:
        return "incomplete"
    if any(step > first and accuracy < 0.99 for step, accuracy in test):
        return "collapse"
    return "incomplete"


def _collapse_recoveries(run: Run) -> list[tuple[int, int, int]]:
    points = _eval_series(run, "test_accuracy")
    first_99 = _first_at_least(points, 0.99)
    if first_99 is None:
        return []
    events: list[tuple[int, int, int]] = []
    event_start: int | None = None
    event_end: int | None = None
    for step, accuracy in points:
        if step < first_99:
            continue
        if accuracy < 0.90:
            if event_start is None:
                event_start = step
            event_end = step
        elif event_start is not None and event_end is not None:
            recovery = next(
                (
                    later_step
                    for later_step, later_accuracy in points
                    if later_step > event_end and later_accuracy >= 0.99
                ),
                None,
            )
            if recovery is not None:
                events.append((event_start, event_end, recovery))
            event_start = None
            event_end = None
    return events


def _select_representative_collapse(run: Run) -> tuple[int, int, int]:
    events = _collapse_recoveries(run)
    for event in events:
        start, end, recovery = event
        if end - start >= 20 and recovery - end <= 300:
            return event
    if events:
        return events[0]
    raise ValueError(
        f"No test-accuracy collapse with recovery found in requested run {run.path}"
    )


def _metric_plot(
    ax: plt.Axes,
    run: Run,
    key: str,
    label: str,
    *,
    start: int | None = None,
    end: int | None = None,
    marker: str = ".",
    linewidth: float = 1.0,
    color: str | None = None,
    linestyle: str = "-",
) -> None:
    points = _eval_series(run, key)
    if start is not None:
        points = [point for point in points if point[0] >= start]
    if end is not None:
        points = [point for point in points if point[0] <= end]
    if points:
        ax.plot(
            [step for step, _ in points],
            [value for _, value in points],
            label=label,
            marker=marker,
            markersize=2.5,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
        )


def _find_run(runs: list[Run], *, nodes: int, delayed: int, delay: int) -> Run:
    matches = [
        run for run in runs
        if run.nodes == nodes and run.delayed_nodes == delayed and run.delay == delay
    ]
    if not matches:
        raise ValueError(
            f"Missing required run: nodes={nodes}, delayed_workers={delayed}, delay={delay}"
        )
    return sorted(matches, key=lambda run: str(run.seed))[0]


def _write_placeholder_pdf(path: Path, title: str, message: str) -> None:
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.72, title, ha="center", va="center", fontsize=20, weight="bold")
        ax.text(
            0.5, 0.51, "NOT YET AVAILABLE", ha="center", va="center",
            fontsize=16, color="#9b3d3d", weight="bold",
        )
        ax.text(
            0.5, 0.34, message, ha="center", va="center",
            fontsize=12, wrap=True,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _phase_master_rows(runs: list[Run]) -> list[dict[str, Any]]:
    master: list[dict[str, Any]] = []
    for run in runs:
        test = _eval_series(run, "test_accuracy")
        train = _eval_series(run, "train_accuracy")
        first_test = _first_at_least(test, 0.99)
        first_train = _first_at_least(train, 0.99)
        master.append({
            "run_id": run.run_id,
            "seed": run.seed,
            "nodes": run.nodes,
            "m_delayed": run.delayed_nodes,
            "m_over_6": run.delayed_nodes / 6 if run.nodes == 6 else NA,
            "delay_steps": run.delay,
            "category": _category(run),
            "first_test99_step": first_test if first_test is not None else NA,
            "first_train99_step": first_train if first_train is not None else NA,
            "stable99_start_step": (
                _stable_start(test, STABLE_WINDOW_STEPS)
                if _stable_start(test, STABLE_WINDOW_STEPS) is not None else NA
            ),
            "final_test_accuracy": test[-1][1] if test else NA,
        })
    return master


def _write_master_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _phase_diagram(runs: list[Run], output: Path, master_csv: Path) -> None:
    eligible = [run for run in runs if run.nodes == 6]
    if not eligible:
        _write_placeholder_pdf(
            output,
            "Figure 1. Phase diagram: where is generalization stable?",
            "No six-node mixed fixed-delay runs were found in the results directory.",
        )
        _write_master_csv(master_csv, [])
        return

    delays = sorted({run.delay for run in eligible})
    categories = ("sustained", "collapse", "never")
    colors = [CATEGORY_COLORS[item] for item in categories]
    x_values = list(range(7))
    category_code = {category: index for index, category in enumerate(categories)}
    cells: dict[tuple[int, int], list[Run]] = {}
    for run in eligible:
        cells.setdefault((run.delay, run.delayed_nodes), []).append(run)

    master = _phase_master_rows(eligible)
    _write_master_csv(master_csv, master)
    cmap = ListedColormap(colors)
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for y_index, delay in enumerate(delays):
            for delayed in x_values:
                group = cells.get((delay, delayed), [])
                x = delayed / 6
                if not group:
                    ax.add_patch(plt.Rectangle(
                        (x - 1 / 12, y_index - 0.45),
                        1 / 6, 0.9, facecolor="#eeeeee",
                        edgecolor="white", hatch="///",
                    ))
                    ax.text(x, y_index, "missing", ha="center", va="center", fontsize=7)
                    continue
                counts = {category: 0 for category in categories}
                for run in group:
                    outcome = _category(run)
                    if outcome in counts:
                        counts[outcome] += 1
                majority = max(categories, key=lambda category: counts[category])
                category_index = category_code[majority]
                ax.add_patch(plt.Rectangle(
                    (x - 1 / 12, y_index - 0.45),
                    1 / 6, 0.9, facecolor=cmap(category_index),
                    edgecolor="white",
                ))
                ax.text(
                    x, y_index,
                    f"S:{counts['sustained']} C:{counts['collapse']} N:{counts['never']}\nn={len(group)}",
                    ha="center", va="center", fontsize=7,
                )
        ax.set(
            title="Figure 1. Where is generalization stable?",
            xlabel="Delayed workers m / 6",
            ylabel="Fixed delay d (global optimizer steps)",
            xticks=[value / 6 for value in x_values],
            xticklabels=[f"{value}/6" for value in x_values],
            yticks=list(range(len(delays))),
            yticklabels=[str(value) for value in delays],
            xlim=(-1 / 12, 1 + 1 / 12),
            ylim=(-0.6, len(delays) - 0.4),
        )
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[key], label=CATEGORY_LABELS[key])
            for key in categories
        ]
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="#eeeeee", hatch="///", label="Missing configuration")
        )
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(axis="x", alpha=0.15)
        fig.text(
            0.01, 0.01,
            f"Classification: sustained if test accuracy has {STABLE_WINDOW_STEPS} consecutive measured global steps >=0.99; "
            "collapse if it first reaches 0.99 and later drops below 0.99; otherwise never. "
            "Cell labels show seed counts S/C/N and total n. All series use global optimizer steps. "
            f"Only observed delays are shown: {', '.join(map(str, delays))}; other delay values were not run.",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.06, 0.8, 1))
        pdf.savefig(fig)
        plt.close(fig)

        table_fig, table_ax = plt.subplots(figsize=(14, 8.5))
        table_ax.axis("off")
        columns = [
            "run_id", "seed", "m_delayed", "m_over_6", "delay_steps",
            "category", "first_test99_step", "stable99_start_step",
            "final_test_accuracy",
        ]
        table_values = [
            [
                f"d{row['delay_steps']}_m{row['m_delayed']}_s{row['seed']}"
                if key == "run_id" else str(row[key])
                for key in columns
            ]
            for row in master
        ]
        table_ax.set_title("Numerical master table attached to Figure 1", fontsize=15, pad=20)
        table_ax.text(
            0.5, 0.92,
            "Compact run labels are d<delay>_m<delayed workers>_s<seed>; full run IDs are in phase_diagram_master_table.csv.",
            ha="center", va="center", transform=table_ax.transAxes, fontsize=8,
        )
        table = table_ax.table(
            cellText=table_values,
            colLabels=columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
        pdf.savefig(table_fig, bbox_inches="tight")
        plt.close(table_fig)


def _collapse_diagnostics(run: Run, output: Path) -> tuple[int, int, int]:
    start_event, end_event, recovery = _select_representative_collapse(run)
    center = (start_event + end_event) // 2
    left = max(0, min(start_event - 300, center - 300))
    right = max(recovery + 20, center + 300)
    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
    panels = (
        ("accuracy", ("train_accuracy", "test_accuracy"), "Accuracy"),
        ("loss", ("train_loss", "test_loss"), "Cross-entropy loss"),
    )
    for index, (title, fields, ylabel) in enumerate(panels):
        for field in fields:
            _metric_plot(axes[index], run, field, field.replace("_", " "), start=left, end=right)
        axes[index].set_ylabel(ylabel)
        axes[index].legend(loc="best", fontsize=8)
        axes[index].grid(alpha=0.2)

    for ax, title, explanation in (
        (
            axes[2], "Aggregated gradient norm",
            "NOT AVAILABLE: only per-node gradient norms are logged; gradient vectors are not saved, "
            "so the norm of their mean cannot be recovered.",
        ),
        (
            axes[4], "Actual parameter-update norm",
            "NOT AVAILABLE in this event window: parameter snapshots are not saved at adjacent steps "
            "here, and no update-norm series is logged.",
        ),
    ):
        ax.axis("off")
        ax.text(0.5, 0.66, title, ha="center", va="center", transform=ax.transAxes, weight="bold")
        ax.text(0.5, 0.42, explanation, ha="center", va="center", transform=ax.transAxes, wrap=True)
        ax.text(0.5, 0.12, f"Global steps {left}–{right}", ha="center", transform=ax.transAxes)

    _metric_plot(
        axes[3], run, "aggregated_momentum_l2_norm",
        "Central AdamW exp_avg L2 norm", start=left, end=right,
    )
    axes[3].set_ylabel("L2 norm")
    axes[3].legend(loc="best", fontsize=8)
    axes[3].grid(alpha=0.2)

    for key, label in (
        ("embedding_fourier_gini", "Embedding Fourier Gini"),
        ("logit_map_fourier_gini", "Output-map Fourier Gini"),
    ):
        _metric_plot(
            axes[5], run, key, label, start=left, end=right,
            marker="o", linewidth=1,
        )
    axes[5].set_ylabel("Gini")
    axes[5].set_xlabel("Global optimizer step")
    axes[5].legend(loc="best", fontsize=8)
    axes[5].grid(alpha=0.2)

    for ax in axes:
        if ax.axison:
            ax.axvspan(start_event, end_event, color="#d97941", alpha=0.18)
            ax.axvline(recovery, color="#4c9f70", linestyle="--", linewidth=1)
    fig.suptitle(
        f"Figure 2. Collapse diagnostics: {_short_run_label(run)}\n"
        f"6/6 delayed workers, d=5; collapse {start_event}–{end_event}, "
        f"first measured recovery to >=99% at {recovery}",
        fontsize=13,
    )
    fig.text(
        0.01, 0.01,
        "Points mark actual logged observations; losses/accuracies/momentum are logged per step, "
        "Fourier Gini every 100 steps. Shading marks test accuracy <0.90; dashed line marks recovery. "
        "This figure shows temporal alignment, not causality.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return start_event, end_event, recovery


def _spectrum_amplitudes(run: Run, matrix: str, keys: set[int], left: int, right: int):
    components: dict[tuple[int, str], dict[str, float]] = {}
    for row in run.spectra:
        if row.get("matrix") != matrix:
            continue
        step = int(row["step"])
        if step < left or step > right:
            continue
        component = row["component"]
        if component == "DC":
            continue
        if "_" not in component:
            continue
        kind, raw_frequency = component.split("_", 1)
        if kind not in {"cos", "sin"}:
            continue
        frequency = int(raw_frequency)
        if frequency not in keys:
            continue
        item = components.setdefault((step, frequency), {})
        item[kind] = float(row["norm"])
    values: dict[int, list[tuple[int, float]]] = {}
    for (step, frequency), parts in components.items():
        amplitude = math.sqrt(parts.get("cos", 0.0) ** 2 + parts.get("sin", 0.0) ** 2)
        values.setdefault(frequency, []).append((step, amplitude))
    return {frequency: sorted(points) for frequency, points in values.items()}


def _detected_keys_at(run: Run, step: int) -> set[int]:
    row = next(
        (record for record in run.metrics if int(record.get("step", -1)) == step),
        None,
    )
    if row is None:
        return set()
    return {int(value) for value in row.get("detected_key_frequencies", [])}


def _circuit_dynamics(run: Run, collapse: tuple[int, int, int], output: Path) -> None:
    event_start, event_end, recovery = collapse
    center = (event_start + event_end) // 2
    left = max(0, min(event_start - 300, center - 300))
    right = max(recovery + 20, center + 300)
    event_rows = [
        row for row in run.metrics
        if left <= int(row.get("step", -1)) <= right
    ]
    fixed_keys = set(FIXED_KEYS)
    discovered_keys = {
        int(frequency)
        for row in event_rows
        for frequency in row.get("detected_key_frequencies", [])
    }
    union = fixed_keys | discovered_keys

    fig, axes = plt.subplots(
        5, 1, figsize=(12, 15), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.25, 1.15, 1.3, 0.9]},
    )
    for matrix, ax, title in (
        ("embedding", axes[0], "(a1) Embedding Fourier amplitudes"),
        ("logit_map", axes[1], "(a2) Neuron-to-logit Fourier amplitudes"),
    ):
        amplitudes = _spectrum_amplitudes(run, matrix, union, left, right)
        for frequency, points in sorted(amplitudes.items()):
            fixed = frequency in fixed_keys
            if not fixed:
                points = [
                    (step, amplitude)
                    for step, amplitude in points
                    if frequency in _detected_keys_at(run, step)
                ]
            if not points:
                continue
            line_style = "-" if fixed else "--"
            label = f"k={frequency} ({'fixed' if fixed else 'discovered'})"
            ax.plot(
                [step for step, _ in points],
                [value for _, value in points],
                linestyle=line_style,
                marker=".",
                markersize=2,
                linewidth=0.9,
                label=label,
            )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("sqrt(cos norm² + sin norm²)")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(ncol=4, fontsize=6, loc="upper left")

    coefficient_series: dict[int, list[tuple[int, float, bool]]] = {}
    for row in event_rows:
        values = row.get("key_logit_coefficients")
        step = int(row.get("step", -1))
        if isinstance(values, dict):
            for frequency, value in values.items():
                k = int(frequency)
                if k in union and isinstance(value, (int, float)):
                    coefficient_series.setdefault(k, []).append(
                        (step, float(value), True)
                    )
        discovered_values = row.get("discovered_key_logit_coefficients")
        if isinstance(discovered_values, dict):
            for frequency, value in discovered_values.items():
                k = int(frequency)
                if k not in fixed_keys and isinstance(value, (int, float)):
                    coefficient_series.setdefault(k, []).append(
                        (step, float(value), False)
                    )
    for frequency, points in sorted(coefficient_series.items()):
        ax = axes[2]
        ax.plot(
            [step for step, _, _ in points],
            [value for _, value, _ in points],
            marker=".",
            markersize=2,
            linewidth=0.9,
            linestyle="-" if frequency in fixed_keys else "--",
            label=f"k={frequency} ({'fixed' if frequency in fixed_keys else 'discovered'})",
        )
    axes[2].set_title("(b) Logged functional cosine-template projections (not full phase)", loc="left", fontsize=10)
    axes[2].set_ylabel("Projection coefficient")
    axes[2].grid(alpha=0.2)
    if coefficient_series:
        axes[2].legend(ncol=4, fontsize=6, loc="upper left")
    else:
        axes[2].text(
            0.5, 0.5, "NOT AVAILABLE: no functional Fourier coefficients logged",
            ha="center", va="center", transform=axes[2].transAxes,
        )

    _metric_plot(axes[3], run, "test_accuracy", "Original test accuracy", start=left, end=right)
    _metric_plot(
        axes[3], run, "excluded_test_accuracy_fixed",
        "All fixed keys excluded", start=left, end=right, marker="o",
    )
    _metric_plot(
        axes[3], run, "excluded_test_accuracy_discovered",
        "All discovered keys excluded", start=left, end=right, marker="s",
    )
    per_frequency: dict[int, list[tuple[int, float]]] = {}
    for row in event_rows:
        excluded = row.get("excluded_loss_by_frequency")
        step = int(row.get("step", -1))
        if not isinstance(excluded, dict):
            continue
        for frequency, values in excluded.items():
            if isinstance(values, dict) and isinstance(values.get("test_accuracy"), (int, float)):
                per_frequency.setdefault(int(frequency), []).append(
                    (step, float(values["test_accuracy"]))
                )
    for frequency, points in sorted(per_frequency.items()):
        axes[3].plot(
            [step for step, _ in points],
            [value for _, value in points],
            linestyle=":",
            marker=".",
            markersize=2,
            label=f"Exclude k={frequency}",
        )
    axes[3].set_title("(c) Original and frequency-ablated test accuracy", loc="left", fontsize=10)
    axes[3].set_ylabel("Test accuracy")
    axes[3].set_ylim(-0.03, 1.03)
    axes[3].grid(alpha=0.2)
    axes[3].legend(ncol=3, fontsize=6, loc="lower left")

    axes[4].axis("off")
    axes[4].text(0.5, 0.68, "(d) Classification margins", ha="center", va="center", weight="bold")
    axes[4].text(
        0.5, 0.38,
        "NOT AVAILABLE around this collapse: raw per-example logits/margins are not logged, "
        "and this run has no saved checkpoint at the event steps. A final checkpoint cannot recover "
        "the historical margin trajectory.",
        ha="center", va="center", wrap=True,
    )

    for ax in axes:
        if ax.axison:
            ax.axvspan(event_start, event_end, color="#d97941", alpha=0.14)
            ax.axvline(recovery, color="#4c9f70", linestyle="--", linewidth=0.9)
    axes[-1].set_xlabel("Global optimizer step")
    fig.suptitle(
        f"Figure 4. Fourier circuit diagnostics (partial where marked unavailable)\n"
        f"{_short_run_label(run)}\n"
        f"6/6 delayed workers, d=5; collapse {event_start}–{event_end}, recovery {recovery}",
        fontsize=13,
    )
    fig.text(
        0.01, 0.01,
        "Fourier amplitudes are combined real-basis cos/sin component norms; fixed keys are solid, "
        "checkpoint-discovered keys are dashed. Signed cosine-template projections are not full phases; "
        "embedding phase is not treated as functionally invariant.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.09, hspace=0.48)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def _momentum_spikes(points: list[tuple[int, float]]) -> tuple[list[tuple[int, float]], float]:
    """Detect local peaks with 50-step prominence windows and robust-range cutoff."""
    if len(points) < 3:
        return [], 0.0
    steps = np.asarray([step for step, _ in points], dtype=int)
    values = np.asarray([value for _, value in points], dtype=float)
    prominence_threshold = 0.05 * float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    window = 50
    candidates: list[tuple[int, float, float]] = []
    for index in range(1, len(values) - 1):
        if values[index] < values[index - 1] or values[index] <= values[index + 1]:
            continue
        left = values[max(0, index - window):index]
        right = values[index + 1:min(len(values), index + window + 1)]
        if left.size == 0 or right.size == 0:
            continue
        prominence = values[index] - max(float(left.min()), float(right.min()))
        if prominence >= prominence_threshold:
            candidates.append((index, prominence, values[index]))
    selected: list[tuple[int, float]] = []
    for index, prominence, value in sorted(
        candidates, key=lambda candidate: (candidate[1], candidate[2]), reverse=True
    ):
        if all(abs(int(steps[index]) - step) >= 50 for step, _ in selected):
            selected.append((int(steps[index]), float(value)))
    return sorted(selected), prominence_threshold


def _oscillation_analysis(runs: list[Run], output: Path) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 8), sharex="col", sharey="row"
    )
    report_lines: list[str] = []
    for column, delay in enumerate((5, 10)):
        run = _find_run(runs, nodes=6, delayed=4, delay=delay)
        points = _eval_series(run, "aggregated_momentum_l2_norm")
        peaks, threshold = _momentum_spikes(points)
        times = [step for step, _ in peaks]
        intervals = np.diff(times)
        mean = float(np.mean(intervals)) if len(intervals) else math.nan
        std = float(np.std(intervals)) if len(intervals) else math.nan
        axes[0, column].plot(
            [step for step, _ in points],
            [value for _, value in points],
            linewidth=0.65,
            color="#355c7d",
            label="Central AdamW exp_avg L2 norm",
        )
        if peaks:
            axes[0, column].scatter(
                times, [value for _, value in peaks],
                marker="x", color="#c44e52", s=20, label="Detected momentum spike",
                zorder=3,
            )
        axes[0, column].set_title(
            f"4/6 delayed workers, d={delay}, seed={run.seed}\n{_short_run_label(run)}",
            fontsize=9,
        )
        axes[0, column].set_ylabel("First-moment L2 norm")
        axes[0, column].legend(fontsize=7)
        axes[0, column].grid(alpha=0.2)

        if len(intervals):
            axes[1, column].plot(
                times[1:], intervals, marker="o", markersize=3,
                linewidth=0.8, label="Inter-peak interval",
            )
        axes[1, column].set_xlabel("Global optimizer step (later peak)")
        axes[1, column].set_ylabel("Interval (global steps)")
        axes[1, column].grid(alpha=0.2)
        axes[1, column].text(
            0.02, 0.96,
            f"peak events={len(peaks)}; intervals={len(intervals)}\n"
            f"mean={mean:.1f} steps; SD={std:.1f} steps\n"
            f"prominence threshold={threshold:.4g}",
            transform=axes[1, column].transAxes,
            va="top",
            fontsize=8,
        )

        checkpoint_interval = int(run.config.get("logging", {}).get("checkpoint_interval", 0))
        mechanistic_interval = int(run.config.get("logging", {}).get("mechanistic_metrics_interval", 0))
        near_checkpoint = sum(
            1 for step in times
            if checkpoint_interval and min(step % checkpoint_interval, checkpoint_interval - step % checkpoint_interval) <= 1
        )
        near_mechanistic = sum(
            1 for step in times
            if mechanistic_interval and min(step % mechanistic_interval, mechanistic_interval - step % mechanistic_interval) <= 1
        )
        report_lines.append(
            f"d={delay}: events={len(peaks)}, intervals={len(intervals)}, "
            f"mean={mean:.6g}, SD={std:.6g}, near checkpoint cadence ±1={near_checkpoint}, "
            f"near mechanistic cadence ±1={near_mechanistic}"
        )

    fig.suptitle("Figure 5. Is there a genuine period? Momentum-spike intervals", fontsize=14)
    fig.text(
        0.01, 0.01,
        "Peak rule: strict local maxima; prominence is peak minus the larger neighboring minimum within 50 steps; "
        "threshold is 5% of the series' 95th–5th percentile range; retain strongest peaks >=50 steps apart.\n"
        "Mean/SD are population statistics of adjacent intervals. Aggregation, AdamW/scheduler and gradient-buffer "
        "zeroing run every step; mechanistic analysis is scheduled every 100 steps; checkpoints every 1,000 by config.\n"
        "Checkpoint files are absent in these run directories. Cadence coincidence alone is not evidence of a limit cycle.",
        fontsize=7.2,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.24, hspace=0.3, wspace=0.18)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Oscillation peak summary:")
    for line in report_lines:
        print(f"  {line}")


def create_figures(results_root: str | Path, output_dir: str | Path) -> list[Path]:
    root = Path(results_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {root}")
    destination.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(root)
    if not runs:
        raise ValueError(f"No mixed fixed-delay distributed runs found under {root}")

    generated = [
        destination / "phase_diagram.pdf",
        destination / "collapse_diagnostics.pdf",
        destination / "gradient_alignment.pdf",
        destination / "fourier_circuit_dynamics.pdf",
        destination / "oscillation_analysis.pdf",
    ]
    master_path = destination / "phase_diagram_master_table.csv"
    _phase_diagram(runs, generated[0], master_path)
    collapse_run = _find_run(runs, nodes=6, delayed=6, delay=5)
    event = _collapse_diagnostics(collapse_run, generated[1])
    plot_gradient_alignment(collapse_run.path, generated[2])
    _circuit_dynamics(collapse_run, event, generated[3])
    _oscillation_analysis(runs, generated[4])
    if len(generated) != 5:
        raise RuntimeError("Exactly five figure paths must be generated")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the five thesis figures from saved distributed results."
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Root folder containing completed run directories (default: results).",
    )
    parser.add_argument(
        "--output",
        default="results/thesis_figures",
        help="Output directory for five PDFs and the Figure 1 master CSV.",
    )
    args = parser.parse_args()
    try:
        paths = create_figures(args.results, args.output)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))
    print("Generated five figures:")
    for path in paths:
        print(f"  {path}")
    print(f"Numerical master table: {Path(args.output) / 'phase_diagram_master_table.csv'}")


if __name__ == "__main__":
    main()
