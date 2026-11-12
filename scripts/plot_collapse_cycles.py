from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("collapse_cycles")
METRICS = (
    "delayed_gradient_relative_discrepancy_mean",
    "delayed_gradient_cosine_mean",
    "aggregated_momentum_l2_norm",
    "test_accuracy",
)
METRIC_COLORS = ("#e28e2c", "#8172b2", "#c44e52", "#4c8bc8")
METRIC_LABELS = (
    "Gradient relative discrepancy",
    "Gradient cosine similarity",
    "Momentum-state L2 norm",
    "Test accuracy",
)
NA = "NA"


@dataclass(frozen=True)
class Run:
    path: Path
    run_id: str
    seed: str
    config: dict[str, Any]
    settings_id: str
    settings_label: str
    metrics: pd.DataFrame
    metrics_source: str


@dataclass(frozen=True)
class CollapseEvent:
    event_id: str
    settings_id: str
    settings_label: str
    run_id: str
    seed: str
    onset_step: int
    trough_step: int
    trough_accuracy: float
    trough_delay_steps: int
    recovery_step: int | None
    recovery_duration: int | None
    next_onset_step: int | None
    next_trough_step: int | None
    next_trough_accuracy: float | None
    next_recovery_step: int | None
    next_recovery_duration: int | None
    intercollapse_steps: int | None
    stable_plateau_steps: int | None
    discrepancy_crossing_step: int | None
    discrepancy_lead_steps: int | None
    cosine_crossing_step: int | None
    cosine_lead_steps: int | None


def _read_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_json(path, lines=True)
    if "step" not in frame.columns and "epoch" not in frame.columns:
        raise ValueError("metrics file has neither step nor epoch column")
    if "step" not in frame.columns:
        frame["step"] = frame["epoch"]
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame = frame.dropna(subset=["step"]).copy()
    frame["step"] = frame["step"].astype(int)
    frame = frame.sort_values("step").drop_duplicates("step", keep="last")
    for metric in METRICS:
        if metric not in frame:
            frame[metric] = np.nan
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    accuracy = frame["test_accuracy"]
    finite_accuracy = accuracy[np.isfinite(accuracy)]
    if not finite_accuracy.empty and finite_accuracy.max() > 1.5:
        frame["test_accuracy"] = accuracy / 100.0
    return frame.reset_index(drop=True)


def _normalize_settings(config: dict[str, Any], grouping: str) -> dict[str, Any]:
    settings = json.loads(json.dumps(config, sort_keys=True, default=str))
    settings.pop("runtime", None)
    settings.pop("output", None)
    experiment = settings.get("experiment", {})
    if isinstance(experiment, dict):
        experiment.pop("seed", None)
    if grouping == "optimizer-delay":
        settings = {
            "optimizer": settings.get("optimizer"),
            "distributed": settings.get("distributed"),
            "experiment": {
                key: value
                for key, value in settings.get("experiment", {}).items()
                if key in {"name", "modulus", "train_fraction", "epochs"}
            },
            "model": settings.get("model"),
            "training": settings.get("training"),
        }
    return settings


def _settings_label(config: dict[str, Any]) -> str:
    optimizer = config.get("optimizer", {})
    opt_name = str(optimizer.get("name", "optimizer"))
    if opt_name.lower() == "adamw":
        opt_name += (
            f"(b1={optimizer.get('beta1', '?')},"
            f"b2={optimizer.get('beta2', '?')})"
        )
    elif opt_name.lower() == "sgd":
        opt_name += f"(mom={optimizer.get('momentum', 0.0)})"
    distributed = config.get("distributed", {})
    if distributed.get("enabled"):
        nodes = distributed.get("nodes", "?")
        zero = distributed.get("zero_delay_nodes")
        delayed = nodes - zero if isinstance(zero, int) else "?"
        tau = distributed.get("tau", "?")
        delay_label = f"{delayed}/{nodes} delayed, d={tau}"
    else:
        delay_label = "single-process baseline"
    lr = optimizer.get("learning_rate", "?")
    wd = optimizer.get("weight_decay", "?")
    experiment = config.get("experiment", {})
    modulus = experiment.get("modulus", "?")
    epochs = experiment.get("epochs", "?")
    experiment_name = config.get("experiment", {}).get("name")
    model = config.get("model", {})
    label = (
        f"p={modulus}; {opt_name}; lr={lr}; wd={wd}; {delay_label}; "
        f"{model.get('n_layers', '?')}L/{model.get('n_heads', '?')}H/"
        f"d{model.get('d_model', '?')}/MLP{model.get('d_mlp', '?')}; "
        f"steps={epochs}"
    )
    return f"{experiment_name}; {label}" if experiment_name else label


def discover_runs(
    results_dir: Path,
    output_dir: Path,
    grouping: str,
) -> tuple[list[Run], list[dict[str, str]]]:
    runs: list[Run] = []
    skipped: list[dict[str, str]] = []
    output_resolved = output_dir.resolve()
    run_dirs = {
        candidate.parent
        for metrics_name in ("metrics.csv", "metrics.jsonl")
        for candidate in results_dir.rglob(metrics_name)
    }
    for run_dir in sorted(run_dirs):
        if run_dir.resolve() == output_resolved or output_resolved in run_dir.resolve().parents:
            continue
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            metrics_path = run_dir / "metrics.jsonl"
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            skipped.append({
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "reason": "missing config.yaml; settings and seed are ambiguous",
            })
            continue
        try:
            with config_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if not isinstance(config, dict):
                raise ValueError("config.yaml is not a mapping")
            metrics = _read_metrics(metrics_path)
            experiment = config.get("experiment", {})
            seed_value = experiment.get("seed")
            if seed_value is None:
                raise ValueError("config has no experiment.seed")
            settings = _normalize_settings(config, grouping)
            settings_json = json.dumps(settings, sort_keys=True, separators=(",", ":"))
            settings_id = hashlib.sha1(settings_json.encode()).hexdigest()[:10]
            runtime_path = config.get("runtime", {}).get("run_dir")
            run_id = Path(runtime_path).name if runtime_path else run_dir.name
            runs.append(Run(
                path=run_dir,
                run_id=run_id,
                seed=str(seed_value),
                config=config,
                settings_id=settings_id,
                settings_label=_settings_label(config),
                metrics=metrics,
                metrics_source=str(metrics_path),
            ))
        except (OSError, ValueError, TypeError, yaml.YAMLError, pd.errors.ParserError) as error:
            skipped.append({
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "reason": str(error),
            })
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.run_id] = counts.get(run.run_id, 0) + 1
    runs = [
        Run(
            path=run.path,
            run_id=(
                f"{run.run_id}_{hashlib.sha1(str(run.path).encode()).hexdigest()[:8]}"
                if counts[run.run_id] > 1 else run.run_id
            ),
            seed=run.seed,
            config=run.config,
            settings_id=run.settings_id,
            settings_label=run.settings_label,
            metrics=run.metrics,
            metrics_source=run.metrics_source,
        )
        for run in runs
    ]
    return runs, skipped


def _valid_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    points = frame.loc[
        np.isfinite(frame["test_accuracy"]),
        ["step", "test_accuracy"],
    ].sort_values("step").reset_index(drop=True)
    if not points.empty and points["test_accuracy"].max() > 1.5:
        points["test_accuracy"] = points["test_accuracy"] / 100.0
    return points


def detect_events(
    frame: pd.DataFrame,
    *,
    onset_threshold: float = 0.99,
    severe_threshold: float = 0.90,
    min_step: int = 3500,
) -> list[dict[str, Any]]:
    """Return major collapse episodes; each episode ends at first >= onset threshold."""
    points = _valid_accuracy(frame)
    if points.empty:
        return []
    records = list(points.itertuples(index=False, name=None))
    events: list[dict[str, Any]] = []
    previous_accuracy: float | None = None
    active: dict[str, Any] | None = None
    for step_value, accuracy_value in records:
        step, accuracy = int(step_value), float(accuracy_value)
        if active is None:
            crossed = previous_accuracy is not None and previous_accuracy >= onset_threshold
            if step >= min_step and crossed and accuracy < onset_threshold:
                active = {
                    "onset_step": step,
                    "trough_step": step,
                    "trough_accuracy": accuracy,
                    "severe": accuracy < severe_threshold,
                }
            previous_accuracy = accuracy
            continue

        if accuracy < active["trough_accuracy"]:
            active["trough_step"] = step
            active["trough_accuracy"] = accuracy
        active["severe"] = active["severe"] or accuracy < severe_threshold
        if accuracy >= onset_threshold:
            if active["severe"]:
                active["recovery_step"] = step
                events.append(active)
            active = None
        previous_accuracy = accuracy

    if active is not None and active["severe"]:
        active["recovery_step"] = None
        events.append(active)
    return events


def _crossing_before(
    frame: pd.DataFrame,
    metric: str,
    onset_step: int,
    threshold: float,
    *,
    direction: str,
) -> int | None:
    values = frame.loc[
        (frame["step"] < onset_step) & np.isfinite(frame[metric]),
        ["step", metric],
    ].sort_values("step")
    previous_value: float | None = None
    crossing_step: int | None = None
    for row in values.itertuples(index=False, name=None):
        step, value = int(row[0]), float(row[1])
        crossed = (
            previous_value is not None
            and previous_value < threshold <= value
            if direction == "above"
            else previous_value is not None
            and previous_value > threshold >= value
        )
        if crossed:
            crossing_step = step
        previous_value = value
    return crossing_step


def make_event_records(
    runs: list[Run],
    *,
    onset_threshold: float,
    severe_threshold: float,
    min_step: int,
) -> list[CollapseEvent]:
    events: list[CollapseEvent] = []
    for run in runs:
        detected = detect_events(
            run.metrics,
            onset_threshold=onset_threshold,
            severe_threshold=severe_threshold,
            min_step=min_step,
        )
        for index, event in enumerate(detected):
            next_event = detected[index + 1] if index + 1 < len(detected) else None
            discrepancy_crossing = _crossing_before(
                run.metrics,
                "delayed_gradient_relative_discrepancy_mean",
                int(event["onset_step"]),
                0.5,
                direction="above",
            )
            cosine_crossing = _crossing_before(
                run.metrics,
                "delayed_gradient_cosine_mean",
                int(event["onset_step"]),
                0.9,
                direction="below",
            )
            recovery = event.get("recovery_step")
            events.append(CollapseEvent(
                event_id=f"{run.run_id}__event_{index + 1:03d}",
                settings_id=run.settings_id,
                settings_label=run.settings_label,
                run_id=run.run_id,
                seed=run.seed,
                onset_step=int(event["onset_step"]),
                trough_step=int(event["trough_step"]),
                trough_accuracy=float(event["trough_accuracy"]),
                trough_delay_steps=int(event["trough_step"]) - int(event["onset_step"]),
                recovery_step=int(recovery) if recovery is not None else None,
                recovery_duration=(
                    int(recovery) - int(event["onset_step"])
                    if recovery is not None else None
                ),
                next_onset_step=(
                    int(next_event["onset_step"]) if next_event is not None else None
                ),
                next_trough_step=(
                    int(next_event["trough_step"]) if next_event is not None else None
                ),
                next_trough_accuracy=(
                    float(next_event["trough_accuracy"])
                    if next_event is not None else None
                ),
                next_recovery_step=(
                    int(next_event["recovery_step"])
                    if next_event is not None and next_event.get("recovery_step") is not None
                    else None
                ),
                next_recovery_duration=(
                    int(next_event["recovery_step"]) - int(next_event["onset_step"])
                    if next_event is not None and next_event.get("recovery_step") is not None
                    else None
                ),
                intercollapse_steps=(
                    int(next_event["onset_step"]) - int(event["onset_step"])
                    if next_event is not None else None
                ),
                stable_plateau_steps=(
                    int(next_event["onset_step"]) - int(recovery)
                    if next_event is not None and recovery is not None else None
                ),
                discrepancy_crossing_step=discrepancy_crossing,
                discrepancy_lead_steps=(
                    int(event["onset_step"]) - discrepancy_crossing
                    if discrepancy_crossing is not None else None
                ),
                cosine_crossing_step=cosine_crossing,
                cosine_lead_steps=(
                    int(event["onset_step"]) - cosine_crossing
                    if cosine_crossing is not None else None
                ),
            ))
        if not detected:
            LOGGER.info("%s: no qualifying major collapse events", run.run_id)
    return events


def _event_window(
    event: CollapseEvent,
    *,
    pre_window: int,
    post_window: int,
    adaptive_window: bool,
) -> tuple[int, int]:
    end = post_window
    if adaptive_window and event.next_onset_step is not None:
        through_next_recovery = (
            event.next_recovery_step - event.onset_step
            if event.next_recovery_step is not None
            else event.next_onset_step - event.onset_step
        )
        end = max(end, through_next_recovery)
    return -pre_window, end


def aligned_rows(
    runs: list[Run],
    events: list[CollapseEvent],
    *,
    pre_window: int,
    post_window: int,
    adaptive_window: bool,
) -> pd.DataFrame:
    run_by_id = {run.run_id: run for run in runs}
    rows: list[dict[str, Any]] = []
    for event in events:
        run = run_by_id[event.run_id]
        start, end = _event_window(
            event,
            pre_window=pre_window,
            post_window=post_window,
            adaptive_window=adaptive_window,
        )
        indexed = run.metrics.set_index("step")
        steps = np.arange(start, end + 1, dtype=int)
        for relative_step in steps:
            absolute_step = event.onset_step + int(relative_step)
            values = (
                indexed.loc[absolute_step]
                if absolute_step in indexed.index
                else None
            )
            row: dict[str, Any] = {
                "configuration_id": event.settings_id,
                "configuration": event.settings_label,
                "run_id": event.run_id,
                "seed": event.seed,
                "event_id": event.event_id,
                "collapse_onset_step": event.onset_step,
                "step_relative": int(relative_step),
                "step_absolute": absolute_step,
            }
            for metric in METRICS:
                value = values.get(metric) if values is not None else np.nan
                if value is None or pd.isna(value):
                    row[metric] = np.nan
                else:
                    row[metric] = (
                        float(value) / 100.0
                        if metric == "test_accuracy" and float(value) > 1.5
                        else float(value)
                    )
            rows.append(row)
    frame = pd.DataFrame(rows)
    for metric in METRICS:
        if frame.empty:
            frame[f"{metric}_n_events_at_step"] = pd.Series(dtype=int)
            frame[f"{metric}_n_seeds_at_step"] = pd.Series(dtype=int)
            continue
        group_keys = ["configuration_id", "step_relative"]
        counts = frame.groupby(group_keys, dropna=False)[metric].transform("count")
        seed_counts = (
            frame.loc[frame[metric].notna(), group_keys + ["seed"]]
            .drop_duplicates()
            .groupby(group_keys, dropna=False)["seed"]
            .nunique()
        )
        key_index = pd.MultiIndex.from_frame(frame[group_keys])
        frame[f"{metric}_n_events_at_step"] = counts
        frame[f"{metric}_n_seeds_at_step"] = seed_counts.reindex(key_index).fillna(0).to_numpy()
    return frame


def _summary_arrays(
    frame: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby("step_relative")[metric]
    steps = np.array(sorted(frame["step_relative"].unique()), dtype=int)
    medians: list[float] = []
    q25: list[float] = []
    q75: list[float] = []
    counts: list[int] = []
    for step in steps:
        values = grouped.get_group(step).dropna().to_numpy(dtype=float)
        counts.append(len(values))
        if len(values):
            medians.append(float(np.median(values)))
            q25.append(float(np.quantile(values, 0.25)))
            q75.append(float(np.quantile(values, 0.75)))
        else:
            medians.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)
    median_array = np.array(medians)
    observed = np.isfinite(median_array)
    return (
        steps[observed],
        median_array[observed],
        np.array(q25)[observed],
        np.array(q75)[observed],
        np.array(counts)[observed],
    )


def _gap_limit(runs: list[Run], maximum: int = 100) -> int:
    cadences: list[float] = []
    for run in runs:
        steps = run.metrics["step"].dropna().sort_values().unique()
        gaps = np.diff(steps)
        if len(gaps):
            cadences.append(float(np.median(gaps[gaps > 0])))
    inferred = max(1, int(np.ceil(2 * np.median(cadences)))) if cadences else 1
    return min(maximum, inferred)


def _plot_observed(
    ax: plt.Axes,
    steps: np.ndarray,
    median: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    color: Any,
    max_gap: int,
    linewidth: float = 1.6,
    alpha: float = 0.18,
    label: str | None = None,
) -> None:
    if not len(steps):
        return
    breaks = np.flatnonzero(np.diff(steps) > max_gap) + 1
    for index, segment in enumerate(np.split(np.arange(len(steps)), breaks)):
        if not len(segment):
            continue
        x = steps[segment]
        y = median[segment]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            marker=".",
            markersize=2.2,
            label=label if index == 0 else None,
        )
        ax.fill_between(x, low[segment], high[segment], color=color, alpha=alpha, linewidth=0)


def _annotate_landmarks(
    ax: plt.Axes,
    events: list[CollapseEvent],
    *,
    show_next_collapse: bool = True,
    seed_balanced: bool = False,
) -> None:
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.0, label="Collapse onset")
    if not show_next_collapse:
        return
    next_offsets = _event_values(events, "intercollapse_steps", seed_balanced)
    if next_offsets:
        median_next = float(np.median(next_offsets))
        ax.axvline(
            median_next,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label=f"Median next collapse ({median_next:.0f})",
        )


def _format_axes(
    axes: np.ndarray,
    events: list[CollapseEvent],
    *,
    show_next_collapse: bool = True,
    seed_balanced: bool = False,
) -> None:
    for index, (ax, metric, color, label) in enumerate(
        zip(axes, METRICS, METRIC_COLORS, METRIC_LABELS)
    ):
        ax.set_ylabel(label, color=color)
        ax.yaxis.label.set_size(9)
        ax.yaxis.labelpad = 6
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, color="#cccccc", alpha=0.28, linewidth=0.6)
        if metric == METRICS[2]:
            ax.set_yscale("log")
        if metric == "delayed_gradient_relative_discrepancy_mean":
            ax.axhline(0.5, color=color, linestyle=":", alpha=0.7, linewidth=0.9)
        elif metric == "delayed_gradient_cosine_mean":
            ax.axhline(0.9, color=color, linestyle=":", alpha=0.7, linewidth=0.9)
            ax.set_ylim(-1.05, 1.05)
        elif metric == "test_accuracy":
            ax.axhline(0.99, color=color, linestyle=":", alpha=0.7, linewidth=0.9)
            ax.set_ylim(-0.02, 1.02)
        _annotate_landmarks(
            ax,
            events,
            show_next_collapse=show_next_collapse,
            seed_balanced=seed_balanced,
        )
    axes[-1].set_xlabel("Training steps relative to collapse onset")


def _event_values(
    events: list[CollapseEvent],
    attribute: str,
    seed_balanced: bool,
) -> list[int]:
    grouped: dict[str, list[int]] = {}
    for event in events:
        value = getattr(event, attribute)
        if value is not None:
            grouped.setdefault(event.seed, []).append(int(value))
    if seed_balanced:
        return [int(np.median(values)) for values in grouped.values()]
    return [value for values in grouped.values() for value in values]


def _event_timing_text(
    events: list[CollapseEvent],
    *,
    seed_balanced: bool = False,
) -> str:
    def median(values: list[int]) -> str:
        return f"{np.median(values):.0f}" if values else "NA"

    trough = _event_values(events, "trough_delay_steps", seed_balanced)
    recovery = _event_values(events, "recovery_duration", seed_balanced)
    next_collapse = _event_values(events, "intercollapse_steps", seed_balanced)
    next_recovery = _event_values(events, "next_recovery_duration", seed_balanced)
    return (
        f"Events={len(events)}; median trough={median(trough)} steps; "
        f"recovery={median(recovery)} steps (n={len(recovery)}); "
        f"next collapse={median(next_collapse)} steps (n={len(next_collapse)}); "
        f"next recovery after onset={median(next_recovery)} steps "
        f"(n={len(next_recovery)})"
    )


def _plot_aggregate(
    frame: pd.DataFrame,
    events: list[CollapseEvent],
    path: Path,
    *,
    title: str,
    subtitle: str,
    max_gap: int = 1,
    seed_balanced_landmarks: bool = False,
) -> None:
    if frame.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax, metric, color in zip(axes, METRICS, METRIC_COLORS):
        steps, median, low, high, _ = _summary_arrays(frame, metric)
        _plot_observed(
            ax, steps, median, low, high, color=color, max_gap=max_gap,
            linewidth=1.7,
        )
    _format_axes(axes, events, seed_balanced=seed_balanced_landmarks)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    details = textwrap.fill(subtitle, width=72)
    timing = textwrap.fill(
        _event_timing_text(events, seed_balanced=seed_balanced_landmarks),
        width=76,
    )
    fig.suptitle(f"{title}\n{details}\n{timing}", fontsize=9, y=0.99)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.78, bottom=0.07, hspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _configuration_summary(
    events: list[CollapseEvent],
    runs: list[Run],
) -> pd.DataFrame:
    frame = pd.DataFrame(asdict(event) for event in events)
    rows: list[dict[str, Any]] = []
    measures = (
        "trough_delay_steps",
        "trough_accuracy",
        "recovery_duration",
        "next_recovery_duration",
        "intercollapse_steps",
        "stable_plateau_steps",
        "discrepancy_lead_steps",
        "cosine_lead_steps",
    )
    configurations = {
        run.settings_id: (run.settings_label, [r for r in runs if r.settings_id == run.settings_id])
        for run in runs
    }
    for settings_id, (label, config_runs) in configurations.items():
        group = (
            frame[frame["settings_id"] == settings_id]
            if not frame.empty else pd.DataFrame()
        )
        row: dict[str, Any] = {
            "configuration_id": settings_id,
            "configuration": label,
            "seed_count": len({run.seed for run in config_runs}),
            "run_count": len(config_runs),
            "event_count": len(group),
            "paired_next_event_count": (
                int(group["intercollapse_steps"].notna().sum()) if not group.empty else 0
            ),
            "complete_recovery_count": (
                int(group["recovery_duration"].notna().sum()) if not group.empty else 0
            ),
        }
        for measure in measures:
            values = (
                pd.to_numeric(group[measure], errors="coerce").dropna()
                if not group.empty else pd.Series(dtype=float)
            )
            seed_medians = pd.Series(dtype=float)
            if not group.empty:
                numeric_group = group.assign(
                    _measure=pd.to_numeric(group[measure], errors="coerce")
                )
                seed_medians = numeric_group.groupby(
                    "seed", dropna=False
                )["_measure"].agg(
                    lambda seed_values: seed_values.dropna().median()
                    if seed_values.notna().any() else np.nan
                ).dropna()
            row[f"{measure}_median"] = (
                float(seed_medians.median()) if len(seed_medians) else np.nan
            )
            row[f"{measure}_q25"] = (
                float(seed_medians.quantile(0.25)) if len(seed_medians) else np.nan
            )
            row[f"{measure}_q75"] = (
                float(seed_medians.quantile(0.75)) if len(seed_medians) else np.nan
            )
            row[f"{measure}_n_seeds"] = int(len(seed_medians))
            row[f"{measure}_n_events"] = int(len(values))
        rows.append(row)
    columns = [
        "configuration_id",
        "configuration",
        "seed_count",
        "run_count",
        "event_count",
        "paired_next_event_count",
        "complete_recovery_count",
        *[
            f"{measure}_{stat}"
            for measure in measures
            for stat in ("median", "q25", "q75", "n_seeds", "n_events")
        ],
    ]
    return pd.DataFrame(rows, columns=columns)


def _plot_configurations(
    runs: list[Run],
    events: list[CollapseEvent],
    aligned: pd.DataFrame,
    output_dir: Path,
    *,
    overlay_seeds: bool,
    make_individual: bool,
    make_by_config: bool,
    make_comparison: bool,
    max_gap_steps: int,
) -> None:
    config_dir = output_dir / "by_configuration"
    individual_dir = output_dir / "individual_runs"
    for run in runs if make_individual else []:
        run_events = [event for event in events if event.run_id == run.run_id]
        if not run_events:
            continue
        run_aligned = aligned[aligned["run_id"] == run.run_id].copy()
        run_aligned = (
            run_aligned.groupby(["event_id", "step_relative"], dropna=False)[list(METRICS)]
            .median()
            .reset_index()
        )
        run_events_frame = {event.event_id: event for event in run_events}
        _plot_aggregate(
            run_aligned,
            [run_events_frame[event_id] for event_id in run_aligned["event_id"].unique()],
            individual_dir / f"{_safe_name(run.run_id)}.png",
            title=f"Collapse–recovery cycle: {run.run_id}",
            subtitle=f"{run.settings_label}; seed={run.seed}; {len(run_events)} qualifying events",
            max_gap=_gap_limit([run], max_gap_steps),
        )

    by_config: dict[str, list[Run]] = {}
    for run in runs:
        by_config.setdefault(run.settings_id, []).append(run)
    for settings_id, config_runs in by_config.items() if make_by_config else []:
        config_events = [event for event in events if event.settings_id == settings_id]
        if not config_events:
            continue
        config_aligned = aligned[aligned["configuration_id"] == settings_id].copy()
        seed_frames: list[pd.DataFrame] = []
        for seed, seed_data in config_aligned.groupby("seed", dropna=False):
            seed_frame = (
                seed_data.groupby("step_relative", dropna=False)[list(METRICS)]
                .median()
                .reset_index()
            )
            seed_frame["seed"] = seed
            seed_frames.append(seed_frame)
        seed_medians = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
        if seed_medians.empty:
            continue
        if overlay_seeds:
            fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
            for seed, seed_frame in seed_medians.groupby("seed", dropna=False):
                for ax, metric, color in zip(axes, METRICS, METRIC_COLORS):
                    seed_values = seed_frame.dropna(subset=[metric])
                    steps = seed_values["step_relative"].to_numpy(dtype=int)
                    values = seed_values[metric].to_numpy(dtype=float)
                    breaks = np.flatnonzero(
                        np.diff(steps) > _gap_limit(config_runs, max_gap_steps)
                    ) + 1
                    for segment in np.split(np.arange(len(steps)), breaks):
                        if len(segment):
                            ax.plot(
                                steps[segment],
                                values[segment],
                                color=color,
                                alpha=0.18,
                                linewidth=0.8,
                                marker=".",
                                markersize=1.7,
                            )
            for ax, metric, color in zip(axes, METRICS, METRIC_COLORS):
                summary_frame = (
                    seed_medians.groupby("step_relative", dropna=False)[metric]
                    .agg(
                        median="median",
                        q25=lambda values: values.quantile(0.25),
                        q75=lambda values: values.quantile(0.75),
                    )
                    .reset_index()
                )
                summary_frame = summary_frame.dropna(subset=["median"])
                _plot_observed(
                    ax,
                    summary_frame["step_relative"].to_numpy(dtype=int),
                    summary_frame["median"].to_numpy(dtype=float),
                    summary_frame["q25"].to_numpy(dtype=float),
                    summary_frame["q75"].to_numpy(dtype=float),
                    color=color,
                    max_gap=_gap_limit(config_runs, max_gap_steps),
                    linewidth=1.7,
                )
            _format_axes(axes, config_events, seed_balanced=True)
            axes[0].legend(loc="upper left", fontsize=8, frameon=False)
            seeds_count = seed_medians["seed"].nunique()
            fig.suptitle(
                textwrap.fill(
                    f"Seed-balanced collapse cycles: {config_runs[0].settings_label}",
                    width=72,
                )
                + "\n"
                f"Seeds={seeds_count}; events={len(config_events)}\n"
                f"{textwrap.fill(_event_timing_text(config_events, seed_balanced=True), width=76)}",
                fontsize=9,
            )
            fig.subplots_adjust(
                left=0.19, right=0.98, top=0.76, bottom=0.07, hspace=0.12
            )
            config_dir.mkdir(parents=True, exist_ok=True)
            path = config_dir / f"{settings_id}.png"
            fig.savefig(path, dpi=220)
            fig.savefig(path.with_suffix(".pdf"))
            plt.close(fig)
        else:
            _plot_aggregate(
                seed_medians,
                config_events,
                config_dir / f"{settings_id}.png",
                title=f"Seed-balanced collapse cycles: {config_runs[0].settings_label}",
                subtitle=(
                    f"Seeds={seed_medians['seed'].nunique()}; events={len(config_events)}"
                ),
                max_gap=_gap_limit(config_runs, max_gap_steps),
                seed_balanced_landmarks=True,
            )

    if not make_comparison:
        return
    comparison_dir = output_dir / "comparisons"
    grouped = aligned.groupby("configuration_id", dropna=False)
    config_ids = sorted(aligned["configuration_id"].dropna().unique())
    page_size = 8
    for page_start in range(0, len(config_ids), page_size):
        page_ids = config_ids[page_start:page_start + page_size]
        page_rows = 4
        fig, axes = plt.subplots(page_rows, 1, figsize=(12, 12), sharex=True)
        palette = plt.get_cmap("tab10")
        for config_index, settings_id in enumerate(page_ids):
            frame = grouped.get_group(settings_id)
            seed_medians = (
                frame.groupby(["seed", "step_relative"], dropna=False)[list(METRICS)]
                .median()
                .reset_index()
            )
            trajectory = (
                seed_medians.groupby("step_relative", dropna=False)[list(METRICS)]
                .median()
            )
            color = palette(config_index % 10)
            label = next(
                event.settings_label for event in events
                if event.settings_id == settings_id
            )
            for ax, metric in zip(axes, METRICS):
                summary_frame = trajectory[[metric]].dropna().reset_index()
                steps = summary_frame["step_relative"].to_numpy(dtype=int)
                values = summary_frame[metric].to_numpy(dtype=float)
                breaks = np.flatnonzero(
                    np.diff(steps) > _gap_limit(
                        [run for run in runs if run.settings_id == settings_id],
                        max_gap_steps,
                    )
                ) + 1
                for segment in np.split(np.arange(len(steps)), breaks):
                    if not len(segment):
                        continue
                    ax.plot(
                        steps[segment],
                        values[segment],
                        color=color,
                        linewidth=1.3,
                        marker=".",
                        markersize=1.8,
                        label=label if segment[0] == 0 else None,
                    )
        all_page_events = [event for event in events if event.settings_id in page_ids]
        _format_axes(axes, all_page_events, show_next_collapse=False)
        for settings_id in page_ids:
            config_events = [
                event for event in events
                if event.settings_id == settings_id
                and event.intercollapse_steps is not None
            ]
            offsets = _event_values(
                config_events, "intercollapse_steps", seed_balanced=True
            )
            if not offsets:
                continue
            color = palette(page_ids.index(settings_id) % 10)
            for ax in axes:
                ax.axvline(
                    float(np.median(offsets)),
                    color=color,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.55,
                )
        axes[0].legend(loc="best", fontsize=7, frameon=False, ncol=2)
        fig.suptitle(
            f"Cross-configuration event-aligned comparison "
            f"({page_start // page_size + 1}/{(len(config_ids) + page_size - 1) // page_size})",
            fontsize=12,
        )
        fig.subplots_adjust(
            left=0.19, right=0.98, top=0.92, bottom=0.07, hspace=0.12
        )
        comparison_dir.mkdir(parents=True, exist_ok=True)
        path = comparison_dir / f"all_configurations_{page_start // page_size + 1:02d}.png"
        fig.savefig(path, dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:150] or "run"


def _write_outputs(
    runs: list[Run],
    skipped: list[dict[str, str]],
    events: list[CollapseEvent],
    aligned: pd.DataFrame,
    output_dir: Path,
    *,
    make_individual: bool,
    make_by_config: bool,
    make_comparison: bool,
    overlay_seeds: bool,
    max_gap_steps: int,
) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    event_frame = pd.DataFrame(
        [asdict(event) for event in events],
        columns=[field for field in CollapseEvent.__dataclass_fields__],
    )
    event_frame.to_csv(data_dir / "collapse_events.csv", index=False)
    _configuration_summary(events, runs).to_csv(
        data_dir / "configuration_summary.csv", index=False
    )
    if aligned.empty:
        aligned = pd.DataFrame(columns=[
            "configuration_id", "configuration", "run_id", "seed", "event_id",
            "collapse_onset_step", "step_relative", "step_absolute", *METRICS,
            *[
                f"{metric}_{suffix}"
                for metric in METRICS
                for suffix in ("n_events_at_step", "n_seeds_at_step")
            ],
        ])
    aligned.to_csv(data_dir / "aligned_trajectories.csv", index=False)
    pd.DataFrame(skipped, columns=["run_id", "run_dir", "reason"]).to_csv(
        data_dir / "skipped_runs.csv", index=False
    )
    pd.DataFrame([
        {
            "run_id": run.run_id,
            "run_dir": str(run.path),
            "seed": run.seed,
            "configuration_id": run.settings_id,
            "configuration": run.settings_label,
            "metrics_source": run.metrics_source,
            "metric_steps": len(run.metrics),
            "first_step": int(run.metrics["step"].min()) if not run.metrics.empty else NA,
            "last_step": int(run.metrics["step"].max()) if not run.metrics.empty else NA,
            "event_count": sum(event.run_id == run.run_id for event in events),
        }
        for run in runs
    ], columns=[
        "run_id", "run_dir", "seed", "configuration_id", "configuration",
        "metrics_source", "metric_steps", "first_step", "last_step", "event_count",
    ]).to_csv(data_dir / "included_runs.csv", index=False)

    event_run_ids = {event.run_id for event in events}
    plot_runs = [run for run in runs if run.run_id in event_run_ids]
    if events and (make_individual or make_by_config or make_comparison):
        _plot_configurations(
            plot_runs,
            events,
            aligned,
            output_dir,
            overlay_seeds=overlay_seeds,
            make_individual=make_individual,
            make_by_config=make_by_config,
            make_comparison=make_comparison,
            max_gap_steps=max_gap_steps,
        )


def run_analysis(
    results_dir: Path,
    output_dir: Path,
    *,
    onset_threshold: float = 0.99,
    severe_threshold: float = 0.90,
    min_step: int = 3500,
    pre_window: int = 220,
    post_window: int = 1100,
    adaptive_window: bool = False,
    grouping: str = "full",
    make_individual: bool = True,
    make_by_config: bool = True,
    make_comparison: bool = True,
    overlay_seeds: bool = True,
    max_gap_steps: int = 100,
) -> tuple[list[Run], list[CollapseEvent], pd.DataFrame]:
    runs, skipped = discover_runs(results_dir, output_dir, grouping)
    LOGGER.info("Discovered %d runs with usable metrics/configuration", len(runs))
    for item in skipped:
        LOGGER.warning("Skipped %s: %s", item["run_dir"], item["reason"])
    events = make_event_records(
        runs,
        onset_threshold=onset_threshold,
        severe_threshold=severe_threshold,
        min_step=min_step,
    )
    for run in runs:
        count = sum(event.run_id == run.run_id for event in events)
        LOGGER.info(
            "%s seed=%s setting=%s: %d major collapse events",
            run.run_id,
            run.seed,
            run.settings_id,
            count,
        )
    aligned = aligned_rows(
        runs,
        events,
        pre_window=pre_window,
        post_window=post_window,
        adaptive_window=adaptive_window,
    )
    _write_outputs(
        runs,
        skipped,
        events,
        aligned,
        output_dir,
        make_individual=make_individual,
        make_by_config=make_by_config,
        make_comparison=make_comparison,
        overlay_seeds=overlay_seeds,
        max_gap_steps=max_gap_steps,
    )
    LOGGER.info(
        "Analysis complete: %d events across %d runs, %d configurations; output=%s",
        len(events),
        len(runs),
        len({run.settings_id for run in runs}),
        output_dir,
    )
    return runs, events, aligned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and plot event-aligned test-accuracy collapse cycles."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/collapse_cycles"),
    )
    parser.add_argument("--onset-threshold", type=float, default=0.99)
    parser.add_argument("--severe-threshold", type=float, default=0.90)
    parser.add_argument("--min-step", type=int, default=3500)
    parser.add_argument("--pre-window", type=int, default=220)
    parser.add_argument("--post-window", type=int, default=1100)
    parser.add_argument(
        "--max-gap-steps",
        type=int,
        default=100,
        help="Break plotted lines across larger unobserved gaps (default: 100).",
    )
    parser.add_argument(
        "--adaptive-window",
        action="store_true",
        help="Extend each event through its next collapse and, when available, recovery.",
    )
    parser.add_argument(
        "--grouping",
        choices=("full", "optimizer-delay"),
        default="full",
        help="Group by full effective configuration or only optimizer/delay/model/task settings.",
    )
    parser.add_argument("--no-individual", action="store_true")
    parser.add_argument("--no-by-configuration", action="store_true")
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument(
        "--overlay-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show faint per-seed median trajectories behind configuration aggregates.",
    )
    args = parser.parse_args()
    if not 0 <= args.severe_threshold < args.onset_threshold <= 1:
        parser.error("thresholds must satisfy 0 <= severe < onset <= 1")
    if args.min_step < 0 or args.pre_window < 0 or args.post_window < 0:
        parser.error("step and window arguments must be nonnegative")
    if args.max_gap_steps < 1:
        parser.error("--max-gap-steps must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.results_dir.is_dir():
        parser.error(f"results directory does not exist: {args.results_dir}")
    run_analysis(
        args.results_dir,
        args.output_dir,
        onset_threshold=args.onset_threshold,
        severe_threshold=args.severe_threshold,
        min_step=args.min_step,
        pre_window=args.pre_window,
        post_window=args.post_window,
        adaptive_window=args.adaptive_window,
        grouping=args.grouping,
        make_individual=not args.no_individual,
        make_by_config=not args.no_by_configuration,
        make_comparison=not args.no_comparison,
        overlay_seeds=args.overlay_seeds,
        max_gap_steps=args.max_gap_steps,
    )


if __name__ == "__main__":
    main()
