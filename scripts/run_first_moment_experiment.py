from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml
from torch import Tensor, nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_primary_experiment import (
    _adamw_step_with_second_moment,
    _block_update_norms,
    _evaluate,
    _gradient,
    _run_directory,
    _set_gradients,
)
from src.config import load_config, validate_config
from src.dataset import make_modular_addition_data
from src.device import select_device
from src.logging_utils import RunLogger
from src.optimizers import build_optimizer
from src.trainer import _build_model
from src.utils import environment_info, seed_everything, write_json


ARMS = {
    "fresh_first_moment": "A: Fresh-only first moment",
    "mixed_first_moment": "B: Fresh/stale mixed first moment",
    "delayed_only_first_moment": "C: Delayed-only first moment",
}
MAX_EXPERIMENT_STEPS = 20_000
PROGRESS_INTERVAL = 1_000


def _effective_step_limit(configured_steps: int) -> int:
    return min(configured_steps, MAX_EXPERIMENT_STEPS)


def _mix(
    fresh: list[Tensor | None],
    stale: list[Tensor | None],
    delayed_fraction: float,
) -> tuple[list[Tensor | None], list[Tensor | None]]:
    mixed: list[Tensor | None] = []
    square_inputs: list[Tensor | None] = []
    for current, delayed in zip(fresh, stale):
        if current is None and delayed is None:
            mixed.append(None)
            square_inputs.append(None)
            continue
        if current is None:
            assert delayed is not None
            current = torch.zeros_like(delayed)
        if delayed is None:
            delayed = torch.zeros_like(current)
        mixed.append(
            (1.0 - delayed_fraction) * current + delayed_fraction * delayed
        )
        square_inputs.append(
            ((1.0 - delayed_fraction) * current
             + delayed_fraction * delayed).square()
        )
    return mixed, square_inputs


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_arm_metrics(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        metrics_path = path / arm / "metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing metrics for {arm}: {metrics_path}")
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        converted: list[dict[str, Any]] = []
        for row in rows:
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if value in (None, ""):
                    normalized[key] = None
                    continue
                try:
                    normalized[key] = float(value)
                except ValueError:
                    normalized[key] = value
            normalized["step"] = int(normalized["step"])
            converted.append(normalized)
        if not converted:
            raise ValueError(f"metrics file is empty: {metrics_path}")
        result[arm] = converted
    return result


def _optimizer_state_norms(
    optimizer: torch.optim.Optimizer,
    parameters: list[nn.Parameter],
) -> dict[str, float]:
    first_squared = 0.0
    second_sqrt_squared = 0.0
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if "exp_avg" in state:
            first_squared += float(state["exp_avg"].double().square().sum().item())
        if "exp_avg_sq" in state:
            second_sqrt_squared += float(
                state["exp_avg_sq"].double().sum().item()
            )
    return {
        "first_moment_exp_avg_l2_norm": first_squared**0.5,
        "second_moment_sqrt_l2_norm": second_sqrt_squared**0.5,
    }


def _make_plots(run_dir: Path, rows_by_arm: dict[str, list[dict[str, Any]]]) -> Path:
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "fresh_first_moment": "#2878b5",
        "mixed_first_moment": "#e1812c",
        "delayed_only_first_moment": "#3a923a",
    }
    plots = (
        ("train_accuracy", "Training accuracy", "Accuracy"),
        ("test_accuracy", "Test accuracy", "Accuracy"),
        ("train_loss", "Training cross-entropy", "Loss"),
        ("test_loss", "Test cross-entropy", "Loss"),
        ("restricted_test_loss_paper_keys", "Restricted test loss (paper frequencies)", "Loss"),
        ("excluded_train_loss_paper_keys", "Excluded train loss (paper frequencies)", "Loss"),
        ("restricted_test_loss_all_addition", "Restricted test loss (all addition frequencies)", "Loss"),
        ("excluded_train_loss_all_addition", "Excluded train loss (all addition frequencies)", "Loss"),
        ("adaptive_update_l2_norm", "Adaptive data-update L2 norm", "Update norm"),
        ("relative_adaptive_update_l2_norm", "Relative adaptive update norm", "Update / weight norm"),
        ("first_moment_exp_avg_l2_norm", "AdamW first-moment L2 norm", "First-moment norm"),
        ("second_moment_sqrt_l2_norm", "AdamW second-moment square-root L2 norm", "Second-moment norm"),
        ("weight_l2_norm", "Total parameter L2 norm", "Weight norm"),
    )
    for field, title, ylabel in plots:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for arm, rows in rows_by_arm.items():
            points: list[tuple[int, float]] = []
            for row in rows:
                try:
                    value = float(row[field])
                except (KeyError, TypeError, ValueError):
                    continue
                if value == value and abs(value) != float("inf"):
                    points.append((int(row["step"]), value))
            if points:
                ax.plot(
                    [step for step, _ in points],
                    [value for _, value in points],
                    label=ARMS[arm],
                    color=colors[arm],
                    linewidth=1.2,
                )
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set(title=title, xlabel="Global optimizer updates", ylabel=ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{field}.png", dpi=180)
        fig.savefig(figure_dir / f"{field}.pdf")
        plt.close(fig)
    return figure_dir


def _arm_outcomes(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    branch_step: int,
    final_step: int,
    confirmation_steps: int = 1000,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for arm, rows in rows_by_arm.items():
        by_step = {int(row["step"]): row for row in rows}
        start = by_step.get(branch_step, {})
        horizon = by_step.get(branch_step + 1000, {})
        loss_start = start.get("restricted_test_loss_paper_keys")
        loss_end = horizon.get("restricted_test_loss_paper_keys")
        sustained = None
        for candidate in range(branch_step, final_step - confirmation_steps + 1):
            window = [
                by_step.get(step, {}).get("test_accuracy")
                for step in range(candidate, candidate + confirmation_steps + 1)
            ]
            if len(window) == confirmation_steps + 1 and all(
                value is not None and float(value) >= 0.99 for value in window
            ):
                sustained = candidate
                break
        results[arm] = {
            "arm": arm,
            "branch_step": branch_step,
            "final_step": final_step,
            "restricted_loss_reduction_1000_steps": (
                float(loss_start) - float(loss_end)
                if loss_start is not None and loss_end is not None else None
            ),
            "sustained_grokking_step": sustained,
            "sustained_grokking_right_censored": sustained is None,
            "confirmation_window_steps": confirmation_steps,
        }
    return results


def run_first_moment_experiment(
    config: dict[str, Any],
    *,
    seed: int,
    run_dir: str | Path | None = None,
    nodes: int = 6,
    delayed_nodes: int = 4,
    tau: int = 4,
    selection_train_accuracy: float = 0.99,
    selection_test_max: float = 0.20,
    selection_streak: int = 100,
    diagnostic_interval: int = 20,
) -> Path:
    validate_config(config)
    if str(config["optimizer"]["name"]).lower() != "adamw":
        raise ValueError("the first-moment experiment requires AdamW")
    if nodes < 2 or not 1 <= delayed_nodes < nodes:
        raise ValueError("delayed_nodes must be between 1 and nodes - 1")
    if tau < 1 or selection_streak < 1 or diagnostic_interval < 1:
        raise ValueError("tau and evaluation/logging intervals must be positive")

    experiment_config = copy.deepcopy(config)
    experiment_config["experiment"]["seed"] = int(seed)
    experiment_config["experiment"]["name"] += "_first_moment_comparison"
    configured_steps = int(experiment_config["experiment"]["epochs"])
    max_steps = _effective_step_limit(configured_steps)
    experiment_config["experiment"]["epochs"] = max_steps
    seed_everything(seed)
    device = select_device(
        experiment_config["device"]["selection"],
        bool(experiment_config["device"]["allow_cpu_fallback"]),
    )
    output_dir = _run_directory(experiment_config, run_dir)
    baseline_dir = output_dir / "baseline_to_t0"
    baseline_dir.mkdir()
    arm_dirs = {name: output_dir / name for name in ARMS}
    for path in arm_dirs.values():
        path.mkdir()

    rho = delayed_nodes / nodes
    effective_config = copy.deepcopy(experiment_config)
    effective_config["first_moment_experiment"] = {
        "seed": seed,
        "nodes": nodes,
        "delayed_nodes": delayed_nodes,
        "zero_delay_nodes": nodes - delayed_nodes,
        "rho": rho,
        "tau": tau,
        "branch_rule": {
            "train_accuracy_at_least": selection_train_accuracy,
            "test_accuracy_below": selection_test_max,
            "consecutive_evaluations": selection_streak,
            "restricted_loss_used_for_selection": False,
        },
        "second_moment_input": "square of the same (1-rho)*fresh + rho*stale mixed gradient in every arm",
        "first_moment_inputs": {
            "fresh_first_moment": "current fresh full-batch gradient",
            "mixed_first_moment": "mean over all fresh and fixed-delay worker gradients",
            "delayed_only_first_moment": "mean over the fixed-delay workers only; identical delayed gradients under this full-batch fixed-delay setup",
        },
        "diagnostic_interval": diagnostic_interval,
        "configured_max_steps": configured_steps,
        "effective_max_steps": max_steps,
        "progress_interval": PROGRESS_INTERVAL,
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective_config, handle, sort_keys=False)

    data = make_modular_addition_data(
        int(experiment_config["experiment"]["modulus"]),
        float(experiment_config["experiment"]["train_fraction"]),
        seed,
    )
    write_json(output_dir / "environment.json", environment_info(device))
    model = _build_model(experiment_config).to(device)
    optimizer = build_optimizer(model.parameters(), experiment_config["optimizer"])
    warmup = int(experiment_config["optimizer"]["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda count: min(count / warmup, 1.0) if warmup else 1.0,
    )
    paper_keys = list(
        experiment_config["mechanistic"]["key_frequencies"]["paper_mainline"]
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    gradient_history: deque[list[Tensor | None]] = deque(maxlen=tau)
    baseline_rows: list[dict[str, Any]] = []
    consecutive = 0
    branch_step: int | None = None
    with RunLogger(baseline_dir) as logger:
        for step in range(max_steps + 1):
            metrics = _evaluate(
                model,
                data,
                device,
                paper_keys,
                include_diagnostics=(step % diagnostic_interval == 0),
            )
            row = {
                "step": step,
                **metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            logger.log(row)
            baseline_rows.append(row)
            if step > 0 and step % PROGRESS_INTERVAL == 0:
                print(
                    f"baseline completed {step}/{max_steps} updates | "
                    f"train={metrics['train_accuracy']:.4f} "
                    f"test={metrics['test_accuracy']:.4f}",
                    flush=True,
                )
            if (
                metrics["train_accuracy"] >= selection_train_accuracy
                and metrics["test_accuracy"] < selection_test_max
            ):
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= selection_streak and len(gradient_history) >= tau:
                branch_step = step
                break
            if step == max_steps:
                break
            gradients = _gradient(model, data, device)
            gradient_history.append(gradients)
            _set_gradients(parameters, gradients)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    if branch_step is None:
        write_json(output_dir / "first_moment_experiment_status.json", {
            "status": "t0_not_found",
            "max_steps": max_steps,
            "selection_train_accuracy": selection_train_accuracy,
            "selection_test_max": selection_test_max,
            "selection_streak": selection_streak,
            "last_train_accuracy": baseline_rows[-1]["train_accuracy"],
            "last_test_accuracy": baseline_rows[-1]["test_accuracy"],
        })
        raise RuntimeError(
            f"No pre-grokking branch checkpoint found in {max_steps} steps; "
            f"baseline is saved at {baseline_dir}"
        )

    preceding_gradients = [copy.deepcopy(value) for value in gradient_history]
    torch.save({
        "step": branch_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "preceding_gradients": preceding_gradients,
        "config": effective_config,
        "train_indices": data.train_indices,
        "test_indices": data.test_indices,
    }, output_dir / "checkpoint_t0.pt")
    write_json(output_dir / "t0_selection.json", {
        "step": branch_step,
        "train_accuracy": baseline_rows[-1]["train_accuracy"],
        "test_accuracy": baseline_rows[-1]["test_accuracy"],
        "criterion": effective_config["first_moment_experiment"]["branch_rule"],
        "restricted_test_loss_paper_keys": baseline_rows[-1].get(
            "restricted_test_loss_paper_keys"
        ),
    })

    arm_models: dict[str, nn.Module] = {}
    arm_optimizers: dict[str, torch.optim.Optimizer] = {}
    arm_schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
    arm_histories: dict[str, deque[list[Tensor | None]]] = {}
    for arm in ARMS:
        arm_model = _build_model(experiment_config).to(device)
        arm_model.load_state_dict(copy.deepcopy(model.state_dict()))
        arm_optimizer = build_optimizer(
            arm_model.parameters(), experiment_config["optimizer"]
        )
        arm_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        arm_scheduler = torch.optim.lr_scheduler.LambdaLR(
            arm_optimizer,
            lr_lambda=lambda count: min(count / warmup, 1.0) if warmup else 1.0,
        )
        arm_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        arm_models[arm] = arm_model
        arm_optimizers[arm] = arm_optimizer
        arm_schedulers[arm] = arm_scheduler
        arm_histories[arm] = deque(
            (copy.deepcopy(value) for value in preceding_gradients),
            maxlen=tau,
        )

    arm_loggers = {arm: RunLogger(arm_dirs[arm]) for arm in ARMS}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    final_step = branch_step
    try:
        for step in range(branch_step, max_steps + 1):
            offset = step - branch_step
            records: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                metrics = _evaluate(
                    arm_models[arm],
                    data,
                    device,
                    paper_keys,
                    include_diagnostics=(
                        offset % diagnostic_interval == 0 or step == branch_step
                    ),
                )
                record = {
                    "step": step,
                    **metrics,
                    "learning_rate": float(arm_optimizers[arm].param_groups[0]["lr"]),
                    "first_moment_mode": arm,
                    "second_moment_input": "mixed_gradient_squared",
                    "nodes": nodes,
                    "delayed_nodes": delayed_nodes,
                    "zero_delay_nodes": nodes - delayed_nodes,
                    "delayed_fraction": rho,
                    "delay_steps": tau,
                }
                for name in parameter_names:
                    record[f"adaptive_update_l2_norm_{name}"] = None
                    record[f"relative_adaptive_update_l2_norm_{name}"] = None
                records[arm] = record

            completed_arm_updates = step - branch_step
            if (
                completed_arm_updates > 0
                and completed_arm_updates % PROGRESS_INTERVAL == 0
            ):
                accuracy_text = " | ".join(
                    f"{arm} test={records[arm]['test_accuracy']:.4f}"
                    for arm in ARMS
                )
                print(
                    f"arms completed {completed_arm_updates}/{max_steps - branch_step} "
                    f"updates (global step {step}/{max_steps}) | {accuracy_text}",
                    flush=True,
                )

            if step == max_steps:
                for arm, record in records.items():
                    arm_loggers[arm].log(record)
                    rows_by_arm[arm].append(record)
                final_step = step
                break

            current_gradients: dict[str, list[Tensor | None]] = {}
            stale_gradients: dict[str, list[Tensor | None]] = {}
            for arm in ARMS:
                fresh = _gradient(arm_models[arm], data, device)
                current_gradients[arm] = fresh
                stale_gradients[arm] = (
                    arm_histories[arm][0]
                    if len(arm_histories[arm]) == tau else fresh
                )

            for arm in ARMS:
                fresh = current_gradients[arm]
                stale = stale_gradients[arm]
                mixed, mixed_squares = _mix(fresh, stale, rho)
                if arm == "fresh_first_moment":
                    first_moment_input = fresh
                elif arm == "mixed_first_moment":
                    first_moment_input = mixed
                else:
                    first_moment_input = stale

                arm_model = arm_models[arm]
                arm_optimizer = arm_optimizers[arm]
                arm_parameters = list(arm_model.parameters())
                _set_gradients(arm_parameters, first_moment_input)
                _adamw_step_with_second_moment(
                    arm_optimizer, arm_parameters, mixed_squares
                )
                update_norms = _block_update_norms(
                    arm_optimizer, list(arm_model.named_parameters())
                )
                records[arm].update(update_norms)
                records[arm].update(
                    _optimizer_state_norms(arm_optimizer, arm_parameters)
                )
                arm_schedulers[arm].step()
                arm_optimizer.zero_grad(set_to_none=True)
                arm_loggers[arm].log(records[arm])
                rows_by_arm[arm].append(records[arm])
                arm_histories[arm].append(copy.deepcopy(fresh))
            final_step = step + 1
    finally:
        for logger in arm_loggers.values():
            logger.close()

    for arm, arm_model in arm_models.items():
        torch.save({
            "step": final_step,
            "model_state_dict": arm_model.state_dict(),
            "optimizer_state_dict": arm_optimizers[arm].state_dict(),
            "scheduler_state_dict": arm_schedulers[arm].state_dict(),
            "config": effective_config,
            "train_indices": data.train_indices,
            "test_indices": data.test_indices,
        }, arm_dirs[arm] / "final.pt")
    _save_csv(output_dir / "arm_outcomes.csv", [
        result
        for result in _arm_outcomes(rows_by_arm, branch_step, final_step).values()
    ])
    _make_plots(output_dir, rows_by_arm)
    write_json(output_dir / "first_moment_experiment_status.json", {
        "status": "completed",
        "branch_step": branch_step,
        "final_step": final_step,
        "max_steps": max_steps,
        "arm_count": len(ARMS),
    })
    return output_dir


def plot_first_moment_experiment(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    if not (path / "config.yaml").is_file():
        raise FileNotFoundError(f"not a first-moment experiment directory: {path}")
    rows_by_arm = _read_arm_metrics(path)
    figure_dir = _make_plots(path, rows_by_arm)
    branch_step = int(json.loads((path / "t0_selection.json").read_text())["step"])
    final_step = min(int(rows[-1]["step"]) for rows in rows_by_arm.values())
    _save_csv(path / "arm_outcomes.csv", list(
        _arm_outcomes(rows_by_arm, branch_step, final_step).values()
    ))
    return figure_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fresh, mixed, and delayed-only AdamW first moments."
    )
    parser.add_argument("--config", default="configs/modular_addition_p113.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--delayed-nodes", type=int, default=4)
    parser.add_argument("--tau", type=int, default=4)
    parser.add_argument("--run-dir")
    parser.add_argument("--selection-streak", type=int, default=100)
    parser.add_argument("--diagnostic-interval", type=int, default=20)
    parser.add_argument("--plot-only", metavar="RUN_DIR")
    args = parser.parse_args()
    if args.plot_only:
        figures = plot_first_moment_experiment(args.plot_only)
        print(f"Figures saved to {figures}")
        return
    if args.seed is None:
        parser.error("--seed is required unless --plot-only is used")
    config = load_config(args.config)
    run_dir = run_first_moment_experiment(
        config,
        seed=args.seed,
        run_dir=args.run_dir,
        nodes=args.nodes,
        delayed_nodes=args.delayed_nodes,
        tau=args.tau,
        selection_streak=args.selection_streak,
        diagnostic_interval=args.diagnostic_interval,
    )
    print(f"First-moment experiment complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
