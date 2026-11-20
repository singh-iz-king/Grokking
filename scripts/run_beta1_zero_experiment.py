from __future__ import annotations

import argparse
import copy
import csv
import math
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml
from torch import Tensor, nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_first_moment_experiment import (
    MAX_EXPERIMENT_STEPS,
    PROGRESS_INTERVAL,
    _optimizer_state_norms,
)
from scripts.run_primary_experiment import (
    _adamw_step_with_second_moment,
    _arm_outcomes,
    _block_update_norms,
    _evaluate,
    _gradient,
    _run_directory,
    _set_gradients,
)
from scripts.run_second_moment_experiment import _mixed_gradient, _second_moment_inputs
from src.config import load_config, validate_config
from src.dataset import make_modular_addition_data
from src.device import select_device
from src.logging_utils import RunLogger
from src.optimizers import build_optimizer
from src.trainer import _build_model
from src.utils import environment_info, seed_everything, write_json


ARMS = {
    "fresh_gradient": "Fresh gradient",
    "mixed_gradient": "Mixed fresh/stale gradient",
    "stale_gradient": "Fully stale gradient",
}
COLORS = {
    "fresh_gradient": "#2878b5",
    "mixed_gradient": "#e1812c",
    "stale_gradient": "#3a923a",
}


def _read_arm_metrics(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        path = run_dir / arm / "metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing metrics for {arm}: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty metrics file: {path}")
        converted: list[dict[str, Any]] = []
        for row in rows:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in (None, ""):
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            parsed["step"] = int(parsed["step"])
            converted.append(parsed)
        result[arm] = converted
    return result


def _write_outcomes(path: Path, outcomes: dict[str, dict[str, Any]]) -> None:
    if not outcomes:
        return
    rows = list(outcomes.values())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(run_dir: Path, rows_by_arm: dict[str, list[dict[str, Any]]]) -> Path:
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    panels = (
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
        ("first_moment_exp_avg_l2_norm", "AdamW first-moment L2 norm (beta1=0)", "First-moment norm"),
        ("second_moment_sqrt_l2_norm", "AdamW second-moment square-root L2 norm", "Second-moment norm"),
        ("weight_l2_norm", "Total parameter L2 norm", "Weight norm"),
    )
    for metric, title, ylabel in panels:
        fig, ax = plt.subplots(figsize=(9, 5))
        has_values = False
        for arm, rows in rows_by_arm.items():
            points: list[tuple[int, float]] = []
            for row in rows:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    points.append((int(row["step"]), value))
            if points:
                ax.plot(
                    [step for step, _ in points],
                    [value for _, value in points],
                    color=COLORS[arm],
                    label=ARMS[arm],
                    linewidth=1.2,
                )
                has_values = True
        if not has_values:
            plt.close(fig)
            continue
        ax.set(title=title, xlabel="Global optimizer updates", ylabel=ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{metric}.png", dpi=180)
        fig.savefig(figure_dir / f"{metric}.pdf")
        plt.close(fig)
    return figure_dir


def run_beta1_zero_experiment(
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
        raise ValueError("the beta1=0 experiment requires AdamW")
    if nodes < 2 or not 1 <= delayed_nodes < nodes:
        raise ValueError("delayed_nodes must be between 1 and nodes - 1")
    if tau < 1 or selection_streak < 1 or diagnostic_interval < 1:
        raise ValueError("tau and evaluation/logging intervals must be positive")

    experiment_config = copy.deepcopy(config)
    experiment_config["experiment"]["seed"] = int(seed)
    experiment_config["experiment"]["name"] += "_beta1_zero_comparison"
    configured_steps = int(experiment_config["experiment"]["epochs"])
    max_steps = min(configured_steps, MAX_EXPERIMENT_STEPS)
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
    effective_config["beta1_zero_experiment"] = {
        "seed": seed,
        "nodes": nodes,
        "delayed_nodes": delayed_nodes,
        "zero_delay_nodes": nodes - delayed_nodes,
        "rho": rho,
        "tau": tau,
        "beta1": 0.0,
        "branch_rule": {
            "train_accuracy_at_least": selection_train_accuracy,
            "test_accuracy_below": selection_test_max,
            "consecutive_evaluations": selection_streak,
            "restricted_loss_used_for_selection": False,
        },
        "first_moment_inputs": {
            "fresh_gradient": "current fresh full-batch gradient",
            "mixed_gradient": "weighted fresh/stale gradient mixture",
            "stale_gradient": "fully stale full-batch gradient at step t-tau",
        },
        "second_moment_input": "square of the weighted fresh/stale gradient mixture in every arm",
        "configured_max_steps": configured_steps,
        "effective_max_steps": max_steps,
        "progress_interval": PROGRESS_INTERVAL,
        "diagnostic_interval": diagnostic_interval,
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
            logger.log({
                "step": step,
                **metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            baseline_rows.append(metrics)
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
        write_json(output_dir / "beta1_zero_experiment_status.json", {
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
        "criterion": effective_config["beta1_zero_experiment"]["branch_rule"],
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
        for group in arm_optimizer.param_groups:
            beta2 = float(group["betas"][1])
            group["betas"] = (0.0, beta2)
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

    loggers = {arm: RunLogger(arm_dirs[arm]) for arm in ARMS}
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
                record: dict[str, Any] = {
                    "step": step,
                    **metrics,
                    "learning_rate": float(arm_optimizers[arm].param_groups[0]["lr"]),
                    "beta1": 0.0,
                    "first_moment_input": arm,
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

            completed = step - branch_step
            if completed > 0 and completed % PROGRESS_INTERVAL == 0:
                accuracy = " | ".join(
                    f"{ARMS[arm]} test={records[arm]['test_accuracy']:.4f}"
                    for arm in ARMS
                )
                print(
                    f"arms completed {completed}/{max_steps - branch_step} updates "
                    f"(global step {step}/{max_steps}) | {accuracy}",
                    flush=True,
                )

            if step == max_steps:
                for arm, record in records.items():
                    loggers[arm].log(record)
                    rows_by_arm[arm].append(record)
                final_step = step
                break

            gradients: dict[str, list[Tensor | None]] = {}
            stales: dict[str, list[Tensor | None]] = {}
            second_inputs: dict[str, list[Tensor | None]] = {}
            mixed_inputs: dict[str, list[Tensor | None]] = {}
            for arm in ARMS:
                fresh = _gradient(arm_models[arm], data, device)
                stale = arm_histories[arm][0] if len(arm_histories[arm]) == tau else fresh
                gradients[arm] = fresh
                stales[arm] = stale
                mixed_inputs[arm] = _mixed_gradient(fresh, stale, rho)
                second_inputs[arm] = _second_moment_inputs(fresh, stale, rho)[
                    "mixed_square"
                ]

            for arm in ARMS:
                fresh = gradients[arm]
                stale = stales[arm]
                first_input = {
                    "fresh_gradient": fresh,
                    "mixed_gradient": mixed_inputs[arm],
                    "stale_gradient": stale,
                }[arm]
                arm_model = arm_models[arm]
                arm_optimizer = arm_optimizers[arm]
                arm_parameters = list(arm_model.parameters())
                _set_gradients(arm_parameters, first_input)
                _adamw_step_with_second_moment(
                    arm_optimizer, arm_parameters, second_inputs[arm]
                )
                records[arm].update(_block_update_norms(
                    arm_optimizer, list(arm_model.named_parameters())
                ))
                records[arm].update(
                    _optimizer_state_norms(arm_optimizer, arm_parameters)
                )
                loggers[arm].log(records[arm])
                rows_by_arm[arm].append(records[arm])
                arm_schedulers[arm].step()
                arm_optimizer.zero_grad(set_to_none=True)
                arm_histories[arm].append(copy.deepcopy(fresh))
            final_step = step + 1
    finally:
        for logger in loggers.values():
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
    outcomes = _arm_outcomes(rows_by_arm, branch_step, final_step)
    with (output_dir / "arm_outcomes.csv").open("w", newline="", encoding="utf-8") as handle:
        outcome_rows = list(outcomes.values())
        writer = csv.DictWriter(handle, fieldnames=list(outcome_rows[0]))
        writer.writeheader()
        writer.writerows(outcome_rows)
    _plot(output_dir, rows_by_arm)
    write_json(output_dir / "beta1_zero_experiment_status.json", {
        "status": "completed",
        "branch_step": branch_step,
        "final_step": final_step,
        "max_steps": max_steps,
        "arm_count": len(ARMS),
    })
    return output_dir


def plot_beta1_zero_experiment(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    if not (path / "config.yaml").is_file():
        raise FileNotFoundError(f"not a beta1=0 experiment directory: {path}")
    return _plot(path, _read_arm_metrics(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fresh, mixed, and stale AdamW inputs with beta1=0."
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
        print(f"Figures saved to {plot_beta1_zero_experiment(args.plot_only)}")
        return
    if args.seed is None:
        parser.error("--seed is required unless --plot-only is used")
    output = run_beta1_zero_experiment(
        load_config(args.config),
        seed=args.seed,
        run_dir=args.run_dir,
        nodes=args.nodes,
        delayed_nodes=args.delayed_nodes,
        tau=args.tau,
        selection_streak=args.selection_streak,
        diagnostic_interval=args.diagnostic_interval,
    )
    print(f"Beta1=0 experiment complete: {output}", flush=True)


if __name__ == "__main__":
    main()
