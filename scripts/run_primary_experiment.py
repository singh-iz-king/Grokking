from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import torch
import yaml
from torch import Tensor, nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, validate_config
from src.dataset import ModularAdditionData, make_modular_addition_data
from src.device import select_device
from src.evaluation import cross_entropy, evaluate_model
from src.fourier_analysis import project_frequency_components, restrict_logits
from src.logging_utils import RunLogger
from src.optimizers import build_optimizer
from src.trainer import _build_model
from src.utils import environment_info, seed_everything, write_json


ARM_NAMES = ("A_fresh_adamw", "B_delayed_adamw", "C_no_second_moment_cancellation")
METRIC_FIELDS = (
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "weight_l2_norm",
    "restricted_train_loss_paper_keys",
    "restricted_test_loss_paper_keys",
    "excluded_train_loss_paper_keys",
    "excluded_test_loss_paper_keys",
    "restricted_train_loss_all_addition",
    "restricted_test_loss_all_addition",
    "excluded_train_loss_all_addition",
    "excluded_test_loss_all_addition",
    "adaptive_update_l2_norm",
    "relative_adaptive_update_l2_norm",
    "learning_rate",
)


def _run_directory(config: dict[str, Any], requested: str | Path | None) -> Path:
    if requested is not None:
        path = Path(requested)
        path.mkdir(parents=True, exist_ok=False)
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"primary_{config['experiment']['name']}_seed{config['experiment']['seed']}_{stamp}"
    path = Path(config["output"]["root"]) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _gradient(model: nn.Module, data: ModularAdditionData, device: torch.device) -> list[Tensor | None]:
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(data.train_inputs.to(device))
    assert isinstance(logits, Tensor)
    loss = cross_entropy(logits[:, -1], data.train_labels.to(device))
    loss.backward()
    gradients = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in model.parameters()
    ]
    model.zero_grad(set_to_none=True)
    return gradients


def _mix_gradients(
    fresh: list[Tensor | None],
    stale: list[Tensor | None],
    rho: float,
) -> tuple[list[Tensor | None], list[Tensor | None]]:
    mixed: list[Tensor | None] = []
    average_squares: list[Tensor | None] = []
    for current, previous in zip(fresh, stale):
        if current is None and previous is None:
            mixed.append(None)
            average_squares.append(None)
            continue
        if current is None:
            assert previous is not None
            current = torch.zeros_like(previous)
        if previous is None:
            previous = torch.zeros_like(current)
        mixed.append((1.0 - rho) * current + rho * previous)
        average_squares.append(
            (1.0 - rho) * current.square() + rho * previous.square()
        )
    return mixed, average_squares


def _set_gradients(parameters: list[nn.Parameter], gradients: list[Tensor | None]) -> None:
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.detach() if gradient is not None else None


@torch.no_grad()
def _adamw_step_with_second_moment(
    optimizer: torch.optim.Optimizer,
    parameters: list[nn.Parameter],
    second_moment_inputs: list[Tensor | None],
) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("the primary experiment requires AdamW")
    parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        for parameter in group["params"]:
            index = parameter_indices[id(parameter)]
            gradient = parameter.grad
            second_input = second_moment_inputs[index]
            if gradient is None or second_input is None:
                continue
            state = optimizer.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"].add_(1)
            step = int(state["step"].item())
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).add_(second_input, alpha=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            parameter.mul_(1.0 - lr * weight_decay)
            parameter.addcdiv_(exp_avg, denominator, value=-lr / bias_correction1)


def _parameter_norm(parameters: list[nn.Parameter]) -> float:
    total = sum(
        float(parameter.detach().double().square().sum().item())
        for parameter in parameters
        if parameter.requires_grad
    )
    return math.sqrt(total)


def _block_update_norms(
    optimizer: torch.optim.Optimizer,
    named_parameters: list[tuple[str, nn.Parameter]],
    epsilon: float = 1e-12,
) -> dict[str, float]:
    norms: list[float] = []
    total_parameter_squared = 0.0
    total_update_squared = 0.0
    result: dict[str, float] = {}
    for name, parameter in named_parameters:
        state = optimizer.state.get(parameter, {})
        if not state or "exp_avg" not in state:
            update_norm = 0.0
        else:
            group = next(
                item for item in optimizer.param_groups
                if any(value is parameter for value in item["params"])
            )
            step = int(state["step"].item())
            beta1, beta2 = group["betas"]
            first = state["exp_avg"] / (1.0 - beta1**step)
            second = state["exp_avg_sq"] / (1.0 - beta2**step)
            direction = first / (second.sqrt() + float(group["eps"]))
            update_norm = float(direction.double().norm().item()) * float(group["lr"])
        param_norm = float(parameter.detach().double().norm().item())
        result[f"adaptive_update_l2_norm_{name}"] = update_norm
        result[f"relative_adaptive_update_l2_norm_{name}"] = update_norm / (param_norm + epsilon)
        norms.append(update_norm)
        total_parameter_squared += param_norm**2
        total_update_squared += update_norm**2
    result["adaptive_update_l2_norm"] = math.sqrt(total_update_squared)
    result["relative_adaptive_update_l2_norm"] = (
        result["adaptive_update_l2_norm"] / (math.sqrt(total_parameter_squared) + epsilon)
    )
    return result


def _diagnostic_losses(
    logits: Tensor,
    labels: Tensor,
    train_indices: Tensor,
    test_indices: Tensor,
    modulus: int,
    paper_keys: list[int],
) -> dict[str, float]:
    logits = logits[:, -1, :modulus].detach().cpu().double()
    labels = labels.cpu()
    train_indices = train_indices.cpu()
    test_indices = test_indices.cpu()
    all_keys = list(range(1, (modulus + 1) // 2))
    results: dict[str, float] = {}
    for name, keys in (("paper_keys", paper_keys), ("all_addition", all_keys)):
        restricted = restrict_logits(logits, modulus, keys, preserve_mean=True)
        excluded = logits.clone()
        for frequency in keys:
            excluded -= project_frequency_components(logits, modulus, frequency)
        for split, indices in (("train", train_indices), ("test", test_indices)):
            results[f"restricted_{split}_loss_{name}"] = float(
                cross_entropy(restricted[indices], labels[indices]).item()
            )
            results[f"excluded_{split}_loss_{name}"] = float(
                cross_entropy(excluded[indices], labels[indices]).item()
            )
    return results


def _evaluate(
    model: nn.Module,
    data: ModularAdditionData,
    device: torch.device,
    paper_keys: list[int],
    include_diagnostics: bool,
) -> dict[str, float]:
    metrics = evaluate_model(
        model, data.all_inputs, data.all_labels,
        data.train_indices, data.test_indices, device,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    metrics["weight_l2_norm"] = _parameter_norm(parameters)
    if include_diagnostics:
        was_training = model.training
        model.eval()
        logits = model(data.all_inputs.to(device))
        assert isinstance(logits, Tensor)
        metrics.update(_diagnostic_losses(
            logits, data.all_labels, data.train_indices, data.test_indices,
            int(model.modulus), paper_keys,
        ))
        if was_training:
            model.train()
    return metrics


def _circuit_gradient(
    model: nn.Module,
    data: ModularAdditionData,
    device: torch.device,
    paper_keys: list[int],
) -> list[Tensor | None]:
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(data.all_inputs.to(device))
    assert isinstance(logits, Tensor)
    restricted = restrict_logits(
        logits[:, -1, :int(model.modulus)],
        int(model.modulus),
        paper_keys,
        preserve_mean=True,
    )
    loss = cross_entropy(
        restricted[data.test_indices.to(device)],
        data.all_labels[data.test_indices].to(device),
    )
    loss.backward()
    result = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in model.parameters()
    ]
    model.zero_grad(set_to_none=True)
    return result


def _dot(left: list[Tensor | None], right: list[Tensor | None]) -> float:
    return sum(
        float(torch.sum(a.double() * b.double()).item())
        for a, b in zip(left, right)
        if a is not None and b is not None
    )


def _squared_norm(values: list[Tensor | None]) -> float:
    return sum(
        float(value.double().square().sum().item())
        for value in values if value is not None
    )


def _persistence_summary(
    pairs: list[tuple[list[Tensor | None], list[Tensor | None]]],
    circuit_gradient: list[Tensor | None],
) -> dict[str, float]:
    q_squared = _squared_norm(circuit_gradient)
    result: dict[str, float] = {}
    for label, complement in (("circuit", False), ("remaining", True)):
        current_energy = stale_energy = cross_energy = signed_sum = 0.0
        for current, stale in pairs:
            coefficient = _dot(current, circuit_gradient) / q_squared if q_squared else 0.0
            projected_current = [
                (value - coefficient * q if complement else coefficient * q)
                if value is not None and q is not None else None
                for value, q in zip(current, circuit_gradient)
            ]
            stale_coefficient = _dot(stale, circuit_gradient) / q_squared if q_squared else 0.0
            projected_stale = [
                (value - stale_coefficient * q if complement else stale_coefficient * q)
                if value is not None and q is not None else None
                for value, q in zip(stale, circuit_gradient)
            ]
            current_energy += _squared_norm(projected_current)
            stale_energy += _squared_norm(projected_stale)
            cross_energy += _dot(projected_current, projected_stale)
            signed_sum += _dot(current, circuit_gradient)
        denominator = math.sqrt(current_energy * stale_energy)
        result[f"persistence_{label}"] = cross_energy / denominator if denominator else math.nan
        result[f"lagged_energy_ratio_{label}"] = stale_energy / current_energy if current_energy else math.nan
        result[f"signed_circuit_alignment_{label}"] = (
            signed_sum / math.sqrt(q_squared * current_energy)
            if q_squared and current_energy and not complement else math.nan
        )
    return result


def _adaptive_direction(optimizer: torch.optim.Optimizer, parameters: list[nn.Parameter]) -> list[Tensor | None]:
    result: list[Tensor | None] = []
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if not state or "exp_avg" not in state:
            result.append(torch.zeros_like(parameter))
            continue
        group = next(
            item for item in optimizer.param_groups
            if any(value is parameter for value in item["params"])
        )
        step = int(state["step"].item())
        beta1, beta2 = group["betas"]
        first = state["exp_avg"] / (1 - beta1**step)
        second = state["exp_avg_sq"] / (1 - beta2**step)
        result.append(first / (second.sqrt() + float(group["eps"])))
    return result


def _common_path_shadows(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> dict[str, tuple[nn.Module, torch.optim.Optimizer]]:
    shadows: dict[str, tuple[nn.Module, torch.optim.Optimizer]] = {}
    for name in ("A", "B", "C"):
        shadow_model = _build_model(config).to(next(model.parameters()).device)
        shadow_model.load_state_dict(copy.deepcopy(model.state_dict()))
        shadow_optimizer = build_optimizer(shadow_model.parameters(), config["optimizer"])
        shadow_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        shadows[name] = (shadow_model, shadow_optimizer)
    return shadows


def _shadow_step(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    gradients: list[Tensor | None],
    second_moment_inputs: list[Tensor | None] | None = None,
) -> None:
    parameters = list(model.parameters())
    _set_gradients(parameters, gradients)
    if second_moment_inputs is None:
        optimizer.step()
    else:
        _adamw_step_with_second_moment(optimizer, parameters, second_moment_inputs)
    optimizer.zero_grad(set_to_none=True)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_plots(output_dir: Path, arm_rows: dict[str, list[dict[str, Any]]]) -> None:
    colors = {"A_fresh_adamw": "#2878b5", "B_delayed_adamw": "#e1812c",
              "C_no_second_moment_cancellation": "#3a923a"}
    labels = {
        "A_fresh_adamw": "A: Fresh AdamW",
        "B_delayed_adamw": "B: Delayed AdamW",
        "C_no_second_moment_cancellation": "C: No second-moment cancellation",
    }
    panels = [
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
        ("weight_l2_norm", "Total parameter L2 norm", "Weight norm"),
    ]
    for field, title, ylabel in panels:
        fig, ax = plt.subplots(figsize=(9, 5))
        for arm, rows in arm_rows.items():
            points = [(row["step"], row[field]) for row in rows
                      if row.get(field) is not None]
            if points:
                ax.plot([point[0] for point in points], [point[1] for point in points],
                        label=labels[arm], color=colors[arm], linewidth=1.2)
        ax.set(title=title, xlabel="Global optimizer updates", ylabel=ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{field}.png", dpi=180)
        fig.savefig(output_dir / f"{field}.pdf")
        plt.close(fig)


def _arm_outcomes(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    branch_step: int,
    final_step: int,
    confirmation_steps: int = 1000,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm, rows in rows_by_arm.items():
        by_step = {int(row["step"]): row for row in rows}
        start = by_step.get(branch_step, {})
        horizon = by_step.get(branch_step + 1000, {})
        primary_start = start.get("restricted_test_loss_paper_keys")
        primary_end = horizon.get("restricted_test_loss_paper_keys")
        sustained = None
        ordered_steps = list(range(branch_step, final_step + 1))
        bad_prefix = [0]
        for current_step in ordered_steps:
            accuracy = by_step.get(current_step, {}).get("test_accuracy")
            bad_prefix.append(
                bad_prefix[-1]
                + int(accuracy is None or float(accuracy) < 0.99)
            )
        window_size = confirmation_steps + 1
        for index in range(max(0, len(ordered_steps) - window_size + 1)):
            if bad_prefix[index + window_size] == bad_prefix[index]:
                step = ordered_steps[index]
                sustained = step
                break
        result[arm] = {
            "arm": arm,
            "branch_step": branch_step,
            "final_step": final_step,
            "restricted_loss_reduction_1000_steps": (
                float(primary_start) - float(primary_end)
                if primary_start is not None and primary_end is not None else None
            ),
            "sustained_grokking_step": sustained,
            "sustained_grokking_right_censored": sustained is None,
            "confirmation_window_steps": confirmation_steps,
        }
    return result


def run_primary_experiment(
    config: dict[str, Any],
    *,
    seed: int,
    run_dir: str | Path | None = None,
    nodes: int = 6,
    delayed_nodes: int = 3,
    tau: int = 5,
    selection_train_accuracy: float = 0.99,
    selection_test_max: float = 0.20,
    selection_streak: int = 100,
    diagnostic_interval: int = 20,
    window_steps: int = 500,
) -> Path:
    validate_config(config)
    if str(config["optimizer"]["name"]).lower() != "adamw":
        raise ValueError("the primary experiment requires AdamW")
    if nodes < 1 or not 1 <= delayed_nodes < nodes:
        raise ValueError("delayed_nodes must be between 1 and nodes - 1")
    if tau < 1 or selection_streak < 1 or diagnostic_interval < 1 or window_steps < 1:
        raise ValueError("tau and logging/window intervals must be positive")
    config = copy.deepcopy(config)
    config["experiment"]["seed"] = int(seed)
    config["experiment"]["name"] = f"{config['experiment']['name']}_primary_three_arm"
    seed_everything(seed)
    device = select_device(config["device"]["selection"], bool(config["device"]["allow_cpu_fallback"]))
    output_dir = _run_directory(config, run_dir)
    baseline_dir = output_dir / "baseline_to_t0"
    baseline_dir.mkdir()
    arm_dirs = {name: output_dir / name for name in ARM_NAMES}
    for directory in arm_dirs.values():
        directory.mkdir()
    effective_config = copy.deepcopy(config)
    effective_config["primary_experiment"] = {
        "seed": seed,
        "nodes": nodes,
        "delayed_nodes": delayed_nodes,
        "rho": delayed_nodes / nodes,
        "tau": tau,
        "branch_rule": {
            "train_accuracy_at_least": selection_train_accuracy,
            "test_accuracy_below": selection_test_max,
            "consecutive_evaluations": selection_streak,
            "restricted_loss_used_for_selection": False,
        },
        "arms": {
            "A": "fresh gradient for first and second moments",
            "B": "fresh/stale mixture for both AdamW moments",
            "C": "fresh/stale mixture for first moment; weighted average of gradient squares for second moment",
        },
        "diagnostic_interval": diagnostic_interval,
        "persistence_window_steps": window_steps,
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective_config, handle, sort_keys=False)
    data = make_modular_addition_data(
        int(config["experiment"]["modulus"]),
        float(config["experiment"]["train_fraction"]),
        seed,
    )
    environment = environment_info(device)
    write_json(output_dir / "environment.json", environment)
    model = _build_model(config).to(device)
    optimizer = build_optimizer(model.parameters(), config["optimizer"])
    warmup = int(config["optimizer"]["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(step / warmup, 1.0) if warmup else 1.0
    )
    paper_keys = list(config["mechanistic"]["key_frequencies"]["paper_mainline"])
    max_steps = int(config["experiment"]["epochs"])
    param_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    history: deque[tuple[int, list[Tensor | None]]] = deque(maxlen=tau)
    selection_count = 0
    branch_step: int | None = None
    baseline_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    with RunLogger(baseline_dir) as logger:
        for step in range(max_steps + 1):
            metrics = _evaluate(
                model, data, device, paper_keys,
                include_diagnostics=(step % diagnostic_interval == 0),
            )
            row: dict[str, Any] = {
                "step": step,
                **{field: metrics.get(field) for field in METRIC_FIELDS},
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            for name in param_names:
                row[f"adaptive_update_l2_norm_{name}"] = None
                row[f"relative_adaptive_update_l2_norm_{name}"] = None
            logger.log(row)
            baseline_rows.append(row)
            if (
                metrics["train_accuracy"] >= selection_train_accuracy
                and metrics["test_accuracy"] < selection_test_max
            ):
                selection_count += 1
            else:
                selection_count = 0
            if selection_count >= selection_streak and len(history) >= tau:
                branch_step = step
                break
            if step == max_steps:
                break
            gradients = _gradient(model, data, device)
            history.append((step, gradients))
            parameters = list(model.parameters())
            _set_gradients(parameters, gradients)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if step % int(config["logging"]["progress_interval"]) == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"baseline step {step}/{max_steps} | "
                    f"train={metrics['train_accuracy']:.4f} "
                    f"test={metrics['test_accuracy']:.4f} | {elapsed:.1f}s",
                    flush=True,
                )
    if branch_step is None:
        write_json(output_dir / "primary_experiment_status.json", {
            "status": "t0_not_found",
            "max_steps": max_steps,
            "selection_train_accuracy": selection_train_accuracy,
            "selection_test_max": selection_test_max,
            "selection_streak": selection_streak,
            "last_train_accuracy": baseline_rows[-1]["train_accuracy"],
            "last_test_accuracy": baseline_rows[-1]["test_accuracy"],
        })
        raise RuntimeError(
            f"No pre-grokking t0 met the selection rule within {max_steps} updates; "
            f"partial baseline is at {baseline_dir}"
        )

    baseline_history = [copy.deepcopy(gradients) for _, gradients in history]
    torch.save({
        "step": branch_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "preceding_gradients": baseline_history,
        "config": effective_config,
        "train_indices": data.train_indices,
        "test_indices": data.test_indices,
    }, output_dir / "checkpoint_t0.pt")
    write_json(output_dir / "t0_selection.json", {
        "step": branch_step,
        "criterion": effective_config["primary_experiment"]["branch_rule"],
        "train_accuracy": baseline_rows[-1]["train_accuracy"],
        "test_accuracy": baseline_rows[-1]["test_accuracy"],
        "restricted_test_loss_paper_keys": baseline_rows[-1].get("restricted_test_loss_paper_keys"),
        "excluded_train_loss_paper_keys": baseline_rows[-1].get("excluded_train_loss_paper_keys"),
    })

    arm_models: dict[str, nn.Module] = {}
    arm_optimizers: dict[str, torch.optim.Optimizer] = {}
    arm_schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
    arm_histories: dict[str, deque[list[Tensor | None]]] = {}
    for arm in ARM_NAMES:
        arm_model = _build_model(config).to(device)
        arm_model.load_state_dict(copy.deepcopy(model.state_dict()))
        arm_optimizer = build_optimizer(arm_model.parameters(), config["optimizer"])
        arm_scheduler = torch.optim.lr_scheduler.LambdaLR(
            arm_optimizer, lr_lambda=lambda count: min(count / warmup, 1.0) if warmup else 1.0
        )
        arm_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        arm_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        arm_models[arm] = arm_model
        arm_optimizers[arm] = arm_optimizer
        arm_schedulers[arm] = arm_scheduler
        arm_histories[arm] = deque(
            (copy.deepcopy(gradients) for gradients in baseline_history),
            maxlen=tau,
        )

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_NAMES}
    arm_loggers = {arm: RunLogger(arm_dirs[arm]) for arm in ARM_NAMES}
    persistence_rows: list[dict[str, Any]] = []
    common_path_rows: list[dict[str, Any]] = []
    window_pairs: dict[str, list[tuple[list[Tensor | None], list[Tensor | None]]]] = {}
    window_q: dict[str, list[Tensor | None]] = {}
    shadow_pack: dict[str, tuple[nn.Module, torch.optim.Optimizer]] | None = None
    rho = delayed_nodes / nodes
    final_step = branch_step
    for step in range(branch_step, max_steps + 1):
        offset = step - branch_step
        if offset % window_steps == 0:
            window_pairs = {arm: [] for arm in ARM_NAMES}
            window_q = {
                arm: _circuit_gradient(arm_models[arm], data, device, paper_keys)
                for arm in ARM_NAMES
            }
            shadow_pack = _common_path_shadows(
                arm_models["A_fresh_adamw"],
                arm_optimizers["A_fresh_adamw"],
                config,
            )
        current_gradients: dict[str, list[Tensor | None]] = {}
        stale_gradients: dict[str, list[Tensor | None]] = {}
        rows_this_step: dict[str, dict[str, Any]] = {}
        for arm in ARM_NAMES:
            arm_model = arm_models[arm]
            include_diagnostics = (
                offset % diagnostic_interval == 0 or step == branch_step
            )
            metrics = _evaluate(
                arm_model, data, device, paper_keys, include_diagnostics
            )
            row = {
                "step": step,
                **{field: metrics.get(field) for field in METRIC_FIELDS},
                "learning_rate": float(arm_optimizers[arm].param_groups[0]["lr"]),
                "delayed_fraction": rho,
                "delay_steps": tau,
                "delayed_nodes": delayed_nodes,
                "nodes": nodes,
            }
            for name in param_names:
                row[f"adaptive_update_l2_norm_{name}"] = None
                row[f"relative_adaptive_update_l2_norm_{name}"] = None
            rows_this_step[arm] = row

        if step == max_steps:
            for arm in ARM_NAMES:
                arm_loggers[arm].log(rows_this_step[arm])
                arm_rows[arm].append(rows_this_step[arm])
            final_step = step
            break
        for arm in ARM_NAMES:
            current = _gradient(arm_models[arm], data, device)
            current_gradients[arm] = current
            stale = arm_histories[arm][0] if len(arm_histories[arm]) == tau else current
            stale_gradients[arm] = stale

        a_fresh = current_gradients["A_fresh_adamw"]
        a_stale = stale_gradients["A_fresh_adamw"]
        mixed_a, squares_a = _mix_gradients(a_fresh, a_stale, rho)
        if shadow_pack is not None:
            shadow_gradients = {
                "A": a_fresh,
                "B": mixed_a,
                "C": mixed_a,
            }
            for shadow_name, gradient in shadow_gradients.items():
                shadow_model, shadow_optimizer = shadow_pack[shadow_name]
                for shadow_group, source_group in zip(
                    shadow_optimizer.param_groups,
                    arm_optimizers["A_fresh_adamw"].param_groups,
                ):
                    shadow_group["lr"] = source_group["lr"]
                _shadow_step(
                    shadow_optimizer,
                    shadow_model,
                    gradient,
                    squares_a if shadow_name == "C" else None,
                )

        for arm in ARM_NAMES:
            fresh = current_gradients[arm]
            stale = stale_gradients[arm]
            if arm == "A_fresh_adamw":
                applied_gradient = fresh
                second_inputs = None
            else:
                applied_gradient, second_inputs = _mix_gradients(fresh, stale, rho)
                if arm == "B_delayed_adamw":
                    second_inputs = None
            arm_model = arm_models[arm]
            arm_optimizer = arm_optimizers[arm]
            parameters = list(arm_model.parameters())
            _set_gradients(parameters, applied_gradient)
            if arm == "C_no_second_moment_cancellation":
                assert second_inputs is not None
                _adamw_step_with_second_moment(
                    arm_optimizer, parameters, second_inputs
                )
            else:
                arm_optimizer.step()
            update_stats = _block_update_norms(
                arm_optimizer,
                list(arm_model.named_parameters()),
            )
            rows_this_step[arm].update(update_stats)
            arm_schedulers[arm].step()
            arm_optimizer.zero_grad(set_to_none=True)
            arm_loggers[arm].log(rows_this_step[arm])
            arm_histories[arm].append(copy.deepcopy(fresh))
            window_pairs[arm].append((fresh, stale))

        if shadow_pack is not None and (offset + 1) % window_steps == 0:
            circuit_at_end = _circuit_gradient(
                arm_models["A_fresh_adamw"], data, device, paper_keys
            )
            parameter_norm = math.sqrt(_squared_norm(circuit_at_end))
            for short_name, (shadow_model, shadow_optimizer) in shadow_pack.items():
                direction = _adaptive_direction(
                    shadow_optimizer, list(shadow_model.parameters())
                )
                progress = _dot(circuit_at_end, direction)
                direction_norm = math.sqrt(_squared_norm(direction))
                common_path_rows.append({
                    "window_start_step": step - window_steps + 1,
                    "window_end_step": step + 1,
                    "arm": short_name,
                    "circuit_progress": progress,
                    "circuit_alignment": (
                        progress / (parameter_norm * direction_norm)
                        if parameter_norm and direction_norm else math.nan
                    ),
                    "adaptive_direction_norm": direction_norm,
                })
            for arm in ARM_NAMES:
                persistence_rows.append({
                    "arm": arm,
                    "window_start_step": step - window_steps + 1,
                    "window_end_step": step + 1,
                    "paired_updates": len(window_pairs[arm]),
                    **_persistence_summary(window_pairs[arm], window_q[arm]),
                })
            shadow_pack = None
        final_step = step + 1

    for logger in arm_loggers.values():
        logger.close()
    for arm in ARM_NAMES:
        arm_model = arm_models[arm]
        torch.save({
            "step": final_step,
            "model_state_dict": arm_model.state_dict(),
            "optimizer_state_dict": arm_optimizers[arm].state_dict(),
            "scheduler_state_dict": arm_schedulers[arm].state_dict(),
            "config": effective_config,
            "train_indices": data.train_indices,
            "test_indices": data.test_indices,
        }, arm_dirs[arm] / "final.pt")
    _write_rows(output_dir / "persistence_windows.csv", persistence_rows)
    _write_rows(output_dir / "common_path_replay.csv", common_path_rows)
    outcomes = _arm_outcomes(arm_rows, branch_step, final_step)
    write_json(output_dir / "arm_outcomes.json", outcomes)
    _summary_plots(output_dir, arm_rows)
    write_json(output_dir / "primary_experiment_status.json", {
        "status": "completed",
        "branch_step": branch_step,
        "final_step": final_step,
        "max_steps": max_steps,
        "arm_count": len(ARM_NAMES),
    })
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paired three-arm primary staleness-mechanism experiment."
    )
    parser.add_argument("--config", default="configs/modular_addition_p113.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--delayed-nodes", type=int, default=3)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--run-dir", help="Optional new output directory.")
    parser.add_argument("--selection-streak", type=int, default=100)
    parser.add_argument("--diagnostic-interval", type=int, default=20)
    parser.add_argument("--window-steps", type=int, default=500)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = run_primary_experiment(
        config,
        seed=args.seed,
        run_dir=args.run_dir,
        nodes=args.nodes,
        delayed_nodes=args.delayed_nodes,
        tau=args.tau,
        selection_streak=args.selection_streak,
        diagnostic_interval=args.diagnostic_interval,
        window_steps=args.window_steps,
    )
    print(f"Primary experiment complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
