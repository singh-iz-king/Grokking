from __future__ import annotations

import json
import random
import shutil
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from src.config import validate_config
from src.dataset import ModularAdditionData, make_modular_addition_data
from src.device import select_device
from src.evaluation import cross_entropy, evaluate_model
from src.logging_utils import RunLogger
from src.model import NandaOneLayerTransformer
from src.progress_measures import calculate_progress_measures
from src.trainer import _build_model, _print_run_summary
from src.utils import environment_info, seed_everything, write_json

NODE_COUNT = 5


def _snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }


def _save_distributed_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    history: deque[tuple[int, dict[str, Tensor]]],
    config: dict[str, Any],
    environment: dict[str, Any],
    data: ModularAdditionData,
    staleness_rng: random.Random,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "config": config,
            "environment": environment,
            "train_indices": data.train_indices,
            "test_indices": data.test_indices,
            "distributed_state": {
                "nodes": NODE_COUNT,
                "tau": config["distributed"]["tau"],
                "history": [
                    {"version": version, "state_dict": state}
                    for version, state in history
                ],
                "staleness_rng_state": staleness_rng.getstate(),
            },
        },
        path,
    )


def _log_nodes(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()


def train_distributed(
    config: dict[str, Any],
    tau: int = 5,
    nodes: int = NODE_COUNT,
    run_dir: str | Path | None = None,
) -> Path:
    """Simulate async centralized gradient aggregation with node-local staleness.

    Each node computes a full-batch gradient on its selected historical global
    model, using the same complete training split. The central AdamW optimizer
    applies the mean of those gradients to the latest global model once.
    """
    validate_config(config)
    if nodes != NODE_COUNT:
        raise ValueError(f"this experiment is defined for exactly {NODE_COUNT} nodes")
    if tau < 0:
        raise ValueError("tau must be nonnegative")

    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = select_device(
        config["device"]["selection"],
        bool(config["device"]["allow_cpu_fallback"]),
    )
    effective_config = dict(config)
    effective_config["distributed"] = {
        "enabled": True,
        "nodes": nodes,
        "tau": tau,
        "aggregation": "mean_of_full_batch_node_gradients",
        "optimizer_semantics": "canonical_adamw_applied_once_per_global_step",
        "startup": "synchronous_until_tau_history_steps_exist",
        "node_metrics": "selected_gradient_source_model",
    }
    experiment = dict(config["experiment"])
    experiment["name"] = f"{experiment['name']}_distributed_async_tau{tau}"
    effective_config["experiment"] = experiment

    if run_dir is not None:
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{experiment['name']}_seed{seed}_{stamp}"
        output_dir = Path(config["output"]["root"]) / name
        output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = Path(config["output"]["checkpoint_root"]) / output_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    effective_config["runtime"] = {
        "selected_device": str(device),
        "seed": seed,
        "run_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "torch_deterministic_algorithms": True,
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective_config, handle, sort_keys=False)

    environment = environment_info(device)
    data = make_modular_addition_data(
        modulus=int(config["experiment"]["modulus"]),
        train_fraction=float(config["experiment"]["train_fraction"]),
        seed=seed,
    )
    environment.update({
        "train_examples": int(data.train_indices.numel()),
        "test_examples": int(data.test_indices.numel()),
        "total_examples": int(data.all_inputs.shape[0]),
        "distributed_nodes": nodes,
        "tau": tau,
    })
    write_json(output_dir / "environment.json", environment)

    model = _build_model(config).to(device)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    warmup = int(optimizer_config["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / warmup, 1.0) if warmup else 1.0,
    )
    _print_run_summary(
        effective_config, data, device, environment, output_dir, checkpoint_dir, model
    )
    print(
        f"Distributed simulation: {nodes} nodes | tau={tau} | "
        "full-batch gradients | mean aggregation | canonical AdamW",
        flush=True,
    )
    if tau > 0:
        print(
            f"Staleness startup: synchronous through global step {tau}; "
            f"then each node samples lag uniformly from 1..{tau}.",
            flush=True,
        )

    total_steps = int(config["experiment"]["epochs"])
    logging = config["logging"]
    mech_config = config["mechanistic"]
    key_config = mech_config["key_frequencies"]
    fixed_frequencies = list(key_config["paper_mainline"])
    staleness_rng = random.Random(seed + 1_000_003)
    history: deque[tuple[int, dict[str, Tensor]]] = deque(
        [(0, _snapshot(model))],
        maxlen=tau + 1,
    )
    train_inputs = data.train_inputs.to(device)
    train_labels = data.train_labels.to(device)
    version_basic_metrics: dict[int, dict[str, float]] = {
        0: evaluate_model(
            model, data.all_inputs, data.all_labels,
            data.train_indices, data.test_indices, device,
        )
    }
    metric_cache: dict[int, dict[str, Any]] = {}
    nodes_path = output_dir / "node_metrics.jsonl"
    nodes_path.touch()
    training_started_at = time.perf_counter()
    last_progress_at = training_started_at

    try:
        with RunLogger(output_dir) as logger:
            initial_row = {
                "step": 0,
                "model_version": 0,
                **version_basic_metrics[0],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "staleness": [0] * nodes,
                "node_source_versions": [0] * nodes,
                "distributed_tau": tau,
                "distributed_nodes": nodes,
                "progress_interval_seconds": None,
                "elapsed_seconds": None,
            }
            if 0 % int(logging["mechanistic_metrics_interval"]) == 0:
                initial_measures, initial_spectra = calculate_progress_measures(
                    model,
                    data,
                    fixed_frequencies,
                    float(key_config["neuron_explained_fraction"]),
                    include_ablation_metrics=bool(
                        mech_config.get("frequency_ablations", True)
                    ),
                    frequency_mode=str(key_config["mode"]),
                )
                metric_cache[0] = initial_measures
                initial_row.update(initial_measures)
                if bool(mech_config.get("save_fourier_spectra", True)):
                    logger.log_spectra(0, 0, initial_spectra)
            logger.log(initial_row)
            _log_nodes(nodes_path, [
                {
                    "step": 0,
                    "node": node,
                    "staleness": 0,
                    "source_model_version": 0,
                    **version_basic_metrics[0],
                    "progress_measures": metric_cache.get(0),
                }
                for node in range(nodes)
            ])

            for step in range(total_steps):
                global_version = step
                current_state = _snapshot(model)
                available_versions = {version for version, _ in history}
                if tau == 0 or step < tau:
                    lags = [0] * nodes
                else:
                    lags = [
                        staleness_rng.randint(1, tau)
                        for _ in range(nodes)
                    ]
                source_versions = [
                    global_version - lag for lag in lags
                ]
                history_by_version = dict(history)
                if any(version not in available_versions for version in source_versions):
                    raise RuntimeError(
                        f"requested stale version absent from history at step {step}"
                    )

                accumulated_gradients: list[Tensor | None] = [
                    None for _ in model.parameters()
                ]
                for source_version in source_versions:
                    model.load_state_dict(history_by_version[source_version])
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(train_inputs)
                    assert isinstance(logits, Tensor)
                    loss = cross_entropy(logits[:, -1], train_labels)
                    loss.backward()
                    for index, parameter in enumerate(model.parameters()):
                        if parameter.grad is not None:
                            gradient = parameter.grad.detach().clone()
                            if accumulated_gradients[index] is None:
                                accumulated_gradients[index] = gradient
                            else:
                                accumulated_gradients[index].add_(gradient)

                model.load_state_dict(current_state)
                optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(model.parameters(), accumulated_gradients):
                    if gradient is not None:
                        parameter.grad = gradient.div_(nodes)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                new_version = global_version + 1
                new_state = _snapshot(model)
                history.append((new_version, new_state))
                current_metrics = evaluate_model(
                    model, data.all_inputs, data.all_labels,
                    data.train_indices, data.test_indices, device,
                )
                version_basic_metrics[new_version] = current_metrics
                minimum_cached_version = max(0, new_version - tau - 1)
                for old_version in list(version_basic_metrics):
                    if old_version < minimum_cached_version:
                        del version_basic_metrics[old_version]
                for old_version in list(metric_cache):
                    if old_version < minimum_cached_version:
                        del metric_cache[old_version]

                row: dict[str, Any] = {
                    "step": new_version,
                    "model_version": new_version,
                    **current_metrics,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "staleness": lags,
                    "node_source_versions": source_versions,
                    "distributed_tau": tau,
                    "distributed_nodes": nodes,
                }
                mechanistic_due = (
                    new_version % int(logging["mechanistic_metrics_interval"]) == 0
                )
                node_progress: dict[int, dict[str, Any]] = {}
                if mechanistic_due:
                    source_versions_to_analyze = sorted(set(source_versions))
                    for source_version in source_versions_to_analyze:
                        model.load_state_dict(history_by_version[source_version])
                        measures, _ = calculate_progress_measures(
                            model,
                            data,
                            fixed_frequencies,
                            float(key_config["neuron_explained_fraction"]),
                            include_ablation_metrics=bool(
                                mech_config.get("frequency_ablations", True)
                            ) and new_version % int(logging["ablation_interval"]) == 0,
                            frequency_mode=str(key_config["mode"]),
                        )
                        metric_cache[source_version] = measures

                    model.load_state_dict(new_state)
                    new_measures, spectra = calculate_progress_measures(
                        model,
                        data,
                        fixed_frequencies,
                        float(key_config["neuron_explained_fraction"]),
                        include_ablation_metrics=bool(
                            mech_config.get("frequency_ablations", True)
                        ) and new_version % int(logging["ablation_interval"]) == 0,
                        frequency_mode=str(key_config["mode"]),
                    )
                    metric_cache[new_version] = new_measures
                    row.update(new_measures)
                    if bool(mech_config.get("save_fourier_spectra", True)):
                        logger.log_spectra(new_version, new_version, spectra)
                model.load_state_dict(new_state)

                node_records = []
                for node, (lag, source_version) in enumerate(zip(lags, source_versions)):
                    node_record: dict[str, Any] = {
                        "step": new_version,
                        "node": node,
                        "staleness": lag,
                        "source_model_version": source_version,
                        **version_basic_metrics[source_version],
                    }
                    if mechanistic_due:
                        node_record["progress_measures"] = metric_cache[source_version]
                    node_records.append(node_record)
                _log_nodes(nodes_path, node_records)

                now = time.perf_counter()
                progress_due = (
                    new_version % int(logging["progress_interval"]) == 0
                    or new_version == total_steps
                )
                if progress_due:
                    interval_seconds = now - last_progress_at
                    elapsed_seconds = now - training_started_at
                    row["progress_interval_seconds"] = interval_seconds
                    row["elapsed_seconds"] = elapsed_seconds
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
                    interval = time.strftime("%H:%M:%S", time.gmtime(interval_seconds))
                    print(
                        f"global step {new_version:>6}/{total_steps} | "
                        f"elapsed {elapsed} | last interval {interval} | "
                        f"train acc {current_metrics['train_accuracy']:.4f} | "
                        f"val acc {current_metrics['test_accuracy']:.4f} | "
                        f"mean staleness "
                        f"{sum(lags) / len(lags):.2f} | lags {lags}",
                        flush=True,
                    )
                    last_progress_at = now
                logger.log(row)

                if new_version % int(logging["checkpoint_interval"]) == 0:
                    _save_distributed_checkpoint(
                        checkpoint_dir / f"step_{new_version:07d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        new_version,
                        history,
                        effective_config,
                        environment,
                        data,
                        staleness_rng,
                    )
                early_stop_threshold = config["training"]["early_stopping_test_loss_below"]
                if (
                    early_stop_threshold is not None
                    and current_metrics["test_loss"] < float(early_stop_threshold)
                ):
                    total_steps = new_version
                    break

        _save_distributed_checkpoint(
            checkpoint_dir / "final.pt",
            model,
            optimizer,
            scheduler,
            total_steps,
            history,
            effective_config,
            environment,
            data,
            staleness_rng,
        )
        shutil.copy2(checkpoint_dir / "final.pt", output_dir / "final.pt")
        return output_dir
    except Exception:
        write_json(output_dir / "failure.json", {
            "error": "Distributed training failed; see the raised exception in the console.",
            "latest_step": step + 1 if "step" in locals() else 0,
        })
        raise
