from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from src.config import validate_config
from src.dataset import ModularAdditionData, make_modular_addition_data
from src.device import select_device
from src.evaluation import cross_entropy, evaluate_model
from src.logging_utils import RunLogger
from src.model import NandaOneLayerTransformer
from src.progress_measures import calculate_progress_measures
from src.utils import environment_info, seed_everything, write_json


def _new_run_dir(config: dict[str, Any], requested: str | Path | None) -> Path:
    if requested is not None:
        path = Path(requested)
        path.mkdir(parents=True, exist_ok=False)
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{config['experiment']['name']}_seed{config['experiment']['seed']}_{stamp}"
    path = Path(config["output"]["root"]) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _build_model(config: dict[str, Any]) -> NandaOneLayerTransformer:
    model_config = config["model"]
    return NandaOneLayerTransformer(
        modulus=int(config["experiment"]["modulus"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        d_head=int(model_config["d_head"]),
        d_mlp=int(model_config["d_mlp"]),
        context_length=int(config["data"]["input_length"]),
        activation=str(model_config["activation"]).lower(),
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: dict[str, Any],
    environment: dict[str, Any],
    data: ModularAdditionData,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "environment": environment,
            "train_indices": data.train_indices,
            "test_indices": data.test_indices,
        },
        path,
    )


def _print_run_summary(
    config: dict[str, Any],
    data: ModularAdditionData,
    device: torch.device,
    environment: dict[str, Any],
    run_dir: Path,
    checkpoint_dir: Path,
    model: NandaOneLayerTransformer,
) -> None:
    experiment = config["experiment"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    logging = config["logging"]
    keys = config["mechanistic"]["key_frequencies"]
    accelerator = environment.get("gpu")
    device_detail = f"{device}" + (f" ({accelerator})" if accelerator else "")
    train_count = int(data.train_indices.numel())
    test_count = int(data.test_indices.numel())
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "\n".join([
            "",
            "=" * 72,
            f"Starting run: {experiment['name']}",
            f"Device: {device_detail}",
            (
                f"Data: modular addition mod {experiment['modulus']} | "
                f"train {train_count}/{data.all_inputs.shape[0]} "
                f"({train_count / data.all_inputs.shape[0]:.1%}) | "
                f"validation/test {test_count} | seed {experiment['seed']}"
            ),
            (
                f"Model: {model_config['n_layers']} layer(s), "
                f"{model_config['n_heads']} heads, d_model={model_config['d_model']}, "
                f"d_head={model_config['d_head']}, d_mlp={model_config['d_mlp']}, "
                f"activation={model_config['activation']} | "
                f"{total_params:,} parameters"
            ),
            (
                f"Training: {experiment['epochs']:,} epochs | "
                f"batch={config['training']['batch_size']} | "
                f"loss={config['training']['loss']} "
                f"(float64; training includes equals-token logit)"
            ),
            (
                f"Optimizer: {optimizer_config['name']} | "
                f"lr={optimizer_config['learning_rate']:g} | "
                f"weight_decay={optimizer_config['weight_decay']:g} | "
                f"betas=({optimizer_config['beta1']:g}, "
                f"{optimizer_config['beta2']:g}) | "
                f"eps={optimizer_config['epsilon']:g} | "
                f"warmup={optimizer_config['warmup_steps']} steps"
            ),
            (
                f"Logging: progress every {logging['progress_interval']} epochs | "
                f"train/validation every "
                f"{logging['train_metrics_interval']}/"
                f"{logging['validation_interval']} | "
                f"mechanistic every {logging['mechanistic_metrics_interval']} | "
                f"checkpoints every {logging['checkpoint_interval']} epochs"
            ),
            (
                f"Fourier keys: mode={keys['mode']} | "
                f"configured={keys['paper_mainline']} | "
                f"neuron threshold={keys['neuron_explained_fraction']}"
            ),
            (
                f"Runtime: Python {environment['python'].split()[0]} | "
                f"PyTorch {environment['torch']} | {environment['platform']}"
            ),
            f"Run output: {run_dir}",
            f"Checkpoints: {checkpoint_dir}",
            "=" * 72,
            "",
        ]),
        flush=True,
    )


def train(
    config: dict[str, Any],
    run_dir: str | Path | None = None,
) -> Path:
    validate_config(config)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = select_device(
        config["device"]["selection"],
        bool(config["device"]["allow_cpu_fallback"]),
    )
    output_dir = _new_run_dir(config, run_dir)
    checkpoint_dir = Path(config["output"]["checkpoint_root"]) / output_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    effective_config = dict(config)
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
    write_json(output_dir / "environment.json", environment)

    data = make_modular_addition_data(
        modulus=int(config["experiment"]["modulus"]),
        train_fraction=float(config["experiment"]["train_fraction"]),
        seed=seed,
    )
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
    mech_config = config["mechanistic"]
    logging = config["logging"]
    # Keep the paper's mainline keys as a fixed baseline. Discovery is logged
    # separately and can be selected as the active restricted-loss key set.
    fixed_frequencies = list(mech_config["key_frequencies"]["paper_mainline"])

    total_epochs = int(config["experiment"]["epochs"])
    environment["train_examples"] = int(data.train_indices.numel())
    environment["test_examples"] = int(data.test_indices.numel())
    environment["total_examples"] = int(data.all_inputs.shape[0])
    write_json(output_dir / "environment.json", environment)
    _print_run_summary(
        config, data, device, environment, output_dir, checkpoint_dir, model
    )

    try:
        training_started_at = time.perf_counter()
        last_progress_at = training_started_at
        with RunLogger(output_dir) as logger:
            for epoch in range(total_epochs + 1):
                row: dict[str, Any] = {"epoch": epoch, "step": epoch}
                progress_due = (
                    epoch % int(logging["progress_interval"]) == 0
                    or epoch == total_epochs
                )
                train_due = epoch % int(logging["train_metrics_interval"]) == 0
                early_stop_threshold = config["training"]["early_stopping_test_loss_below"]
                validation_due = (
                    epoch % int(logging["validation_interval"]) == 0
                    or early_stop_threshold is not None
                )
                if train_due or validation_due or progress_due:
                    evaluated = evaluate_model(
                        model,
                        data.all_inputs,
                        data.all_labels,
                        data.train_indices,
                        data.test_indices,
                        device,
                    )
                    if train_due or progress_due:
                        row.update({
                            "train_loss": evaluated["train_loss"],
                            "train_accuracy": evaluated["train_accuracy"],
                        })
                    if validation_due or progress_due:
                        row.update({
                            "test_loss": evaluated["test_loss"],
                            "test_accuracy": evaluated["test_accuracy"],
                        })

                mechanistic_due = (
                    epoch % int(logging["mechanistic_metrics_interval"]) == 0
                )
                if mechanistic_due:
                    measures, spectra = calculate_progress_measures(
                        model=model,
                        data=data,
                        fixed_frequencies=fixed_frequencies,
                        explained_threshold=float(
                            mech_config["key_frequencies"]["neuron_explained_fraction"]
                        ),
                        include_ablation_metrics=(
                            bool(mech_config.get("frequency_ablations", True))
                            and
                            epoch % int(logging["ablation_interval"]) == 0
                        ),
                        frequency_mode=str(
                            mech_config["key_frequencies"]["mode"]
                        ),
                    )
                    row.update(measures)
                    if bool(mech_config.get("save_fourier_spectra", True)):
                        logger.log_spectra(epoch, epoch, spectra)

                row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                now = time.perf_counter()
                if progress_due:
                    interval_seconds = now - last_progress_at
                    elapsed_seconds = now - training_started_at
                    row["progress_interval_seconds"] = interval_seconds
                    row["elapsed_seconds"] = elapsed_seconds
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
                    interval = time.strftime("%H:%M:%S", time.gmtime(interval_seconds))
                    print(
                        f"epoch {epoch:>6}/{total_epochs} | "
                        f"elapsed {elapsed} | last interval {interval} | "
                        f"train acc {row['train_accuracy']:.4f} | "
                        f"val acc {row['test_accuracy']:.4f} | "
                        f"train loss {row['train_loss']:.6f} | "
                        f"val loss {row['test_loss']:.6f}",
                        flush=True,
                    )
                    last_progress_at = now
                logger.log(row)

                if epoch % int(logging["checkpoint_interval"]) == 0:
                    _save_checkpoint(
                        checkpoint_dir / f"step_{epoch:07d}.pt",
                        model, optimizer, scheduler, epoch, effective_config,
                        environment, data,
                    )
                if (
                    early_stop_threshold is not None
                    and row.get("test_loss", float("inf")) < float(early_stop_threshold)
                ):
                    break
                if epoch == total_epochs:
                    break

                model.train()
                train_inputs = data.train_inputs.to(device)
                train_labels = data.train_labels.to(device)
                logits = model(train_inputs)
                assert isinstance(logits, torch.Tensor)
                loss = cross_entropy(logits[:, -1], train_labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        _save_checkpoint(
            checkpoint_dir / "final.pt",
            model, optimizer, scheduler, total_epochs, effective_config,
            environment, data,
        )
        shutil.copy2(checkpoint_dir / "final.pt", output_dir / "final.pt")
        return output_dir
    except Exception:
        write_json(output_dir / "failure.json", {
            "epoch": epoch if "epoch" in locals() else None,
            "error": "Training failed; see the raised exception in the console.",
        })
        raise
