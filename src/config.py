from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "experiment",
        "data",
        "model",
        "optimizer",
        "training",
        "logging",
        "mechanistic",
        "device",
        "output",
    }
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    p = int(config["experiment"]["modulus"])
    model = config["model"]
    if p < 3:
        raise ValueError("modulus must be at least 3")
    if int(config["experiment"]["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    if model["d_model"] % model["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    if model["d_head"] != model["d_model"] // model["n_heads"]:
        raise ValueError("d_head must equal d_model // n_heads")
    if int(model["n_layers"]) != 1:
        raise ValueError("this replication implements the one-layer reference model")
    if model["normalization"] is not None:
        raise ValueError("the reference model does not use normalization")
    if model["positional_embeddings"] != "learned":
        raise ValueError("the reference model uses learned positional embeddings")
    if bool(model["causal_attention"]) is not True:
        raise ValueError("the reference model uses causal attention")
    if float(model["dropout"]) != 0.0 or bool(model["tie_embeddings"]):
        raise ValueError("dropout and tied embeddings are not used by the reference model")
    if model["initialization"] != "nanda_manual_normal":
        raise ValueError("unsupported initialization for the reference architecture")
    if int(config["data"]["token_equals"]) != p:
        raise ValueError("equals token id must equal the modulus")
    if int(config["data"]["input_length"]) != 3:
        raise ValueError("the reference input has three tokens: a, b, and equals")
    if config["data"]["split_method"] != "shuffled_all_pairs_python_random":
        raise ValueError("unsupported split method for the reference experiment")
    if not 0 < float(config["experiment"]["train_fraction"]) < 1:
        raise ValueError("train_fraction must be between zero and one")
    if config["training"]["batch_size"] != "full":
        raise ValueError("the reference experiment uses full-batch training")
    if config["training"]["loss"] != "cross_entropy":
        raise ValueError("only reference cross-entropy training is supported")
    if config["training"]["loss_precision"] != "float64":
        raise ValueError("the reference cross-entropy uses float64 log-softmax")
    threshold = config["training"]["early_stopping_test_loss_below"]
    if threshold is not None and float(threshold) < 0:
        raise ValueError("early_stopping_test_loss_below must be nonnegative or null")
    if not bool(config["optimizer"]["weight_decay_all_parameters"]):
        raise ValueError("the reference applies AdamW weight decay to all parameters")
    if config["optimizer"]["name"].lower() != "adamw":
        raise ValueError("the reference experiment uses AdamW")
    if float(config["optimizer"]["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if float(config["optimizer"]["weight_decay"]) < 0:
        raise ValueError("weight_decay must be nonnegative")
    if float(config["optimizer"]["epsilon"]) <= 0:
        raise ValueError("AdamW epsilon must be positive")
    if int(config["optimizer"]["warmup_steps"]) < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if config["mechanistic"]["key_frequencies"]["mode"] not in {"discover", "fixed"}:
        raise ValueError("key-frequency mode must be 'discover' or 'fixed'")
    explained_threshold = float(
        config["mechanistic"]["key_frequencies"]["neuron_explained_fraction"]
    )
    if not 0 < explained_threshold <= 1:
        raise ValueError("neuron_explained_fraction must be in (0, 1]")
    for frequency in config["mechanistic"]["key_frequencies"]["paper_mainline"]:
        if not 1 <= int(frequency) < p // 2:
            raise ValueError(f"invalid key frequency for modulus {p}: {frequency}")
    for name in ("train_metrics_interval", "validation_interval",
                 "progress_interval", "mechanistic_metrics_interval", "checkpoint_interval",
                 "ablation_interval"):
        if int(config["logging"][name]) < 1:
            raise ValueError(f"logging.{name} must be positive")
