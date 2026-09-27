from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset import make_modular_addition_data
from src.device import select_device
from src.model import NandaOneLayerTransformer
from src.progress_measures import calculate_progress_measures
from src.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a saved experiment checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = select_device(
        config["device"]["selection"],
        bool(config["device"]["allow_cpu_fallback"]),
    )
    experiment, model_config, data_config = (
        config["experiment"], config["model"], config["data"]
    )
    model = NandaOneLayerTransformer(
        modulus=int(experiment["modulus"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        d_head=int(model_config["d_head"]),
        d_mlp=int(model_config["d_mlp"]),
        context_length=int(data_config["input_length"]),
        activation=str(model_config["activation"]).lower(),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    data = make_modular_addition_data(
        int(experiment["modulus"]),
        float(experiment["train_fraction"]),
        int(experiment["seed"]),
    )
    frequencies = config["mechanistic"]["key_frequencies"]["paper_mainline"]
    metrics, _ = calculate_progress_measures(
        model,
        data,
        list(frequencies),
        float(config["mechanistic"]["key_frequencies"]["neuron_explained_fraction"]),
        include_ablation_metrics=True,
        frequency_mode=str(config["mechanistic"]["key_frequencies"]["mode"]),
    )
    result = {"epoch": checkpoint["epoch"], **metrics}
    output = args.output or args.checkpoint.with_name("analysis.json")
    write_json(output, result)
    print(f"Checkpoint analysis written to {output} (device: {device})")


if __name__ == "__main__":
    main()

