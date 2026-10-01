from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from src.config import load_config
from src.distributed_trainer import NODE_COUNT, train_distributed
from src.plotting import plot_node_metrics, plot_results


def _learning_rate_tag(factor: float, learning_rate: float) -> str:
    factor_tag = f"{factor:g}".replace(".", "p")
    rate_tag = f"{learning_rate:.12g}".replace(".", "p").replace("-", "m")
    return f"lr_factor_{factor_tag}_lr_{rate_tag}"


def run_learning_rate_sweep(
    config: dict,
    tau: int,
    run_dir: str | Path | None = None,
) -> list[tuple[float, float, Path]]:
    """Run the configured LR and its 0.1x / 0.01x variants sequentially."""
    base_learning_rate = float(config["optimizer"]["learning_rate"])
    results: list[tuple[float, float, Path]] = []
    for factor in (1.0, 0.1, 0.01):
        run_config = deepcopy(config)
        learning_rate = base_learning_rate * factor
        tag = _learning_rate_tag(factor, learning_rate)
        run_config["optimizer"]["learning_rate"] = learning_rate
        run_config["experiment"]["name"] = (
            f"{config['experiment']['name']}_{tag}"
        )
        specific_run_dir = Path(run_dir) / tag if run_dir is not None else None

        print(
            f"\nStep-size sweep: factor={factor:g} | "
            f"learning_rate={learning_rate:.12g} | tau={tau}",
            flush=True,
        )
        output_dir = train_distributed(
            run_config,
            tau=tau,
            nodes=NODE_COUNT,
            run_dir=specific_run_dir,
        )
        if config["output"]["save_plots"]:
            figures = plot_results(output_dir)
            node_figures = plot_node_metrics(output_dir)
            print(
                f"Figures saved to {figures}; per-node figures saved to {node_figures}",
                flush=True,
            )
        results.append((factor, learning_rate, output_dir))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate five-node asynchronous centralized Grokking training."
    )
    parser.add_argument(
        "--config",
        default="configs/modular_addition_p113.yaml",
        help="Experiment YAML; its epoch count is interpreted as global update steps.",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=5,
        help="Maximum staleness in global updates; 0 gives synchronous centralized training.",
    )
    parser.add_argument("--run-dir", help="Optional new output directory (must not exist).")
    parser.add_argument(
        "--step-size-sweep",
        action="store_true",
        help=(
            "Run three sequential experiments at the configured learning rate, "
            "0.1x, and 0.01x."
        ),
    )
    args = parser.parse_args()
    if args.tau < 0:
        parser.error("--tau must be nonnegative")

    config = load_config(args.config)
    if args.step_size_sweep:
        results = run_learning_rate_sweep(config, tau=args.tau, run_dir=args.run_dir)
        print("\nStep-size sweep complete:", flush=True)
        for factor, learning_rate, output_dir in results:
            print(
                f"  factor={factor:g} | learning_rate={learning_rate:.12g} | "
                f"run={output_dir}",
                flush=True,
            )
        return

    run_dir = train_distributed(
        config,
        tau=args.tau,
        nodes=NODE_COUNT,
        run_dir=args.run_dir,
    )
    if config["output"]["save_plots"]:
        figures = plot_results(run_dir)
        node_figures = plot_node_metrics(run_dir)
        print(
            f"Distributed run saved to {run_dir}; figures saved to {figures} "
            f"and per-node figures to {node_figures}"
        )
    else:
        print(f"Distributed run saved to {run_dir}")


if __name__ == "__main__":
    main()
