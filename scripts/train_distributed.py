from __future__ import annotations

import argparse

from src.config import load_config
from src.distributed_trainer import NODE_COUNT, train_distributed
from src.plotting import plot_results


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
    args = parser.parse_args()
    if args.tau < 0:
        parser.error("--tau must be nonnegative")

    config = load_config(args.config)
    run_dir = train_distributed(
        config,
        tau=args.tau,
        nodes=NODE_COUNT,
        run_dir=args.run_dir,
    )
    if config["output"]["save_plots"]:
        figures = plot_results(run_dir)
        print(f"Distributed run saved to {run_dir}; figures saved to {figures}")
    else:
        print(f"Distributed run saved to {run_dir}")


if __name__ == "__main__":
    main()
