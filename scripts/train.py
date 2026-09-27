from __future__ import annotations

import argparse

from src.config import load_config
from src.plotting import plot_results
from src.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the p=113 Grokking replication.")
    parser.add_argument("--config", default="configs/modular_addition_p113.yaml")
    parser.add_argument("--run-dir", help="Optional new output directory (must not exist).")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = train(config, args.run_dir)
    if config["output"]["save_plots"]:
        figures = plot_results(run_dir)
        print(f"Run saved to {run_dir}; figures saved to {figures}")
    else:
        print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()

