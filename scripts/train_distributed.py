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
    staleness_mode: str = "uniform",
    zero_delay_nodes: int | None = None,
    nodes: int = NODE_COUNT,
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
        if staleness_mode == "mixed_fixed":
            mode_tag = (
                f"mixed_fixed_tau_{tau}_zero_delay_nodes_{zero_delay_nodes}"
                f"_nodes_{nodes}"
            )
        else:
            mode_tag = f"uniform_tau_{tau}_nodes_{nodes}"
        run_tag = f"{mode_tag}_{tag}"
        specific_run_dir = Path(run_dir) / run_tag if run_dir is not None else None

        print(
            f"\nStep-size sweep: factor={factor:g} | "
            f"learning_rate={learning_rate:.12g} | tau={tau} | "
            f"staleness_mode={staleness_mode}",
            flush=True,
        )
        output_dir = train_distributed(
            run_config,
            tau=tau,
            nodes=nodes,
            run_dir=specific_run_dir,
            staleness_mode=staleness_mode,
            zero_delay_nodes=zero_delay_nodes,
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
        description="Simulate asynchronous centralized Grokking training."
    )
    parser.add_argument(
        "--config",
        default="configs/modular_addition_p113.yaml",
        help="Experiment YAML; its epoch count is interpreted as global update steps.",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=None,
        help="Maximum staleness in global updates; 0 gives synchronous centralized training.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=NODE_COUNT,
        help=f"Number of logical nodes (default: {NODE_COUNT}).",
    )
    parser.add_argument(
        "--fixed-delay",
        type=int,
        help=(
            "Use a mixed fixed-delay mode with --zero-delay-nodes: remaining "
            "nodes use exactly this many update steps of delay."
        ),
    )
    parser.add_argument(
        "--zero-delay-nodes",
        type=int,
        help=(
            "In --fixed-delay mode, assign node IDs 0..N-1 to zero delay; "
            "remaining nodes use the fixed delay."
        ),
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
    if args.nodes < 1:
        parser.error("--nodes must be a positive integer")
    if args.fixed_delay is not None or args.zero_delay_nodes is not None:
        if args.fixed_delay is None or args.zero_delay_nodes is None:
            parser.error("--fixed-delay and --zero-delay-nodes must be provided together")
        if args.tau is not None:
            parser.error("use either --tau or --fixed-delay, not both")
        if args.fixed_delay < 0:
            parser.error("--fixed-delay must be nonnegative")
        if not 0 <= args.zero_delay_nodes <= args.nodes:
            parser.error(f"--zero-delay-nodes must be between 0 and {args.nodes}")
        tau = args.fixed_delay
        staleness_mode = "mixed_fixed"
        zero_delay_nodes = args.zero_delay_nodes
    else:
        tau = 5 if args.tau is None else args.tau
        if tau < 0:
            parser.error("--tau must be nonnegative")
        staleness_mode = "uniform"
        zero_delay_nodes = None

    config = load_config(args.config)
    if args.step_size_sweep:
        results = run_learning_rate_sweep(
            config,
            tau=tau,
            run_dir=args.run_dir,
            staleness_mode=staleness_mode,
            zero_delay_nodes=zero_delay_nodes,
            nodes=args.nodes,
        )
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
        tau=tau,
        nodes=args.nodes,
        run_dir=args.run_dir,
        staleness_mode=staleness_mode,
        zero_delay_nodes=zero_delay_nodes,
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
