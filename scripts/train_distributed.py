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


def run_fixed_delay_composition_sweep(
    config: dict,
    tau: int,
    nodes: int = NODE_COUNT,
    run_dir: str | Path | None = None,
) -> list[tuple[int, Path]]:
    """Run fixed-delay experiments for every zero-delay node count."""
    results: list[tuple[int, Path]] = []
    for zero_delay_nodes in range(nodes + 1):
        tag = (
            f"mixed_fixed_tau_{tau}_zero_delay_nodes_{zero_delay_nodes}"
            f"_nodes_{nodes}"
        )
        specific_run_dir = Path(run_dir) / tag if run_dir is not None else None
        print(
            f"\nFixed-delay composition sweep: tau={tau} | nodes={nodes} | "
            f"zero-delay nodes={zero_delay_nodes} | delayed nodes="
            f"{nodes - zero_delay_nodes}",
            flush=True,
        )
        output_dir = train_distributed(
            config,
            tau=tau,
            nodes=nodes,
            run_dir=specific_run_dir,
            staleness_mode="mixed_fixed",
            zero_delay_nodes=zero_delay_nodes,
        )
        if config["output"]["save_plots"]:
            figures = plot_results(output_dir)
            node_figures = plot_node_metrics(output_dir)
            print(
                f"Figures saved to {figures}; per-node figures saved to {node_figures}",
                flush=True,
            )
        results.append((zero_delay_nodes, output_dir))
    return results


def run_gradient_rejection_sweep(
    config: dict,
    tau: int = 5,
    nodes: int = 6,
    mean_cosine_cutoff: float = -0.25,
    run_dir: str | Path | None = None,
) -> list[tuple[int, Path]]:
    """Run every fixed-delay composition with cosine-triggered update rejection."""
    results: list[tuple[int, Path]] = []
    for delayed_workers in range(nodes + 1):
        zero_delay_nodes = nodes - delayed_workers
        threshold_tag = (
            f"{mean_cosine_cutoff:g}".replace("-", "m").replace(".", "p")
        )
        tag = (
            f"reject_mean_cos_below_{threshold_tag}_fixed_tau_{tau}"
            f"_delayed_nodes_{delayed_workers}_nodes_{nodes}"
        )
        specific_run_dir = Path(run_dir) / tag if run_dir is not None else None
        print(
            f"\nGradient-rejection sweep: tau={tau} | nodes={nodes} | "
            f"delayed workers={delayed_workers} | zero-delay nodes={zero_delay_nodes} | "
            f"reject mean cosine < {mean_cosine_cutoff:g}",
            flush=True,
        )
        output_dir = train_distributed(
            config,
            tau=tau,
            nodes=nodes,
            run_dir=specific_run_dir,
            staleness_mode="mixed_fixed",
            zero_delay_nodes=zero_delay_nodes,
            reject_below_mean_cosine=mean_cosine_cutoff,
        )
        if config["output"]["save_plots"]:
            figures = plot_results(output_dir)
            node_figures = plot_node_metrics(output_dir)
            print(
                f"Figures saved to {figures}; per-node figures saved to {node_figures}",
                flush=True,
            )
        results.append((delayed_workers, output_dir))
    return results


def run_staged_delay_schedule(
    config: dict,
    tau: int = 5,
    nodes: int = 6,
    phase_steps: int = 10_000,
    run_dir: str | Path | None = None,
) -> Path:
    """Run the grokking-triggered 3/6 -> 4/6 -> 3/6 -> 5/6 -> 6/6 schedule."""
    if nodes != 6:
        raise ValueError("the staged delay schedule requires exactly 6 nodes")
    output_dir = train_distributed(
        config,
        tau=tau,
        nodes=nodes,
        run_dir=run_dir,
        staleness_mode="mixed_fixed",
        zero_delay_nodes=nodes - 3,
        staged_delay_schedule=True,
        staged_phase_steps=phase_steps,
    )
    if config["output"]["save_plots"]:
        figures = plot_results(output_dir)
        node_figures = plot_node_metrics(output_dir)
        print(
            f"Figures saved to {figures}; per-node figures saved to {node_figures}",
            flush=True,
        )
    return output_dir


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
        default=None,
        help="Number of logical nodes (default: 5 for existing modes, 6 for new experiments).",
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
    parser.add_argument(
        "--fixed-delay-sweep",
        action="store_true",
        help=(
            "With --fixed-delay and --nodes, run once for every zero-delay "
            "node count from 0 through the total."
        ),
    )
    parser.add_argument(
        "--reject-misaligned-sweep",
        action="store_true",
        help=(
            "Run every fixed-delay worker composition, rejecting the entire central "
            "update when mean delayed-worker cosine is below the cutoff."
        ),
    )
    parser.add_argument(
        "--mean-cosine-cutoff",
        type=float,
        default=-0.25,
        help="Rejection-sweep cutoff; reject when mean delayed-worker cosine is strictly below this value.",
    )
    parser.add_argument(
        "--staged-delay-schedule",
        action="store_true",
        help=(
            "Run 3/6 delayed until measured test accuracy reaches 0.99, then "
            "4/6, 3/6, 5/6, 3/6, 6/6, and 3/6."
        ),
    )
    parser.add_argument(
        "--phase-steps",
        type=int,
        default=10_000,
        help="Attempted global steps in each post-trigger staged phase (default: 10000).",
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
    selected_nodes = (
        args.nodes
        if args.nodes is not None
        else (6 if args.reject_misaligned_sweep or args.staged_delay_schedule else NODE_COUNT)
    )
    if selected_nodes < 1:
        parser.error("--nodes must be a positive integer")
    new_experiment_modes = sum((
        args.reject_misaligned_sweep,
        args.staged_delay_schedule,
    ))
    if new_experiment_modes > 1:
        parser.error("--reject-misaligned-sweep and --staged-delay-schedule are separate experiments")
    if args.reject_misaligned_sweep:
        if args.tau is not None or args.zero_delay_nodes is not None:
            parser.error("--reject-misaligned-sweep cannot be combined with --tau or --zero-delay-nodes")
        if args.step_size_sweep or args.fixed_delay_sweep:
            parser.error("--reject-misaligned-sweep cannot be combined with existing sweep flags")
        fixed_delay = 5 if args.fixed_delay is None else args.fixed_delay
        if fixed_delay < 0:
            parser.error("--fixed-delay must be nonnegative")
        if not -1.0 <= args.mean_cosine_cutoff <= 1.0:
            parser.error("--mean-cosine-cutoff must be between -1 and 1")
        config = load_config(args.config)
        results = run_gradient_rejection_sweep(
            config,
            tau=fixed_delay,
            nodes=selected_nodes,
            mean_cosine_cutoff=args.mean_cosine_cutoff,
            run_dir=args.run_dir,
        )
        print("\nGradient-rejection sweep complete:", flush=True)
        for delayed_workers, output_dir in results:
            print(
                f"  delayed workers={delayed_workers}/{selected_nodes} | run={output_dir}",
                flush=True,
            )
        return
    if args.staged_delay_schedule:
        if selected_nodes != 6:
            parser.error("--staged-delay-schedule requires --nodes 6")
        if args.tau is not None or args.zero_delay_nodes is not None:
            parser.error("--staged-delay-schedule cannot be combined with --tau or --zero-delay-nodes")
        if args.step_size_sweep or args.fixed_delay_sweep:
            parser.error("--staged-delay-schedule cannot be combined with other sweep flags")
        if args.phase_steps < 1:
            parser.error("--phase-steps must be positive")
        fixed_delay = 5 if args.fixed_delay is None else args.fixed_delay
        if fixed_delay < 0:
            parser.error("--fixed-delay must be nonnegative")
        config = load_config(args.config)
        run_dir = run_staged_delay_schedule(
            config,
            tau=fixed_delay,
            nodes=selected_nodes,
            phase_steps=args.phase_steps,
            run_dir=args.run_dir,
        )
        print(f"Staged delay schedule saved to {run_dir}")
        return
    if args.fixed_delay_sweep and args.nodes is None:
        # Preserve the original fixed-delay sweep's five-node default.
        selected_nodes = NODE_COUNT
    if args.fixed_delay_sweep:
        if args.fixed_delay is None:
            parser.error("--fixed-delay-sweep requires --fixed-delay")
        if args.zero_delay_nodes is not None:
            parser.error("--fixed-delay-sweep cannot be combined with --zero-delay-nodes")
        if args.tau is not None:
            parser.error("use --fixed-delay-sweep with --fixed-delay, not --tau")
        if args.fixed_delay < 0:
            parser.error("--fixed-delay must be nonnegative")
        if args.step_size_sweep:
            parser.error("--fixed-delay-sweep cannot be combined with --step-size-sweep")
        config = load_config(args.config)
        results = run_fixed_delay_composition_sweep(
            config,
            tau=args.fixed_delay,
            nodes=selected_nodes,
            run_dir=args.run_dir,
        )
        print("\nFixed-delay composition sweep complete:", flush=True)
        for zero_delay_nodes, output_dir in results:
            print(
                f"  zero-delay nodes={zero_delay_nodes} | "
                f"delayed nodes={selected_nodes - zero_delay_nodes} | run={output_dir}",
                flush=True,
            )
        return
    if args.fixed_delay is not None or args.zero_delay_nodes is not None:
        if args.fixed_delay is None or args.zero_delay_nodes is None:
            parser.error("--fixed-delay and --zero-delay-nodes must be provided together")
        if args.tau is not None:
            parser.error("use either --tau or --fixed-delay, not both")
        if args.fixed_delay < 0:
            parser.error("--fixed-delay must be nonnegative")
        if not 0 <= args.zero_delay_nodes <= selected_nodes:
            parser.error(f"--zero-delay-nodes must be between 0 and {selected_nodes}")
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
            nodes=selected_nodes,
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
        nodes=selected_nodes,
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
