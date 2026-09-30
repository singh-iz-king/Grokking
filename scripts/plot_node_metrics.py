from __future__ import annotations

import argparse

from src.plotting import plot_node_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-node plots from a distributed run's node_metrics.jsonl."
    )
    parser.add_argument("run_dir", help="Distributed run output directory.")
    args = parser.parse_args()
    figures_dir = plot_node_metrics(args.run_dir)
    print(f"Per-node figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
