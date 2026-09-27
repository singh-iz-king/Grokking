from __future__ import annotations

import argparse

from src.plotting import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures from saved run metrics.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(f"Figures saved to {plot_results(args.run_dir)}")


if __name__ == "__main__":
    main()

