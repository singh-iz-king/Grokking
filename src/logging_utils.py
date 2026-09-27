from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl = (self.run_dir / "metrics.jsonl").open("w", encoding="utf-8")
        self.metrics_csv_path = self.run_dir / "metrics.csv"
        self.metrics_csv = None
        self.csv_writer = None
        self.spectra_path = self.run_dir / "fourier_spectra.csv"
        self.spectra_file = self.spectra_path.open("w", newline="", encoding="utf-8")
        self.spectra_writer = csv.DictWriter(
            self.spectra_file,
            fieldnames=[
                "epoch", "step", "matrix", "component_index",
                "component", "norm", "squared_norm",
            ],
        )
        self.spectra_writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        self.metrics_jsonl.write(json.dumps(row, default=str) + "\n")
        self.metrics_jsonl.flush()
        if self.csv_writer is None:
            fields = list(row.keys())
            self.metrics_csv = self.metrics_csv_path.open(
                "w", newline="", encoding="utf-8"
            )
            self.csv_writer = csv.DictWriter(
                self.metrics_csv,
                fieldnames=fields,
                extrasaction="ignore",
            )
            self.csv_writer.writeheader()
        self.csv_writer.writerow({
            key: json.dumps(value) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
        self.metrics_csv.flush()

    def log_spectra(self, epoch: int, step: int, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.spectra_writer.writerow({"epoch": epoch, "step": step, **row})
        self.spectra_file.flush()

    def close(self) -> None:
        self.metrics_jsonl.close()
        self.spectra_file.close()
        if self.metrics_csv is not None:
            self.metrics_csv.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

