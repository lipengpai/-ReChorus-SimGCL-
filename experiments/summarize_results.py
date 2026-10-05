"""Consolidate baseline and validation-selected SimGCL results."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DATASETS = ["Grocery_Pilot", "ML_1M_Pilot"]


def parse_split(log_path: Path, split: str) -> dict[str, float]:
    text = log_path.read_bytes().decode("utf-8", errors="ignore")
    matches = re.findall(rf"{split}\s+After Training: \(([^\n]+)\)", text)
    if not matches:
        raise RuntimeError(f"Missing {split} result in {log_path}")
    return {
        key: float(value)
        for key, value in (pair.strip().split(":") for pair in matches[-1].split(","))
    }


def main() -> None:
    rows = []
    for dataset in DATASETS:
        for model in ("BPRMF", "LightGCN"):
            log_path = RESULTS / "logs" / dataset / f"{model}_seed2026.txt"
            dev, test = parse_split(log_path, "Dev"), parse_split(log_path, "Test")
            rows.append({
                "dataset": dataset, "model": model, "configuration": "default comparison",
                **{f"dev_{key}": value for key, value in dev.items()},
                **{f"test_{key}": value for key, value in test.items()},
            })

        default_log = RESULTS / "logs" / dataset / "SimGCL_seed2026.txt"
        default_dev, default_test = parse_split(default_log, "Dev"), parse_split(default_log, "Test")
        rows.append({
            "dataset": dataset, "model": "SimGCL-default",
            "configuration": "cl_rate=0.2, include_ego=0",
            **{f"dev_{key}": value for key, value in default_dev.items()},
            **{f"test_{key}": value for key, value in default_test.items()},
        })

    tuning = pd.read_csv(RESULTS / "simgcl_tuning.csv")
    for dataset in DATASETS:
        subset = tuning[tuning.dataset == dataset]
        selected = subset.loc[subset["dev_NDCG@20"].idxmax()]
        rows.append({
            "dataset": dataset, "model": "SimGCL-tuned",
            "configuration": f"cl_rate={selected.cl_rate:g}, include_ego={int(selected.include_ego)}",
            **{column: float(selected[column]) for column in subset.columns if column.startswith(("dev_", "test_"))},
        })

    fieldnames = [
        "dataset", "model", "configuration",
        "dev_HR@10", "dev_NDCG@10", "dev_HR@20", "dev_NDCG@20",
        "test_HR@10", "test_NDCG@10", "test_HR@20", "test_NDCG@20",
    ]
    output = RESULTS / "final_results.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows)[fieldnames].to_string(index=False))


if __name__ == "__main__":
    main()
