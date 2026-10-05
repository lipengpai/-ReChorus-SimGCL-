"""Small validation-only SimGCL grid for the CPU experiment."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT = ROOT / "experiments" / "results" / "simgcl_tuning.csv"


def parse_split(log_path: Path, split: str) -> dict[str, float]:
    text = log_path.read_bytes().decode("utf-8", errors="ignore")
    matches = re.findall(rf"{split}\s+After Training: \(([^\n]+)\)", text)
    if not matches:
        raise RuntimeError(f"No {split} metrics in {log_path}")
    result = {}
    for pair in matches[-1].split(","):
        key, value = pair.strip().split(":")
        result[key] = float(value)
    return result


def run(dataset: str, rate: float, include_ego: int, epochs: int, seed: int) -> dict:
    tag = f"cl{rate:g}_ego{include_ego}"
    log_path = ROOT / "experiments" / "results" / "tuning_logs" / dataset / f"{tag}.txt"
    model_path = ROOT / "experiments" / "results" / "tuning_models" / dataset / f"{tag}.pt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "main.py", "--model_name", "SimGCL",
        "--dataset", dataset, "--path", "../data",
        "--emb_size", "32", "--n_layers", "2", "--cl_rate", str(rate),
        "--temperature", "0.2", "--eps", "0.1", "--include_ego", str(include_ego),
        "--lr", "0.001", "--l2", "0.0001", "--batch_size", "1024",
        "--eval_batch_size", "1024", "--epoch", str(epochs), "--early_stop", "3",
        "--num_workers", "0", "--random_seed", str(seed),
        "--topk", "10,20", "--metric", "NDCG,HR", "--main_metric", "NDCG@20",
        "--save_final_results", "0", "--log_file", str(log_path),
        "--model_path", str(model_path),
    ]
    started = time.perf_counter()
    subprocess.run(command, cwd=SRC, check=True)
    seconds = time.perf_counter() - started
    dev, test = parse_split(log_path, "Dev"), parse_split(log_path, "Test")
    return {
        "dataset": dataset, "cl_rate": rate, "include_ego": include_ego,
        "seed": seed, "seconds": round(seconds, 2),
        **{f"dev_{key}": value for key, value in dev.items()},
        **{f"test_{key}": value for key, value in test.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Grocery_Pilot", "ML_1M_Pilot"])
    parser.add_argument("--rates", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--include-ego", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = []
    for dataset in args.datasets:
        for rate in args.rates:
            for include_ego in args.include_ego:
                print(f"\n=== {dataset} rate={rate} include_ego={include_ego} ===", flush=True)
                row = run(dataset, rate, include_ego, args.epochs, args.seed)
                rows.append(row)
                OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                with OUTPUT.open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(row, flush=True)
