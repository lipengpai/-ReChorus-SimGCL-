"""Run the reproducible CPU comparison and collect final metrics."""

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
RESULTS = ROOT / "experiments" / "results"

MODEL_ARGS = {
    "BPRMF": ["--emb_size", "32"],
    "LightGCN": ["--emb_size", "32", "--n_layers", "2"],
    "SimGCL": [
        "--emb_size", "32", "--n_layers", "2", "--cl_rate", "0.2",
        "--temperature", "0.2", "--eps", "0.1", "--include_ego", "0",
    ],
}


def parse_final_metrics(log_path: Path) -> dict[str, float]:
    # ReChorus uses the platform logging encoding on Windows.  Metric lines are
    # ASCII, so tolerant decoding is sufficient and portable across locales.
    text = log_path.read_bytes().decode("utf-8", errors="ignore")
    matches = re.findall(r"Test After Training: \(([^\n]+)\)", text)
    if not matches:
        raise RuntimeError(f"No final test result in {log_path}")
    metrics = {}
    for pair in matches[-1].split(","):
        name, value = pair.strip().split(":")
        metrics[name] = float(value)
    return metrics


def write_results(rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "model", "seed", "epochs", "seconds", "HR@10", "NDCG@10", "HR@20", "NDCG@20"]
    with (RESULTS / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_results() -> list[dict]:
    path = RESULTS / "results.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def run_one(dataset: str, model: str, epochs: int, seed: int, batch_size: int) -> dict:
    log_path = RESULTS / "logs" / dataset / f"{model}_seed{seed}.txt"
    model_path = RESULTS / "models" / dataset / f"{model}_seed{seed}.pt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "main.py", "--model_name", model,
        "--dataset", dataset, "--path", "../data",
        "--lr", "0.001", "--l2", "0.0001",
        "--batch_size", str(batch_size), "--eval_batch_size", "1024",
        "--epoch", str(epochs), "--early_stop", "3",
        "--num_workers", "0", "--random_seed", str(seed),
        "--topk", "10,20", "--metric", "NDCG,HR", "--main_metric", "NDCG@20",
        "--save_final_results", "0", "--log_file", str(log_path),
        "--model_path", str(model_path),
    ] + MODEL_ARGS[model]

    started = time.perf_counter()
    subprocess.run(command, cwd=SRC, check=True)
    elapsed = time.perf_counter() - started
    metrics = parse_final_metrics(log_path)
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "epochs": epochs,
        "seconds": round(elapsed, 2),
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Grocery_Pilot", "ML_1M_Pilot"])
    parser.add_argument("--models", nargs="+", choices=MODEL_ARGS, default=list(MODEL_ARGS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = read_results()
    for dataset in args.datasets:
        for model in args.models:
            print(f"\n=== {dataset} / {model} ===", flush=True)
            row = run_one(dataset, model, args.epochs, args.seed, args.batch_size)
            rows = [existing for existing in rows if not (
                existing["dataset"] == dataset
                and existing["model"] == model
                and str(existing["seed"]) == str(args.seed)
            )]
            rows.append(row)
            write_results(rows)
            print(row, flush=True)
