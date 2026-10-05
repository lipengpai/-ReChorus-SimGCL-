"""Run full-data CPU experiments with a hard wall-clock budget.

Completed runs are detected from their logs and skipped on restart.  By
default, one newly completed dataset-model run is executed per invocation.
The wall-clock budget is cumulative across invocations: completed epoch times
already present in the canonical logs are charged against the budget.  If the
remaining budget expires, the active child process is terminated and all
previous logs/checkpoints/results are kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT = ROOT / "experiments" / "results_full_cpu"

MODEL_ARGS = {
    "BPRMF": ["--emb_size", "64"],
    "LightGCN": ["--emb_size", "64", "--n_layers", "3"],
    "SimGCL": [
        "--emb_size", "64", "--n_layers", "3", "--cl_rate", "0.2",
        "--temperature", "0.2", "--eps", "0.1", "--include_ego", "0",
    ],
}


def parse_metrics(log_path: Path, split: str = "Test") -> dict[str, float] | None:
    if not log_path.exists():
        return None
    text = log_path.read_bytes().decode("utf-8", errors="ignore")
    matches = re.findall(rf"{split}\s+After Training: \(([^\n]+)\)", text)
    if not matches:
        return None
    return {
        key: float(value)
        for key, value in (pair.strip().split(":") for pair in matches[-1].split(","))
    }


def parse_logged_patience(log_path: Path, fallback: int) -> int:
    """Read the actual early-stop value used by a completed historical run."""
    text = log_path.read_bytes().decode("utf-8", errors="ignore")
    matches = re.findall(r"early_stop\s+\|\s+(\d+)", text)
    return int(matches[-1]) if matches else fallback


def parse_last_completed_epoch(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_bytes().decode("utf-8", errors="ignore")
    matches = re.findall(r"Epoch\s+(\d+)\s+loss=", text)
    return int(matches[-1]) if matches else 0


def logged_training_seconds(plan: list[tuple[str, str]], seed: int) -> float:
    """Return training time already consumed by canonical full-data logs.

    ReChorus records each completed epoch as ``loss=... [N.N s]``.  Summing
    these entries makes the time limit survive process restarts.  Evaluation
    and startup overhead are intentionally not reconstructed, so the active
    invocation still uses a real subprocess timeout for its remaining budget.
    """
    total = 0.0
    for dataset, model in plan:
        log_path = OUTPUT / "logs" / dataset / f"{model}_seed{seed}.txt"
        if not log_path.exists():
            continue
        text = log_path.read_bytes().decode("utf-8", errors="ignore")
        total += sum(
            float(value)
            for value in re.findall(r"Epoch\s+\d+\s+loss=.*?\[([0-9.]+) s\]", text)
        )
    return total


def write_rows(rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "model", "status", "seed", "max_epochs", "patience",
        "elapsed_seconds",
        "HR@10", "NDCG@10", "HR@20", "NDCG@20",
    ]
    with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command_for(
    dataset: str, model: str, seed: int, epochs: int, patience: int
) -> tuple[list[str], Path]:
    log_path = OUTPUT / "logs" / dataset / f"{model}_seed{seed}.txt"
    model_path = OUTPUT / "models" / dataset / f"{model}_seed{seed}.pt"
    resume_state = model_path.with_suffix(".resume.pt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "main.py", "--model_name", model,
        "--dataset", dataset, "--path", "../data", "--gpu", "",
        "--lr", "0.001", "--l2", "0.0001",
        "--batch_size", "2048", "--eval_batch_size", "512",
        "--epoch", str(epochs), "--early_stop", str(patience),
        "--num_workers", "0", "--random_seed", str(seed),
        "--topk", "10,20", "--metric", "NDCG,HR", "--main_metric", "NDCG@20",
        "--save_final_results", "0", "--log_file", str(log_path),
        "--model_path", str(model_path),
        "--resume_state", str(resume_state),
    ] + MODEL_ARGS[model]
    resume_epoch = parse_last_completed_epoch(log_path)
    if resume_epoch > 0 and model_path.exists() and parse_metrics(log_path) is None:
        if resume_state.exists():
            command += ["--resume_epoch", str(resume_epoch)]
        else:
            command += ["--load", "1", "--resume_epoch", str(resume_epoch)]
        print(
            f"RESUME partial run after epoch {resume_epoch}: {dataset} / {model}",
            flush=True,
        )
    return command, log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit-hours", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--runs-per-invocation", type=int, default=1,
        help="Stop after this many newly completed runs so later commands can resume.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")
    if args.runs_per_invocation < 1:
        raise ValueError("--runs-per-invocation must be at least 1")
    plan = [
        ("Grocery_and_Gourmet_Food", "BPRMF"),
        ("Grocery_and_Gourmet_Food", "LightGCN"),
        ("Grocery_and_Gourmet_Food", "SimGCL"),
        ("ML_1MTOPK", "BPRMF"),
        ("ML_1MTOPK", "LightGCN"),
        ("ML_1MTOPK", "SimGCL"),
    ]
    budget_seconds = args.time_limit_hours * 3600
    historical_seconds = logged_training_seconds(plan, args.seed)
    started_all = time.monotonic()
    rows: list[dict] = []
    newly_completed = 0

    for dataset, model in plan:
        command, log_path = command_for(
            dataset, model, args.seed, args.epochs, args.patience
        )
        existing = parse_metrics(log_path)
        if existing:
            existing_patience = parse_logged_patience(log_path, args.patience)
            rows.append({
                "dataset": dataset, "model": model, "status": "completed-existing",
                "seed": args.seed, "max_epochs": args.epochs,
                "patience": existing_patience, "elapsed_seconds": 0,
                **existing,
            })
            write_rows(rows)
            print(f"SKIP completed: {dataset} / {model}", flush=True)
            continue

        elapsed_all = time.monotonic() - started_all
        remaining = budget_seconds - historical_seconds - elapsed_all
        if remaining <= 0:
            print("TIME LIMIT reached before starting the next run.", flush=True)
            break

        print(
            f"\n=== FULL CPU: {dataset} / {model}; "
            f"remaining budget {remaining / 3600:.2f} h ===",
            flush=True,
        )
        started_run = time.monotonic()
        try:
            subprocess.run(command, cwd=SRC, check=True, timeout=remaining)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started_run
            rows.append({
                "dataset": dataset, "model": model, "status": "aborted-time-limit",
                "seed": args.seed, "max_epochs": args.epochs,
                "patience": args.patience,
                "elapsed_seconds": round(elapsed, 2),
            })
            write_rows(rows)
            (OUTPUT / "status.json").write_text(json.dumps({
                "status": "aborted-time-limit", "dataset": dataset, "model": model,
                "budget_hours": args.time_limit_hours, "patience": args.patience,
                "historical_training_seconds": round(historical_seconds, 2),
                "invocation_elapsed_seconds": round(time.monotonic() - started_all, 2),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print("TIME LIMIT reached; active run terminated and prior results kept.", flush=True)
            return 124

        metrics = parse_metrics(log_path)
        if metrics is None:
            raise RuntimeError(f"Run ended without final metrics: {log_path}")
        elapsed = time.monotonic() - started_run
        rows.append({
            "dataset": dataset, "model": model, "status": "completed",
            "seed": args.seed, "max_epochs": args.epochs,
            "patience": args.patience,
            "elapsed_seconds": round(elapsed, 2), **metrics,
        })
        write_rows(rows)
        newly_completed += 1
        if newly_completed >= args.runs_per_invocation:
            print(
                "Invocation run limit reached; stopping before the next model. "
                "Rerun this command to resume.",
                flush=True,
            )
            break

    completed = sum(row["status"].startswith("completed") for row in rows)
    status = {
        "status": "completed" if completed == len(plan) else "partial",
        "completed_runs": completed,
        "planned_runs": len(plan),
        "elapsed_seconds": round(
            historical_seconds + time.monotonic() - started_all, 2
        ),
        "historical_training_seconds": round(historical_seconds, 2),
        "invocation_elapsed_seconds": round(time.monotonic() - started_all, 2),
        "budget_hours": args.time_limit_hours,
        "patience": args.patience,
        "runs_per_invocation": args.runs_per_invocation,
    }
    (OUTPUT / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
