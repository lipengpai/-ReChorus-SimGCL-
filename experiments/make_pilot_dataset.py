"""Create a deterministic, resource-bounded ReChorus dataset subset."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_part(source: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(source / f"{name}.csv", sep="\t")
    if "neg_items" in frame:
        frame["neg_items"] = frame["neg_items"].map(
            lambda value: ast.literal_eval(value) if isinstance(value, str) else value
        )
    return frame


def create_pilot(source: Path, output: Path, users: int, negatives: int, seed: int) -> dict:
    train, dev, test = [load_part(source, part) for part in ("train", "dev", "test")]
    eligible = sorted(set(train.user_id) & set(dev.user_id) & set(test.user_id))
    if len(eligible) < users:
        raise ValueError(f"Only {len(eligible)} users occur in all three partitions; requested {users}.")

    rng = np.random.default_rng(seed)
    selected_users = set(rng.choice(np.array(eligible), size=users, replace=False).tolist())
    parts = [frame[frame.user_id.isin(selected_users)].copy() for frame in (train, dev, test)]

    all_positive = pd.concat(parts, ignore_index=True)
    user_map = {int(old): new for new, old in enumerate(sorted(selected_users), 1)}
    item_map = {
        int(old): new for new, old in enumerate(sorted(all_positive.item_id.unique()), 1)
    }
    for frame in parts:
        frame["user_id"] = frame["user_id"].map(user_map).astype(int)
        frame["item_id"] = frame["item_id"].map(item_map).astype(int)
        frame.sort_values(["user_id", "time"], inplace=True)

    combined = pd.concat(parts, ignore_index=True)
    all_items = np.array(sorted(combined.item_id.unique()), dtype=int)
    clicked = combined.groupby("user_id")["item_id"].apply(set).to_dict()
    for offset, frame in enumerate(parts[1:], 1):
        part_rng = np.random.default_rng(seed + offset)
        sampled = []
        for user_id in frame.user_id:
            candidates = np.setdiff1d(
                all_items, np.fromiter(clicked[user_id], dtype=int), assume_unique=False
            )
            if len(candidates) < negatives:
                raise ValueError(
                    f"User {user_id} has {len(candidates)} eligible negatives; requested {negatives}."
                )
            sampled.append(part_rng.choice(candidates, size=negatives, replace=False).tolist())
        frame["neg_items"] = sampled

    output.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("train", "dev", "test"), parts):
        keep = [column for column in frame.columns if column != "neg_items"]
        if name != "train":
            keep.append("neg_items")
        frame[keep].to_csv(output / f"{name}.csv", sep="\t", index=False)

    stats = {
        "source": source.name,
        "seed": seed,
        "users": users,
        "items": len(item_map),
        "train_interactions": len(parts[0]),
        "dev_interactions": len(parts[1]),
        "test_interactions": len(parts[2]),
        "negative_candidates_per_eval_case": negatives,
    }
    (output / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(create_pilot(args.source, args.output, args.users, args.negatives, args.seed),
                     ensure_ascii=False, indent=2))
