"""Prepare MovieLens-1M for ReChorus Top-K recommendation.

This script mirrors the official ReChorus notebook while fixing its typo and
making the pipeline deterministic.  It never downloads data by itself; pass
the extracted ``ratings.dat`` (and optionally ``movies.dat``) explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_ratings(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        names=["original_user_id", "original_item_id", "rating", "time"],
        engine="python",
        encoding="latin-1",
    )


def positive_k_core(data: pd.DataFrame, min_rating: float, min_core: int) -> pd.DataFrame:
    """Iteratively retain users/items with at least ``min_core`` positives."""
    result = data.copy()
    while True:
        positives = result[result["rating"] >= min_rating]
        user_count = positives.groupby("original_user_id").size()
        item_count = positives.groupby("original_item_id").size()
        keep_users = set(user_count[user_count >= min_core].index)
        keep_items = set(item_count[item_count >= min_core].index)
        filtered = result[
            result["original_user_id"].isin(keep_users)
            & result["original_item_id"].isin(keep_items)
        ].copy()
        if len(filtered) == len(result):
            return filtered
        if filtered.empty:
            raise ValueError("The k-core filter removed every interaction.")
        result = filtered


def add_time_context(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    dt = pd.to_datetime(result["time"], unit="s", utc=True)
    result["c_hour_c"] = dt.dt.hour.astype(int)
    result["c_weekday_c"] = dt.dt.weekday.astype(int)

    def period(hour: int) -> int:
        if 5 <= hour <= 8:
            return 0
        if 8 < hour < 11:
            return 1
        if 11 <= hour <= 12:
            return 2
        if 12 < hour <= 15:
            return 3
        if 15 < hour <= 17:
            return 4
        if 18 <= hour <= 19:
            return 5
        if 19 < hour <= 21:
            return 6
        if hour > 21:
            return 7
        return 8

    result["c_period_c"] = result["c_hour_c"].map(period).astype(int)
    dates = dt.dt.floor("D")
    result["c_day_f"] = (dates - dates.min()).dt.days.astype(int)
    return result


def sample_users(data: pd.DataFrame, max_users: int | None) -> pd.DataFrame:
    if not max_users or data["original_user_id"].nunique() <= max_users:
        return data
    counts = (
        data.groupby("original_user_id")
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "original_user_id"], ascending=[False, True])
    )
    selected = set(counts.head(max_users)["original_user_id"])
    return data[data["original_user_id"].isin(selected)].copy()


def split_by_global_time(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_day = int(data["c_day_f"].max())
    split1, split2 = int(max_day * 0.8), int(max_day * 0.9)
    train = data[data["c_day_f"] <= split1].copy()
    dev = data[(data["c_day_f"] > split1) & (data["c_day_f"] <= split2)].copy()
    test = data[data["c_day_f"] > split2].copy()

    train_users, train_items = set(train["original_user_id"]), set(train["original_item_id"])
    dev = dev[
        dev["original_user_id"].isin(train_users)
        & dev["original_item_id"].isin(train_items)
    ].copy()
    test = test[
        test["original_user_id"].isin(train_users)
        & test["original_item_id"].isin(train_items)
    ].copy()
    if train.empty or dev.empty or test.empty:
        raise ValueError("The temporal split produced an empty train/dev/test partition.")
    return train, dev, test


def remap_ids(parts: list[pd.DataFrame]) -> tuple[dict[int, int], dict[int, int]]:
    combined = pd.concat(parts, ignore_index=True)
    user_map = {int(old): new for new, old in enumerate(sorted(combined["original_user_id"].unique()), 1)}
    item_map = {int(old): new for new, old in enumerate(sorted(combined["original_item_id"].unique()), 1)}
    for part in parts:
        part["user_id"] = part["original_user_id"].map(user_map).astype(int)
        part["item_id"] = part["original_item_id"].map(item_map).astype(int)
        part.sort_values(["user_id", "time"], inplace=True)
    return user_map, item_map


def add_negative_items(parts: list[pd.DataFrame], negatives: int, seed: int) -> None:
    combined = pd.concat(parts, ignore_index=True)
    all_items = np.array(sorted(combined["item_id"].unique()), dtype=int)
    clicked = combined.groupby("user_id")["item_id"].apply(set).to_dict()

    for offset, part in enumerate(parts[1:], 1):
        rng = np.random.default_rng(seed + offset)
        sampled = []
        for user_id in part["user_id"]:
            candidates = np.setdiff1d(all_items, np.fromiter(clicked[user_id], dtype=int), assume_unique=False)
            if len(candidates) < negatives:
                raise ValueError(
                    f"User {user_id} has only {len(candidates)} eligible negatives; "
                    f"requested {negatives}."
                )
            sampled.append(rng.choice(candidates, size=negatives, replace=False).tolist())
        part["neg_items"] = sampled


def write_item_meta(movies_path: Path, output_dir: Path, item_map: dict[int, int]) -> None:
    movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    movies = movies[movies["movie_id"].isin(item_map)].copy()
    movies["item_id"] = movies["movie_id"].map(item_map).astype(int)
    genre_map = {value: idx for idx, value in enumerate(sorted(movies["genres"].unique()), 1)}
    title_map = {value: idx for idx, value in enumerate(sorted(movies["title"].unique()), 1)}
    movies["i_genre_c"] = movies["genres"].map(genre_map)
    movies["i_title_c"] = movies["title"].map(title_map)
    movies[["item_id", "i_genre_c", "i_title_c"]].to_csv(
        output_dir / "item_meta.csv", sep="\t", index=False
    )


def prepare(
    ratings_path: Path,
    output_dir: Path,
    movies_path: Path | None = None,
    min_rating: float = 4.0,
    min_core: int = 5,
    negatives: int = 99,
    seed: int = 0,
    max_users: int | None = None,
) -> dict:
    raw = read_ratings(ratings_path)
    core = positive_k_core(raw, min_rating, min_core)
    positives = core[core["rating"] >= min_rating].copy()
    positives = sample_users(positives, max_users)
    positives = add_time_context(positives)
    train, dev, test = split_by_global_time(positives)
    parts = [train, dev, test]
    user_map, item_map = remap_ids(parts)
    add_negative_items(parts, negatives, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_columns = [
        "user_id", "item_id", "time", "c_hour_c", "c_weekday_c", "c_period_c", "c_day_f"
    ]
    train[base_columns].to_csv(output_dir / "train.csv", sep="\t", index=False)
    dev[base_columns + ["neg_items"]].to_csv(output_dir / "dev.csv", sep="\t", index=False)
    test[base_columns + ["neg_items"]].to_csv(output_dir / "test.csv", sep="\t", index=False)
    (output_dir / "user2newid.json").write_text(json.dumps(user_map, indent=2), encoding="utf-8")
    (output_dir / "item2newid.json").write_text(json.dumps(item_map, indent=2), encoding="utf-8")
    if movies_path:
        write_item_meta(movies_path, output_dir, item_map)

    stats = {
        "dataset": "MovieLens-1M",
        "users": len(user_map),
        "items": len(item_map),
        "train_interactions": len(train),
        "dev_interactions": len(dev),
        "test_interactions": len(test),
        "negative_candidates_per_eval_case": negatives,
        "min_rating": min_rating,
        "min_core": min_core,
        "max_users": max_users,
        "seed": seed,
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True, help="Path to extracted ratings.dat")
    parser.add_argument("--movies", type=Path, help="Optional path to extracted movies.dat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--min-core", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-users", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = prepare(
        args.ratings,
        args.output_dir,
        args.movies,
        args.min_rating,
        args.min_core,
        args.negatives,
        args.seed,
        args.max_users,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
