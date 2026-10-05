import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.prepare_movielens import prepare  # noqa: E402


class PrepareMovieLensTest(unittest.TestCase):
    def test_temporal_split_remap_and_negative_sampling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            ratings = temp / "ratings.dat"
            rows = []
            base_time = 946684800
            for user_id in range(1, 11):
                for day in range(10):
                    item_id = ((user_id * 3 + day) % 20) + 1
                    rows.append(f"{user_id}::{item_id}::5::{base_time + day * 86400}\n")
            ratings.write_text("".join(rows), encoding="latin-1")

            output = temp / "prepared"
            stats = prepare(
                ratings, output, min_core=2, negatives=3, seed=7
            )

            self.assertGreater(stats["train_interactions"], 0)
            self.assertGreater(stats["dev_interactions"], 0)
            self.assertGreater(stats["test_interactions"], 0)
            train = pd.read_csv(output / "train.csv", sep="\t")
            dev = pd.read_csv(output / "dev.csv", sep="\t")
            test = pd.read_csv(output / "test.csv", sep="\t")
            self.assertEqual(train["user_id"].min(), 1)
            self.assertEqual(len(eval(dev.iloc[0]["neg_items"])), 3)

            all_data = pd.concat([train, dev, test], ignore_index=True)
            clicked = all_data.groupby("user_id")["item_id"].apply(set).to_dict()
            for frame in (dev, test):
                for row in frame.itertuples():
                    self.assertTrue(clicked[row.user_id].isdisjoint(eval(row.neg_items)))


if __name__ == "__main__":
    unittest.main()
