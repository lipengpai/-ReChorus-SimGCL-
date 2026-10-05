import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from helpers.BaseRunner import BaseRunner  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.optimizer = None
        self.device = torch.device("cpu")

    def customize_parameters(self):
        return self.parameters()


def runner_args(temp: Path):
    return SimpleNamespace(
        train=1,
        epoch=3,
        resume_epoch=1,
        resume_state=str(temp / "resume.pt"),
        check_epoch=1,
        test_epoch=-1,
        early_stop=5,
        lr=0.001,
        l2=0.0,
        batch_size=2,
        eval_batch_size=2,
        optimizer="Adam",
        num_workers=0,
        pin_memory=0,
        topk="20",
        metric="NDCG,HR",
        main_metric="NDCG@20",
        log_file=str(temp / "run.log"),
    )


class ResumeStateTest(unittest.TestCase):
    def test_full_state_restores_model_optimizer_history_and_rng(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            runner = BaseRunner(runner_args(temp))
            model = TinyModel()
            model.optimizer = runner._build_optimizer(model)

            loss = model.linear(torch.ones(1, 2)).sum()
            loss.backward()
            model.optimizer.step()
            saved_weight = model.linear.weight.detach().clone()

            np.random.seed(11)
            random.seed(12)
            torch.manual_seed(13)
            runner._save_resume_state(
                model, 1, [0.25], [{"NDCG@20": 0.25}]
            )
            expected_np = np.random.rand()
            expected_py = random.random()
            expected_torch = torch.rand(1)

            with torch.no_grad():
                model.linear.weight.add_(10)
            model.optimizer = None
            np.random.seed(99)
            random.seed(99)
            torch.manual_seed(99)

            main_history, dev_history = runner._load_resume_state(model)
            self.assertTrue(torch.allclose(model.linear.weight, saved_weight))
            self.assertTrue(model.optimizer.state)
            self.assertEqual(main_history, [0.25])
            self.assertEqual(dev_history, [{"NDCG@20": 0.25}])
            self.assertAlmostEqual(np.random.rand(), expected_np)
            self.assertAlmostEqual(random.random(), expected_py)
            self.assertTrue(torch.allclose(torch.rand(1), expected_torch))


if __name__ == "__main__":
    unittest.main()
