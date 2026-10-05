import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from models.general.LightGCN import LightGCN  # noqa: E402
from models.general.SimGCL import SimGCL  # noqa: E402


class DummyCorpus:
    n_users = 4
    n_items = 6
    train_clicked_set = {
        1: {1, 2},
        2: {2, 3},
        3: {4},
    }


def common_args(**overrides):
    values = {
        "device": torch.device("cpu"),
        "model_path": "unused.pt",
        "buffer": 0,
        "num_neg": 1,
        "dropout": 0.0,
        "test_all": 0,
        "emb_size": 8,
        "n_layers": 2,
        "cl_rate": 0.2,
        "temperature": 0.2,
        "eps": 0.1,
        "include_ego": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def training_feed():
    return {
        "user_id": torch.tensor([1, 2], dtype=torch.long),
        "item_id": torch.tensor([[1, 5], [2, 4]], dtype=torch.long),
        "pos_item_id": torch.tensor([1, 2], dtype=torch.long),
        "batch_size": 2,
        "phase": "train",
    }


class SimGCLTest(unittest.TestCase):
    def test_forward_loss_and_backward_on_cpu(self):
        model = SimGCL(common_args(), DummyCorpus()).to("cpu")
        output = model(training_feed())

        self.assertEqual(tuple(output["prediction"].shape), (2, 2))
        self.assertEqual(tuple(output["user_view1"].shape), (2, 8))
        self.assertEqual(tuple(output["item_view1"].shape), (2, 8))

        loss = model.loss(output)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.encoder.user_emb.grad)
        self.assertTrue(torch.isfinite(model.encoder.user_emb.grad).all())

    def test_two_perturbed_views_are_distinct(self):
        model = SimGCL(common_args(), DummyCorpus()).to("cpu")
        output = model(training_feed())
        self.assertFalse(torch.equal(output["user_view1"], output["user_view2"]))
        self.assertFalse(torch.equal(output["item_view1"], output["item_view2"]))

    def test_inference_does_not_build_contrastive_views(self):
        model = SimGCL(common_args(), DummyCorpus()).to("cpu")
        feed = training_feed()
        feed["phase"] = "test"
        output = model(feed)
        self.assertEqual(set(output), {"prediction"})

    def test_noise_has_fixed_norm_and_same_hyperoctant(self):
        model = SimGCL(common_args(n_layers=1, eps=0.15), DummyCorpus()).to("cpu")
        with torch.no_grad():
            base_u, base_i = model.encoder(perturbed=False)
            noisy_u, noisy_i = model.encoder(perturbed=True)
        base = torch.cat([base_u, base_i])
        delta = torch.cat([noisy_u - base_u, noisy_i - base_i])
        connected = base.norm(dim=-1) > 0
        expected = torch.full_like(delta[connected].norm(dim=-1), 0.15)
        self.assertTrue(torch.allclose(delta[connected].norm(dim=-1), expected, atol=1e-6))
        self.assertTrue(torch.all(delta[connected] * base[connected] >= -1e-7))

    def test_info_nce_matches_paper_batch_formula(self):
        model = SimGCL(common_args(temperature=0.3), DummyCorpus()).to("cpu")
        view1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        view2 = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        normalized1 = F.normalize(view1, dim=-1)
        normalized2 = F.normalize(view2, dim=-1)
        logits = normalized1 @ normalized2.T / 0.3
        expected = F.cross_entropy(logits, torch.arange(2))
        self.assertTrue(torch.allclose(model.info_nce(view1, view2), expected))

    def test_duplicate_batch_nodes_are_contrasted_once(self):
        model = SimGCL(common_args(), DummyCorpus()).to("cpu")
        feed = {
            "user_id": torch.tensor([1, 1], dtype=torch.long),
            "item_id": torch.tensor([[1, 5], [1, 4]], dtype=torch.long),
            "pos_item_id": torch.tensor([1, 1], dtype=torch.long),
            "batch_size": 2,
            "phase": "train",
        }
        output = model(feed)
        self.assertEqual(output["user_view1"].shape[0], 1)
        self.assertEqual(output["item_view1"].shape[0], 1)


class LightGCNCPUTest(unittest.TestCase):
    def test_forward_and_backward_on_cpu(self):
        model = LightGCN(common_args(), DummyCorpus()).to("cpu")
        output = model(training_feed())
        self.assertEqual(tuple(output["prediction"].shape), (2, 2))
        loss = model.loss(output)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        user_grad = model.encoder.embedding_dict["user_emb"].grad
        self.assertIsNotNone(user_grad)
        self.assertTrue(torch.isfinite(user_grad).all())


if __name__ == "__main__":
    unittest.main()
