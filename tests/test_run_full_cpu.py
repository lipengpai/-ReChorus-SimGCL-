import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_full_cpu  # noqa: E402


class FullCpuBudgetTest(unittest.TestCase):
    def test_logged_training_time_is_cumulative(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_output = run_full_cpu.OUTPUT
            try:
                run_full_cpu.OUTPUT = Path(temp_dir)
                log = (
                    run_full_cpu.OUTPUT / "logs" / "Tiny" / "SimGCL_seed7.txt"
                )
                log.parent.mkdir(parents=True)
                log.write_text(
                    "Epoch 1     loss=2.0 [1.2 s]\tdev=(NDCG@20:0.1)\n"
                    "Epoch 2     loss=1.0 [3.4 s]\tdev=(NDCG@20:0.2)\n",
                    encoding="utf-8",
                )
                elapsed = run_full_cpu.logged_training_seconds(
                    [("Tiny", "SimGCL")], seed=7
                )
                self.assertAlmostEqual(elapsed, 4.6)
            finally:
                run_full_cpu.OUTPUT = original_output


if __name__ == "__main__":
    unittest.main()
