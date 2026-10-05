"""Create the compact comparison figure used in the experiment report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
data = pd.read_csv(RESULTS / "final_results.csv")
data = data[data.model.isin(["BPRMF", "LightGCN", "SimGCL-tuned"])].copy()

model_order = ["BPRMF", "LightGCN", "SimGCL-tuned"]
dataset_order = ["Grocery_Pilot", "ML_1M_Pilot"]
colors = ["#4C78A8", "#F58518", "#54A24B"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for axis, dataset in zip(axes, dataset_order):
    subset = data[data.dataset == dataset].set_index("model").loc[model_order]
    x = np.arange(len(model_order))
    width = 0.36
    axis.bar(x - width / 2, subset["test_HR@20"], width, label="HR@20", color=colors[0])
    axis.bar(x + width / 2, subset["test_NDCG@20"], width, label="NDCG@20", color=colors[1])
    axis.set_title(dataset.replace("_", " "))
    axis.set_xticks(x, ["BPRMF", "LightGCN", "SimGCL\n(tuned)"])
    axis.set_ylim(0, max(subset["test_HR@20"]) * 1.2)
    axis.grid(axis="y", alpha=0.25)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

axes[0].set_ylabel("Metric value")
axes[1].legend(loc="upper right")
fig.suptitle("CPU pilot comparison (one positive + 99 negatives)")
fig.savefig(RESULTS / "comparison.png", dpi=180)
