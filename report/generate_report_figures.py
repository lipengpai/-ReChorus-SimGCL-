"""Generate the four external figures used by 实验报告_最终版.tex.

All outputs are written next to this script so that the report can be uploaded
to Overleaf as one flat folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
    }
)

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#7A6FF0"
INK = "#263238"
GRID = "#D9DEE3"


def rounded_box(ax, center, text, width=1.9, height=0.72, face="#EEF4FA"):
    x, y = center
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.07",
        linewidth=1.1,
        edgecolor=INK,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", color=INK, linespacing=1.35)
    return box


def arrow(ax, start, end, *, rad=0.0, text=None, text_offset=(0, 0)):
    item = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color="#59636B",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(item)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=9, color=INK)


def save(fig, filename):
    fig.savefig(ROOT / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_rechorus_pipeline():
    fig, ax = plt.subplots(figsize=(10.8, 3.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    rounded_box(ax, (1.25, 2.85), "原始交互数据\nGrocery\nMovieLens-1M", width=2.15, height=0.95, face="#EAF1FC")
    rounded_box(ax, (4.05, 2.85), "Reader\n划分、历史与负采样", width=2.15, face="#EAF7EC")
    rounded_box(ax, (7.00, 2.85), "Model\nBPRMF / LightGCN\nSimGCL", width=2.65, height=0.95, face="#EAF7EC")
    rounded_box(ax, (7.00, 1.15), "Runner\n训练、验证与早停", width=2.35, face="#EAF7EC")
    rounded_box(ax, (4.05, 1.15), "实验产物\n指标、日志与检查点", face="#FFF1E3")
    arrow(ax, (2.34, 2.85), (2.95, 2.85))
    arrow(ax, (5.13, 2.85), (5.66, 2.85))
    arrow(ax, (7.00, 2.47), (7.00, 1.56))
    arrow(ax, (5.80, 1.15), (5.03, 1.15))
    arrow(ax, (8.18, 1.15), (8.18, 2.85), rad=-0.35, text="验证 NDCG@20", text_offset=(0.75, 0))
    ax.text(1.25, 0.55, "数据", ha="center", color=BLUE, weight="bold")
    ax.text(5.55, 0.55, "统一框架", ha="center", color=GREEN, weight="bold")
    ax.text(9.25, 0.55, "可审计输出", ha="center", color=ORANGE, weight="bold")
    save(fig, "fig1_rechorus_pipeline.png")


def make_experiment_process():
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    rounded_box(ax, (1.35, 4.0), "论文与框架阅读", face="#EEF0FD")
    rounded_box(ax, (4.10, 4.0), "模型实现与 CPU 修复", width=2.25, face="#EEF0FD")
    rounded_box(ax, (7.10, 4.0), "子集验证与调参", face="#EEF0FD")
    rounded_box(ax, (7.10, 2.35), "完整数据训练", face="#EEF0FD")
    diamond = Polygon(
        [[4.10, 2.95], [5.23, 2.35], [4.10, 1.75], [2.97, 2.35]],
        closed=True,
        facecolor="#FFF5D6",
        edgecolor=INK,
        linewidth=1.1,
    )
    ax.add_patch(diamond)
    ax.text(4.10, 2.35, "中断状态完整？", ha="center", va="center", color=INK)
    rounded_box(ax, (1.35, 2.35), "重跑、独立评估与测试", width=2.25, face="#FDEDEA")
    rounded_box(ax, (4.10, 0.72), "汇总结果与报告", face="#EAF7EC")
    arrow(ax, (2.32, 4.0), (2.93, 4.0))
    arrow(ax, (5.23, 4.0), (6.12, 4.0))
    arrow(ax, (7.10, 3.62), (7.10, 2.74))
    arrow(ax, (6.12, 2.35), (5.23, 2.35))
    arrow(ax, (2.97, 2.35), (2.48, 2.35), text="否", text_offset=(0, 0.23))
    arrow(ax, (1.35, 1.96), (3.45, 0.88), rad=0.0)
    arrow(ax, (4.10, 1.75), (4.10, 1.12), text="是", text_offset=(0.25, 0))
    ax.text(9.35, 3.15, "原则", ha="center", color=PURPLE, weight="bold")
    ax.text(9.35, 2.55, "只有权重的旧断点\n不能保证优化轨迹连续", ha="center", va="center", color=INK)
    save(fig, "fig2_experiment_process.png")


def make_early_stop_curve():
    epochs = np.arange(1, 7)
    values = np.array([0.2059, 0.1975, 0.1957, 0.1957, 0.1908, 0.1939])
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.plot(epochs, values, color=BLUE, marker="o", linewidth=2.2, markersize=6, label="验证 NDCG@20")
    ax.axhline(values[0], color=RED, linestyle="--", linewidth=1.5, label="第 1 轮最佳值")
    ax.scatter([1], [values[0]], s=80, color=RED, zorder=5)
    ax.annotate(
        "patience=5，第 6 轮停止",
        xy=(6, values[-1]),
        xytext=(4.1, 0.1922),
        arrowprops=dict(arrowstyle="->", color=INK),
        color=INK,
    )
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("验证集 NDCG@20")
    ax.set_xticks(epochs)
    ax.set_ylim(0.188, 0.208)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig3_early_stop_curve.png")


def make_full_results():
    models = ["BPRMF", "LightGCN", "SimGCL"]
    grocery_hr = [0.3990, 0.4186, 0.3896]
    grocery_ndcg = [0.1716, 0.1750, 0.2042]
    ml_hr = [0.6785, 0.6952, 0.4140]
    ml_ndcg = [0.3188, 0.3284, 0.1893]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    x = np.arange(len(models))
    width = 0.34
    for ax, title, hr, ndcg, ymax in [
        (axes[0], "Amazon Grocery", grocery_hr, grocery_ndcg, 0.48),
        (axes[1], "MovieLens-1M", ml_hr, ml_ndcg, 0.78),
    ]:
        bars1 = ax.bar(x - width / 2, hr, width, color=BLUE, label="HR@20")
        bars2 = ax.bar(x + width / 2, ndcg, width, color=ORANGE, label="NDCG@20")
        ax.bar_label(bars1, fmt="%.4f", padding=3, fontsize=9)
        ax.bar_label(bars2, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(title, weight="bold")
        ax.set_xticks(x, models)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("指标值")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(w_pad=2.3)
    save(fig, "fig4_full_results.png")


if __name__ == "__main__":
    make_rechorus_pipeline()
    make_experiment_process()
    make_early_stop_curve()
    make_full_results()
    for name in [
        "fig1_rechorus_pipeline.png",
        "fig2_experiment_process.png",
        "fig3_early_stop_curve.png",
        "fig4_full_results.png",
    ]:
        print(ROOT / name)
