"""
06 — Memory usage profile.

Plots:
  A) Box plot: ram_peak_mb distribution per solution.
  B) Line: mean ram_peak_mb vs percentage, by solution.

Output: output/figures/06_memory_profile.{png,pdf}
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("load_data", HERE / "load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)

OUT = HERE / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def plot_box(train, models):
    if "ram_peak_mb" not in train.columns or train.empty:
        return None

    solutions = sorted(train["solution"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)

    for col, model in enumerate(models):
        ax = axes[0][col]
        sub = train[train["model"] == model] if "model" in train.columns else train
        data = [sub[sub["solution"] == sol]["ram_peak_mb"].dropna().values for sol in solutions]
        bp = ax.boxplot(data, patch_artist=True, notch=False)
        cmap = plt.cm.get_cmap("tab10", len(solutions))
        for patch, color in zip(bp["boxes"], [cmap(i) for i in range(len(solutions))]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(solutions) + 1))
        ax.set_xticklabels(solutions, rotation=15, ha="right")
        ax.set_ylabel("Peak RAM (MB)")
        ax.set_title(f"RAM Peak Distribution — {model}")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_memory_vs_percentage(train, models):
    packing = train[train["percentage"].notna()].copy() if "percentage" in train.columns else train
    if "ram_peak_mb" not in packing.columns or packing.empty:
        return None

    solutions = sorted(packing["solution"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(solutions), 1))

    fig, axes = plt.subplots(len(models), 1, figsize=(6, 4 * len(models)), squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = packing[packing["model"] == model] if "model" in packing.columns else packing
        agg = sub.groupby(["solution", "percentage"], dropna=False)["ram_peak_mb"].mean().reset_index()
        for i, sol in enumerate(solutions):
            s = agg[agg["solution"] == sol].sort_values("percentage")
            if s.empty:
                continue
            ax.plot(s["percentage"], s["ram_peak_mb"], marker="o",
                    label=sol, color=cmap(i), linewidth=1.5)
        ax.set_xlabel("Sparsification percentage")
        ax.set_ylabel("Mean peak RAM (MB)")
        ax.set_title(f"Memory Usage vs Percentage — {model}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train = dfs["train"]
    if train.empty:
        print("No train data found.")
        return

    dims = load_data.discover_dims(train)
    models = dims.get("model", [])

    fig_a = plot_box(train, models)
    if fig_a:
        for ext in ("png", "pdf"):
            fig_a.savefig(OUT / f"06_memory_boxplot.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 06_memory_boxplot.png/pdf")
        plt.close(fig_a)

    fig_b = plot_memory_vs_percentage(train, models)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT / f"06_memory_vs_percentage.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 06_memory_vs_percentage.png/pdf")
        plt.close(fig_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
