"""
03 — Computational overhead breakdown.

Plots:
  A) Stacked bar: per-operation time breakdown per (solution x percentage) — median round.
  B) Line: total HE overhead (encrypt + serialize + pack + mask_select) vs percentage.
  C) Line: server aggregation_time_s vs round, by solution.

Output: output/figures/03_overhead_breakdown.{png,pdf}
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("load_data", HERE / "load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)

OUT = HERE / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TIME_OPS = [
    ("train_time_s", "Train"),
    ("flatten_time_s", "Flatten"),
    ("pack_time_s", "Pack"),
    ("mask_select_time_s", "Mask"),
    ("encrypt_time_s", "Encrypt"),
    ("serialize_time_s", "Serialize"),
    ("decrypt_time_s", "Decrypt"),
]
HE_OPS = ["encrypt_time_s", "serialize_time_s", "pack_time_s", "mask_select_time_s"]


def _median_per_config(train_df):
    present_ops = [c for c, _ in TIME_OPS if c in train_df.columns]
    gc = ["model", "solution", "niid", "percentage"]
    gc = [c for c in gc if c in train_df.columns]
    return train_df.groupby(gc, dropna=False)[present_ops].median().reset_index()


def plot_stacked_bar(medians, models):
    packing = medians[medians["percentage"].notna()]
    if packing.empty:
        return None

    solutions = sorted(packing["solution"].unique())
    percentages = sorted(packing["percentage"].dropna().unique())

    present_ops = [(c, lbl) for c, lbl in TIME_OPS if c in packing.columns]
    colors = plt.cm.get_cmap("Set2", len(present_ops))

    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(10, len(solutions) * len(percentages) * 0.6), 4 * n_rows), squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = packing[packing["model"] == model].copy() if "model" in packing.columns else packing.copy()
        labels = [f"{sol}\np={pct}" for sol in solutions for pct in percentages]
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))

        for k, (col, lbl) in enumerate(present_ops):
            vals = []
            for sol in solutions:
                for pct in percentages:
                    row_val = sub[(sub["solution"] == sol) & (sub["percentage"] == pct)][col]
                    vals.append(row_val.values[0] if not row_val.empty else 0.0)
            ax.bar(x, vals, bottom=bottoms, label=lbl, color=colors(k), alpha=0.85)
            bottoms += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Time (s) — median round")
        ax.set_title(f"Overhead Breakdown — {model}")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_he_overhead_line(train_df, models):
    packing = train_df[train_df["percentage"].notna()].copy()
    if packing.empty:
        return None

    present_he = [c for c in HE_OPS if c in packing.columns]
    if not present_he:
        return None
    packing["he_overhead"] = packing[present_he].sum(axis=1)

    gc = ["model", "solution", "percentage"]
    gc = [c for c in gc if c in packing.columns]
    agg = packing.groupby(gc, dropna=False)["he_overhead"].mean().reset_index()

    solutions = sorted(agg["solution"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(solutions), 1))

    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 4 * n_rows), squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = agg[agg["model"] == model] if "model" in agg.columns else agg
        for i, sol in enumerate(solutions):
            s = sub[sub["solution"] == sol].sort_values("percentage")
            if s.empty:
                continue
            ax.plot(s["percentage"], s["he_overhead"], marker="o",
                    label=sol, color=cmap(i), linewidth=1.5)
        ax.set_xlabel("Sparsification percentage")
        ax.set_ylabel("Mean HE overhead (s/round)")
        ax.set_title(f"HE Overhead vs Percentage — {model}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_server_agg_time(server_agg):
    if server_agg.empty or "aggregation_time_s" not in server_agg.columns:
        return None

    solutions = sorted(server_agg["solution"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(solutions), 1))
    models = sorted(server_agg["model"].unique()) if "model" in server_agg.columns else ["all"]

    fig, axes = plt.subplots(len(models), 1, figsize=(8, 3 * len(models)), squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = server_agg[server_agg["model"] == model] if "model" in server_agg.columns else server_agg
        for i, sol in enumerate(solutions):
            s = sub[sub["solution"] == sol].groupby("round")["aggregation_time_s"].mean()
            if s.empty:
                continue
            ax.plot(s.index, s.values, label=sol, color=cmap(i), linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Round")
        ax.set_ylabel("Aggregation time (s)")
        ax.set_title(f"Server Aggregation Time — {model}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train = dfs["train"]
    sagg = dfs["server_agg"]

    if train.empty:
        print("No train data found.")
        return

    dims = load_data.discover_dims(train)
    models = dims.get("model", [])

    medians = _median_per_config(train)

    fig_a = plot_stacked_bar(medians, models)
    if fig_a:
        for ext in ("png", "pdf"):
            fig_a.savefig(OUT / f"03_overhead_stacked.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 03_overhead_stacked.png/pdf")
        plt.close(fig_a)

    fig_b = plot_he_overhead_line(train, models)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT / f"03_he_overhead_line.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 03_he_overhead_line.png/pdf")
        plt.close(fig_b)

    fig_c = plot_server_agg_time(sagg)
    if fig_c:
        for ext in ("png", "pdf"):
            fig_c.savefig(OUT / f"03_server_agg_time.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 03_server_agg_time.png/pdf")
        plt.close(fig_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
