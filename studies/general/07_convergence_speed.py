"""
07 — Convergence speed analysis.

Plots:
  A) Bar: rounds to reach 70%, 80%, 85% accuracy per (solution x percentage x niid).
  B) Bar: AUC-normalized accuracy per (solution x percentage).

Output: output/figures/07_convergence_speed.{png,pdf}
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

THRESHOLDS = [0.70, 0.80, 0.85]


def plot_convergence_rounds(server_eval, models):
    packing = server_eval[server_eval["percentage"].notna()]
    if packing.empty:
        return None

    solutions = sorted(packing["solution"].unique())
    percentages = sorted(packing["percentage"].dropna().unique())
    niids = sorted(packing["niid"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(solutions), 1))

    n_rows = len(models) * len(niids)
    n_cols = len(THRESHOLDS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    row = 0
    for model in models:
        for niid in niids:
            for col, thr in enumerate(THRESHOLDS):
                ax = axes[row][col]
                conv = load_data.rounds_to_threshold(
                    packing[(packing["model"] == model) & (packing["niid"] == niid)]
                    if "model" in packing.columns else packing[packing["niid"] == niid],
                    metric="mean_acc", threshold=thr
                )
                if conv.empty:
                    ax.set_visible(False)
                    continue

                x = np.arange(len(percentages))
                width = 0.8 / max(len(solutions), 1)
                for i, sol in enumerate(solutions):
                    vals = [
                        conv[(conv["solution"] == sol) & (conv["percentage"] == pct)]["convergence_round"].mean()
                        for pct in percentages
                    ]
                    ax.bar(x + i * width, vals, width, label=sol,
                           color=cmap(i), alpha=0.85)

                ax.set_xticks(x + width * (len(solutions) - 1) / 2)
                ax.set_xticklabels([str(p) for p in percentages])
                ax.set_xlabel("Percentage")
                ax.set_ylabel("Rounds to reach threshold")
                ax.set_title(f"{model} niid={niid} | ≥{thr:.0%}")
                if i == 0:
                    ax.legend(fontsize=7)
                ax.grid(True, axis="y", alpha=0.3)
            row += 1

    fig.tight_layout()
    return fig


def plot_auc_bar(server_eval, models):
    # Compute AUC for all solutions including plaintext
    auc_df = load_data.auc_accuracy(server_eval)
    if auc_df.empty or "auc_norm" not in auc_df.columns:
        return None

    all_solutions = sorted(auc_df["solution"].unique())
    percentages = sorted(auc_df["percentage"].dropna().unique())
    cmap = plt.cm.get_cmap("tab10", max(len(all_solutions), 1))

    # x-axis: percentages + "baseline"
    x_labels = [str(p) for p in percentages] + ["baseline"]
    x = np.arange(len(x_labels))
    width = 0.8 / max(len(all_solutions), 1)

    fig, axes = plt.subplots(len(models), 1, figsize=(9, 4 * len(models)), squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = auc_df[auc_df["model"] == model] if "model" in auc_df.columns else auc_df
        agg = sub.groupby(["solution", "percentage"], dropna=False)["auc_norm"].mean().reset_index()

        for i, sol in enumerate(all_solutions):
            vals = [agg[(agg["solution"] == sol) & (agg["percentage"] == pct)]["auc_norm"].mean()
                    for pct in percentages]
            # baseline column (percentage=None → plaintext)
            pt_val = agg[(agg["solution"] == sol) & (agg["percentage"].isna())]["auc_norm"]
            vals.append(pt_val.mean() if not pt_val.empty else np.nan)

            bar_vals = [0 if np.isnan(v) else v for v in vals]
            bars = ax.bar(x + i * width, bar_vals, width, label=sol, color=cmap(i), alpha=0.85)
            for bar, orig in zip(bars, vals):
                if np.isnan(orig):
                    bar.set_visible(False)

        # separator before baseline
        sep_x = len(percentages) - 0.5 + width * (len(all_solutions) - 1) / 2
        ax.axvline(sep_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xticks(x + width * (len(all_solutions) - 1) / 2)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Sparsification percentage  |  baseline")
        ax.set_ylabel("AUC (normalised)")
        ax.set_title(f"Accuracy AUC — {model}")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    se = dfs["server_eval"]
    if se.empty:
        print("No server_eval data found.")
        return

    dims = load_data.discover_dims(se)
    models = dims.get("model", [])

    fig_a = plot_convergence_rounds(se, models)
    if fig_a:
        for ext in ("png", "pdf"):
            fig_a.savefig(OUT / f"07_convergence_rounds.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 07_convergence_rounds.png/pdf")
        plt.close(fig_a)

    fig_b = plot_auc_bar(se, models)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT / f"07_auc_bar.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 07_auc_bar.png/pdf")
        plt.close(fig_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
