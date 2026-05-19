"""
05 — Non-IID data heterogeneity impact.

Plots:
  A) Paired bar: niid=X vs niid=Y final accuracy per (solution x percentage).
  B) Heatmap: Δacc = acc(max_niid) - acc(min_niid) per (solution x percentage).

Output: output/figures/05_niid_impact.{png,pdf}
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


def _final_acc_by_niid(server_eval):
    # Include both packing and plaintext (percentage=None)
    lr = load_data.last_round(server_eval, group_cols=["model", "solution", "niid", "percentage", "seed"])
    return lr.groupby(["model", "solution", "niid", "percentage"], dropna=False)["mean_acc"].mean().reset_index()


def plot_paired_bar(summary, models):
    if summary is None or summary.empty:
        return None

    niids = sorted(summary["niid"].unique())
    if len(niids) < 2:
        print("Only one niid value found — skipping paired bar.")
        return None

    all_solutions = sorted(summary["solution"].unique())
    percentages = sorted(summary["percentage"].dropna().unique())
    # x-axis: percentages + "baseline"
    x_labels = [str(p) for p in percentages] + ["baseline"]
    x = np.arange(len(x_labels))
    width = 0.8 / max(len(niids), 1)
    niid_colors = plt.cm.get_cmap("Set1", max(len(niids), 1))

    fig, axes = plt.subplots(len(models), len(all_solutions),
                             figsize=(4 * len(all_solutions), 4 * len(models)),
                             squeeze=False)

    for row, model in enumerate(models):
        for col, sol in enumerate(all_solutions):
            ax = axes[row][col]
            sub = summary[(summary["model"] == model) & (summary["solution"] == sol)] \
                if "model" in summary.columns else summary[summary["solution"] == sol]

            for j, niid in enumerate(niids):
                vals = [
                    sub[(sub["niid"] == niid) & (sub["percentage"] == pct)]["mean_acc"].mean()
                    for pct in percentages
                ]
                # baseline column: percentage=None
                pt_val = sub[(sub["niid"] == niid) & (sub["percentage"].isna())]["mean_acc"]
                vals.append(pt_val.mean() if not pt_val.empty else np.nan)

                bar_vals = [0 if np.isnan(v) else v for v in vals]
                bars = ax.bar(x + j * width, bar_vals, width, label=f"niid={niid}",
                              color=niid_colors(j), alpha=0.85)
                for bar, orig in zip(bars, vals):
                    if np.isnan(orig):
                        bar.set_visible(False)

            # vertical separator before baseline
            sep_x = len(percentages) - 0.5 + width * (len(niids) - 1) / 2
            ax.axvline(sep_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

            ax.set_xticks(x + width * (len(niids) - 1) / 2)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Percentage  |  baseline")
            ax.set_ylabel("Final accuracy")
            ax.set_title(f"{model} — {sol}")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_delta_heatmap(summary, models):
    if summary is None or summary.empty:
        return None

    niids = sorted(summary["niid"].unique())
    if len(niids) < 2:
        return None

    niid_low, niid_high = niids[0], niids[-1]
    all_solutions = sorted(summary["solution"].unique())
    percentages = sorted(summary["percentage"].dropna().unique())
    col_labels = [str(p) for p in percentages] + ["baseline"]
    n_cols_mat = len(col_labels)

    fig, axes = plt.subplots(len(models), 1,
                             figsize=(max(6, n_cols_mat * 1.2), 2.5 * len(models)),
                             squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = summary[summary["model"] == model] if "model" in summary.columns else summary
        mat = np.full((len(all_solutions), n_cols_mat), np.nan)

        for i, sol in enumerate(all_solutions):
            for j, pct in enumerate(percentages):
                hi = sub[(sub["solution"] == sol) & (sub["niid"] == niid_high) & (sub["percentage"] == pct)]["mean_acc"]
                lo = sub[(sub["solution"] == sol) & (sub["niid"] == niid_low) & (sub["percentage"] == pct)]["mean_acc"]
                if not hi.empty and not lo.empty:
                    mat[i, j] = hi.mean() - lo.mean()
            # baseline column (percentage=None)
            hi_pt = sub[(sub["solution"] == sol) & (sub["niid"] == niid_high) & (sub["percentage"].isna())]["mean_acc"]
            lo_pt = sub[(sub["solution"] == sol) & (sub["niid"] == niid_low)  & (sub["percentage"].isna())]["mean_acc"]
            if not hi_pt.empty and not lo_pt.empty:
                mat[i, n_cols_mat - 1] = hi_pt.mean() - lo_pt.mean()

        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.1
        im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_cols_mat))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(all_solutions)))
        ax.set_yticklabels(all_solutions)
        ax.set_xlabel("Sparsification percentage  |  baseline")
        ax.set_title(f"Δacc (niid={niid_high} − niid={niid_low}) — {model}")
        ax.axvline(n_cols_mat - 1.5, color="white", linewidth=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(all_solutions)):
            for j in range(n_cols_mat):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)

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

    summary = _final_acc_by_niid(se)

    fig_a = plot_paired_bar(summary, models)
    if fig_a:
        for ext in ("png", "pdf"):
            fig_a.savefig(OUT / f"05_niid_paired_bar.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 05_niid_paired_bar.png/pdf")
        plt.close(fig_a)

    fig_b = plot_delta_heatmap(summary, models)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT / f"05_niid_delta_heatmap.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 05_niid_delta_heatmap.png/pdf")
        plt.close(fig_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
