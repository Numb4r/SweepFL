"""
02 — Solution comparison heatmap and bar chart.

Plots:
  A) Heatmap: rows=solution, cols=percentage, colour=final accuracy.
     One heatmap per (model x niid) combination.
  B) Grouped bar: final accuracy by solution x percentage, one subplot per model.

Output: output/figures/02_solution_comparison.{png,pdf}
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


def _final_acc(server_eval):
    lr = load_data.last_round(
        server_eval, group_cols=["model", "dataset", "solution", "niid", "percentage", "seed"]
    )
    return (
        lr.groupby(["model", "solution", "niid", "percentage"], dropna=False)["mean_acc"]
        .mean()
        .reset_index()
    )


def plot_heatmaps(summary, models, niids):
    all_solutions = sorted(summary["solution"].unique())
    percentages = sorted(summary["percentage"].dropna().unique())
    # columns: one per percentage + "baseline" at the end
    col_labels = [str(p) for p in percentages] + ["baseline"]
    n_cols_mat = len(col_labels)

    if not percentages:
        return None

    n_rows = len(models) * len(niids)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(6, n_cols_mat * 1.2), 2.5 * n_rows),
        squeeze=False,
    )

    row = 0
    for model in models:
        for niid in niids:
            ax = axes[row][0]
            sub = summary[(summary["model"] == model) & (summary["niid"] == niid)]
            mat = np.full((len(all_solutions), n_cols_mat), np.nan)

            for i, sol in enumerate(all_solutions):
                for j, pct in enumerate(percentages):
                    v = sub[(sub["solution"] == sol) & (sub["percentage"] == pct)]["mean_acc"]
                    if not v.empty:
                        mat[i, j] = v.mean()
                # "baseline" column: value from plaintext (percentage=None)
                if sol == "plaintext":
                    v = sub[(sub["solution"] == sol) & (sub["percentage"].isna())]["mean_acc"]
                    if not v.empty:
                        mat[i, n_cols_mat - 1] = v.mean()

            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(n_cols_mat))
            ax.set_xticklabels(col_labels)
            ax.set_yticks(range(len(all_solutions)))
            ax.set_yticklabels(all_solutions)
            ax.set_xlabel("Sparsification percentage  |  baseline")
            ax.set_title(f"Final Accuracy — {model} | niid={niid}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # vertical separator before baseline column
            ax.axvline(n_cols_mat - 1.5, color="white", linewidth=2)

            for i in range(len(all_solutions)):
                for j in range(n_cols_mat):
                    val = mat[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=8, color="black" if 0.3 < val < 0.8 else "white")
            row += 1

    fig.tight_layout()
    return fig


def plot_grouped_bar(summary, models, server_eval):
    packing = summary[summary["percentage"].notna()]
    if packing.empty:
        return None

    all_solutions = sorted(summary["solution"].unique())
    packing_solutions = sorted(packing["solution"].unique())
    percentages = sorted(packing["percentage"].dropna().unique())

    # x groups: one per solution (packing) + "plaintext" at end
    x_labels = packing_solutions + (["plaintext"] if "plaintext" in all_solutions else [])
    x = np.arange(len(x_labels))
    # bars per group: one per percentage + "baseline" for the plaintext group
    all_pct_labels = [str(p) for p in percentages] + ["baseline"]
    width = 0.8 / max(len(all_pct_labels), 1)

    cmap = plt.cm.get_cmap("viridis", max(len(percentages), 1))
    pct_colors = {p: cmap(i) for i, p in enumerate(percentages)}
    baseline_color = "dimgray"

    pt_lr = load_data.last_round(server_eval[server_eval["percentage"].isna()])

    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(8, len(x_labels) * 1.8), 4 * n_rows), squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = packing[packing["model"] == model]

        # bars for packing solutions, coloured by percentage
        for j, pct in enumerate(percentages):
            vals = []
            for sol in x_labels:
                if sol == "plaintext":
                    vals.append(np.nan)
                else:
                    v = sub[(sub["solution"] == sol) & (sub["percentage"] == pct)]["mean_acc"]
                    vals.append(v.mean() if not v.empty else np.nan)
            bar_vals = [0 if np.isnan(v) else v for v in vals]
            bars = ax.bar(x + j * width, bar_vals, width, label=f"p={pct}",
                          color=pct_colors[pct], alpha=0.85)
            for bar, orig in zip(bars, vals):
                if np.isnan(orig):
                    bar.set_visible(False)

        # "baseline" bar — only for the plaintext x-group
        if "plaintext" in x_labels:
            pt_idx = x_labels.index("plaintext")
            pt_sub = pt_lr[pt_lr["model"] == model] if not pt_lr.empty else None
            pt_val = pt_sub["mean_acc"].mean() if (pt_sub is not None and not pt_sub.empty) else np.nan
            baseline_x = x[pt_idx] + len(percentages) * width / 2
            if not np.isnan(pt_val):
                ax.bar(baseline_x, pt_val, width * len(percentages),
                       label="baseline", color=baseline_color, alpha=0.7,
                       hatch="//", edgecolor="white")

        # separator before plaintext group
        if "plaintext" in x_labels:
            sep = x[x_labels.index("plaintext")] - 0.5
            ax.axvline(sep, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xticks(x + width * (len(percentages) - 1) / 2)
        ax.set_xticklabels(x_labels, rotation=15, ha="right")
        ax.set_ylabel("Final mean accuracy")
        ax.set_title(f"Solution Comparison — {model}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, ncol=3)
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
    niids = dims.get("niid", [])

    summary = _final_acc(se)

    fig_a = plot_heatmaps(summary, models, niids)
    if fig_a:
        for ext in ("png", "pdf"):
            fig_a.savefig(OUT / f"02_solution_heatmap.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 02_solution_heatmap.png/pdf")
        plt.close(fig_a)

    fig_b = plot_grouped_bar(summary, models, se)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT / f"02_solution_comparison.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 02_solution_comparison.png/pdf")
        plt.close(fig_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
