"""
01 — Accuracy convergence curves.

Plots:
  A) Grid (rows=model, cols=niid): mean_acc vs round, one line per (solution x percentage).
     Plaintext baseline drawn as a dashed horizontal line.
  B) Bar chart: final accuracy by solution, grouped by percentage.

Output: output/figures/01_accuracy_curves.{png,pdf}
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("load_data", HERE / "load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)

OUT = HERE / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _solution_color_map(solutions):
    cmap = cm.get_cmap("tab10", max(len(solutions), 1))
    return {s: cmap(i) for i, s in enumerate(sorted(solutions))}


def _percentage_linestyle(percentages):
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    pcts = sorted([p for p in percentages if p is not None])
    return {p: styles[i % len(styles)] for i, p in enumerate(pcts)}


def plot_convergence_grid(server_eval, models, niids, sol_colors, pct_styles):
    n_rows, n_cols = len(models), len(niids)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        sharey=False,
    )

    packing_solutions = server_eval[server_eval["percentage"].notna()]["solution"].unique()
    plaintext_df = server_eval[server_eval["percentage"].isna()]

    for r, model in enumerate(models):
        for c, niid in enumerate(niids):
            ax = axes[r][c]
            ax.set_title(f"{model} | niid={niid}", fontsize=9)
            ax.set_xlabel("Round")
            ax.set_ylabel("Mean Accuracy")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            # plaintext baseline
            pt = plaintext_df[
                (plaintext_df["model"] == model) & (plaintext_df["niid"] == niid)
            ]
            if not pt.empty:
                final_pt = pt.groupby("round")["mean_acc"].mean()
                if not final_pt.empty:
                    baseline_val = final_pt.iloc[-1]
                    ax.axhline(
                        baseline_val, color="black", linestyle=":", linewidth=1.5,
                        label="plaintext", alpha=0.7,
                    )

            # packing solutions
            for sol in sorted(packing_solutions):
                sol_df = server_eval[
                    (server_eval["model"] == model)
                    & (server_eval["niid"] == niid)
                    & (server_eval["solution"] == sol)
                    & server_eval["percentage"].notna()
                ]
                for pct in sorted(sol_df["percentage"].dropna().unique()):
                    sub = sol_df[sol_df["percentage"] == pct].groupby("round")["mean_acc"].mean()
                    if sub.empty:
                        continue
                    ax.plot(
                        sub.index, sub.values,
                        color=sol_colors.get(sol, "gray"),
                        linestyle=pct_styles.get(pct, "-"),
                        linewidth=1.2,
                        label=f"{sol} p={pct}",
                    )

            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=6, loc="lower right", ncol=2)

    fig.suptitle("Accuracy Convergence by Model & NIID", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def plot_final_acc_bar(server_eval, models, sol_colors):
    lr = load_data.last_round(server_eval, group_cols=["model", "dataset", "solution", "niid", "percentage", "seed"])
    if lr.empty:
        return None
    summary = lr.groupby(["model", "solution", "percentage"], dropna=False)["mean_acc"].mean().reset_index()

    packing = summary[summary["percentage"].notna()]
    if packing.empty:
        return None

    packing_solutions = sorted(packing["solution"].unique())
    percentages = sorted(packing["percentage"].dropna().unique())

    # x-axis: percentages + "baseline" column at the end
    x_labels = [str(p) for p in percentages] + ["baseline"]
    x = np.arange(len(x_labels))
    all_solutions = sorted(summary["solution"].unique())
    width = 0.8 / max(len(all_solutions), 1)

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4), squeeze=False)

    for col, model in enumerate(models):
        ax = axes[0][col]
        sub_packing = packing[packing["model"] == model]
        sub_all = summary[summary["model"] == model]

        for i, sol in enumerate(all_solutions):
            vals = []
            for p in percentages:
                v = sub_packing[(sub_packing["solution"] == sol) & (sub_packing["percentage"] == p)]["mean_acc"]
                vals.append(v.mean() if not v.empty else np.nan)
            # baseline column: only plaintext has a value here
            if sol == "plaintext":
                pt_val = sub_all[sub_all["solution"] == "plaintext"]["mean_acc"]
                vals.append(pt_val.mean() if not pt_val.empty else np.nan)
            else:
                vals.append(np.nan)

            # skip bars that are entirely NaN
            if all(np.isnan(v) for v in vals):
                continue

            bar_vals = [0 if np.isnan(v) else v for v in vals]
            alpha_vals = [0 if np.isnan(v) else 0.85 for v in vals]
            bars = ax.bar(
                x + i * width, bar_vals, width,
                label=sol, color=sol_colors.get(sol, "gray"), alpha=0.85,
            )
            # make NaN bars invisible
            for bar, orig in zip(bars, vals):
                if np.isnan(orig):
                    bar.set_visible(False)

        # vertical separator before "baseline"
        sep_x = len(percentages) - 0.5 + width * (len(all_solutions) - 1) / 2
        ax.axvline(sep_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xticks(x + width * (len(all_solutions) - 1) / 2)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Sparsification percentage  |  baseline")
        ax.set_ylabel("Final mean accuracy")
        ax.set_title(f"Final Accuracy — {model}")
        ax.set_ylim(0, 1.05)
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
    niids = dims.get("niid", [])
    solutions = [s for s in dims.get("solution", []) if s != "plaintext"]
    percentages = dims.get("percentage", [])

    print(f"Models: {models}")
    print(f"NIIDs: {niids}")
    print(f"Solutions: {dims.get('solution', [])}")
    print(f"Percentages: {percentages}")

    sol_colors = _solution_color_map(dims.get("solution", []))
    pct_styles = _percentage_linestyle(percentages)

    fig_a = plot_convergence_grid(se, models, niids, sol_colors, pct_styles)
    for ext in ("png", "pdf"):
        fig_a.savefig(OUT / f"01_accuracy_curves.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved 01_accuracy_curves.png/pdf")
    plt.close(fig_a)

    fig_b = plot_final_acc_bar(se, models, sol_colors)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT / f"01_final_accuracy_bar.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 01_final_accuracy_bar.png/pdf")
        plt.close(fig_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
