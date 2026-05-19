"""
08 — Efficiency frontier (Pareto: accuracy vs communication cost).

Plots:
  Scatter: final accuracy vs mean payload (MB), colour=solution, marker=percentage.
  Pareto-optimal points highlighted with a dashed frontier line.

Output: output/figures/08_efficiency_frontier.{png,pdf}
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


def _pareto_front(costs, accuracies):
    """Return boolean mask of Pareto-optimal points (min cost, max acc)."""
    n = len(costs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if costs[j] <= costs[i] and accuracies[j] >= accuracies[i]:
                if costs[j] < costs[i] or accuracies[j] > accuracies[i]:
                    dominated[i] = True
                    break
    return ~dominated


def plot_frontier(train, server_eval, models):
    if "payload_bytes" not in train.columns:
        return None

    # Build payload aggregation for ALL solutions (packing + plaintext)
    payload_agg = (
        train.groupby(["model", "solution", "niid", "percentage"], dropna=False)
        ["payload_bytes"].mean().reset_index()
    )
    payload_agg["payload_mb"] = payload_agg["payload_bytes"] / 1e6

    lr = load_data.last_round(server_eval)
    acc_agg = lr.groupby(["model", "solution", "niid", "percentage"], dropna=False)["mean_acc"].mean().reset_index()

    merged = acc_agg.merge(payload_agg, on=["model", "solution", "niid", "percentage"], how="inner")
    if merged.empty:
        return None

    all_solutions = sorted(merged["solution"].unique())
    percentages = sorted(merged["percentage"].dropna().unique())
    niids = sorted(merged["niid"].unique())

    cmap = plt.cm.get_cmap("tab10", max(len(all_solutions), 1))
    sol_colors = {s: cmap(i) for i, s in enumerate(all_solutions)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
    pct_markers = {p: markers[i % len(markers)] for i, p in enumerate(percentages)}
    PLAINTEXT_MARKER = "*"

    n_cols = len(models)
    n_rows = len(niids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for row, niid in enumerate(niids):
        for col, model in enumerate(models):
            ax = axes[row][col]
            sub = merged[(merged["model"] == model) & (merged["niid"] == niid)] \
                if "model" in merged.columns else merged[merged["niid"] == niid]

            if sub.empty:
                ax.set_visible(False)
                continue

            # Pareto computed over packing solutions only (plaintext is reference, not competing)
            packing_sub = sub[sub["percentage"].notna()]
            pareto_mask = _pareto_front(packing_sub["payload_mb"].values, packing_sub["mean_acc"].values) \
                if not packing_sub.empty else np.array([], dtype=bool)

            for i, sol in enumerate(all_solutions):
                s = sub[sub["solution"] == sol]
                for _, r in s.iterrows():
                    is_pt = r["percentage"] is None or (hasattr(r["percentage"], "__float__") and np.isnan(float(r["percentage"])))
                    mk = PLAINTEXT_MARKER if is_pt else pct_markers.get(r["percentage"], "o")
                    size = 160 if is_pt else 90
                    ax.scatter(r["payload_mb"], r["mean_acc"],
                               color=sol_colors[sol], marker=mk,
                               s=size, zorder=3, alpha=0.9,
                               edgecolors="black" if is_pt else "none", linewidths=1)

            # Pareto frontier (packing only)
            if len(pareto_mask) > 0:
                pareto_pts = packing_sub[pareto_mask].sort_values("payload_mb")
                if not pareto_pts.empty:
                    ax.plot(pareto_pts["payload_mb"], pareto_pts["mean_acc"],
                            "k--", linewidth=1.5, zorder=4, label="Pareto front")

            # Legend — solutions
            for sol in all_solutions:
                is_pt = sol == "plaintext"
                ax.scatter([], [], color=sol_colors[sol],
                           marker=PLAINTEXT_MARKER if is_pt else "o",
                           label=sol, s=70,
                           edgecolors="black" if is_pt else "none")
            # Legend — percentages
            for pct in percentages:
                ax.scatter([], [], color="gray", marker=pct_markers[pct], label=f"p={pct}", s=50)
            ax.scatter([], [], color="gray", marker=PLAINTEXT_MARKER, label="baseline (plaintext)", s=70,
                       edgecolors="black")

            ax.set_xlabel("Mean payload (MB)")
            ax.set_ylabel("Final accuracy")
            ax.set_title(f"Efficiency Frontier — {model} | niid={niid}")
            ax.legend(fontsize=6, ncol=2, loc="lower right")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train = dfs["train"]
    se = dfs["server_eval"]
    if train.empty or se.empty:
        print("No data found.")
        return

    dims = load_data.discover_dims(train)
    models = dims.get("model", [])

    fig = plot_frontier(train, se, models)
    if fig:
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"08_efficiency_frontier.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 08_efficiency_frontier.png/pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
