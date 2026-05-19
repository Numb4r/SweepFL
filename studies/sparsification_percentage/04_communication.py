"""
04 — Communication efficiency.

Plots:
  A) Line: mean payload_bytes vs percentage, by solution.
  B) Line: packs_sent / n_packs ratio vs percentage, by solution.
  C) Scatter: final accuracy vs mean payload (MB) per configuration.

Output: output/figures/04_communication.{png,pdf}
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


def plot_payload_line(train, models):
    packing = train[train["percentage"].notna()].copy()
    if "payload_bytes" not in packing.columns or packing.empty:
        return None

    solutions = sorted(packing["solution"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(solutions), 1))

    fig, axes = plt.subplots(len(models), 1, figsize=(6, 4 * len(models)), squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = packing[packing["model"] == model] if "model" in packing.columns else packing
        agg = sub.groupby(["solution", "percentage"], dropna=False)["payload_bytes"].mean().reset_index()
        for i, sol in enumerate(solutions):
            s = agg[agg["solution"] == sol].sort_values("percentage")
            if s.empty:
                continue
            ax.plot(s["percentage"], s["payload_bytes"] / 1e6, marker="o",
                    label=sol, color=cmap(i), linewidth=1.5)
        ax.set_xlabel("Sparsification percentage")
        ax.set_ylabel("Mean payload (MB)")
        ax.set_title(f"Payload Size vs Percentage — {model}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_pack_ratio_line(train, models):
    packing = train[train["percentage"].notna()].copy()
    if "n_packs" not in packing.columns or "packs_sent" not in packing.columns:
        return None
    packing = packing[(packing["n_packs"] > 0)].copy()
    if packing.empty:
        return None
    packing["pack_ratio"] = packing["packs_sent"] / packing["n_packs"]

    solutions = sorted(packing["solution"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(solutions), 1))

    fig, axes = plt.subplots(len(models), 1, figsize=(6, 4 * len(models)), squeeze=False)

    for row, model in enumerate(models):
        ax = axes[row][0]
        sub = packing[packing["model"] == model] if "model" in packing.columns else packing
        agg = sub.groupby(["solution", "percentage"], dropna=False)["pack_ratio"].mean().reset_index()
        for i, sol in enumerate(solutions):
            s = agg[agg["solution"] == sol].sort_values("percentage")
            if s.empty:
                continue
            ax.plot(s["percentage"], s["pack_ratio"], marker="o",
                    label=sol, color=cmap(i), linewidth=1.5)
        ax.set_xlabel("Sparsification percentage")
        ax.set_ylabel("packs_sent / n_packs")
        ax.set_title(f"Pack Transmission Ratio — {model}")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_acc_vs_payload_scatter(train, server_eval, models):
    packing_train = train[train["percentage"].notna() & ("payload_bytes" in train.columns if "payload_bytes" in train.columns else False)].copy()
    if "payload_bytes" not in train.columns:
        return None
    packing_train = train[train["percentage"].notna()].copy()
    if packing_train.empty:
        return None

    payload_agg = packing_train.groupby(
        ["model", "solution", "niid", "percentage"], dropna=False
    )["payload_bytes"].mean().reset_index()
    payload_agg["payload_mb"] = payload_agg["payload_bytes"] / 1e6

    lr = load_data.last_round(server_eval[server_eval["percentage"].notna()])
    acc_agg = lr.groupby(["model", "solution", "niid", "percentage"], dropna=False)["mean_acc"].mean().reset_index()

    merged = acc_agg.merge(payload_agg, on=["model", "solution", "niid", "percentage"], how="inner")
    if merged.empty:
        return None

    solutions = sorted(merged["solution"].unique())
    percentages = sorted(merged["percentage"].dropna().unique())
    cmap = plt.cm.get_cmap("tab10", max(len(solutions), 1))
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    pct_markers = {p: markers[i % len(markers)] for i, p in enumerate(percentages)}

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)

    for col, model in enumerate(models):
        ax = axes[0][col]
        sub = merged[merged["model"] == model] if "model" in merged.columns else merged
        for i, sol in enumerate(solutions):
            s = sub[sub["solution"] == sol]
            for _, row in s.iterrows():
                ax.scatter(
                    row["payload_mb"], row["mean_acc"],
                    color=cmap(i),
                    marker=pct_markers.get(row["percentage"], "o"),
                    s=80, zorder=3,
                )
            if not s.empty:
                ax.scatter([], [], color=cmap(i), label=sol, s=60)

        for pct, mk in pct_markers.items():
            ax.scatter([], [], color="gray", marker=mk, label=f"p={pct}", s=50)

        ax.set_xlabel("Mean payload (MB)")
        ax.set_ylabel("Final accuracy")
        ax.set_title(f"Accuracy vs Payload — {model}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train = dfs["train"]
    se = dfs["server_eval"]

    if train.empty:
        print("No train data found.")
        return

    dims = load_data.discover_dims(train)
    models = dims.get("model", [])

    fig_a = plot_payload_line(train, models)
    if fig_a:
        for ext in ("png", "pdf"):
            fig_a.savefig(OUT / f"04_payload_line.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 04_payload_line.png/pdf")
        plt.close(fig_a)

    fig_b = plot_pack_ratio_line(train, models)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT / f"04_pack_ratio.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 04_pack_ratio.png/pdf")
        plt.close(fig_b)

    fig_c = plot_acc_vs_payload_scatter(train, se, models)
    if fig_c:
        for ext in ("png", "pdf"):
            fig_c.savefig(OUT / f"04_acc_vs_payload.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 04_acc_vs_payload.png/pdf")
        plt.close(fig_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
