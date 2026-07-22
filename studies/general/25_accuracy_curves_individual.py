"""
25 — Accuracy vs Rounds: one figure per (model, dataset, niid, percentage, client_fraction).

Each figure shows accuracy (%) over FL rounds for all packing solutions present at that
percentage, plus the plaintext baseline from the same client_fraction, as a dashed line.

Output: output/figures/25_accuracy_curves_individual/cf<frac>/{model}_{dataset}_niid{n}_p{pct}.{png,pdf}
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

OUT_BASE = HERE / "output" / "figures" / "25_accuracy_curves_individual"

SOL_COLORS = {
    "plaintext":  "#333333",
    "nvnl":       "#2196F3",
    "sw":         "#FF5722",
    "rw":         "#4CAF50",
    "fedphe":     "#9C27B0",
    "topkrr":     "#E91E63",
    "srr":        "#009688",
    "ckks":       "#FF9800",
    "bfv":        "#00BCD4",
    "batchcrypt": "#795548",
}
SOL_MARKERS = {
    "plaintext": "D", "nvnl": "o", "sw": "s", "rw": "^",
    "fedphe": "P", "topkrr": "p", "srr": "h",
    "ckks": "v", "bfv": "X", "batchcrypt": "H",
}

# Dimensions used to identify a unique experiment run (before averaging over seeds/reruns)
SEED_DIMS = ["model", "dataset", "solution", "niid", "percentage", "seed", "client_fraction"]
# Dimensions that define one "cell" (figure)
CELL_DIMS = ["model", "dataset", "niid", "percentage", "client_fraction"]
# Dimensions used to average over seeds and reruns for the plotted curves
AVG_DIMS  = ["model", "dataset", "solution", "niid", "percentage", "client_fraction", "round"]


def _sorted_solutions(sols):
    """Return solutions sorted: packing solutions first, plaintext last."""
    other = sorted(s for s in sols if s != "plaintext")
    return other + (["plaintext"] if "plaintext" in sols else [])


def _cf_label(cf: float) -> str:
    """Format client_fraction as a folder-safe string like cf0.25."""
    return f"cf{cf:g}"


def _pct_label(pct) -> str:
    if pct is None:
        return "plaintext"
    return f"p{int(round(pct * 100)):03d}"


def build_averaged(server_eval: pd.DataFrame) -> pd.DataFrame:
    """Average mean_acc over seeds/reruns for every (config..., round) group."""
    if server_eval.empty:
        return pd.DataFrame()

    # Ensure numeric
    for col in ["round", "niid", "seed", "client_fraction"]:
        if col in server_eval.columns:
            server_eval[col] = pd.to_numeric(server_eval[col], errors="coerce")
    if "mean_acc" in server_eval.columns:
        server_eval["mean_acc"] = pd.to_numeric(server_eval["mean_acc"], errors="coerce")

    avg_cols = [c for c in AVG_DIMS if c in server_eval.columns]
    return (
        server_eval.groupby(avg_cols, dropna=False)["mean_acc"]
        .mean()
        .reset_index()
    )


def plot_cell(
    averaged: pd.DataFrame,
    model: str,
    dataset: str,
    niid: float,
    pct: float,
    cf: float,
) -> plt.Figure:
    """One figure: accuracy vs rounds for all solutions at (model,dataset,niid,pct,cf)."""

    # Packing solutions at this percentage
    pct_mask  = averaged["percentage"].notna() & (averaged["percentage"] == pct)
    # Plaintext baseline at this client_fraction (percentage is None)
    pt_mask   = averaged["percentage"].isna() & (averaged["solution"] == "plaintext")

    sub = averaged[pct_mask | pt_mask].copy()
    if "model"   in sub.columns: sub = sub[sub["model"]   == model]
    if "dataset" in sub.columns: sub = sub[sub["dataset"] == dataset]
    if "niid"    in sub.columns: sub = sub[sub["niid"]    == niid]
    if "client_fraction" in sub.columns:
        sub = sub[sub["client_fraction"] == cf]

    if sub.empty:
        return None

    solutions = _sorted_solutions(sub["solution"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for sol in solutions:
        sg = sub[sub["solution"] == sol].sort_values("round")
        if sg.empty:
            continue
        color  = SOL_COLORS.get(sol, "#888888")
        marker = SOL_MARKERS.get(sol, "o")
        ls     = "--" if sol == "plaintext" else "-"
        lw     = 1.5  if sol == "plaintext" else 2.0
        ms     = 0    if sol == "plaintext" else 4
        label  = "plaintext (baseline)" if sol == "plaintext" else sol

        ax.plot(
            sg["round"], sg["mean_acc"] * 100,
            color=color, linestyle=ls, linewidth=lw,
            marker=marker if ms > 0 else None,
            markersize=ms, markevery=max(1, len(sg) // 10),
            label=label, zorder=3 if sol != "plaintext" else 2,
        )

    pct_lbl  = f"{pct:.0%}"
    niid_lbl = f"niid={niid}"
    cf_lbl   = f"cf={cf:g}"
    ax.set_title(
        f"{model} / {dataset}  |  {niid_lbl}  |  sparsification={pct_lbl}  |  {cf_lbl}",
        fontsize=10,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Accuracy vs Rounds — Federated Learning with HE Sparsification",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    server_eval = dfs["server_eval"]

    if server_eval.empty:
        print("No server_eval data found.")
        return

    averaged = build_averaged(server_eval)
    if averaged.empty:
        print("No data after averaging.")
        return

    dims = load_data.discover_dims(averaged)
    models      = dims.get("model", [])
    datasets    = dims.get("dataset", [])
    niids       = dims.get("niid", [])
    percentages = [p for p in dims.get("percentage", []) if p is not None]
    # client_fraction may not be in discover_dims (not in _GROUP_COLS); compute directly
    if "client_fraction" in averaged.columns:
        cfs = sorted(averaged["client_fraction"].dropna().unique().tolist())
    else:
        cfs = [1.0]

    print(f"Models           : {models}")
    print(f"Datasets         : {datasets}")
    print(f"NIIDs            : {niids}")
    print(f"Percentages      : {percentages}")
    print(f"Client fractions : {cfs}")
    print()

    n_saved = 0

    for cf in cfs:
        cf_dir = OUT_BASE / _cf_label(cf)
        cf_dir.mkdir(parents=True, exist_ok=True)

        for model in models:
            for dataset in datasets:
                for niid in niids:
                    for pct in percentages:
                        fig = plot_cell(averaged, model, dataset, niid, pct, cf)
                        if fig is None:
                            continue
                        niid_s = f"niid{niid:g}"
                        fname  = f"{model}_{dataset}_{niid_s}_{_pct_label(pct)}"
                        for ext in ("png", "pdf"):
                            fig.savefig(
                                cf_dir / f"{fname}.{ext}",
                                dpi=150, bbox_inches="tight", pad_inches=0.02,
                            )
                        print(f"  Saved {_cf_label(cf)}/{fname}.png/pdf")
                        plt.close(fig)
                        n_saved += 1

    print(f"\nTotal figures saved: {n_saved}")
    print(f"Output: {OUT_BASE.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 25: accuracy vs rounds, one figure per config")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
