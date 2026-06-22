"""
19i — Coverage vs accuracy (dual Y axes): one individual figure per (model, dataset, technique, niid, pct).

Same metric as study 19 (19_coverage_accuracy.py), but instead of a combined grid, each
(model, dataset, technique, niid, percentage) combination is saved as its own figure with
dual Y axes: coverage (blue, left) and accuracy (red, right).

Output: output/figures/19_coverage_accuracy_individual/{model}_{dataset}_{tech}_niid{n}_p{pct}.{png,pdf}
No tables generated (use 19_coverage_accuracy.py for tables).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("load_data", HERE / "load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)

OUT_FIG_STUDY = HERE / "output" / "figures" / "19_coverage_accuracy_individual"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

GROUP      = ["model", "dataset", "solution", "niid", "percentage", "seed"]
COV_COLOR  = "tab:blue"
ACC_COLOR  = "tab:red"


def _tok(model, dataset, technique, niid, pct=None) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    parts = [str(model), str(dataset), str(technique), niid_s]
    if pct is not None:
        parts.append(f"p{int(round(pct * 100)):03d}")
    return "_".join(parts)


# ── Core: unique-coverage (identical to study 17) ────────────────────────────

def _unique_cycle_series(group_df: pd.DataFrame) -> list[dict]:
    n_packs = int(group_df["n_packs"].iloc[0])
    if n_packs <= 0:
        return []

    records = []
    covered = set()

    for rnd, rnd_df in group_df.sort_values("round").groupby("round", sort=False):
        server_union = set()
        for sel in rnd_df["selected"]:
            server_union.update(sel)
        covered.update(server_union)
        cov_pct = len(covered) / n_packs * 100.0
        records.append({"round": int(rnd), "coverage_pct": cov_pct})
        if len(covered) >= n_packs:
            covered = set()

    return records


def build_coverage_df(masks: pd.DataFrame) -> pd.DataFrame:
    parts = []
    gcols = [c for c in GROUP if c in masks.columns]
    for key, grp in masks.groupby(gcols, dropna=False):
        series = _unique_cycle_series(grp)
        if not series:
            continue
        df = pd.DataFrame(series)
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        for k, v in meta.items():
            df[k] = v
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _avg_over_seeds(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    gcols = [c for c in ["model", "dataset", "solution", "niid", "percentage", "round"]
             if c in df.columns]
    return (
        df.groupby(gcols, dropna=False)[value_col]
        .mean()
        .reset_index()
    )


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(cs: pd.DataFrame, ac: pd.DataFrame,
              model: str, dataset: str, technique: str,
              niid, pct: float) -> plt.Figure:
    """Single figure with dual Y axes: coverage (left) + accuracy (right)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.set_ylim(0, 110)
    if not cs.empty:
        ax.plot(cs["round"], cs["coverage_pct"],
                linewidth=1.8, color=COV_COLOR)
    ax.axhline(100, color=COV_COLOR, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.tick_params(axis="y", labelcolor=COV_COLOR, labelsize=7)
    ax.set_ylabel("Coverage (%)", color=COV_COLOR, fontsize=9)
    ax.set_xlabel("Round", fontsize=9)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 100)
    if not ac.empty:
        ax2.plot(ac["round"], ac["mean_acc"] * 100,
                 linewidth=1.8, color=ACC_COLOR)
    ax2.tick_params(axis="y", labelcolor=ACC_COLOR, labelsize=7)
    ax2.set_ylabel("Accuracy (%)", color=ACC_COLOR, fontsize=9)

    ax.grid(True, alpha=0.25)

    h_cov = mlines.Line2D([], [], color=COV_COLOR, linewidth=1.8, label="Coverage (%)")
    h_acc = mlines.Line2D([], [], color=ACC_COLOR, linewidth=1.8, label="Accuracy (%)")
    ax.legend(handles=[h_cov, h_acc], loc="lower right", fontsize=8, framealpha=0.85)

    pct_lbl  = f"{pct:.0%}"
    niid_lbl = f"niid={niid}" if niid is not None else "niid=NA"
    ax.set_title(
        f"Coverage vs accuracy — {model} / {dataset} / {technique}\n"
        f"{pct_lbl}  |  {niid_lbl}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    print("Loading mask data...")
    try:
        masks = load_data.load_masks(logs_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if masks.empty:
        print("No mask data found.")
        return
    masks = masks[masks["percentage"].notna()].copy()
    if masks.empty:
        print("No packing-technique mask data found.")
        return

    print("Loading accuracy data...")
    try:
        dfs = load_data.load_all(logs_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    server_eval = dfs["server_eval"]
    server_eval = server_eval[server_eval["percentage"].notna()].copy()

    dims        = load_data.discover_dims(masks)
    models      = dims.get("model", [])
    datasets    = dims.get("dataset", [])
    solutions   = dims.get("solution", [])
    niids       = dims.get("niid", [])
    percentages = sorted(masks["percentage"].dropna().unique())
    print(f"Models     : {models}")
    print(f"Datasets   : {datasets}")
    print(f"Solutions  : {solutions}")
    print(f"Percentages: {percentages}")
    print()

    print("Computing unique-coverage series...")
    cov_df = build_coverage_df(masks)
    if cov_df.empty:
        print("No coverage data computed.")
        return

    cov_avg = _avg_over_seeds(cov_df, "coverage_pct")
    acc_avg = _avg_over_seeds(server_eval, "mean_acc")

    n_saved = 0
    for model in models:
        for dataset in datasets:
            for technique in solutions:
                for niid in niids:
                    for pct in percentages:
                        cs = cov_avg.copy()
                        if "model" in cs.columns:
                            cs = cs[cs["model"] == model]
                        if "dataset" in cs.columns:
                            cs = cs[cs["dataset"] == dataset]
                        if "solution" in cs.columns:
                            cs = cs[cs["solution"] == technique]
                        if niid is not None and "niid" in cs.columns:
                            cs = cs[cs["niid"] == niid]
                        if "percentage" in cs.columns:
                            cs = cs[cs["percentage"] == pct]

                        if cs.empty:
                            continue

                        ac = acc_avg.copy()
                        if "model" in ac.columns:
                            ac = ac[ac["model"] == model]
                        if "dataset" in ac.columns:
                            ac = ac[ac["dataset"] == dataset]
                        if "solution" in ac.columns:
                            ac = ac[ac["solution"] == technique]
                        if niid is not None and "niid" in ac.columns:
                            ac = ac[ac["niid"] == niid]
                        if "percentage" in ac.columns:
                            ac = ac[ac["percentage"] == pct]
                        ac = ac.sort_values("round")

                        fig = plot_cell(cs.sort_values("round"), ac,
                                        model, dataset, technique, niid, pct)
                        fname = _tok(model, dataset, technique, niid, pct)
                        for ext in ("png", "pdf"):
                            fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                        dpi=150, bbox_inches="tight", pad_inches=0.02)
                        print(f"Saved 19_coverage_accuracy_individual/{fname}.png/pdf")
                        plt.close(fig)
                        n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 19i: individual coverage vs accuracy figures")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
