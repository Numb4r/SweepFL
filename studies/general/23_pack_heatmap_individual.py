"""
23i — Pack transmission frequency heatmap: one individual figure per (model, dataset, technique, niid).

Same data as study 23 (23_pack_heatmap.py), but instead of one figure per (model, technique)
with multiple niid subplots side by side, each (model, dataset, technique, niid) combination
is saved as its own figure (2D heatmap: Y=pack index, X=sparsification percentage, color=send_pct).

Output: output/figures/23_pack_heatmap_individual/{model}_{dataset}_{tech}_niid{n}.{png,pdf}
No tables generated (use 23_pack_heatmap.py for tables).
"""

import argparse
import importlib.util
import sys
from collections import defaultdict
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "23_pack_heatmap_individual"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _tok(model, dataset, technique, niid) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    return "_".join([str(model), str(dataset), str(technique), niid_s])


# ── Core: pack send-count (identical to study 23) ────────────────────────────

def _pack_send_counts(group_df: pd.DataFrame) -> dict:
    n_packs = int(group_df["n_packs"].iloc[0])
    counts = defaultdict(int)
    n_rounds = 0
    for rnd, rnd_df in group_df.sort_values("round").groupby("round", sort=False):
        server_union = set()
        for sel in rnd_df["selected"]:
            server_union.update(sel)
        for idx in server_union:
            counts[idx] += 1
        n_rounds += 1
    return {"counts": dict(counts), "n_rounds": n_rounds, "n_packs": n_packs}


def build_freq_df(masks: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in GROUP if c in masks.columns]
    parts = []
    for key, grp in masks.groupby(gcols, dropna=False):
        result  = _pack_send_counts(grp)
        n_packs  = result["n_packs"]
        n_rounds = result["n_rounds"]
        if n_rounds == 0:
            continue
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        for pack_idx in range(n_packs):
            cnt = result["counts"].get(pack_idx, 0)
            parts.append({
                **meta,
                "pack_idx": pack_idx,
                "send_count": cnt,
                "send_pct": cnt / n_rounds * 100.0,
                "n_rounds": n_rounds,
            })
    if not parts:
        return pd.DataFrame()
    return pd.DataFrame(parts)


def _avg_over_seeds(freq_df: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in ["model", "dataset", "solution", "niid", "percentage", "pack_idx"]
             if c in freq_df.columns]
    return (
        freq_df.groupby(gcols, dropna=False)[["send_count", "send_pct", "n_rounds"]]
        .mean()
        .reset_index()
    )


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(sub: pd.DataFrame, model: str, dataset: str,
              technique: str, niid, percentages: list) -> plt.Figure:
    """2D heatmap (Y=pack index, X=pct) for one (model, dataset, technique, niid)."""
    sorted_pcts = sorted(percentages)
    n_pcts      = len(sorted_pcts)
    n_packs     = int(sub["pack_idx"].max()) + 1 if not sub.empty else 1

    fig_h = max(4.0, n_packs * 0.18)
    fig_w = max(3.5, n_pcts * 0.7 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat = np.full((n_packs, n_pcts), np.nan)
    for j, pct in enumerate(sorted_pcts):
        s = sub.copy()
        if "percentage" in s.columns:
            s = s[s["percentage"] == pct]
        s = s.sort_values("pack_idx")
        if not s.empty:
            for _, row in s.iterrows():
                mat[int(row["pack_idx"]), j] = row["send_pct"]

    im = ax.imshow(
        mat, aspect="auto", cmap="viridis",
        vmin=0, vmax=100,
        extent=[-0.5, n_pcts - 0.5, n_packs - 0.5, -0.5],
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label="% rounds sent", pad=0.02)

    ax.set_xticks(range(n_pcts))
    ax.set_xticklabels([f"{p:.0%}" for p in sorted_pcts], fontsize=8)
    ax.set_xlabel("Sparsification", fontsize=9)

    ytick_fs = max(4, min(8, int(180 / max(n_packs, 1))))
    if n_packs <= 50:
        ax.set_yticks(range(n_packs))
        ax.set_yticklabels([str(i) for i in range(n_packs)], fontsize=ytick_fs)
    else:
        step = max(1, n_packs // 20)
        ticks = list(range(0, n_packs, step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(i) for i in ticks], fontsize=ytick_fs)

    ax.set_ylabel("Pack index", fontsize=9)

    niid_lbl = f"niid={niid}" if niid is not None else "niid=NA"
    ax.set_title(
        f"Pack heatmap — {model} / {dataset} / {technique}  |  {niid_lbl}",
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

    print("Computing pack send frequencies...")
    freq_df = build_freq_df(masks)
    if freq_df.empty:
        print("No frequency data computed.")
        return

    freq_avg = _avg_over_seeds(freq_df)

    n_saved = 0
    for model in models:
        for dataset in datasets:
            for technique in solutions:
                for niid in niids:
                    s = freq_avg.copy()
                    if "model" in s.columns:
                        s = s[s["model"] == model]
                    if "dataset" in s.columns:
                        s = s[s["dataset"] == dataset]
                    if "solution" in s.columns:
                        s = s[s["solution"] == technique]
                    if niid is not None and "niid" in s.columns:
                        s = s[s["niid"] == niid]
                    if s.empty:
                        continue

                    fig = plot_cell(s, model, dataset, technique, niid, percentages)
                    fname = _tok(model, dataset, technique, niid)
                    for ext in ("png", "pdf"):
                        fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                    dpi=150, bbox_inches="tight", pad_inches=0.02)
                    print(f"Saved 23_pack_heatmap_individual/{fname}.png/pdf")
                    plt.close(fig)
                    n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 23i: individual pack transmission frequency heatmap figures")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
