"""
20i — Pack transmission frequency: one individual figure per (model, dataset, technique, niid, pct).

Same metric as study 20 (20_pack_frequency.py), but instead of a combined grid, each
(model, dataset, technique, niid, percentage) combination is saved as its own figure
(heatmap strip: X = pack index, color = % rounds sent).

Output: output/figures/20_pack_frequency_individual/{model}_{dataset}_{tech}_niid{n}_p{pct}.{png,pdf}
No tables generated (use 20_pack_frequency.py for tables).
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "20_pack_frequency_individual"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _tok(model, dataset, technique, niid, pct=None) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    parts = [str(model), str(dataset), str(technique), niid_s]
    if pct is not None:
        parts.append(f"p{int(round(pct * 100)):03d}")
    return "_".join(parts)


# ── Core: pack send-count (identical to study 20) ────────────────────────────

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
        result = _pack_send_counts(grp)
        n_packs  = result["n_packs"]
        n_rounds = result["n_rounds"]
        if n_rounds == 0:
            continue
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        for pack_idx in range(n_packs):
            cnt = result["counts"].get(pack_idx, 0)
            row = {**meta,
                   "pack_idx": pack_idx,
                   "send_count": cnt,
                   "send_pct": cnt / n_rounds * 100.0,
                   "n_rounds": n_rounds}
            parts.append(row)
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

def plot_cell(s: pd.DataFrame, model: str, dataset: str,
              technique: str, niid, pct: float) -> plt.Figure:
    """Single heatmap strip for one (model, dataset, technique, niid, pct)."""
    fig, ax = plt.subplots(figsize=(8, 1.6))

    ax.set_yticks([])
    if not s.empty:
        n_packs = int(s["pack_idx"].max()) + 1
        data = s.sort_values("pack_idx")["send_pct"].values.reshape(1, -1)
        im = ax.imshow(
            data, aspect="auto", cmap="viridis",
            vmin=0, vmax=100,
            extent=[0, n_packs, 0, 1],
        )
        ax.set_xlim(0, n_packs)
        fig.colorbar(im, ax=ax, shrink=0.8, label="% rounds sent", pad=0.02)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Pack index", fontsize=9)

    pct_lbl  = f"{pct:.0%}"
    niid_lbl = f"niid={niid}" if niid is not None else "niid=NA"
    ax.set_title(
        f"Pack frequency — {model} / {dataset} / {technique}  |  {pct_lbl}  |  {niid_lbl}",
        fontsize=9,
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
                    for pct in percentages:
                        s = freq_avg.copy()
                        if "model" in s.columns:
                            s = s[s["model"] == model]
                        if "dataset" in s.columns:
                            s = s[s["dataset"] == dataset]
                        if "solution" in s.columns:
                            s = s[s["solution"] == technique]
                        if niid is not None and "niid" in s.columns:
                            s = s[s["niid"] == niid]
                        if "percentage" in s.columns:
                            s = s[s["percentage"] == pct]
                        if s.empty:
                            continue

                        fig = plot_cell(s, model, dataset, technique, niid, pct)
                        fname = _tok(model, dataset, technique, niid, pct)
                        for ext in ("png", "pdf"):
                            fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                        dpi=150, bbox_inches="tight", pad_inches=0.02)
                        print(f"Saved 20_pack_frequency_individual/{fname}.png/pdf")
                        plt.close(fig)
                        n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 20i: individual pack transmission frequency figures")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
