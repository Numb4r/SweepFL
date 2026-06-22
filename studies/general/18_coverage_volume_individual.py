"""
18i — Volume-based coverage: one individual figure per (model, dataset, technique, niid, pct).

Same metric as study 18 (18_coverage_volume.py), but instead of a combined grid
with all percentages×niids in one figure per (model, technique), each
(model, dataset, technique, niid, percentage) combination is saved as its own figure.

Output: output/figures/18_coverage_volume_individual/{model}_{dataset}_{tech}_niid{n}_p{pct}.{png,pdf}
No tables generated (use 18_coverage_volume.py for tables).
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "18_coverage_volume_individual"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _tok(model, dataset, technique, niid, pct=None) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    parts = [str(model), str(dataset), str(technique), niid_s]
    if pct is not None:
        parts.append(f"p{int(round(pct * 100)):03d}")
    return "_".join(parts)


# ── Core: volume-coverage sawtooth ───────────────────────────────────────────

def _volume_cycle_series(group_df: pd.DataFrame) -> list[dict]:
    n_packs = int(group_df["n_packs"].iloc[0])
    if n_packs <= 0:
        return []

    records = []
    vol = 0

    for rnd, rnd_df in group_df.sort_values("round").groupby("round", sort=False):
        server_union = set()
        for sel in rnd_df["selected"]:
            server_union.update(sel)
        vol += len(server_union)
        cov_pct = vol / n_packs * 100.0
        records.append({"round": int(rnd), "coverage_pct": cov_pct})
        if vol >= n_packs:
            vol = 0

    return records


def build_coverage_df(masks: pd.DataFrame) -> pd.DataFrame:
    parts = []
    gcols = [c for c in GROUP if c in masks.columns]
    for key, grp in masks.groupby(gcols, dropna=False):
        series = _volume_cycle_series(grp)
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


def _avg_over_seeds(cov_df: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in ["model", "dataset", "solution", "niid", "percentage", "round"]
             if c in cov_df.columns]
    return (
        cov_df.groupby(gcols, dropna=False)["coverage_pct"]
        .mean()
        .reset_index()
    )


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(s: pd.DataFrame, model: str, dataset: str,
              technique: str, niid, pct: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))

    s = s.sort_values("round")
    if not s.empty:
        ax.plot(s["round"], s["coverage_pct"], linewidth=1.8, color="tab:orange")

    ax.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Round", fontsize=9)
    ax.set_ylabel("Volume coverage (%)", fontsize=9)

    pct_lbl  = f"{pct:.0%}"
    niid_lbl = f"niid={niid}" if niid is not None else "niid=NA"
    ax.set_title(
        f"Volume-based coverage — {model} / {dataset} / {technique}\n"
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

    print("Computing volume-coverage series...")
    cov_df = build_coverage_df(masks)
    if cov_df.empty:
        print("No coverage data computed.")
        return

    cov_avg = _avg_over_seeds(cov_df)

    n_saved = 0
    for model in models:
        for dataset in datasets:
            for technique in solutions:
                for niid in niids:
                    for pct in percentages:
                        s = cov_avg.copy()
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
                        print(f"Saved 18_coverage_volume_individual/{fname}.png/pdf")
                        plt.close(fig)
                        n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 18i: individual volume-based coverage figures")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
