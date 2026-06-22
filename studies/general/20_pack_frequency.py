"""
20 — Pack transmission frequency (horizontal bar chart).

Answers: over all rounds of an experiment, how often was each pack index transmitted?

Metric: for each round, compute the server-side union of selected pack indices across all
clients. A pack "was sent" in a round if it appears in that union.
send_pct[pack_k] = (rounds where pack_k was in the server-union) / total_rounds × 100%

Behaviour by technique:
  nvnl   : non-overlapping RR → each pack sent ~pct% of rounds; bars nearly uniform.
  sw     : 50% stride overlap → each pack sent ~2×pct% of rounds (appears in two windows).
  rw     : random/adaptive → spread depends on weights; high-magnitude packs sent more often.
  fedphe : top-k fixed → a fixed subset of ~pct% packs at 100%, all others at 0%.
  srr    : evenly spaced RR → similar to nvnl but different spacing.
  topkrr : top-k phase then RR phase → visible transition in bar heights.

Source data: mask.csv (via load_data.load_masks). No model snapshot needed.

Layout: 1 figure per (model, technique), grid rows=percentage × cols=niid.
        Each subplot = heatmap strip (X = pack index, color = % rounds sent).
        A shared colorbar per figure indicates the color scale (0–100 %).

Output: output/figures/20_pack_frequency/{model}_{technique}.{png,pdf}
Tables: output/tables/20_pack_frequency_summary.{csv,tex}
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

OUT_FIG = HERE / "output" / "figures"
OUT_FIG_STUDY = OUT_FIG / "20_pack_frequency"
OUT_TAB = HERE / "output" / "tables"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]


# ── Core: pack send-count per (config+seed) ───────────────────────────────────

def _pack_send_counts(group_df: pd.DataFrame) -> dict:
    """Return {pack_idx: send_count, '_n_rounds': total_rounds, '_n_packs': n_packs}."""
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
    """Compute per-pack send-pct for every (config+seed) group."""
    gcols = [c for c in GROUP if c in masks.columns]
    parts = []
    for key, grp in masks.groupby(gcols, dropna=False):
        result = _pack_send_counts(grp)
        n_packs = result["n_packs"]
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


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_model_technique(freq_avg: pd.DataFrame, model: str, technique: str,
                         niids: list, percentages: list) -> plt.Figure:
    sub = freq_avg.copy()
    if "model" in sub.columns:
        sub = sub[sub["model"] == model]
    if "solution" in sub.columns:
        sub = sub[sub["solution"] == technique]

    sorted_pcts = sorted(percentages)
    n_rows = max(len(sorted_pcts), 1)
    n_cols = max(len(niids), 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, max(1.6, 1.3 * n_rows)),
                             squeeze=False)
    fig.suptitle(f"Pack transmission frequency — {model} / {technique}", fontsize=13)

    last_im = None
    for r, pct in enumerate(sorted_pcts):
        for c, niid in enumerate(niids):
            ax = axes[r][c]
            s = sub.copy()
            if "percentage" in s.columns:
                s = s[s["percentage"] == pct]
            if niid is not None and "niid" in s.columns:
                s = s[s["niid"] == niid]
            s = s.sort_values("pack_idx")

            ax.set_title(f"{pct:.0%}  |  niid={niid}", fontsize=9)
            ax.set_yticks([])

            if not s.empty:
                n_packs = int(s["pack_idx"].max()) + 1
                data = s["send_pct"].values.reshape(1, -1)
                last_im = ax.imshow(
                    data, aspect="auto", cmap="viridis",
                    vmin=0, vmax=100,
                    extent=[0, n_packs, 0, 1],
                )
                ax.set_xlim(0, n_packs)

            ax.set_ylim(0, 1)

            if c == 0:
                ax.set_ylabel(f"{pct:.0%}", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("Pack index", fontsize=8)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(),
                     shrink=0.6, label="% rounds sent", pad=0.02)

    fig.tight_layout()
    return fig


# ── Tables ────────────────────────────────────────────────────────────────────

def save_csv(freq_avg: pd.DataFrame):
    cols = [c for c in ["model", "dataset", "solution", "niid", "percentage",
                         "pack_idx", "send_count", "send_pct"] if c in freq_avg.columns]
    out = freq_avg[cols].copy()
    for c in out.select_dtypes(include=float).columns:
        out[c] = out[c].round(3)
    path = OUT_TAB / "20_pack_frequency_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(freq_avg: pd.DataFrame):
    display_cols = {
        "model": "Model", "solution": "Solution", "niid": "NIID",
        "percentage": "Pct.", "pack_idx": "Pack", "send_pct": "Send \\% (rounds)",
    }
    cols = [c for c in display_cols if c in freq_avg.columns]

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        if isinstance(val, float):
            return f"{val:.1f}"
        return str(val)

    header = " & ".join(display_cols.get(c, c) for c in cols) + r" \\"
    ncols = len(cols)
    lines = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Pack transmission frequency (Study 20)}",
        r"\label{tab:pack_frequency} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_key = None
    key_cols = [c for c in ["solution", "model"] if c in cols]
    for _, row in freq_avg.sort_values(
            [c for c in ["solution", "model", "niid", "percentage", "pack_idx"]
             if c in freq_avg.columns]).iterrows():
        cur_key = tuple(row[c] for c in key_cols)
        if prev_key is not None and cur_key != prev_key:
            lines.append(r"\midrule")
        prev_key = cur_key
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "20_pack_frequency_summary.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


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

    dims = load_data.discover_dims(masks)
    models = dims.get("model", [])
    solutions = dims.get("solution", [])
    niids = dims.get("niid", [])
    percentages = sorted(masks["percentage"].dropna().unique())
    print(f"Models     : {models}")
    print(f"Solutions  : {solutions}")
    print(f"Percentages: {percentages}")
    print(f"Mask rows  : {len(masks):,}")
    print()

    print("Computing pack send frequencies...")
    freq_df = build_freq_df(masks)
    if freq_df.empty:
        print("No frequency data computed.")
        return

    freq_avg = _avg_over_seeds(freq_df)

    n_saved = 0
    for model in models:
        for technique in solutions:
            cs = freq_avg.copy()
            if "model" in cs.columns:
                cs = cs[cs["model"] == model]
            if "solution" in cs.columns:
                cs = cs[cs["solution"] == technique]
            if cs.empty:
                continue
            fig = plot_model_technique(freq_avg, model, technique, niids,
                                       percentages)
            fname = f"{model}_{technique}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
            print(f"Saved 20_pack_frequency/{fname}.png/pdf")
            plt.close(fig)
            n_saved += 1
    print(f"Total figures: {n_saved}")
    print()
    save_csv(freq_avg)
    save_latex(freq_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 20: pack transmission frequency per round")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
