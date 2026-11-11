"""
23 — Pack transmission frequency heatmap (2D).

Same data as study 20 (pack send frequency from mask.csv), but displayed as a 2D heatmap
instead of horizontal bars — same layout style as study 22:

  Y = pack index  (rows, 0 … n_packs-1)
  X = sparsification percentage  (columns, 10% … 90%)
  color = send_pct  (% of rounds in which this pack was sent)

One subplot per niid (side by side). Figure height scales with n_packs so large models
(mobilenetv2=176 packs, resnet20=67) remain readable.

Compared with study 20 (bars, one row per percentage): here all percentages are visible
in a single glance per niid, making it easy to see how the pack-selection pattern changes
across sparsification levels (uniform=sweepfl/sw, bimodal=fedphe).

Source data: mask.csv (via load_data.load_masks). No model snapshot needed.

Output: output/figures/23_pack_heatmap/{model}_{technique}.{png,pdf}
Tables: output/tables/23_pack_heatmap_summary.{csv,tex}
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
OUT_FIG_STUDY = OUT_FIG / "23_pack_heatmap"
OUT_TAB = HERE / "output" / "tables"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]


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
        n_packs = result["n_packs"]
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


# ── Plot: 2D heatmap per (model, technique) ───────────────────────────────────
# Y = pack index (rows), X = percentage (cols), one subplot per niid.

def plot_model_technique(freq_avg: pd.DataFrame, model: str, technique: str,
                         niids: list, percentages: list) -> plt.Figure:
    sub = freq_avg.copy()
    if "model" in sub.columns:
        sub = sub[sub["model"] == model]
    if "solution" in sub.columns:
        sub = sub[sub["solution"] == technique]

    sorted_pcts = sorted(percentages)
    n_pcts = len(sorted_pcts)
    n_cols = max(len(niids), 1)
    n_packs = int(sub["pack_idx"].max()) + 1 if not sub.empty else 1

    # Figure height scales with n_packs (0.18 in/pack, min 4 in)
    fig_h = max(4.0, n_packs * 0.18)
    # Figure width: narrow — only 5 percentage columns + y-label
    fig_w = max(3.5, n_pcts * 0.7 + 2.5) * n_cols

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(f"Pack transmission frequency heatmap — {model} / {technique}",
                 fontsize=13)

    last_im = None
    for c, niid in enumerate(niids):
        ax = axes[0][c]

        # Build 2D matrix: rows=pack_idx, cols=percentage
        mat = np.full((n_packs, n_pcts), np.nan)
        for j, pct in enumerate(sorted_pcts):
            s = sub.copy()
            if "percentage" in s.columns:
                s = s[s["percentage"] == pct]
            if niid is not None and "niid" in s.columns:
                s = s[s["niid"] == niid]
            s = s.sort_values("pack_idx")
            if not s.empty:
                for _, row in s.iterrows():
                    mat[int(row["pack_idx"]), j] = row["send_pct"]

        last_im = ax.imshow(
            mat, aspect="auto", cmap="viridis",
            vmin=0, vmax=100,
            extent=[-0.5, n_pcts - 0.5, n_packs - 0.5, -0.5],
        )

        # X ticks = percentage labels
        ax.set_xticks(range(n_pcts))
        ax.set_xticklabels([f"{p:.0%}" for p in sorted_pcts], fontsize=8)
        ax.set_xlabel("Sparsification", fontsize=8)

        # Y ticks = pack indices; font scales down for large models
        ytick_fs = max(4, min(8, int(180 / max(n_packs, 1))))
        if n_packs <= 50:
            ax.set_yticks(range(n_packs))
            ax.set_yticklabels([str(i) for i in range(n_packs)], fontsize=ytick_fs)
        else:
            # For large models, only show every N-th tick
            step = max(1, n_packs // 20)
            ticks = list(range(0, n_packs, step))
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(i) for i in ticks], fontsize=ytick_fs)

        ax.set_title(f"niid = {niid}", fontsize=9)
        if c == 0:
            ax.set_ylabel("Pack index", fontsize=8)
        else:
            ax.set_yticklabels([])

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
    for col in out.select_dtypes(include=float).columns:
        out[col] = out[col].round(3)
    path = OUT_TAB / "23_pack_heatmap_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(freq_avg: pd.DataFrame):
    display_cols = {
        "model": "Model", "solution": "Solution", "niid": "NIID",
        "percentage": "Pct.", "pack_idx": "Pack",
        "send_pct": "Send \\% (rounds)",
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
        r"\caption{Pack transmission frequency heatmap summary (Study 23)}",
        r"\label{tab:pack_heatmap} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_key = None
    sort_cols = [c for c in ["solution", "model", "niid", "percentage", "pack_idx"]
                 if c in freq_avg.columns]
    for _, row in freq_avg.sort_values(sort_cols).iterrows():
        cur_key = (row.get("solution"), row.get("model"))
        if prev_key is not None and cur_key != prev_key:
            lines.append(r"\midrule")
        prev_key = cur_key
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "23_pack_heatmap_summary.tex"
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
    models      = dims.get("model", [])
    solutions   = dims.get("solution", [])
    niids       = dims.get("niid", [])
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
            fig = plot_model_technique(freq_avg, model, technique,
                                       niids, percentages)
            fname = f"{model}_{technique}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                            dpi=150, bbox_inches="tight")
            print(f"Saved 23_pack_heatmap/{fname}.png/pdf")
            plt.close(fig)
            n_saved += 1
    print(f"Total figures: {n_saved}")
    print()
    save_csv(freq_avg)
    save_latex(freq_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 23: pack transmission frequency 2D heatmap")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
