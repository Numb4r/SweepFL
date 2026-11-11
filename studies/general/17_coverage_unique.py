"""
17 — Observed unique-pack coverage per round (no duplicate counting).

Answers: as the rounds progress, what fraction of the model's packs has the server
received at least once (within the current cycle)?

Metric — unique coverage:
  For each round, the server-side union of pack indices across all clients is added to a
  running set `covered`.  Coverage = |covered| / n_packs × 100%.
  When |covered| reaches n_packs (100%), the set resets → sawtooth pattern restarts.
  Duplicates within a cycle (e.g. SW overlap) do NOT inflate the counter; only genuinely
  new packs advance the curve.

Behaviour by technique:
  sweepfl  : non-overlapping window → rises by packs_sent/n_packs each round; resets at 100%.
  sw    : 50% stride overlap → new-pack contribution per round ≈ packs_sent/2 → longer cycle.
  rw    : random adaptive → irregular slope; cycle length ≈ coupon-collector.
  srr   : spread round-robin (evenly spaced) → rises like sweepfl but spacing changes.
  topkrr: top-k phase: saturates below 100% (same packs); RR phase: full coverage cycle.
  fedphe: top-k fixed every round → curve saturates at packs_sent/n_packs%, never resets.

Source data: mask.csv (one row per client/round, loaded via load_data.load_masks).
Complements scripts 10/11 (theoretical formulas) with observed coverage from real logs.

Plots: one figure per (model, technique), 2 subplots side-by-side (columns = niid values).
       In each subplot, one line per sparsification percentage (10%→90%, viridis colormap).
       x=round, y=coverage %.  Multiple seeds are averaged per round.

Output: output/figures/17_coverage_unique/{model}_{technique}.{png,pdf}
Tables: output/tables/17_coverage_unique_summary.{csv,tex}
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

OUT_FIG = HERE / "output" / "figures"
OUT_FIG_STUDY = OUT_FIG / "17_coverage_unique"
OUT_TAB = HERE / "output" / "tables"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _pct_colors(percentages: list) -> dict:
    """Map each percentage to a viridis color (sequential: low%=light, high%=dark)."""
    cm = plt.colormaps.get_cmap("viridis")
    n = max(len(percentages) - 1, 1)
    return {p: cm(i / n) for i, p in enumerate(sorted(percentages))}


# ── Core: compute unique-coverage sawtooth per (config+seed) ─────────────────

def _unique_cycle_series(group_df: pd.DataFrame) -> list[dict]:
    """Return list of {round, coverage_pct} for one (config+seed) group.

    Server-side union of all clients per round → accumulate unique covered set.
    Resets when covered == n_packs.
    """
    n_packs = int(group_df["n_packs"].iloc[0])
    if n_packs <= 0:
        return []

    records = []
    covered = set()

    for rnd, rnd_df in group_df.sort_values("round").groupby("round", sort=False):
        # server union: all packs seen by any client this round
        server_union = set()
        for sel in rnd_df["selected"]:
            server_union.update(sel)

        covered.update(server_union)
        cov_pct = len(covered) / n_packs * 100.0
        records.append({"round": int(rnd), "coverage_pct": cov_pct})

        # reset when full coverage reached
        if len(covered) >= n_packs:
            covered = set()

    return records


def build_coverage_df(masks: pd.DataFrame) -> pd.DataFrame:
    """Compute unique-coverage series for all (config+seed) groups."""
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


def build_summary(cov_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-group summary statistics over all seeds."""
    if cov_df.empty:
        return pd.DataFrame()

    gcols_no_seed = [c for c in ["model", "dataset", "solution", "niid", "percentage"]
                     if c in cov_df.columns]
    gcols = [c for c in GROUP if c in cov_df.columns]

    rows = []
    for key, grp in cov_df.groupby(gcols, dropna=False):
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        n_packs = grp.get("n_packs", pd.Series(dtype=float))
        # detect cycle completions: coverage drops (reset) → count resets
        series = grp.sort_values("round")["coverage_pct"].values
        resets = int(np.sum(np.diff(series) < -50))  # large drop = reset
        n_cycles = resets + (1 if series[-1] > 0 else 0)
        ever_full = bool(resets > 0 or series[-1] >= 99.999)
        rows.append({
            **{k: v for k, v in meta.items() if k in gcols_no_seed},
            "n_cycles": n_cycles,
            "ever_full": ever_full,
        })

    # merge with mask-level info (n_packs, packs_sent median)
    from_masks = cov_df.groupby(gcols, dropna=False).apply(
        lambda g: pd.Series({
            "rounds_per_cycle_mean": len(g) / max(
                int(np.sum(np.diff(g.sort_values("round")["coverage_pct"].values) < -50)) + 1, 1),
        })
    ).reset_index()

    summary = pd.DataFrame(rows)
    summary = summary.merge(from_masks, on=gcols, how="left")
    return summary.sort_values(["solution", "model", "niid", "percentage"], na_position="last")


# ── Plot: one figure per (model, technique) ──────────────────────────────────

def _avg_over_seeds(cov_df: pd.DataFrame) -> pd.DataFrame:
    """Average coverage_pct over seeds at each (model,dataset,solution,niid,percentage,round)."""
    gcols = [c for c in ["model", "dataset", "solution", "niid", "percentage", "round"]
             if c in cov_df.columns]
    return (
        cov_df.groupby(gcols, dropna=False)["coverage_pct"]
        .mean()
        .reset_index()
    )


def plot_model_technique(cov_avg: pd.DataFrame, model: str, technique: str,
                         niids: list, percentages: list,
                         pct_color_map: dict) -> plt.Figure:
    """One figure for a single (model, technique) pair.

    Grid: rows = sparsification percentage, cols = niid.
    Each subplot has one coverage curve (unique per % × niid combination).
    """
    sub = cov_avg.copy()
    if "model" in sub.columns:
        sub = sub[sub["model"] == model]
    if "solution" in sub.columns:
        sub = sub[sub["solution"] == technique]

    sorted_pcts = sorted(percentages)
    n_rows = max(len(sorted_pcts), 1)
    n_cols = max(len(niids), 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.5 * n_rows),
                             squeeze=False, sharex=True)
    fig.suptitle(f"Unique-pack coverage per round — {model} / {technique}",
                 fontsize=13)

    for r, pct in enumerate(sorted_pcts):
        color = pct_color_map.get(pct)
        for c, niid in enumerate(niids):
            ax = axes[r][c]
            s = sub.copy()
            if "percentage" in s.columns:
                s = s[s["percentage"] == pct]
            if niid is not None and "niid" in s.columns:
                s = s[s["niid"] == niid]
            s = s.sort_values("round")

            if not s.empty:
                ax.plot(s["round"], s["coverage_pct"],
                        linewidth=1.8, color=color)

            ax.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.4)
            ax.set_ylim(0, 110)
            ax.set_title(f"{pct:.0%}  |  niid={niid}", fontsize=9)
            ax.grid(True, alpha=0.3)

            # y-label only on left column
            if c == 0:
                ax.set_ylabel("Coverage (%)", fontsize=8)
            # x-label only on bottom row
            if r == n_rows - 1:
                ax.set_xlabel("Round", fontsize=8)

    fig.tight_layout()
    return fig


# ── Tables ────────────────────────────────────────────────────────────────────

def _build_table(masks: pd.DataFrame, cov_df: pd.DataFrame) -> pd.DataFrame:
    """Merge mask stats with cycle stats for the summary table."""
    gcols = [c for c in GROUP if c in masks.columns]
    gcols_no_seed = [c for c in ["model", "dataset", "solution", "niid", "percentage"]
                     if c in masks.columns]

    # aggregate mask info per (config, no seed)
    mask_agg = (
        masks.groupby(gcols, dropna=False)
        .agg(n_packs=("n_packs", "mean"), packs_per_round=("packs_sent", "mean"))
        .reset_index()
        .groupby(gcols_no_seed, dropna=False)
        .agg(n_packs=("n_packs", "mean"), packs_per_round=("packs_per_round", "mean"))
        .reset_index()
    )

    # cycle stats per seed, then average
    cycle_rows = []
    for key, grp in cov_df.groupby(gcols, dropna=False):
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        series = grp.sort_values("round")["coverage_pct"].values
        resets = int(np.sum(np.diff(series) < -50))
        n_cycles = resets + (1 if len(series) > 0 and series[-1] > 0 else 0)
        ever_full = resets > 0 or (len(series) > 0 and series[-1] >= 99.999)
        rpc = len(series) / max(n_cycles, 1) if ever_full else float("nan")
        cycle_rows.append({
            **{k: v for k, v in meta.items() if k != "seed"},
            "seed": meta.get("seed"),
            "n_cycles": n_cycles,
            "ever_full": ever_full,
            "rounds_per_cycle": rpc,
        })

    cycle_df = pd.DataFrame(cycle_rows)
    agg_cols = {
        "n_cycles": "mean",
        "ever_full": "max",
        "rounds_per_cycle": "mean",
    }
    cycle_agg = (
        cycle_df.groupby(gcols_no_seed, dropna=False)
        .agg(**{k: (k, v) for k, v in agg_cols.items()})
        .reset_index()
    )

    table = mask_agg.merge(cycle_agg, on=gcols_no_seed, how="outer")
    return table.sort_values(["solution", "model", "niid", "percentage"], na_position="last")


def save_csv(table: pd.DataFrame):
    cols = [c for c in ["model", "dataset", "solution", "niid", "percentage",
                         "n_packs", "packs_per_round", "rounds_per_cycle",
                         "n_cycles", "ever_full"] if c in table.columns]
    out = table[cols].copy()
    for c in out.select_dtypes(include=float).columns:
        out[c] = out[c].round(3)
    path = OUT_TAB / "17_coverage_unique_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(table: pd.DataFrame):
    display_cols = {
        "model": "Model", "solution": "Solution", "niid": "NIID",
        "percentage": "Pct.", "n_packs": "Total packs",
        "packs_per_round": "Packs/rnd", "rounds_per_cycle": "Rounds/cycle",
        "n_cycles": "Cycles", "ever_full": "Full?",
    }
    cols = [c for c in display_cols if c in table.columns]

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        if col == "ever_full":
            return "yes" if val else "no"
        if isinstance(val, float):
            return f"{val:.1f}" if col in ("rounds_per_cycle", "packs_per_round") else f"{val:.2f}"
        return str(val)

    header = " & ".join(display_cols.get(c, c) for c in cols) + r" \\"
    ncols = len(cols)
    lines = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Observed unique-pack coverage summary (Study 17)}",
        r"\label{tab:coverage_unique} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_sol = None
    for _, row in table.iterrows():
        sol = row.get("solution", "")
        if prev_sol is not None and sol != prev_sol:
            lines.append(r"\midrule")
        prev_sol = sol
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "17_coverage_unique_summary.tex"
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

    # drop plaintext (no packing) and rows with no selected packs info
    if masks.empty:
        print("No mask data found.")
        return
    masks = masks[masks["percentage"].notna()].copy()
    if masks.empty:
        print("No packing-technique mask data found (all were plaintext).")
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

    print("Computing unique-coverage series...")
    cov_df = build_coverage_df(masks)
    if cov_df.empty:
        print("No coverage data computed.")
        return

    cov_avg = _avg_over_seeds(cov_df)
    pct_color_map = _pct_colors(percentages)

    n_saved = 0
    for model in models:
        for technique in solutions:
            sub = cov_avg.copy()
            if "model" in sub.columns:
                sub = sub[sub["model"] == model]
            if "solution" in sub.columns:
                sub = sub[sub["solution"] == technique]
            if sub.empty:
                continue
            fig = plot_model_technique(cov_avg, model, technique, niids,
                                       percentages, pct_color_map)
            fname = f"{model}_{technique}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
            print(f"Saved 17_coverage_unique/{fname}.png/pdf")
            plt.close(fig)
            n_saved += 1
    print(f"Total figures: {n_saved}")

    print()
    table = _build_table(masks, cov_df)
    save_csv(table)
    save_latex(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 17: observed unique-pack coverage per round")
    parser.add_argument("--logs-dir", default=None,
                        help="Path to logs/sparsification_percentage directory")
    args = parser.parse_args()
    main(args.logs_dir)
