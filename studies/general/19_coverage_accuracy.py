"""
19 — Unique-pack coverage vs accuracy per round (dual Y axes).

Same layout as study 17 (one figure per model×technique, grid rows=percentage × cols=niid,
x=round), but each subplot has TWO Y axes:

  Y left  (blue)  = unique-pack coverage % — sawtooth from study 17
                    (server-side union per round, resets when 100% is reached)
  Y right (red)   = model accuracy % per round (server_eval.csv → mean_acc × 100,
                    averaged across seeds)

This reveals how accuracy evolves as the server progressively covers the model, and
highlights whether accuracy gains correlate with coverage cycles (useful for understanding
the impact of sparse packing on learning convergence).

Source data: mask.csv (via load_data.load_masks) + server_eval.csv (via load_data.load_all).
Only for the unique-coverage metric (study 17). See 18_coverage_volume.py for volume metric.

Output: output/figures/19_coverage_accuracy/{model}_{technique}.{png,pdf}
Tables: output/tables/19_coverage_accuracy_summary.{csv,tex}
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

OUT_FIG = HERE / "output" / "figures"
OUT_FIG_STUDY = OUT_FIG / "19_coverage_accuracy"
OUT_TAB = HERE / "output" / "tables"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]

COV_COLOR = "tab:blue"
ACC_COLOR = "tab:red"


def _pct_colors(percentages: list) -> dict:
    """Map each percentage to a viridis color (sequential: low%=light, high%=dark)."""
    cm = plt.colormaps.get_cmap("viridis")
    n = max(len(percentages) - 1, 1)
    return {p: cm(i / n) for i, p in enumerate(sorted(percentages))}


# ── Core: unique-coverage (identical to study 17) ────────────────────────────

def _unique_cycle_series(group_df: pd.DataFrame) -> list[dict]:
    """Server-side union per round → accumulate unique covered set; reset at 100%."""
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


def _avg_over_seeds(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Average value_col over seeds at each (model,dataset,solution,niid,percentage,round)."""
    gcols = [c for c in ["model", "dataset", "solution", "niid", "percentage", "round"]
             if c in df.columns]
    return (
        df.groupby(gcols, dropna=False)[value_col]
        .mean()
        .reset_index()
    )


# ── Plot: one figure per (model, technique) ──────────────────────────────────

def plot_model_technique(cov_avg: pd.DataFrame, acc_avg: pd.DataFrame,
                         model: str, technique: str,
                         niids: list, percentages: list,
                         pct_color_map: dict) -> plt.Figure:
    """One figure for a single (model, technique) pair.

    Grid: rows = sparsification percentage, cols = niid.
    Each subplot: dual Y axes — coverage (left, blue) and accuracy (right, red).
    """
    cov_sub = cov_avg.copy()
    if "model" in cov_sub.columns:
        cov_sub = cov_sub[cov_sub["model"] == model]
    if "solution" in cov_sub.columns:
        cov_sub = cov_sub[cov_sub["solution"] == technique]

    acc_sub = acc_avg.copy()
    if "model" in acc_sub.columns:
        acc_sub = acc_sub[acc_sub["model"] == model]
    if "solution" in acc_sub.columns:
        acc_sub = acc_sub[acc_sub["solution"] == technique]

    sorted_pcts = sorted(percentages)
    n_rows = max(len(sorted_pcts), 1)
    n_cols = max(len(niids), 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.5 * n_rows),
                             squeeze=False, sharex=True)
    fig.suptitle(f"Coverage vs accuracy per round — {model} / {technique}",
                 fontsize=13)

    for r, pct in enumerate(sorted_pcts):
        color = pct_color_map.get(pct)
        for c, niid in enumerate(niids):
            ax = axes[r][c]

            # ── filter coverage ──────────────────────────────────────────
            cs = cov_sub.copy()
            if "percentage" in cs.columns:
                cs = cs[cs["percentage"] == pct]
            if niid is not None and "niid" in cs.columns:
                cs = cs[cs["niid"] == niid]
            cs = cs.sort_values("round")

            # ── filter accuracy ──────────────────────────────────────────
            ac = acc_sub.copy()
            if "percentage" in ac.columns:
                ac = ac[ac["percentage"] == pct]
            if niid is not None and "niid" in ac.columns:
                ac = ac[ac["niid"] == niid]
            ac = ac.sort_values("round")

            # ── left Y: coverage ─────────────────────────────────────────
            ax.set_ylim(0, 110)
            if not cs.empty:
                ax.plot(cs["round"], cs["coverage_pct"],
                        linewidth=1.8, color=COV_COLOR)
            ax.axhline(100, color=COV_COLOR, linestyle="--", linewidth=0.8, alpha=0.35)
            ax.tick_params(axis="y", labelcolor=COV_COLOR, labelsize=7)
            if c == 0:
                ax.set_ylabel("Coverage (%)", color=COV_COLOR, fontsize=8)

            # ── right Y: accuracy ─────────────────────────────────────────
            ax2 = ax.twinx()
            ax2.set_ylim(0, 100)
            if not ac.empty:
                ax2.plot(ac["round"], ac["mean_acc"] * 100,
                         linewidth=1.8, color=ACC_COLOR)
            ax2.tick_params(axis="y", labelcolor=ACC_COLOR, labelsize=7)
            if c == n_cols - 1:
                ax2.set_ylabel("Accuracy (%)", color=ACC_COLOR, fontsize=8)
            else:
                ax2.set_yticklabels([])

            ax.set_title(f"{pct:.0%}  |  niid={niid}", fontsize=9)
            ax.grid(True, alpha=0.25)

            if r == n_rows - 1:
                ax.set_xlabel("Round", fontsize=8)

    # shared legend at figure level
    h_cov = mlines.Line2D([], [], color=COV_COLOR, linewidth=1.8, label="Coverage (%)")
    h_acc = mlines.Line2D([], [], color=ACC_COLOR, linewidth=1.8, label="Accuracy (%)")
    fig.legend(handles=[h_cov, h_acc], loc="upper right", fontsize=8,
               framealpha=0.85, ncol=2)

    fig.tight_layout()
    return fig


# ── Tables ────────────────────────────────────────────────────────────────────

def _build_table(masks: pd.DataFrame, cov_df: pd.DataFrame,
                 acc_avg: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in GROUP if c in masks.columns]
    gcols_no_seed = [c for c in ["model", "dataset", "solution", "niid", "percentage"]
                     if c in masks.columns]

    # coverage cycle stats
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
            "rounds_per_cycle": rpc,
        })

    cycle_df = pd.DataFrame(cycle_rows)
    cycle_agg = (
        cycle_df.groupby(gcols_no_seed, dropna=False)
        .agg(rounds_per_cycle=("rounds_per_cycle", "mean"))
        .reset_index()
    )

    # accuracy stats from acc_avg (already averaged over seeds)
    gcols_acc = [c for c in ["model", "dataset", "solution", "niid", "percentage", "round"]
                 if c in acc_avg.columns]
    acc_no_seed_cols = [c for c in gcols_no_seed if c in acc_avg.columns]
    acc_stats = (
        acc_avg[acc_avg["percentage"].notna()]
        .groupby(acc_no_seed_cols, dropna=False)
        .agg(
            final_acc=("mean_acc", "last"),
            max_acc=("mean_acc", "max"),
        )
        .reset_index()
    )
    for col in ("final_acc", "max_acc"):
        if col in acc_stats.columns:
            acc_stats[col] = acc_stats[col] * 100

    table = cycle_agg.merge(acc_stats, on=acc_no_seed_cols, how="outer")
    return table.sort_values(["solution", "model", "niid", "percentage"], na_position="last")


def save_csv(table: pd.DataFrame):
    cols = [c for c in ["model", "dataset", "solution", "niid", "percentage",
                         "rounds_per_cycle", "final_acc", "max_acc"] if c in table.columns]
    out = table[cols].copy()
    for c in out.select_dtypes(include=float).columns:
        out[c] = out[c].round(3)
    path = OUT_TAB / "19_coverage_accuracy_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(table: pd.DataFrame):
    display_cols = {
        "model": "Model", "solution": "Solution", "niid": "NIID",
        "percentage": "Pct.", "rounds_per_cycle": "Rounds/cycle",
        "final_acc": "Final acc (%)", "max_acc": "Max acc (%)",
    }
    cols = [c for c in display_cols if c in table.columns]

    best_acc = table["max_acc"].dropna().max() if "max_acc" in table.columns else None

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        if isinstance(val, float):
            s = f"{val:.1f}"
        else:
            s = str(val)
        if col == "max_acc" and best_acc is not None and not pd.isna(val) and val == best_acc:
            s = r"\textbf{" + s + "}"
        return s

    header = " & ".join(display_cols.get(c, c) for c in cols) + r" \\"
    ncols = len(cols)
    lines = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Coverage vs accuracy summary (Study 19)}",
        r"\label{tab:coverage_accuracy} \\",
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

    path = OUT_TAB / "19_coverage_accuracy_summary.tex"
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
        print("No packing-technique mask data found (all were plaintext).")
        return

    print("Loading accuracy data...")
    try:
        dfs = load_data.load_all(logs_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    server_eval = dfs["server_eval"]
    server_eval = server_eval[server_eval["percentage"].notna()].copy()

    dims = load_data.discover_dims(masks)
    models = dims.get("model", [])
    solutions = dims.get("solution", [])
    niids = dims.get("niid", [])
    percentages = sorted(masks["percentage"].dropna().unique())
    print(f"Models     : {models}")
    print(f"Solutions  : {solutions}")
    print(f"Percentages: {percentages}")
    print(f"Mask rows  : {len(masks):,}")
    print(f"Eval rows  : {len(server_eval):,}")
    print()

    print("Computing unique-coverage series...")
    cov_df = build_coverage_df(masks)
    if cov_df.empty:
        print("No coverage data computed.")
        return

    cov_avg = _avg_over_seeds(cov_df, "coverage_pct")
    acc_avg = _avg_over_seeds(server_eval, "mean_acc")
    pct_color_map = _pct_colors(percentages)

    n_saved = 0
    for model in models:
        for technique in solutions:
            # skip if no coverage data for this combination
            cs = cov_avg.copy()
            if "model" in cs.columns:
                cs = cs[cs["model"] == model]
            if "solution" in cs.columns:
                cs = cs[cs["solution"] == technique]
            if cs.empty:
                continue

            fig = plot_model_technique(cov_avg, acc_avg, model, technique,
                                       niids, percentages, pct_color_map)
            fname = f"{model}_{technique}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
            print(f"Saved 19_coverage_accuracy/{fname}.png/pdf")
            plt.close(fig)
            n_saved += 1
    print(f"Total figures: {n_saved}")
    print()

    table = _build_table(masks, cov_df, acc_avg)
    save_csv(table)
    save_latex(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 19: unique-pack coverage and accuracy per round (dual Y axes)")
    parser.add_argument("--logs-dir", default=None,
                        help="Path to logs/sparsification_percentage directory")
    args = parser.parse_args()
    main(args.logs_dir)
