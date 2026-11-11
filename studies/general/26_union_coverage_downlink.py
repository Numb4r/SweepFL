"""
26 — Union-coverage per round & downlink inflation.

Motivation: comparing sparsification techniques only by upload bytes per client hides
a real cost difference. The server (server/server.py) aggregates the packs received
from all clients into a SINGLE blob (the union of pack indices selected that round) and
broadcasts that same blob to every client. So the true downlink size per round is driven
by the UNION of packs across clients, not by the per-client upload (~`percentage`).

Techniques differ sharply in how correlated their per-client masks are:
  sweepfl / sw   : mask depends only on `round` (same window for every client) → union ≈
                per-client upload (~percentage). Cheap downlink, but the model updates
                only ~1/percentage rounds worth of packs per full cycle.
  fedphe      : top-k by magnitude → masks are highly correlated across clients (everyone
                agrees on the "important" packs) → union only slightly above upload.
  rw          : each client draws an INDEPENDENT weighted-random mask → the union across
                many clients quickly covers most of the model even though each client only
                uploads `percentage` of it. This is why `rw` can rival plaintext accuracy:
                it is a near-full-coverage broadcast disguised as a sparse upload — the
                cost shows up as downlink inflation, not as extra client-side compute.

Metrics computed per (config, seed, round):
  client_cov_pct = mean_i(|selected_i|) / n_packs * 100   (per-client upload, ≈ percentage)
  union_cov_pct  = |union_i(selected_i)| / n_packs * 100  (server broadcast size, downlink)
  inflation      = union_cov_pct / client_cov_pct         (how many times bigger the
                                                            downlink is vs. the upload)
  n_clients_round = number of clients that reported that round (captures CLIENT_FRACTION)

Aggregation: rounds are averaged after dropping the first 5% (warmup), then averaged
over seeds.

Faceting (per repo convention — never mix percentages/techniques improperly):
  - If the logs contain more than one CLIENT_FRACTION value (client_frac experiments),
    figures are split per (model, client_fraction) — comparing techniques at the same cf.
  - Otherwise (sparsification_percentage, cf is always 1.0), figures are split per model
    only, matching the convention of studies 17/18.
  In both cases, each figure has a grid of rows=percentage, cols=niid — the standard
  "same sparsification % across the row" comparison rule from CLAUDE.md — with one
  line/bar per technique.

Output:
  output/figures/26_union_coverage_curves/{model}[_cf{cf}].{png,pdf}
      Grid rows=percentage, cols=niid. y=union coverage %, x=round, one line per technique.
      Dashed reference line at the nominal `percentage` (= per-client upload level).
  output/figures/26_downlink_inflation/{model}[_cf{cf}].{png,pdf}
      Grouped bar chart: x=percentage, bars=technique, y=downlink inflation factor.
      Reference line at 1.0 (downlink == upload, i.e. no inflation).
  output/tables/26_union_coverage_downlink_summary.{csv,tex}
      One row per (model, [client_fraction], niid, percentage, solution).
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
OUT_FIG_CURVES = OUT_FIG / "26_union_coverage_curves"
OUT_FIG_BARS = OUT_FIG / "26_downlink_inflation"
OUT_TAB = HERE / "output" / "tables"
OUT_FIG_CURVES.mkdir(parents=True, exist_ok=True)
OUT_FIG_BARS.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "client_fraction", "seed"]
GROUP_NO_SEED = ["model", "dataset", "solution", "niid", "percentage", "client_fraction"]

WARMUP_FRAC = 0.05  # drop the first 5% of rounds before averaging (transient startup)


def _solution_colors(solutions: list) -> dict:
    cmap = plt.colormaps.get_cmap("tab10")
    return {s: cmap(i % 10) for i, s in enumerate(sorted(solutions))}


# ── Core: per-round union/client coverage series ──────────────────────────────

def _round_series(group_df: pd.DataFrame) -> list[dict]:
    """Return list of {round, union_cov_pct, client_cov_pct, n_clients_round} for one
    (config+seed) group."""
    n_packs = int(group_df["n_packs"].iloc[0])
    if n_packs <= 0:
        return []

    records = []
    for rnd, rnd_df in group_df.groupby("round", sort=False):
        sets = [set(s) for s in rnd_df["selected"]]
        if not sets:
            continue
        union = set().union(*sets)
        client_covs = [len(s) / n_packs for s in sets]
        records.append({
            "round": int(rnd),
            "union_cov_pct": len(union) / n_packs * 100.0,
            "client_cov_pct": float(np.mean(client_covs)) * 100.0,
            "n_clients_round": len(sets),
        })
    return records


def build_series_df(masks: pd.DataFrame) -> pd.DataFrame:
    """Compute per-round coverage series for all (config+seed) groups."""
    parts = []
    gcols = [c for c in GROUP if c in masks.columns]
    for key, grp in masks.groupby(gcols, dropna=False):
        series = _round_series(grp)
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


def _avg_over_seeds(series_df: pd.DataFrame) -> pd.DataFrame:
    """Average coverage metrics over seeds at each (config, round)."""
    gcols = [c for c in GROUP_NO_SEED + ["round"] if c in series_df.columns]
    return (
        series_df.groupby(gcols, dropna=False)[["union_cov_pct", "client_cov_pct", "n_clients_round"]]
        .mean()
        .reset_index()
    )


def build_summary(series_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over rounds (post-warmup) and seeds into one row per config."""
    if series_df.empty:
        return pd.DataFrame()

    rows = []
    gcols = [c for c in GROUP if c in series_df.columns]
    for key, grp in series_df.groupby(gcols, dropna=False):
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        max_round = grp["round"].max()
        start = max(int(max_round * WARMUP_FRAC), 1)
        warm = grp[grp["round"] >= start]
        if warm.empty:
            warm = grp
        rows.append({
            **meta,
            "client_cov_pct": warm["client_cov_pct"].mean(),
            "union_cov_pct": warm["union_cov_pct"].mean(),
            "clients_per_round": warm["n_clients_round"].mean(),
        })

    per_seed = pd.DataFrame(rows)
    gcols_no_seed = [c for c in GROUP_NO_SEED if c in per_seed.columns]
    summary = (
        per_seed.groupby(gcols_no_seed, dropna=False)[
            ["client_cov_pct", "union_cov_pct", "clients_per_round"]
        ]
        .mean()
        .reset_index()
    )
    summary["inflation"] = np.where(
        summary["client_cov_pct"] > 0,
        summary["union_cov_pct"] / summary["client_cov_pct"],
        np.nan,
    )
    return summary.sort_values(["model", "client_fraction", "niid", "percentage", "solution"])


# ── Plot 1: union-coverage curves (grid rows=percentage, cols=niid) ──────────

def plot_curves(cov_avg: pd.DataFrame, model: str, cf: float,
                 niids: list, percentages: list, sol_colors: dict) -> plt.Figure | None:
    sub = cov_avg[(cov_avg["model"] == model) & (cov_avg["client_fraction"] == cf)]
    if sub.empty:
        return None

    sorted_pcts = sorted(percentages)
    n_rows = max(len(sorted_pcts), 1)
    n_cols = max(len(niids), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.5 * n_rows),
                              squeeze=False, sharex=True)
    title = f"Union-coverage per round (downlink proxy) — {model}"
    if cf != 1.0:
        title += f" — client_fraction={cf}"
    fig.suptitle(title, fontsize=13)

    for r, pct in enumerate(sorted_pcts):
        for c, niid in enumerate(niids):
            ax = axes[r][c]
            s = sub[(sub["percentage"] == pct) & (sub["niid"] == niid)]
            for sol in sorted(s["solution"].unique()):
                ss = s[s["solution"] == sol].sort_values("round")
                if ss.empty:
                    continue
                ax.plot(ss["round"], ss["union_cov_pct"], linewidth=1.6,
                        color=sol_colors.get(sol), label=sol)

            ax.axhline(pct * 100, color="black", linestyle="--", linewidth=1, alpha=0.5,
                       label="per-client upload" if r == 0 and c == 0 else None)
            ax.set_ylim(0, 105)
            ax.set_title(f"{pct:.0%}  |  niid={niid}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel("Union coverage (%)", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("Round", fontsize=8)
            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    return fig


# ── Plot 2: downlink inflation bar chart ──────────────────────────────────────

def plot_inflation_bars(summary: pd.DataFrame, model: str, cf: float,
                         percentages: list, sol_colors: dict) -> plt.Figure | None:
    sub = summary[(summary["model"] == model) & (summary["client_fraction"] == cf)]
    if sub.empty:
        return None

    sorted_pcts = sorted(percentages)
    solutions = sorted(sub["solution"].unique())
    n = len(solutions)
    width = 0.8 / max(n, 1)
    x = np.arange(len(sorted_pcts))

    fig, ax = plt.subplots(figsize=(1.8 * len(sorted_pcts) + 2, 4))
    for i, sol in enumerate(solutions):
        vals = []
        for pct in sorted_pcts:
            row = sub[(sub["solution"] == sol) & (sub["percentage"] == pct)]
            vals.append(row["inflation"].iloc[0] if not row.empty else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=sol,
               color=sol_colors.get(sol))

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label="no inflation (downlink=upload)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0%}" for p in sorted_pcts])
    ax.set_xlabel("Sparsification percentage")
    ax.set_ylabel("Downlink inflation (union / per-client upload)")
    title = f"Downlink inflation — {model}"
    if cf != 1.0:
        title += f" — client_fraction={cf}"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Tables ────────────────────────────────────────────────────────────────────

def save_csv(summary: pd.DataFrame, multi_cf: bool):
    cols = ["model", "client_fraction", "niid", "percentage", "solution",
            "client_cov_pct", "union_cov_pct", "inflation", "clients_per_round"]
    if not multi_cf:
        cols = [c for c in cols if c != "client_fraction"]
    out = summary[[c for c in cols if c in summary.columns]].copy()
    for c in out.select_dtypes(include=float).columns:
        out[c] = out[c].round(3)
    path = OUT_TAB / "26_union_coverage_downlink_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(summary: pd.DataFrame, multi_cf: bool):
    display_cols = {
        "model": "Model", "client_fraction": "Client frac.", "niid": "NIID",
        "percentage": "Pct.", "solution": "Solution",
        "client_cov_pct": "Upload (%)", "union_cov_pct": "Downlink (%)",
        "inflation": "Inflation (x)", "clients_per_round": "Clients/rnd",
    }
    if not multi_cf:
        display_cols.pop("client_fraction")
    cols = [c for c in display_cols if c in summary.columns]

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        if isinstance(val, float):
            return f"{val:.1f}" if col == "clients_per_round" else f"{val:.2f}"
        return str(val)

    header = " & ".join(display_cols[c] for c in cols) + r" \\"
    ncols = len(cols)
    lines = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Union-coverage and downlink inflation summary (Study 26)}",
        r"\label{tab:union_coverage_downlink} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_model = None
    for _, row in summary.iterrows():
        model = row.get("model", "")
        if prev_model is not None and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "26_union_coverage_downlink_summary.tex"
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
    if "client_fraction" not in masks.columns:
        masks["client_fraction"] = 1.0

    # Some experiment sets contain duplicate re-runs of the exact same config+seed
    # (same model/solution/niid/percentage/client_fraction/seed, different timestamp
    # folder) — load_masks() concatenates both, which would double-count clients per
    # round (inflating union coverage and clients_per_round). Deduplicate defensively
    # on (config-key, round, cid): keep only the first occurrence.
    dedup_keys = [c for c in GROUP if c in masks.columns] + ["round", "cid"]
    before = len(masks)
    masks = masks.drop_duplicates(subset=dedup_keys, keep="first")
    dropped = before - len(masks)
    if dropped:
        print(f"Note: dropped {dropped:,} duplicate (config, round, cid) rows "
              f"from re-run folders sharing the same seed.")

    dims = load_data.discover_dims(masks)
    models = dims.get("model", [])
    solutions = dims.get("solution", [])
    niids = dims.get("niid", [])
    percentages = sorted(masks["percentage"].dropna().unique())
    cf_values = sorted(masks["client_fraction"].dropna().unique())
    multi_cf = len(cf_values) > 1
    print(f"Models          : {models}")
    print(f"Solutions       : {solutions}")
    print(f"Percentages     : {percentages}")
    print(f"NIIDs           : {niids}")
    print(f"Client fractions: {cf_values}  (faceting by cf: {multi_cf})")
    print(f"Mask rows       : {len(masks):,}")
    print()

    print("Computing per-round union/client coverage series...")
    series_df = build_series_df(masks)
    if series_df.empty:
        print("No coverage data computed.")
        return

    cov_avg = _avg_over_seeds(series_df)
    sol_colors = _solution_colors(solutions)

    n_curve_figs = 0
    n_bar_figs = 0
    summary = build_summary(series_df)

    for model in models:
        for cf in cf_values:
            suffix = f"_cf{cf}" if multi_cf else ""

            fig = plot_curves(cov_avg, model, cf, niids, percentages, sol_colors)
            if fig is not None:
                fname = f"{model}{suffix}"
                for ext in ("png", "pdf"):
                    fig.savefig(OUT_FIG_CURVES / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
                print(f"Saved 26_union_coverage_curves/{fname}.png/pdf")
                plt.close(fig)
                n_curve_figs += 1

            fig2 = plot_inflation_bars(summary, model, cf, percentages, sol_colors)
            if fig2 is not None:
                fname = f"{model}{suffix}"
                for ext in ("png", "pdf"):
                    fig2.savefig(OUT_FIG_BARS / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
                print(f"Saved 26_downlink_inflation/{fname}.png/pdf")
                plt.close(fig2)
                n_bar_figs += 1

    print(f"Total curve figures: {n_curve_figs}")
    print(f"Total bar figures  : {n_bar_figs}")

    print()
    save_csv(summary, multi_cf)
    save_latex(summary, multi_cf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 26: union-coverage per round & downlink inflation")
    parser.add_argument("--logs-dir", default=None,
                        help="Path to logs/<experiment_name> directory")
    args = parser.parse_args()
    main(args.logs_dir)
