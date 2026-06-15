"""
10 — Rounds for full model coverage by packing strategy.

Answers: how many rounds must pass before the server has received every pack
(i.e., every model parameter) at least once?

Coverage formulas (derived from client/literature/):
  nvnl  : stride = window (no overlap)  → rounds = ceil(n_packs / packs_sent)
  sw    : stride = ceil(window × 0.5)   → rounds = ceil(n_packs / stride_sw)
  rw    : probabilistic adaptive         → rounds ≈ (n_packs/packs_sent) × ln(n_packs)
  fedphe: top-k fixed → same packs every round → full coverage never guaranteed (NaN)
  plaintext: 100% per round             → 1 round (reference)

NOTE on payload_bytes: in packing solutions the logged `payload_bytes` captures only
pack-index metadata (a few hundred bytes), NOT the actual CKKS ciphertext bytes.
Therefore byte-based cost is reported but labelled accordingly.  The rounds metric
is the fully reliable primary output.

Plots:
  A) Bar : rounds for full coverage, x=percentage, grouped by solution
  B) Line: rounds vs percentage with theoretical 1/p and 2/p curves
  C) Line: NVNL vs SW focused comparison (main story)
  D) Bar : total MB as logged (payload_bytes × rounds) — for reference

Output: output/figures/10_*.{png,pdf}
Tables: output/tables/10_coverage_summary.{csv,tex}
"""

import argparse
import importlib.util
import math
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
OUT_TAB = HERE / "output" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage"]

PACKING_SOLUTIONS = {"nvnl", "sw", "rw"}   # solutions with meaningful coverage formula
COVERAGE_NA_NOTE = "fedphe: top-k fixed — same packs every round, full coverage not guaranteed"


def _group(df):
    return [c for c in GROUP if c in df.columns]


def _cmap(solutions):
    cm = plt.colormaps.get_cmap("tab10")
    sorted_sols = sorted(solutions)
    return {s: cm(i / max(len(sorted_sols) - 1, 1)) for i, s in enumerate(sorted_sols)}


# ── Coverage formula ──────────────────────────────────────────────────────────

def _rounds_for_coverage(solution: str, n_packs: float, packs_sent: float) -> float:
    if packs_sent <= 0:
        return 1.0 if solution == "plaintext" else float("nan")
    if solution == "plaintext":
        return 1.0
    if solution == "nvnl":
        # stride = packs_sent (non-overlapping round-robin)
        return math.ceil(n_packs / packs_sent)
    if solution == "sw":
        # stride = ceil(packs_sent × 0.5)  — 50% overlap
        stride = max(math.ceil(packs_sent * 0.5), 1)
        return math.ceil(n_packs / stride)
    if solution == "rw":
        # coupon-collector approximation (adaptive weights accelerate but ln(n) is baseline)
        return (n_packs / packs_sent) * math.log(max(n_packs, 2))
    # fedphe: top-k is identical every round → NaN
    return float("nan")


# ── Core computation ──────────────────────────────────────────────────────────

def build_coverage_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = _group(train)
    required = {"n_packs", "packs_sent"}
    missing = required - set(train.columns)
    if missing:
        print(f"[warn] train.csv missing columns: {missing}")
        return pd.DataFrame()

    agg_dict = {
        "n_packs": ("n_packs", "mean"),
        "packs_sent": ("packs_sent", "mean"),
    }
    if "payload_bytes" in train.columns:
        agg_dict["mean_payload_mb"] = ("payload_bytes", lambda x: x.mean() / 1e6)

    agg = train.groupby(g, dropna=False).agg(**agg_dict).reset_index()

    agg["rounds_coverage"] = agg.apply(
        lambda r: _rounds_for_coverage(r["solution"], r["n_packs"], r["packs_sent"]),
        axis=1,
    )
    if "mean_payload_mb" in agg.columns:
        agg["total_mb_logged"] = agg["rounds_coverage"] * agg["mean_payload_mb"]

    return agg.sort_values(["solution", "model", "niid", "percentage"], na_position="last")


# ── Plot A: bar — rounds for coverage by solution ─────────────────────────────

def plot_rounds_bar(summary: pd.DataFrame, models: list) -> plt.Figure | None:
    plotable = summary[summary["rounds_coverage"].notna() &
                       (summary["solution"] != "plaintext")]
    if plotable.empty:
        return None

    all_solutions = sorted(plotable["solution"].unique())
    percentages = sorted(plotable["percentage"].dropna().unique())
    x = np.arange(len(percentages))
    width = 0.8 / max(len(all_solutions), 1)
    sol_colors = _cmap(all_solutions)

    n_rows = len(models)
    niids = sorted(plotable["niid"].dropna().unique()) if "niid" in plotable.columns else [None]
    n_cols = len(niids)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(7, len(percentages) * 1.4) * n_cols, 5 * n_rows),
                             squeeze=False)

    for row, model in enumerate(models):
        for col, niid in enumerate(niids):
            ax = axes[row][col]
            sub = plotable.copy()
            if "model" in sub.columns:
                sub = sub[sub["model"] == model]
            if niid is not None and "niid" in sub.columns:
                sub = sub[sub["niid"] == niid]

            for i, sol in enumerate(all_solutions):
                vals = [
                    sub[(sub["solution"] == sol) & (sub["percentage"] == p)]["rounds_coverage"].mean()
                    for p in percentages
                ]
                bar_vals = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in vals]
                bars = ax.bar(x + i * width, bar_vals, width,
                              label=sol, color=sol_colors[sol], alpha=0.85)
                for bar, orig in zip(bars, vals):
                    if isinstance(orig, float) and np.isnan(orig):
                        bar.set_visible(False)
                    else:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.1,
                                f"{orig:.0f}", ha="center", va="bottom", fontsize=7)

            # plaintext reference = 1 round
            ax.axhline(1, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.7, label="plaintext (1 round)")

            ax.set_xticks(x + width * (len(all_solutions) - 1) / 2)
            ax.set_xticklabels([str(p) for p in percentages])
            ax.set_xlabel("Sparsification percentage")
            ax.set_ylabel("Rounds for full model coverage")
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"Full coverage rounds — {model}{niid_lbl}")
            ax.legend(fontsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# ── Plot B: line — rounds vs percentage with theoretical curves ───────────────

def plot_rounds_line(summary: pd.DataFrame, models: list) -> plt.Figure | None:
    plotable = summary[summary["rounds_coverage"].notna() &
                       (summary["solution"] != "plaintext")]
    if plotable.empty:
        return None

    all_solutions = sorted(plotable["solution"].unique())
    sol_colors = _cmap(all_solutions)
    niids = sorted(plotable["niid"].dropna().unique()) if "niid" in plotable.columns else [None]

    n_rows = len(models)
    n_cols = len(niids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for row, model in enumerate(models):
        for col, niid in enumerate(niids):
            ax = axes[row][col]
            sub = plotable.copy()
            if "model" in sub.columns:
                sub = sub[sub["model"] == model]
            if niid is not None and "niid" in sub.columns:
                sub = sub[sub["niid"] == niid]

            for sol in all_solutions:
                s = sub[sub["solution"] == sol].dropna(subset=["rounds_coverage"])
                if s.empty:
                    continue
                s = s.sort_values("percentage")
                ax.plot(s["percentage"], s["rounds_coverage"],
                        marker="o", markersize=7, linewidth=2,
                        label=sol, color=sol_colors[sol])

            # Theoretical reference curves
            pcts = np.linspace(0.05, 1.0, 300)
            ax.plot(pcts, 1.0 / pcts, "k:", linewidth=1, alpha=0.6, label="1/p  (NVNL theory)")
            ax.plot(pcts, np.maximum(1.0, 2.0 / pcts - 1),
                    "k--", linewidth=1, alpha=0.6, label="2/p−1  (SW theory)")

            ax.axhline(1, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Sparsification percentage")
            ax.set_ylabel("Rounds for full model coverage")
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"Coverage rounds vs percentage — {model}{niid_lbl}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# ── Plot C: NVNL vs SW focused comparison ────────────────────────────────────

def plot_nvnl_vs_sw(summary: pd.DataFrame, models: list) -> plt.Figure | None:
    focus = summary[summary["solution"].isin({"nvnl", "sw"}) &
                    summary["rounds_coverage"].notna()].copy()
    if focus.empty:
        return None

    percentages = sorted(focus["percentage"].dropna().unique())
    niids = sorted(focus["niid"].dropna().unique()) if "niid" in focus.columns else [None]
    x = np.arange(len(percentages))
    width = 0.35
    colors = {"nvnl": "#2196F3", "sw": "#FF5722"}

    n_rows = len(models)
    n_cols = len(niids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(7, len(percentages) * 1.2) * n_cols, 5 * n_rows),
                             squeeze=False)

    for row, model in enumerate(models):
        for col, niid in enumerate(niids):
            ax = axes[row][col]
            sub = focus.copy()
            if "model" in sub.columns:
                sub = sub[sub["model"] == model]
            if niid is not None and "niid" in sub.columns:
                sub = sub[sub["niid"] == niid]

            for offset, sol in zip([-width / 2, width / 2], ["nvnl", "sw"]):
                vals = [
                    sub[(sub["solution"] == sol) & (sub["percentage"] == p)]["rounds_coverage"].mean()
                    for p in percentages
                ]
                bar_vals = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in vals]
                bars = ax.bar(x + offset, bar_vals, width,
                              label=sol.upper(), color=colors[sol], alpha=0.85)
                for bar, orig in zip(bars, vals):
                    if isinstance(orig, float) and np.isnan(orig):
                        bar.set_visible(False)
                    else:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.15,
                                f"{int(orig)}", ha="center", va="bottom",
                                fontsize=8, fontweight="bold")

            ax.axhline(1, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.7, label="plaintext (1 round)")
            ax.set_xticks(x)
            ax.set_xticklabels([str(p) for p in percentages])
            ax.set_xlabel("Sparsification percentage")
            ax.set_ylabel("Rounds for full model coverage")
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"NVNL vs SW coverage rounds — {model}{niid_lbl}")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# ── Plot D: total MB as logged (secondary, for reference) ─────────────────────

def plot_logged_mb(summary: pd.DataFrame, models: list) -> plt.Figure | None:
    if "total_mb_logged" not in summary.columns:
        return None
    plotable = summary[summary["total_mb_logged"].notna()].copy()
    if plotable.empty:
        return None

    all_solutions = sorted(plotable["solution"].unique())
    percentages = sorted(plotable["percentage"].dropna().unique())
    sol_colors = _cmap(all_solutions)
    niids = sorted(plotable["niid"].dropna().unique()) if "niid" in plotable.columns else [None]

    n_rows = len(models)
    n_cols = len(niids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for row, model in enumerate(models):
        for col, niid in enumerate(niids):
            ax = axes[row][col]
            sub = plotable.copy()
            if "model" in sub.columns:
                sub = sub[sub["model"] == model]
            if niid is not None and "niid" in sub.columns:
                sub = sub[sub["niid"] == niid]

            for sol in all_solutions:
                s = sub[sub["solution"] == sol].dropna(subset=["total_mb_logged"])
                if s.empty:
                    continue
                if sol == "plaintext":
                    s_pt = s[s["percentage"].isna()]
                    if not s_pt.empty:
                        ax.axhline(s_pt["total_mb_logged"].mean(),
                                   color=sol_colors[sol], linestyle="--",
                                   linewidth=1.5, label="plaintext", alpha=0.8)
                else:
                    s = s.sort_values("percentage")
                    ax.plot(s["percentage"], s["total_mb_logged"],
                            marker="o", label=sol, color=sol_colors[sol], linewidth=1.8)

            ax.set_xlabel("Sparsification percentage")
            ax.set_ylabel("Total logged MB (rounds × payload_bytes/round)\n"
                          "[NOTE: packing payload_bytes = mask metadata only]")
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"Logged data per full cycle — {model}{niid_lbl}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── Tables ────────────────────────────────────────────────────────────────────

def save_csv(summary: pd.DataFrame):
    cols = [c for c in [
        "model", "dataset", "solution", "niid", "percentage",
        "n_packs", "packs_sent",
        "rounds_coverage",
        "mean_payload_mb", "total_mb_logged",
    ] if c in summary.columns]
    out = summary[cols].copy()
    for c in out.select_dtypes(include=float).columns:
        out[c] = out[c].round(4)
    path = OUT_TAB / "10_coverage_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(summary: pd.DataFrame):
    display_cols = {
        "solution": "Solution",
        "percentage": "Pct.",
        "niid": "NIID",
        "n_packs": "Total packs",
        "packs_sent": "Sent/rnd",
        "rounds_coverage": "Rounds (cycle)",
    }
    cols = [c for c in display_cols if c in summary.columns]
    if "model" in summary.columns:
        cols = ["model"] + cols
        display_cols["model"] = "Model"

    lower_better = {"rounds_coverage"}
    best = {}
    for col in cols:
        if col in lower_better:
            vals = summary[col].dropna()
            if not vals.empty:
                best[col] = vals.min()

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        if isinstance(val, float):
            s = f"{val:.0f}" if val == int(val) else f"{val:.2f}"
        elif isinstance(val, (int, np.integer)):
            s = str(val)
        else:
            return str(val)
        if col in best and not pd.isna(best.get(col)) and val == best[col]:
            s = r"\textbf{" + s + "}"
        return s

    header = " & ".join(display_cols.get(c, c) for c in cols) + r" \\"
    ncols = len(cols)
    lines = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Rounds for full model coverage per packing strategy}",
        r"\label{tab:coverage_rounds} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_sol = None
    for _, row in summary[summary["solution"] != "plaintext"].iterrows():
        sol = row.get("solution", "")
        if prev_sol is not None and sol != prev_sol:
            lines.append(r"\midrule")
        prev_sol = sol
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "10_coverage_summary.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train = dfs["train"]
    if train.empty:
        print("No train data found.")
        return

    dims = load_data.discover_dims(train)
    models = dims.get("model", [])
    print(f"Models     : {models}")
    print(f"Solutions  : {dims.get('solution', [])}")
    print(f"Percentages: {dims.get('percentage', [])}")
    print()
    print(f"NOTE: {COVERAGE_NA_NOTE}")
    print()

    summary = build_coverage_summary(train)
    if summary.empty:
        print("Could not build coverage summary (missing n_packs/packs_sent).")
        return

    # Print summary table — packing solutions only
    pk = summary[summary["solution"] != "plaintext"].dropna(subset=["rounds_coverage"])
    print(pk[["solution", "percentage", "n_packs", "packs_sent", "rounds_coverage"]].to_string(index=False))

    fig_a = plot_rounds_bar(summary, models)
    if fig_a:
        for ext in ("png", "pdf"):
            fig_a.savefig(OUT_FIG / f"10_coverage_rounds_bar.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 10_coverage_rounds_bar.png/pdf")
        plt.close(fig_a)

    fig_b = plot_rounds_line(summary, models)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT_FIG / f"10_coverage_rounds_line.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 10_coverage_rounds_line.png/pdf")
        plt.close(fig_b)

    fig_c = plot_nvnl_vs_sw(summary, models)
    if fig_c:
        for ext in ("png", "pdf"):
            fig_c.savefig(OUT_FIG / f"10_nvnl_vs_sw.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 10_nvnl_vs_sw.png/pdf")
        plt.close(fig_c)

    fig_d = plot_logged_mb(summary, models)
    if fig_d:
        for ext in ("png", "pdf"):
            fig_d.savefig(OUT_FIG / f"10_logged_mb.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 10_logged_mb.png/pdf")
        plt.close(fig_d)

    save_csv(summary)
    save_latex(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
