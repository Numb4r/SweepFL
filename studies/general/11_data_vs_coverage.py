"""
11 — Data transferred vs model coverage: SweepFL vs SW.

Answers: given a fixed byte budget (or round budget), how much of the model
has each strategy exposed to the server?

Background:
  SweepFL sends `packs_sent` packs per round with NO overlap (stride = packs_sent).
  SW   sends `packs_sent` packs per round with 50% overlap (stride ≈ packs_sent/2).
  → Same bytes per round, but SweepFL covers 2× new territory per byte spent.

Coverage per round (theoretical):
  SweepFL: new_coverage_frac = packs_sent / n_packs   (= percentage, linear)
  SW  : new_coverage_frac = stride / n_packs        (≈ percentage / 2)

Bytes estimate (raw pre-encryption, float32):
  per_round_mb = packs_sent × pack_size × 4 / 1e6

Marginal cost (MB per 1% new coverage):
  SweepFL: n_packs × pack_size × 4 / (1e6 × 100)   ← constant, independent of %
  SW  : 2 × SweepFL marginal cost                   ← constant at 2×

Plots:
  A) Line: cumulative coverage % vs round — SweepFL vs SW per percentage
  B) Line: cumulative coverage % vs cumulative MB — the key efficiency plot
  C) Bar : coverage % at fixed round budgets (N=10,20,50) — SweepFL vs SW
  D) Bar : MB per 1% new coverage (marginal cost) — SweepFL vs SW

Output: output/figures/11_*.{png,pdf}
Tables: output/tables/11_data_coverage_summary.{csv,tex}
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

FOCUS_SOLUTIONS = {"sweepfl", "sw"}
COLORS = {"sweepfl": "#2196F3", "sw": "#FF5722"}
FIXED_BUDGETS = [5, 10, 20, 50]

# ── Coverage & bytes formulas ─────────────────────────────────────────────────

def _stride(solution: str, packs_sent: float) -> float:
    if solution == "sweepfl":
        return packs_sent
    if solution == "sw":
        return max(math.ceil(packs_sent * 0.5), 1)
    return packs_sent


def _coverage_at_round(solution: str, r: int, n_packs: float, packs_sent: float) -> float:
    if n_packs <= 0:
        return 1.0
    stride = _stride(solution, packs_sent)
    return min(r * stride / n_packs, 1.0)


def _mb_per_round(packs_sent: float, pack_size: float) -> float:
    return packs_sent * pack_size * 4 / 1e6  # float32 raw bytes


def _marginal_cost(solution: str, n_packs: float, packs_sent: float, pack_size: float) -> float:
    """MB per 1% new model coverage."""
    stride = _stride(solution, packs_sent)
    if stride <= 0:
        return float("nan")
    new_coverage_pct_per_round = stride / n_packs * 100
    mb_per_round = _mb_per_round(packs_sent, pack_size)
    return mb_per_round / new_coverage_pct_per_round


# ── Build summary from observed data ─────────────────────────────────────────

GROUP = ["model", "dataset", "solution", "niid", "percentage"]


def build_params(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed n_packs, packs_sent, pack_size per group."""
    required = {"n_packs", "packs_sent"}
    missing = required - set(train.columns)
    if missing:
        print(f"[warn] train.csv missing columns: {missing}")
        return pd.DataFrame()

    g = [c for c in GROUP if c in train.columns]
    agg_spec = {
        "n_packs": ("n_packs", "mean"),
        "packs_sent": ("packs_sent", "mean"),
    }
    if "pack_size" in train.columns:
        agg_spec["pack_size"] = ("pack_size", "mean")
    if "rounds" in train.columns:
        agg_spec["total_rounds"] = ("rounds", "median")

    params = train[train["solution"].isin(FOCUS_SOLUTIONS)].groupby(
        g, dropna=False
    ).agg(**agg_spec).reset_index()

    if "pack_size" not in params.columns:
        print("[warn] pack_size not found in train.csv — byte estimates unavailable")
        params["pack_size"] = float("nan")
    if "total_rounds" not in params.columns:
        params["total_rounds"] = 50

    params["stride"] = params.apply(
        lambda r: _stride(r["solution"], r["packs_sent"]), axis=1
    )
    params["new_coverage_pct_per_round"] = params["stride"] / params["n_packs"] * 100
    params["mb_per_round"] = params.apply(
        lambda r: _mb_per_round(r["packs_sent"], r["pack_size"]), axis=1
    )
    params["marginal_cost_mb_per_pct"] = params.apply(
        lambda r: _marginal_cost(r["solution"], r["n_packs"], r["packs_sent"], r["pack_size"]),
        axis=1,
    )

    return params


def build_trajectory(params: pd.DataFrame, max_rounds: int = 60) -> pd.DataFrame:
    """Expand params into per-round trajectory rows."""
    rows = []
    for _, p in params.iterrows():
        for r in range(1, max_rounds + 1):
            cov = _coverage_at_round(p["solution"], r, p["n_packs"], p["packs_sent"])
            cum_mb = r * p["mb_per_round"]
            rows.append({
                **{k: p[k] for k in GROUP if k in p.index},
                "round": r,
                "coverage_pct": cov * 100,
                "cumulative_mb": cum_mb,
                "n_packs": p["n_packs"],
                "packs_sent": p["packs_sent"],
                "pack_size": p["pack_size"],
            })
    return pd.DataFrame(rows)


# ── Plot A: coverage % vs round ───────────────────────────────────────────────

def plot_coverage_vs_round(traj: pd.DataFrame, models: list, niids: list) -> plt.Figure | None:
    focus = traj[traj["solution"].isin(FOCUS_SOLUTIONS)].copy()
    if focus.empty:
        return None
    percentages = sorted(focus["percentage"].dropna().unique())
    ls_map = {p: ls for p, ls in zip(percentages, ["-", "--", "-.", ":", (0, (3, 1, 1, 1))])}

    n_rows = len(models)
    n_cols = len(niids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    for row, model in enumerate(models):
        for col, niid in enumerate(niids):
            ax = axes[row][col]
            sub = focus.copy()
            if "model" in sub.columns:
                sub = sub[sub["model"] == model]
            if "niid" in sub.columns:
                sub = sub[sub["niid"] == niid]

            for sol in ["sweepfl", "sw"]:
                for pct in percentages:
                    s = sub[(sub["solution"] == sol) & (sub["percentage"] == pct)]
                    if s.empty:
                        continue
                    s = s.sort_values("round")
                    ax.plot(s["round"], s["coverage_pct"],
                            color=COLORS[sol], linestyle=ls_map.get(pct, "-"),
                            linewidth=1.8,
                            label=f"{sol.upper()} p={pct:.0%}")

            ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
                       label="100% coverage")
            ax.set_xlabel("Round")
            ax.set_ylabel("Cumulative coverage (%)")
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"Coverage trajectory — {model}{niid_lbl}")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)

    fig.tight_layout()
    return fig


# ── Plot B: coverage % vs cumulative MB ──────────────────────────────────────

def plot_coverage_vs_bytes(traj: pd.DataFrame, models: list, niids: list) -> plt.Figure | None:
    focus = traj[traj["solution"].isin(FOCUS_SOLUTIONS)].copy()
    if focus.empty or focus["cumulative_mb"].isna().all():
        return None
    percentages = sorted(focus["percentage"].dropna().unique())
    ls_map = {p: ls for p, ls in zip(percentages, ["-", "--", "-.", ":", (0, (3, 1, 1, 1))])}

    n_rows = len(models)
    n_cols = len(niids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    for row, model in enumerate(models):
        for col, niid in enumerate(niids):
            ax = axes[row][col]
            sub = focus.copy()
            if "model" in sub.columns:
                sub = sub[sub["model"] == model]
            if "niid" in sub.columns:
                sub = sub[sub["niid"] == niid]

            for sol in ["sweepfl", "sw"]:
                for pct in percentages:
                    s = sub[(sub["solution"] == sol) & (sub["percentage"] == pct)]
                    if s.empty or s["cumulative_mb"].isna().all():
                        continue
                    s = s.sort_values("cumulative_mb")
                    # clip at 100% for cleaner plot
                    s = s[s["coverage_pct"] <= 100.5]
                    ax.plot(s["cumulative_mb"], s["coverage_pct"],
                            color=COLORS[sol], linestyle=ls_map.get(pct, "-"),
                            linewidth=2.0,
                            label=f"{sol.upper()} p={pct:.0%}")

            ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("Cumulative data transferred (MB, raw float32)")
            ax.set_ylabel("Model coverage (%)")
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"Coverage vs Data Transferred — {model}{niid_lbl}\n"
                         "(same line color = same strategy; linestyle = percentage)")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)

    fig.tight_layout()
    return fig


# ── Plot C: coverage at fixed round budgets ───────────────────────────────────

def plot_fixed_budget_coverage(traj: pd.DataFrame, models: list, niids: list,
                                budgets: list = FIXED_BUDGETS) -> plt.Figure | None:
    focus = traj[traj["solution"].isin(FOCUS_SOLUTIONS) & traj["round"].isin(budgets)].copy()
    if focus.empty:
        return None

    percentages = sorted(focus["percentage"].dropna().unique())
    x = np.arange(len(percentages))
    width = 0.35

    n_rows = len(models)
    n_cols = len(niids)
    n_budgets = len(budgets)
    fig, axes = plt.subplots(n_rows * n_budgets, n_cols,
                             figsize=(7 * n_cols, 4 * n_rows * n_budgets), squeeze=False)

    for row_model, model in enumerate(models):
        for row_budget, budget in enumerate(budgets):
            row = row_model * n_budgets + row_budget
            for col, niid in enumerate(niids):
                ax = axes[row][col]
                sub = focus[focus["round"] == budget].copy()
                if "model" in sub.columns:
                    sub = sub[sub["model"] == model]
                if "niid" in sub.columns:
                    sub = sub[sub["niid"] == niid]

                for offset, sol in zip([-width / 2, width / 2], ["sweepfl", "sw"]):
                    vals = [
                        sub[(sub["solution"] == sol) & (sub["percentage"] == p)]["coverage_pct"].mean()
                        for p in percentages
                    ]
                    bars = ax.bar(x + offset, vals, width,
                                  label=sol.upper(), color=COLORS[sol], alpha=0.85)
                    for bar, v in zip(bars, vals):
                        if not np.isnan(v):
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + 1,
                                    f"{v:.0f}%", ha="center", va="bottom",
                                    fontsize=7, fontweight="bold")

                ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{p:.0%}" for p in percentages])
                ax.set_xlabel("Sparsification percentage")
                ax.set_ylabel("Coverage (%)")
                niid_lbl = f" | niid={niid}" if niid is not None else ""
                ax.set_title(f"Coverage after {budget} rounds — {model}{niid_lbl}")
                ax.legend(fontsize=8)
                ax.grid(True, axis="y", alpha=0.3)
                ax.set_ylim(0, 115)

    fig.tight_layout()
    return fig


# ── Plot D: marginal cost (MB per 1% new coverage) ───────────────────────────

def plot_marginal_cost(params: pd.DataFrame, models: list, niids: list) -> plt.Figure | None:
    focus = params[params["solution"].isin(FOCUS_SOLUTIONS) &
                   params["marginal_cost_mb_per_pct"].notna()].copy()
    if focus.empty:
        return None

    percentages = sorted(focus["percentage"].dropna().unique())
    x = np.arange(len(percentages))
    width = 0.35

    n_rows = len(models)
    n_cols = len(niids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(6, len(percentages) * 1.5) * n_cols, 5 * n_rows),
                             squeeze=False)

    for row, model in enumerate(models):
        for col, niid in enumerate(niids):
            ax = axes[row][col]
            sub = focus.copy()
            if "model" in sub.columns:
                sub = sub[sub["model"] == model]
            if "niid" in sub.columns:
                sub = sub[sub["niid"] == niid]

            for offset, sol in zip([-width / 2, width / 2], ["sweepfl", "sw"]):
                vals = [
                    sub[(sub["solution"] == sol) & (sub["percentage"] == p)][
                        "marginal_cost_mb_per_pct"
                    ].mean()
                    for p in percentages
                ]
                bars = ax.bar(x + offset, vals, width,
                              label=sol.upper(), color=COLORS[sol], alpha=0.85)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() * 1.02,
                                f"{v:.3f}", ha="center", va="bottom",
                                fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([f"{p:.0%}" for p in percentages])
            ax.set_xlabel("Sparsification percentage")
            ax.set_ylabel("MB per 1% new model coverage")
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"Marginal data cost per 1% coverage — {model}{niid_lbl}\n"
                         "(lower = more byte-efficient; SW ≈ 2× SweepFL by design)")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# ── Tables ────────────────────────────────────────────────────────────────────

def save_summary(params: pd.DataFrame):
    cols = [c for c in [
        "model", "dataset", "solution", "niid", "percentage",
        "n_packs", "packs_sent", "pack_size",
        "stride", "new_coverage_pct_per_round",
        "mb_per_round", "marginal_cost_mb_per_pct",
    ] if c in params.columns]
    out = params[cols].copy()
    for c in out.select_dtypes(include=float).columns:
        out[c] = out[c].round(4)
    path = OUT_TAB / "11_data_coverage_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(params: pd.DataFrame):
    display = {
        "solution": "Solution",
        "percentage": "Pct.",
        "new_coverage_pct_per_round": "New cov./round (%)",
        "mb_per_round": "MB/round",
        "marginal_cost_mb_per_pct": "MB per 1% cov.",
    }
    if "model" in params.columns:
        display = {"model": "Model", **display}
    cols = [c for c in display if c in params.columns]

    best = {"marginal_cost_mb_per_pct": params["marginal_cost_mb_per_pct"].dropna().min()}

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        if isinstance(val, float):
            s = f"{val:.3f}" if val < 100 else f"{val:.1f}"
        else:
            s = str(val)
        if col in best and val == best[col]:
            s = r"\textbf{" + s + "}"
        return s

    header = " & ".join(display[c] for c in cols) + r" \\"
    ncols = len(cols)
    lines = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Data-to-coverage efficiency: SweepFL vs SW}",
        r"\label{tab:data_coverage} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_sol = None
    for _, row in params[params["solution"].isin(FOCUS_SOLUTIONS)].sort_values(
        ["solution", "percentage"]
    ).iterrows():
        sol = row.get("solution", "")
        if prev_sol is not None and sol != prev_sol:
            lines.append(r"\midrule")
        prev_sol = sol
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "11_data_coverage_summary.tex"
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
    niids = dims.get("niid", [None])
    print(f"Models     : {models}")
    print(f"Solutions  : {dims.get('solution', [])}")
    print(f"Percentages: {dims.get('percentage', [])}")
    print(f"NIIDs      : {niids}")
    print()

    params = build_params(train)
    if params.empty:
        print("Could not compute parameters (missing n_packs/packs_sent).")
        return

    print("Coverage efficiency summary:")
    print(params[["solution", "percentage", "new_coverage_pct_per_round",
                   "mb_per_round", "marginal_cost_mb_per_pct"]].to_string(index=False))
    print()

    max_rounds = int(train["rounds"].max()) if "rounds" in train.columns else 60
    traj = build_trajectory(params, max_rounds=max(max_rounds, 60))

    fig_a = plot_coverage_vs_round(traj, models, niids)
    if fig_a:
        for ext in ("png", "pdf"):
            fig_a.savefig(OUT_FIG / f"11_coverage_vs_round.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 11_coverage_vs_round.png/pdf")
        plt.close(fig_a)

    fig_b = plot_coverage_vs_bytes(traj, models, niids)
    if fig_b:
        for ext in ("png", "pdf"):
            fig_b.savefig(OUT_FIG / f"11_coverage_vs_bytes.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 11_coverage_vs_bytes.png/pdf")
        plt.close(fig_b)

    fig_c = plot_fixed_budget_coverage(traj, models, niids)
    if fig_c:
        for ext in ("png", "pdf"):
            fig_c.savefig(OUT_FIG / f"11_fixed_budget_coverage.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 11_fixed_budget_coverage.png/pdf")
        plt.close(fig_c)

    fig_d = plot_marginal_cost(params, models, niids)
    if fig_d:
        for ext in ("png", "pdf"):
            fig_d.savefig(OUT_FIG / f"11_marginal_cost.{ext}", dpi=150, bbox_inches="tight")
        print("Saved 11_marginal_cost.png/pdf")
        plt.close(fig_d)

    save_summary(params)
    save_latex(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
