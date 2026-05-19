"""
12 — Cumulative bytes transmitted vs accuracy: all techniques.

One figure per sparsification percentage. Techniques at the same percentage are
compared (plaintext baseline included in every figure).

Layout per figure row (one row per model × niid combination):
  Left panel  (3/4 width): X = cumulative upload MB, Y-left = accuracy (%),
                            Y-right = round number.
                            Curves stop at 80% accuracy (★ marker).
  Right panel (1/4 width): Horizontal bars — rounds to reach 80% per technique,
                            labelled with round count and total MB cost.

Bytes metric (raw float32 model data, pre-encryption):
  packing (nvnl, sw, rw, fedphe): packs_sent × pack_size × 4  bytes per client/round
  other   (plaintext, ckks, bfv): n_params × 4                bytes per client/round

Total per round = sum over all clients.

Output: output/figures/12_bytes_acc_p{pct}.{png,pdf}
Tables: output/tables/12_bytes_to_80pct.{csv,tex}
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

TARGET_ACC = 0.80

PACKING_SOLS = {"nvnl", "sw", "rw", "fedphe", "fedphe-scout", "nvnl-scout"}

SOL_COLORS = {
    "plaintext":  "#333333",
    "nvnl":       "#2196F3",
    "sw":         "#FF5722",
    "rw":         "#4CAF50",
    "fedphe":     "#9C27B0",
    "ckks":       "#FF9800",
    "bfv":        "#00BCD4",
    "batchcrypt": "#795548",
}
SOL_MARKERS = {
    "plaintext": "D", "nvnl": "o", "sw": "s", "rw": "^",
    "fedphe": "P", "ckks": "v", "bfv": "X", "batchcrypt": "h",
}

DIMS = ["model", "dataset", "solution", "niid", "percentage"]
SEED_DIMS = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _sol_color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


def _sol_marker(sol: str) -> str:
    return SOL_MARKERS.get(sol, "o")


# ── Byte estimation ───────────────────────────────────────────────────────────

def _est_bytes_per_client(row) -> float:
    """Raw float32 model data per client per round (pre-encryption)."""
    sol = str(row.get("solution", ""))
    if sol in PACKING_SOLS:
        packs = float(row.get("packs_sent") or 0)
        pack_size = float(row.get("pack_size") or 0)
        return packs * pack_size * 4  # float32
    else:
        n = float(row.get("n_params") or 0)
        return n * 4  # float32


# ── Cumulative bytes computation ──────────────────────────────────────────────

def compute_cumulative_bytes(train: pd.DataFrame) -> pd.DataFrame:
    """Sum bytes over clients per round, then cumsum over rounds per experiment."""
    df = train.copy()
    df["bytes_client"] = df.apply(_est_bytes_per_client, axis=1)

    # Sum over clients within a round
    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["bytes_client"]
        .sum()
        .reset_index()
        .rename(columns={"bytes_client": "bytes_round"})
    )

    # Cumulative sum over rounds per experiment
    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_mb"] = (
        per_round.groupby(seed_group, dropna=False)["bytes_round"]
        .cumsum() / 1e6
    )
    return per_round


def merge_accuracy(cumulative: pd.DataFrame, server_eval: pd.DataFrame) -> pd.DataFrame:
    """Merge cumulative bytes with server evaluation accuracy, average over seeds."""
    join_keys = [c for c in SEED_DIMS + ["round"]
                 if c in cumulative.columns and c in server_eval.columns]
    se = server_eval[[*join_keys, *(c for c in ["mean_acc"] if c in server_eval.columns)]].copy()
    merged = cumulative.merge(se, on=join_keys, how="inner")

    avg_group = [c for c in DIMS + ["round"] if c in merged.columns]
    result = (
        merged.groupby(avg_group, dropna=False)
        .agg(cumulative_mb=("cumulative_mb", "mean"),
             mean_acc=("mean_acc", "mean"))
        .reset_index()
    )
    return result


# ── Threshold helpers ─────────────────────────────────────────────────────────

def _clip_at_target(group: pd.DataFrame) -> pd.DataFrame:
    """Keep rows up to (and including) first round at >= TARGET_ACC."""
    group = group.sort_values("round")
    above = group[group["mean_acc"] >= TARGET_ACC]
    if above.empty:
        return group
    return group[group["round"] <= above["round"].min()]


def _rounds_to_target(group: pd.DataFrame) -> float:
    above = group[group["mean_acc"] >= TARGET_ACC]
    return float(above["round"].min()) if not above.empty else float("nan")


def _mb_to_target(group: pd.DataFrame) -> float:
    above = group[group["mean_acc"] >= TARGET_ACC]
    if above.empty:
        return float("nan")
    r = above["round"].min()
    row = group[group["round"] == r]
    return float(row["cumulative_mb"].iloc[0]) if not row.empty else float("nan")


# ── Reference bytes/round for right Y-axis ───────────────────────────────────

def _ref_mb_per_round(sub: pd.DataFrame) -> float:
    """Median MB/round across all experiments in sub, used for right Y-axis scale."""
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_mb"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


# ── Plot ──────────────────────────────────────────────────────────────────────

def _sorted_solutions(sols):
    """Plaintext last so it's drawn on top in legend; alphabetical otherwise."""
    other = sorted(s for s in sols if s != "plaintext")
    return other + (["plaintext"] if "plaintext" in sols else [])


def plot_for_percentage(data: pd.DataFrame, pct: float | None,
                        models: list, niids: list) -> plt.Figure | None:
    # Select: packing solutions at this percentage + plaintext baseline
    if pct is None:
        sub = data[data["percentage"].isna()].copy()
    else:
        pct_mask = data["percentage"].notna() & (data["percentage"] == pct)
        pt_mask  = data["percentage"].isna() & (data["solution"] == "plaintext")
        sub = data[pct_mask | pt_mask].copy()

    if sub.empty:
        return None

    n_rows = len(models) * len(niids)
    if n_rows == 0:
        return None

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(16, 5.5 * n_rows),
        gridspec_kw={"width_ratios": [3, 1]},
        squeeze=False,
    )

    row_idx = 0
    for model in models:
        for niid in niids:
            ax = axes[row_idx][0]
            ax_bar = axes[row_idx][1]
            ax_r = ax.twinx()  # right Y-axis for round numbers

            m_filt = (sub["model"] == model) if "model" in sub.columns else pd.Series(True, index=sub.index)
            n_filt = (sub["niid"] == niid)   if "niid"  in sub.columns else pd.Series(True, index=sub.index)
            combo = sub[m_filt & n_filt].copy()

            if combo.empty:
                row_idx += 1
                continue

            solutions = _sorted_solutions(combo["solution"].dropna().unique().tolist())
            ref_bpr = _ref_mb_per_round(combo)

            rounds_to_80: dict[str, float] = {}
            mb_to_80:     dict[str, float] = {}
            max_mb = 0.0
            max_round = 0

            for sol in solutions:
                sg = combo[combo["solution"] == sol].sort_values("round")
                if sg.empty:
                    continue

                clipped = _clip_at_target(sg)
                color   = _sol_color(sol)
                ls      = "--" if sol == "plaintext" else "-"
                lw      = 1.8  if sol == "plaintext" else 2.2
                mk      = _sol_marker(sol)

                # ── Left Y: accuracy ──
                ax.plot(
                    clipped["cumulative_mb"],
                    clipped["mean_acc"] * 100,
                    color=color, linestyle=ls, linewidth=lw,
                    label=sol, zorder=3,
                )

                # ── Right Y: rounds (same data, round on y) ──
                ax_r.plot(
                    clipped["cumulative_mb"],
                    clipped["round"],
                    color=color, linestyle=ls, linewidth=0.8,
                    alpha=0.0,   # invisible — just to drive the scale
                    zorder=1,
                )

                # Annotate round number every ~N points
                n_pts = len(clipped)
                step  = max(1, n_pts // 4)
                for _, rr in clipped.iloc[::step].iterrows():
                    ax.annotate(
                        f"R{int(rr['round'])}",
                        xy=(rr["cumulative_mb"], rr["mean_acc"] * 100),
                        fontsize=6, color=color, alpha=0.75,
                        xytext=(3, 2), textcoords="offset points",
                    )

                # 80%-crossing marker
                last = clipped.iloc[-1]
                hit_target = last["mean_acc"] >= TARGET_ACC
                ax.plot(
                    last["cumulative_mb"], last["mean_acc"] * 100,
                    marker="*" if hit_target else "x",
                    color=color, markersize=11 if hit_target else 8,
                    zorder=5, linestyle="",
                )

                max_mb    = max(max_mb, clipped["cumulative_mb"].max())
                max_round = max(max_round, int(clipped["round"].max()))

                rounds_to_80[sol] = _rounds_to_target(sg)
                mb_to_80[sol]     = _mb_to_target(sg)

            # 80% horizontal threshold line
            ax.axhline(TARGET_ACC * 100, color="gray", linestyle=":",
                       linewidth=1.2, alpha=0.7, label=f"{TARGET_ACC:.0%} threshold")

            # Right Y-axis: round scale based on ref_bpr
            if ref_bpr > 0 and max_mb > 0:
                max_r_est = max_mb / ref_bpr
                ax_r.set_ylim(0, max_r_est * 1.05)
                ax_r.set_ylabel("Round (approx.)", fontsize=8, color="#555555")
                ax_r.tick_params(axis="y", labelsize=7, labelcolor="#555555")
                # nice round-number ticks
                nice = [r for r in [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
                        if r <= max_r_est * 1.05]
                ax_r.set_yticks(nice)
            else:
                ax_r.set_yticks([])

            pct_lbl  = f"{pct:.0%}" if pct is not None else "baseline"
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"{model}{niid_lbl}  —  p = {pct_lbl}", fontsize=10)
            ax.set_xlabel("Cumulative upload (MB, raw float32 pre-encryption)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
            ax.set_xlim(left=0)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.25)

            # ── Right panel: bar chart — rounds to 80% ──────────────────────
            valid = [(s, rounds_to_80[s], mb_to_80[s])
                     for s in solutions
                     if not np.isnan(rounds_to_80.get(s, float("nan")))]
            if valid:
                bar_sols = [v[0] for v in valid]
                bar_vals = [v[1] for v in valid]
                bar_mbs  = [v[2] for v in valid]
                bar_colors = [_sol_color(s) for s in bar_sols]
                bars = ax_bar.barh(
                    bar_sols, bar_vals,
                    color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5,
                )
                for bar, sol, mb in zip(bars, bar_sols, bar_mbs):
                    w = bar.get_width()
                    mb_str = f"{mb:.1f} MB" if not np.isnan(mb) else ""
                    ax_bar.text(
                        w + max(bar_vals) * 0.03,
                        bar.get_y() + bar.get_height() / 2,
                        f"R{int(w)}\n{mb_str}",
                        va="center", ha="left", fontsize=8, fontweight="bold",
                    )
                ax_bar.set_xlim(right=max(bar_vals) * 1.45)

            # Solutions that never reached 80%
            never = [s for s in solutions
                     if np.isnan(rounds_to_80.get(s, float("nan")))]
            if never:
                ax_bar.text(
                    0.5, -0.18,
                    "Did not reach 80%: " + ", ".join(never),
                    transform=ax_bar.transAxes,
                    fontsize=7, color="red", ha="center", va="top",
                )

            ax_bar.set_xlabel("Rounds to 80% acc.")
            ax_bar.set_title("Rounds to 80%", fontsize=9)
            ax_bar.grid(True, axis="x", alpha=0.3)
            ax_bar.invert_yaxis()

            row_idx += 1

    pct_str = f"p = {pct:.0%}" if pct is not None else "plaintext baseline"
    fig.suptitle(
        f"Cumulative Data Transferred vs Accuracy  |  {pct_str}\n"
        f"(★ = reached {TARGET_ACC:.0%}  |  bytes = raw float32 model data)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    dims = [c for c in DIMS if c in data.columns]
    rows = []
    for keys, group in data.groupby(dims, dropna=False):
        kd = dict(zip(dims, keys if isinstance(keys, tuple) else [keys]))
        rows.append({
            **kd,
            "rounds_to_80pct": _rounds_to_target(group),
            "mb_to_80pct":     _mb_to_target(group),
            "final_acc":       group.sort_values("round")["mean_acc"].iloc[-1],
        })
    df = pd.DataFrame(rows)
    for c in df.select_dtypes(include=float).columns:
        df[c] = df[c].round(3)
    return df.sort_values(["model", "dataset", "niid", "percentage", "solution"],
                           na_position="last")


def save_csv(summary: pd.DataFrame):
    path = OUT_TAB / "12_bytes_to_80pct.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(summary: pd.DataFrame):
    display = {
        "model": "Model", "solution": "Solution", "percentage": "Pct.",
        "niid": "NIID", "rounds_to_80pct": "Rounds to 80\\%",
        "mb_to_80pct": "MB to 80\\%", "final_acc": "Final acc.",
    }
    cols = [c for c in display if c in summary.columns]
    best_rounds = summary["rounds_to_80pct"].dropna().min()
    best_mb     = summary["mb_to_80pct"].dropna().min()

    def _fmt(row, col):
        v = row[col]
        if pd.isna(v):
            return "—"
        if col == "percentage":
            return f"{v:.0%}" if not pd.isna(v) else "—"
        if col == "final_acc":
            return f"{v:.3f}"
        if isinstance(v, float):
            s = f"{v:.1f}" if v >= 10 else f"{v:.2f}"
        else:
            s = str(v)
        if col == "rounds_to_80pct" and v == best_rounds:
            s = r"\textbf{" + s + "}"
        if col == "mb_to_80pct" and v == best_mb:
            s = r"\textbf{" + s + "}"
        return s

    header = " & ".join(display[c] for c in cols) + r" \\"
    ncols  = len(cols)
    lines  = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Rounds and MB to reach 80\% accuracy per technique}",
        r"\label{tab:bytes_to_80pct} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_pct = None
    for _, row in summary.iterrows():
        pct = row.get("percentage")
        if prev_pct is not None and pct != prev_pct:
            lines.append(r"\midrule")
        prev_pct = pct
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "12_bytes_to_80pct.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train      = dfs["train"]
    server_eval = dfs["server_eval"]

    if train.empty:
        print("No train data found.")
        return
    if server_eval.empty:
        print("No server_eval data found.")
        return

    dims = load_data.discover_dims(train)
    models      = dims.get("model", [])
    niids       = dims.get("niid", [None])
    percentages = dims.get("percentage", [])
    print(f"Models     : {models}")
    print(f"Solutions  : {dims.get('solution', [])}")
    print(f"Percentages: {percentages}")
    print(f"NIIDs      : {niids}")
    print()

    print("Computing cumulative bytes...")
    cumulative = compute_cumulative_bytes(train)
    data = merge_accuracy(cumulative, server_eval)

    if data.empty:
        print("No data after merge (check that train.csv and server_eval.csv share rounds).")
        return

    summary = build_summary(data)
    print("Summary — rounds and MB to 80% accuracy:")
    print(summary[["model", "solution", "percentage", "niid",
                   "rounds_to_80pct", "mb_to_80pct", "final_acc"]].to_string(index=False))
    print()

    save_csv(summary)
    save_latex(summary)

    # One figure per percentage (packing techniques at that percentage + plaintext)
    for pct in sorted(data["percentage"].dropna().unique()):
        fig = plot_for_percentage(data, pct, models, niids)
        if fig:
            tag = f"{int(round(pct * 100)):03d}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG / f"12_bytes_acc_p{tag}.{ext}",
                            dpi=150, bbox_inches="tight")
            print(f"Saved 12_bytes_acc_p{tag}.png/pdf")
            plt.close(fig)

    # Plaintext-only figure (baseline)
    if not data[data["solution"] == "plaintext"].empty:
        fig = plot_for_percentage(data, None, models, niids)
        if fig:
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG / f"12_bytes_acc_plaintext.{ext}",
                            dpi=150, bbox_inches="tight")
            print("Saved 12_bytes_acc_plaintext.png/pdf")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
