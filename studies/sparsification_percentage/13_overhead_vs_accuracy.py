"""
13 — Communication time overhead vs accuracy.

For each sparsification percentage, plots cumulative communication overhead (s)
on X against accuracy on Y, with round annotations. A side stacked-bar panel
shows how that overhead is split across HE operations per technique.

Communication overhead = pack + mask_select + encrypt + serialize + decrypt
(training time is excluded — focus is on the privacy/communication cost).

Layout per figure row (one row per model × niid):
  Left panel  (2/3 width): X = cumulative comm. overhead (s), Y = accuracy (%).
                            Round annotations every ~4 points.
                            ★ marker when 80% is reached.
                            Right Y-axis shows round number (approx).
  Right panel (1/3 width): Stacked horizontal bars — median per-round overhead
                            split by: Pack+Mask | Encrypt | Serialize | Decrypt.

Output: output/figures/13_overhead_acc_p{pct:03d}.{png,pdf}
Tables: output/tables/13_overhead_summary.{csv,tex}
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
OUT_TAB = HERE / "output" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

TARGET_ACC = 0.80

PACKING_SOLS = {"nvnl", "sw", "rw", "fedphe", "fedphe-scout", "nvnl-scout",
                "topkrr", "srr"}

SOL_COLORS = {
    "plaintext":  "#333333",
    "nvnl":       "#2196F3",
    "sw":         "#FF5722",
    "rw":         "#4CAF50",
    "fedphe":     "#9C27B0",
    "topkrr":     "#E91E63",
    "srr":        "#009688",
    "ckks":       "#FF9800",
    "bfv":        "#00BCD4",
    "batchcrypt": "#795548",
}
SOL_MARKERS = {
    "plaintext": "D", "nvnl": "o", "sw": "s", "rw": "^",
    "fedphe": "P", "topkrr": "p", "srr": "h",
    "ckks": "v", "bfv": "X", "batchcrypt": "H",
}

# Communication overhead columns (training excluded)
COMM_COLS = [
    "pack_time_s",
    "mask_select_time_s",
    "encrypt_time_s",
    "serialize_time_s",
    "decrypt_time_s",
]

# How to group columns into stacked bar segments
COMM_OP_GROUPS = [
    ("Pack+Mask",  ["pack_time_s", "mask_select_time_s"]),
    ("Encrypt",    ["encrypt_time_s"]),
    ("Serialize",  ["serialize_time_s"]),
    ("Decrypt",    ["decrypt_time_s"]),
]

BAR_COLORS = [plt.colormaps["Set2"](i) for i in range(len(COMM_OP_GROUPS))]

DIMS      = ["model", "dataset", "solution", "niid", "percentage"]
SEED_DIMS = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _sol_color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


def _sol_marker(sol: str) -> str:
    return SOL_MARKERS.get(sol, "o")


# ── Cumulative overhead computation ──────────────────────────────────────────

def compute_cumulative_overhead(train: pd.DataFrame) -> pd.DataFrame:
    """Sum comm overhead over clients per round, then cumsum over rounds."""
    df = train.copy()
    present = [c for c in COMM_COLS if c in df.columns]
    df["overhead_s"] = df[present].fillna(0.0).sum(axis=1)

    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["overhead_s"]
        .sum()
        .reset_index()
        .rename(columns={"overhead_s": "overhead_round"})
    )

    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_overhead_s"] = (
        per_round.groupby(seed_group, dropna=False)["overhead_round"]
        .cumsum()
    )
    return per_round


def merge_accuracy(cumulative: pd.DataFrame, server_eval: pd.DataFrame) -> pd.DataFrame:
    """Merge cumulative overhead with server evaluation accuracy, avg over seeds."""
    join_keys = [c for c in SEED_DIMS + ["round"]
                 if c in cumulative.columns and c in server_eval.columns]
    se = server_eval[
        [*join_keys, *(c for c in ["mean_acc"] if c in server_eval.columns)]
    ].copy()
    merged = cumulative.merge(se, on=join_keys, how="inner")

    avg_group = [c for c in DIMS + ["round"] if c in merged.columns]
    result = (
        merged.groupby(avg_group, dropna=False)
        .agg(
            cumulative_overhead_s=("cumulative_overhead_s", "mean"),
            mean_acc=("mean_acc", "mean"),
        )
        .reset_index()
    )
    return result


# ── Overhead medians for stacked bar ─────────────────────────────────────────

def compute_comm_overhead_medians(train: pd.DataFrame) -> pd.DataFrame:
    """Median per-round overhead per operation group, averaged over seeds."""
    df = train.copy()
    present_cols = [c for g in COMM_OP_GROUPS for c in g[1] if c in df.columns]
    if not present_cols:
        return pd.DataFrame()

    seed_round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    # Sum over clients per round
    per_round = (
        df.groupby(seed_round_group, dropna=False)[present_cols]
        .sum()
        .reset_index()
    )

    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    # Median over rounds per seed
    medians_seed = per_round.groupby(seed_group, dropna=False)[present_cols].median().reset_index()

    dim_group = [c for c in DIMS if c in medians_seed.columns]
    # Average over seeds
    result = medians_seed.groupby(dim_group, dropna=False)[present_cols].mean().reset_index()
    return result


# ── Threshold helpers ─────────────────────────────────────────────────────────

def _clip_at_target(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("round")
    above = group[group["mean_acc"] >= TARGET_ACC]
    if above.empty:
        return group
    return group[group["round"] <= above["round"].min()]


def _rounds_to_target(group: pd.DataFrame) -> float:
    above = group[group["mean_acc"] >= TARGET_ACC]
    return float(above["round"].min()) if not above.empty else float("nan")


def _seconds_to_target(group: pd.DataFrame) -> float:
    above = group[group["mean_acc"] >= TARGET_ACC]
    if above.empty:
        return float("nan")
    r = above["round"].min()
    row = group[group["round"] == r]
    return float(row["cumulative_overhead_s"].iloc[0]) if not row.empty else float("nan")


def _ref_s_per_round(sub: pd.DataFrame) -> float:
    """Median overhead seconds/round across all experiments in sub."""
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_overhead_s"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _sorted_solutions(sols):
    other = sorted(s for s in sols if s != "plaintext")
    return other + (["plaintext"] if "plaintext" in sols else [])


def _draw_stacked_bar(ax, solutions, overhead_medians, model, niid):
    """Draw stacked horizontal bars for each solution."""
    present_cols = [c for g in COMM_OP_GROUPS for c in g[1]]

    bar_sols = []
    bar_data = {label: [] for label, _ in COMM_OP_GROUPS}

    for sol in solutions:
        m_filt = (overhead_medians["model"] == model) if "model" in overhead_medians.columns else True
        n_filt = (overhead_medians["niid"]  == niid)  if "niid"  in overhead_medians.columns else True
        s_filt = overhead_medians["solution"] == sol
        row = overhead_medians[m_filt & n_filt & s_filt]
        if row.empty:
            continue
        bar_sols.append(sol)
        for label, cols in COMM_OP_GROUPS:
            val = sum(
                float(row[c].iloc[0]) if c in row.columns else 0.0
                for c in cols
            )
            bar_data[label].append(val)

    if not bar_sols:
        ax.set_visible(False)
        return

    y = np.arange(len(bar_sols))
    bottoms = np.zeros(len(bar_sols))

    for i, (label, _) in enumerate(COMM_OP_GROUPS):
        vals = np.array(bar_data[label])
        ax.barh(y, vals, left=bottoms, label=label,
                color=BAR_COLORS[i], alpha=0.88, edgecolor="white", linewidth=0.4)
        bottoms += vals

    ax.set_yticks(y)
    ax.set_yticklabels(bar_sols, fontsize=8)
    ax.set_xlabel("Comm. overhead (s/round, median)", fontsize=8)
    ax.set_title("Overhead breakdown", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Label total on each bar
    for yi, sol in enumerate(bar_sols):
        total = bottoms[yi]
        ax.text(total * 1.02, yi, f"{total:.2f}s",
                va="center", ha="left", fontsize=7, color="#333333")
    ax.set_xlim(right=bottoms.max() * 1.35 if bottoms.max() > 0 else 1)


# ── Main plot function ────────────────────────────────────────────────────────

def plot_for_percentage(data: pd.DataFrame, overhead_medians: pd.DataFrame,
                        pct, models: list, niids: list):
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
        gridspec_kw={"width_ratios": [2, 1]},
        squeeze=False,
    )

    row_idx = 0
    for model in models:
        for niid in niids:
            ax     = axes[row_idx][0]
            ax_bar = axes[row_idx][1]
            ax_r   = ax.twinx()

            m_filt = (sub["model"] == model) if "model" in sub.columns else pd.Series(True, index=sub.index)
            n_filt = (sub["niid"]  == niid)  if "niid"  in sub.columns else pd.Series(True, index=sub.index)
            combo  = sub[m_filt & n_filt].copy()

            if combo.empty:
                row_idx += 1
                continue

            solutions  = _sorted_solutions(combo["solution"].dropna().unique().tolist())
            ref_spr    = _ref_s_per_round(combo)
            max_s      = 0.0
            max_round  = 0

            rounds_to_80:  dict[str, float] = {}
            seconds_to_80: dict[str, float] = {}

            for sol in solutions:
                sg = combo[combo["solution"] == sol].sort_values("round")
                if sg.empty:
                    continue

                clipped = _clip_at_target(sg)
                color   = _sol_color(sol)
                ls      = "--" if sol == "plaintext" else "-"
                lw      = 1.8  if sol == "plaintext" else 2.2

                ax.plot(
                    clipped["cumulative_overhead_s"],
                    clipped["mean_acc"] * 100,
                    color=color, linestyle=ls, linewidth=lw,
                    label=sol, zorder=3,
                )
                ax_r.plot(
                    clipped["cumulative_overhead_s"],
                    clipped["round"],
                    color=color, linestyle=ls, linewidth=0.8,
                    alpha=0.0, zorder=1,
                )

                # Round annotations every ~4 points
                n_pts = len(clipped)
                step  = max(1, n_pts // 4)
                for _, rr in clipped.iloc[::step].iterrows():
                    ax.annotate(
                        f"R{int(rr['round'])}",
                        xy=(rr["cumulative_overhead_s"], rr["mean_acc"] * 100),
                        fontsize=6, color=color, alpha=0.75,
                        xytext=(3, 2), textcoords="offset points",
                    )

                last = clipped.iloc[-1]
                hit  = last["mean_acc"] >= TARGET_ACC
                ax.plot(
                    last["cumulative_overhead_s"], last["mean_acc"] * 100,
                    marker="*" if hit else "x",
                    color=color, markersize=11 if hit else 8,
                    zorder=5, linestyle="",
                )

                max_s     = max(max_s, clipped["cumulative_overhead_s"].max())
                max_round = max(max_round, int(clipped["round"].max()))

                rounds_to_80[sol]  = _rounds_to_target(sg)
                seconds_to_80[sol] = _seconds_to_target(sg)

            # 80% threshold line
            ax.axhline(TARGET_ACC * 100, color="gray", linestyle=":",
                       linewidth=1.2, alpha=0.7, label=f"{TARGET_ACC:.0%} threshold")

            # Right Y-axis: round scale
            if ref_spr > 0 and max_s > 0:
                max_r_est = max_s / ref_spr
                ax_r.set_ylim(0, max_r_est * 1.05)
                ax_r.set_ylabel("Round (approx.)", fontsize=8, color="#555555")
                ax_r.tick_params(axis="y", labelsize=7, labelcolor="#555555")
                nice = [r for r in [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
                        if r <= max_r_est * 1.05]
                ax_r.set_yticks(nice)
            else:
                ax_r.set_yticks([])

            pct_lbl  = f"{pct:.0%}" if pct is not None else "baseline"
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"{model}{niid_lbl}  —  p = {pct_lbl}", fontsize=10)
            ax.set_xlabel("Cumulative communication overhead (s)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
            ax.set_xlim(left=0)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.25)

            # Stacked bar
            _draw_stacked_bar(ax_bar, solutions, overhead_medians, model, niid)

            row_idx += 1

    pct_str = f"p = {pct:.0%}" if pct is not None else "plaintext baseline"
    fig.suptitle(
        f"Communication Overhead vs Accuracy  |  {pct_str}\n"
        f"(★ = reached {TARGET_ACC:.0%}  |  overhead = pack+mask+encrypt+serialize+decrypt)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(data: pd.DataFrame, overhead_medians: pd.DataFrame) -> pd.DataFrame:
    dims = [c for c in DIMS if c in data.columns]
    rows = []
    for keys, group in data.groupby(dims, dropna=False):
        kd = dict(zip(dims, keys if isinstance(keys, tuple) else [keys]))
        # Total median comm overhead per round
        m_filt = True
        n_filt = True
        s_filt = overhead_medians["solution"] == kd.get("solution", "")
        if "model" in overhead_medians.columns and "model" in kd:
            m_filt = overhead_medians["model"] == kd["model"]
        if "niid" in overhead_medians.columns and "niid" in kd:
            n_filt = overhead_medians["niid"] == kd["niid"]
        pct_filt = True
        if "percentage" in overhead_medians.columns:
            pct_val = kd.get("percentage")
            if pd.isna(pct_val) if isinstance(pct_val, float) else pct_val is None:
                pct_filt = overhead_medians["percentage"].isna()
            else:
                pct_filt = overhead_medians["percentage"] == pct_val
        om_row = overhead_medians[m_filt & n_filt & s_filt & pct_filt]
        present = [c for g in COMM_OP_GROUPS for c in g[1] if c in om_row.columns]
        comm_med = float(om_row[present].values.sum()) if not om_row.empty and present else float("nan")

        rows.append({
            **kd,
            "rounds_to_80pct":       _rounds_to_target(group),
            "seconds_to_80pct":      _seconds_to_target(group),
            "comm_overhead_s_median": comm_med,
            "final_acc":             group.sort_values("round")["mean_acc"].iloc[-1],
        })
    df = pd.DataFrame(rows)
    for c in df.select_dtypes(include=float).columns:
        df[c] = df[c].round(3)
    return df.sort_values(["model", "dataset", "niid", "percentage", "solution"],
                          na_position="last")


def save_csv(summary: pd.DataFrame):
    path = OUT_TAB / "13_overhead_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(summary: pd.DataFrame):
    display = {
        "model": "Model", "solution": "Solution", "percentage": "Pct.",
        "niid": "NIID", "rounds_to_80pct": r"Rounds to 80\%",
        "seconds_to_80pct": r"Seconds to 80\%",
        "comm_overhead_s_median": "Overhead (s/round)",
        "final_acc": "Final acc.",
    }
    cols = [c for c in display if c in summary.columns]
    best_rounds = summary["rounds_to_80pct"].dropna().min() if "rounds_to_80pct" in summary else None
    best_secs   = summary["seconds_to_80pct"].dropna().min() if "seconds_to_80pct" in summary else None

    def _fmt(row, col):
        v = row[col]
        if pd.isna(v):
            return "—"
        if col == "percentage":
            return f"{v:.0%}"
        if col == "final_acc":
            return f"{v:.3f}"
        if isinstance(v, float):
            s = f"{v:.1f}" if v >= 10 else f"{v:.2f}"
        else:
            s = str(v)
        if col == "rounds_to_80pct" and best_rounds is not None and v == best_rounds:
            s = r"\textbf{" + s + "}"
        if col == "seconds_to_80pct" and best_secs is not None and v == best_secs:
            s = r"\textbf{" + s + "}"
        return s

    header = " & ".join(display[c] for c in cols) + r" \\"
    ncols  = len(cols)
    lines  = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Communication overhead and rounds to reach 80\% accuracy}",
        r"\label{tab:overhead_summary} \\",
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

    path = OUT_TAB / "13_overhead_summary.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train       = dfs["train"]
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

    print("Computing cumulative communication overhead...")
    cumulative      = compute_cumulative_overhead(train)
    data            = merge_accuracy(cumulative, server_eval)
    overhead_medians = compute_comm_overhead_medians(train)

    if data.empty:
        print("No data after merge (check that train.csv and server_eval.csv share rounds).")
        return

    summary = build_summary(data, overhead_medians)
    print("Summary — overhead and rounds to 80% accuracy:")
    print(summary[["model", "solution", "percentage", "niid",
                   "rounds_to_80pct", "seconds_to_80pct",
                   "comm_overhead_s_median", "final_acc"]].to_string(index=False))
    print()

    save_csv(summary)
    save_latex(summary)

    for pct in sorted(data["percentage"].dropna().unique()):
        fig = plot_for_percentage(data, overhead_medians, pct, models, niids)
        if fig:
            tag = f"{int(round(pct * 100)):03d}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG / f"13_overhead_acc_p{tag}.{ext}",
                            dpi=150, bbox_inches="tight")
            print(f"Saved 13_overhead_acc_p{tag}.png/pdf")
            plt.close(fig)

    if not data[data["solution"] == "plaintext"].empty:
        fig = plot_for_percentage(data, overhead_medians, None, models, niids)
        if fig:
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG / f"13_overhead_acc_plaintext.{ext}",
                            dpi=150, bbox_inches="tight")
            print("Saved 13_overhead_acc_plaintext.png/pdf")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
