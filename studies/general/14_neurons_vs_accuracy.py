"""
14 — Cumulative neurons transmitted vs accuracy: all techniques.

One figure per sparsification percentage. Techniques at the same percentage are
compared (plaintext baseline included in every figure).

Layout per figure row (one row per model × niid combination):
  Left panel  (2/3 width): X = cumulative neurons transmitted (M),
                            Y-left = accuracy (%),
                            Y-right = round number (approx).
                            Round annotations every ~4 points.
                            ★ marker when 80% is reached.
  Right panel (1/3 width): Horizontal bars — median neurons per round per technique,
                            labelled with value in M.

Neurons metric (model parameters actually transmitted per client per round):
  packing (sweepfl, sw, rw, fedphe, topkrr, srr): packs_sent × pack_size
  other   (plaintext, ckks, bfv, batchcrypt):   n_params

Total per round = sum over all clients.

Output: output/figures/14_neurons_acc_p{pct}.{png,pdf}
Tables: output/tables/14_neurons_summary.{csv,tex}
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

PACKING_SOLS = {"sweepfl", "sw", "rw", "fedphe", "fedphe-scout", "sweepfl-scout", "topkrr", "srr"}

SOL_COLORS = {
    "plaintext":  "#333333",
    "sweepfl":       "#2196F3",
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
    "plaintext": "D", "sweepfl": "o", "sw": "s", "rw": "^",
    "fedphe": "P", "topkrr": "p", "srr": "h",
    "ckks": "v", "bfv": "X", "batchcrypt": "H",
}

DIMS      = ["model", "dataset", "solution", "niid", "percentage"]
SEED_DIMS = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _sol_color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


def _sol_marker(sol: str) -> str:
    return SOL_MARKERS.get(sol, "o")


# ── Neuron estimation ─────────────────────────────────────────────────────────

def _neurons_per_client(row) -> float:
    """Parameters actually transmitted per client per round."""
    sol = str(row.get("solution", ""))
    if sol in PACKING_SOLS:
        return float(row.get("packs_sent") or 0) * float(row.get("pack_size") or 0)
    else:
        return float(row.get("n_params") or 0)


# ── Cumulative neurons computation ────────────────────────────────────────────

def compute_cumulative_neurons(train: pd.DataFrame) -> pd.DataFrame:
    """Sum neurons over clients per round, then cumsum over rounds per experiment."""
    df = train.copy()
    df["neurons_client"] = df.apply(_neurons_per_client, axis=1)

    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["neurons_client"]
        .sum()
        .reset_index()
        .rename(columns={"neurons_client": "neurons_round"})
    )

    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_neurons"] = (
        per_round.groupby(seed_group, dropna=False)["neurons_round"]
        .cumsum()
    )
    return per_round


def merge_accuracy(cumulative: pd.DataFrame, server_eval: pd.DataFrame) -> pd.DataFrame:
    """Merge cumulative neurons with server evaluation accuracy, average over seeds."""
    join_keys = [c for c in SEED_DIMS + ["round"]
                 if c in cumulative.columns and c in server_eval.columns]
    se = server_eval[[*join_keys, *(c for c in ["mean_acc"] if c in server_eval.columns)]].copy()
    merged = cumulative.merge(se, on=join_keys, how="inner")

    avg_group = [c for c in DIMS + ["round"] if c in merged.columns]
    result = (
        merged.groupby(avg_group, dropna=False)
        .agg(
            cumulative_neurons=("cumulative_neurons", "mean"),
            mean_acc=("mean_acc", "mean"),
        )
        .reset_index()
    )
    return result


# ── Median neurons/round for side bar ────────────────────────────────────────

def compute_neurons_per_round_medians(train: pd.DataFrame) -> pd.DataFrame:
    """Median neurons/round per solution (summed over clients, median over rounds, avg over seeds)."""
    df = train.copy()
    df["neurons_client"] = df.apply(_neurons_per_client, axis=1)

    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["neurons_client"]
        .sum()
        .reset_index()
        .rename(columns={"neurons_client": "neurons_round"})
    )

    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    medians_seed = (
        per_round.groupby(seed_group, dropna=False)["neurons_round"]
        .median()
        .reset_index()
        .rename(columns={"neurons_round": "neurons_median"})
    )

    dim_group = [c for c in DIMS if c in medians_seed.columns]
    result = (
        medians_seed.groupby(dim_group, dropna=False)["neurons_median"]
        .mean()
        .reset_index()
    )
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


def _neurons_to_target(group: pd.DataFrame) -> float:
    above = group[group["mean_acc"] >= TARGET_ACC]
    if above.empty:
        return float("nan")
    r = above["round"].min()
    row = group[group["round"] == r]
    return float(row["cumulative_neurons"].iloc[0]) if not row.empty else float("nan")


# ── Reference neurons/round for right Y-axis ─────────────────────────────────

def _ref_neurons_per_round(sub: pd.DataFrame) -> float:
    """Median neurons/round across all experiments in sub, for right Y-axis scale."""
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_neurons"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


def _fmt_millions(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


# ── Plot ──────────────────────────────────────────────────────────────────────

def _sorted_solutions(sols):
    other = sorted(s for s in sols if s != "plaintext")
    return other + (["plaintext"] if "plaintext" in sols else [])


def _draw_neurons_bar(ax, solutions, neuron_medians, model, niid):
    """Horizontal bar chart: median neurons/round per solution."""
    bar_sols = []
    bar_vals = []

    for sol in solutions:
        m_filt = (neuron_medians["model"] == model) if "model" in neuron_medians.columns else True
        n_filt = (neuron_medians["niid"]  == niid)  if "niid"  in neuron_medians.columns else True
        s_filt = neuron_medians["solution"] == sol
        row = neuron_medians[m_filt & n_filt & s_filt]
        if row.empty:
            continue
        bar_sols.append(sol)
        bar_vals.append(float(row["neurons_median"].iloc[0]))

    if not bar_sols:
        ax.set_visible(False)
        return

    bar_colors = [_sol_color(s) for s in bar_sols]
    bars = ax.barh(
        bar_sols, bar_vals,
        color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    max_val = max(bar_vals) if bar_vals else 1
    for bar, val in zip(bars, bar_vals):
        ax.text(
            val + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            _fmt_millions(val),
            va="center", ha="left", fontsize=8, fontweight="bold",
        )

    ax.set_xlim(right=max_val * 1.45)
    ax.set_xlabel("Neurons/round (median)", fontsize=8)
    ax.set_title("Neurons per round", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # X-axis ticks in millions
    def _tick_fmt(x, _):
        return _fmt_millions(x)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_tick_fmt))


def plot_for_percentage(data: pd.DataFrame, neuron_medians: pd.DataFrame,
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

            solutions = _sorted_solutions(combo["solution"].dropna().unique().tolist())
            ref_npr   = _ref_neurons_per_round(combo)
            max_n     = 0.0
            max_round = 0

            rounds_to_80:  dict[str, float] = {}
            neurons_to_80: dict[str, float] = {}

            for sol in solutions:
                sg = combo[combo["solution"] == sol].sort_values("round")
                if sg.empty:
                    continue

                clipped = _clip_at_target(sg)
                color   = _sol_color(sol)
                ls      = "--" if sol == "plaintext" else "-"
                lw      = 1.8  if sol == "plaintext" else 2.2

                ax.plot(
                    clipped["cumulative_neurons"] / 1e6,
                    clipped["mean_acc"] * 100,
                    color=color, linestyle=ls, linewidth=lw,
                    label=sol, zorder=3,
                )
                ax_r.plot(
                    clipped["cumulative_neurons"] / 1e6,
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
                        xy=(rr["cumulative_neurons"] / 1e6, rr["mean_acc"] * 100),
                        fontsize=6, color=color, alpha=0.75,
                        xytext=(3, 2), textcoords="offset points",
                    )

                last = clipped.iloc[-1]
                hit  = last["mean_acc"] >= TARGET_ACC
                ax.plot(
                    last["cumulative_neurons"] / 1e6, last["mean_acc"] * 100,
                    marker="*" if hit else "x",
                    color=color, markersize=11 if hit else 8,
                    zorder=5, linestyle="",
                )

                max_n     = max(max_n, clipped["cumulative_neurons"].max() / 1e6)
                max_round = max(max_round, int(clipped["round"].max()))

                rounds_to_80[sol]  = _rounds_to_target(sg)
                neurons_to_80[sol] = _neurons_to_target(sg)

            ax.axhline(TARGET_ACC * 100, color="gray", linestyle=":",
                       linewidth=1.2, alpha=0.7, label=f"{TARGET_ACC:.0%} threshold")

            # Right Y-axis: round scale
            if ref_npr > 0 and max_n > 0:
                max_r_est = (max_n * 1e6) / ref_npr
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
            ax.set_xlabel("Cumulative neurons transmitted (M)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
            ax.set_xlim(left=0)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.25)

            _draw_neurons_bar(ax_bar, solutions, neuron_medians, model, niid)

            row_idx += 1

    pct_str = f"p = {pct:.0%}" if pct is not None else "plaintext baseline"
    fig.suptitle(
        f"Cumulative Neurons Transmitted vs Accuracy  |  {pct_str}\n"
        f"(★ = reached {TARGET_ACC:.0%}  |  neurons = packs_sent×pack_size for packing, n_params otherwise)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(data: pd.DataFrame, neuron_medians: pd.DataFrame) -> pd.DataFrame:
    dims = [c for c in DIMS if c in data.columns]
    rows = []
    for keys, group in data.groupby(dims, dropna=False):
        kd = dict(zip(dims, keys if isinstance(keys, tuple) else [keys]))

        m_filt = True
        n_filt = True
        s_filt = neuron_medians["solution"] == kd.get("solution", "")
        if "model" in neuron_medians.columns and "model" in kd:
            m_filt = neuron_medians["model"] == kd["model"]
        if "niid" in neuron_medians.columns and "niid" in kd:
            n_filt = neuron_medians["niid"] == kd["niid"]
        pct_filt = True
        if "percentage" in neuron_medians.columns:
            pct_val = kd.get("percentage")
            if pd.isna(pct_val) if isinstance(pct_val, float) else pct_val is None:
                pct_filt = neuron_medians["percentage"].isna()
            else:
                pct_filt = neuron_medians["percentage"] == pct_val
        nm_row = neuron_medians[m_filt & n_filt & s_filt & pct_filt]
        nm_val = float(nm_row["neurons_median"].iloc[0]) if not nm_row.empty else float("nan")

        n_to_80 = _neurons_to_target(group)
        rows.append({
            **kd,
            "rounds_to_80pct":          _rounds_to_target(group),
            "neurons_to_80pct_M":       n_to_80 / 1e6 if not np.isnan(n_to_80) else float("nan"),
            "neurons_per_round_median": nm_val,
            "final_acc":                group.sort_values("round")["mean_acc"].iloc[-1],
        })
    df = pd.DataFrame(rows)
    for c in df.select_dtypes(include=float).columns:
        df[c] = df[c].round(3)
    return df.sort_values(["model", "dataset", "niid", "percentage", "solution"],
                          na_position="last")


def save_csv(summary: pd.DataFrame):
    path = OUT_TAB / "14_neurons_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(summary: pd.DataFrame):
    display = {
        "model": "Model", "solution": "Solution", "percentage": "Pct.",
        "niid": "NIID", "rounds_to_80pct": r"Rounds to 80\%",
        "neurons_to_80pct_M": r"Neurons to 80\% (M)",
        "neurons_per_round_median": "Neurons/round (M)",
        "final_acc": "Final acc.",
    }
    cols = [c for c in display if c in summary.columns]
    best_rounds   = summary["rounds_to_80pct"].dropna().min() if "rounds_to_80pct" in summary else None
    best_neurons  = summary["neurons_to_80pct_M"].dropna().min() if "neurons_to_80pct_M" in summary else None

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
        if col == "neurons_to_80pct_M" and best_neurons is not None and v == best_neurons:
            s = r"\textbf{" + s + "}"
        return s

    header = " & ".join(display[c] for c in cols) + r" \\"
    ncols  = len(cols)
    lines  = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Rounds and neurons (M) to reach 80\% accuracy per technique}",
        r"\label{tab:neurons_summary} \\",
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

    path = OUT_TAB / "14_neurons_summary.tex"
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

    print("Computing cumulative neurons...")
    cumulative     = compute_cumulative_neurons(train)
    data           = merge_accuracy(cumulative, server_eval)
    neuron_medians = compute_neurons_per_round_medians(train)

    if data.empty:
        print("No data after merge (check that train.csv and server_eval.csv share rounds).")
        return

    summary = build_summary(data, neuron_medians)
    print("Summary — rounds and neurons (M) to 80% accuracy:")
    print(summary[["model", "solution", "percentage", "niid",
                   "rounds_to_80pct", "neurons_to_80pct_M",
                   "neurons_per_round_median", "final_acc"]].to_string(index=False))
    print()

    save_csv(summary)
    save_latex(summary)

    for pct in sorted(data["percentage"].dropna().unique()):
        fig = plot_for_percentage(data, neuron_medians, pct, models, niids)
        if fig:
            tag = f"{int(round(pct * 100)):03d}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG / f"14_neurons_acc_p{tag}.{ext}",
                            dpi=150, bbox_inches="tight")
            print(f"Saved 14_neurons_acc_p{tag}.png/pdf")
            plt.close(fig)

    if not data[data["solution"] == "plaintext"].empty:
        fig = plot_for_percentage(data, neuron_medians, None, models, niids)
        if fig:
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG / f"14_neurons_acc_plaintext.{ext}",
                            dpi=150, bbox_inches="tight")
            print("Saved 14_neurons_acc_plaintext.png/pdf")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
