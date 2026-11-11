"""
16 — Communication time overhead vs accuracy: techniques only, max-accuracy target.

Mirror of study 13 (communication overhead vs accuracy) with three differences:
  * plaintext is excluded — only the sparsification techniques are shown;
  * instead of a fixed 80% target, the reference line is the **maximum accuracy
    reached by the best technique** of each panel;
  * each technique's curve is drawn up to its own peak accuracy (★ marker), and
    the right panel reports, per technique, the maximum accuracy reached together
    with the round at which it peaked and the cumulative overhead cost (s) up to
    that round.

Communication overhead = pack + mask_select + encrypt + serialize + decrypt
(training time excluded — focus is on the privacy/communication cost).

CAVEAT (decrypt time): `decrypt_time_s` is NOT directly comparable to
`encrypt_time_s`. It bundles deserialize + decrypt + re-pack of the full model +
reshape into a single number, whereas the encrypt side reports flatten / pack /
encrypt / serialize as separate columns — so decrypt naturally dominates. The
magnitude of the pure decryption is also backend-dependent (ckks_python_simplify
is markedly slower than tenseal). Additionally, logs produced *before* the
instrumentation fix in client/literature/set_parameters.py timed two
`np.savez_compressed` disk writes (model snapshot + checkpoint) inside the
decrypt window — a logging artifact (~tens of ms), now excluded. The remaining
cost is genuine model reconstruction + decryption.

One figure per sparsification percentage; one figure row per model × niid.

Layout per figure row:
  Left panel  (3/4 width): X = cumulative comm overhead (s), Y-left = accuracy
                            (%), Y-right = round (approx). Curves clipped at each
                            technique's peak (★). Dashed line = best technique's
                            max accuracy in the panel.
  Right panel (1/4 width): Horizontal bars — max accuracy per technique,
                            labelled with peak accuracy, peak round and seconds.

Output: output/figures/16_overhead_maxacc_p{pct}.{png,pdf}
Tables: output/tables/16_overhead_maxacc.{csv,tex}
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

SOL_COLORS = {
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
    "sweepfl": "o", "sw": "s", "rw": "^", "fedphe": "P", "topkrr": "p", "srr": "h",
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

DIMS      = ["model", "dataset", "solution", "niid", "percentage"]
SEED_DIMS = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _sol_color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


def _sol_marker(sol: str) -> str:
    return SOL_MARKERS.get(sol, "o")


def _fmt_secs(s: float) -> str:
    if s >= 100:
        return f"{s:.0f}s"
    if s >= 10:
        return f"{s:.1f}s"
    return f"{s:.2f}s"


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


# ── Peak helpers ──────────────────────────────────────────────────────────────

def _peak_row(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("round")
    return group.loc[group["mean_acc"].idxmax()]


def _max_acc(group: pd.DataFrame) -> float:
    return float(group["mean_acc"].max())


def _peak_round(group: pd.DataFrame) -> float:
    return float(_peak_row(group)["round"])


def _cost_at_peak(group: pd.DataFrame) -> float:
    return float(_peak_row(group)["cumulative_overhead_s"])


def _clip_at_peak(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("round")
    if group.empty:
        return group
    return group[group["round"] <= _peak_round(group)]


def _ref_s_per_round(sub: pd.DataFrame) -> float:
    """Median overhead seconds/round across all experiments in sub."""
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_overhead_s"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


# ── Plot ──────────────────────────────────────────────────────────────────────

def _draw_maxacc_bar(ax, stats: list):
    """Right panel: horizontal bars of max accuracy per technique."""
    if not stats:
        ax.set_visible(False)
        return

    stats = sorted(stats, key=lambda t: t[1])  # ascending by max_acc
    sols   = [s[0] for s in stats]
    accs   = [s[1] * 100 for s in stats]
    rounds = [s[2] for s in stats]
    costs  = [s[3] for s in stats]
    colors = [_sol_color(s) for s in sols]

    y = np.arange(len(sols))
    bars = ax.barh(y, accs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Zoom near the data with headroom on the right for the labels.
    lo = max(0.0, (min(accs) // 5) * 5 - 5)
    hi = lo + (100 - lo) * 1.55
    ax.set_xlim(lo, hi)
    off = (hi - lo) * 0.012
    for bar, acc, rnd, cost in zip(bars, accs, rounds, costs):
        ax.text(
            bar.get_width() + off,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%  R{int(rnd)}  {_fmt_secs(cost)}",
            va="center", ha="left", fontsize=7.5, fontweight="bold", color="#222222",
        )

    ax.axvline(100, color="#bbbbbb", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(sols, fontsize=8)
    ax.set_xlabel("Max accuracy (%)", fontsize=8)
    ax.set_title("Max accuracy reached\n(label: acc · peak round · overhead s)", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)


def plot_for_percentage(data: pd.DataFrame, pct: float,
                        models: list, niids: list) -> plt.Figure | None:
    sub = data[data["percentage"].notna() & (data["percentage"] == pct)].copy()
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
            ax_r = ax.twinx()

            m_filt = (sub["model"] == model) if "model" in sub.columns else pd.Series(True, index=sub.index)
            n_filt = (sub["niid"] == niid)   if "niid"  in sub.columns else pd.Series(True, index=sub.index)
            combo = sub[m_filt & n_filt].copy()

            if combo.empty:
                row_idx += 1
                continue

            solutions = sorted(combo["solution"].dropna().unique().tolist())
            ref_spr = _ref_s_per_round(combo)

            stats    = []
            max_s    = 0.0
            best_acc = 0.0
            best_sol = None

            for sol in solutions:
                sg = combo[combo["solution"] == sol].sort_values("round")
                if sg.empty:
                    continue

                clipped = _clip_at_peak(sg)
                color   = _sol_color(sol)

                ax.plot(
                    clipped["cumulative_overhead_s"],
                    clipped["mean_acc"] * 100,
                    color=color, linestyle="-", linewidth=2.2,
                    label=sol, zorder=3,
                )
                ax_r.plot(
                    clipped["cumulative_overhead_s"], clipped["round"],
                    color=color, linewidth=0.8, alpha=0.0, zorder=1,
                )

                n_pts = len(clipped)
                step  = max(1, n_pts // 4)
                for _, rr in clipped.iloc[::step].iterrows():
                    ax.annotate(
                        f"R{int(rr['round'])}",
                        xy=(rr["cumulative_overhead_s"], rr["mean_acc"] * 100),
                        fontsize=6, color=color, alpha=0.75,
                        xytext=(3, 2), textcoords="offset points",
                    )

                peak = clipped.iloc[-1]
                ax.plot(
                    peak["cumulative_overhead_s"], peak["mean_acc"] * 100,
                    marker="*", color=color, markersize=13,
                    markeredgecolor="white", markeredgewidth=0.5,
                    zorder=5, linestyle="",
                )

                s_max   = _max_acc(sg)
                s_round = _peak_round(sg)
                s_cost  = _cost_at_peak(sg)
                stats.append((sol, s_max, s_round, s_cost))
                max_s = max(max_s, clipped["cumulative_overhead_s"].max())
                if s_max > best_acc:
                    best_acc, best_sol = s_max, sol

            if best_sol is not None:
                ax.axhline(
                    best_acc * 100, color="#444444", linestyle="--",
                    linewidth=1.4, alpha=0.8,
                    label=f"best: {best_sol} {best_acc:.1%}",
                )

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

            pct_lbl  = f"{pct:.0%}"
            niid_lbl = f" | niid={niid}" if niid is not None else ""
            ax.set_title(f"{model}{niid_lbl}  —  p = {pct_lbl}", fontsize=10)
            ax.set_xlabel("Cumulative communication overhead (s)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 105)
            ax.set_xlim(left=0)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, alpha=0.25)

            _draw_maxacc_bar(ax_bar, stats)

            row_idx += 1

    fig.suptitle(
        f"Communication Overhead vs Accuracy (techniques only)  |  p = {pct:.0%}\n"
        f"(dashed = best technique's max accuracy  |  ★ = each technique's peak  |  "
        f"overhead = pack+mask+encrypt+serialize+decrypt)",
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
            "max_acc":        _max_acc(group),
            "round_at_max":   _peak_round(group),
            "seconds_at_max": _cost_at_peak(group),
            "final_acc":      group.sort_values("round")["mean_acc"].iloc[-1],
        })
    df = pd.DataFrame(rows)
    for c in df.select_dtypes(include=float).columns:
        df[c] = df[c].round(3)
    return df.sort_values(["model", "dataset", "niid", "percentage", "max_acc"],
                          ascending=[True, True, True, True, False],
                          na_position="last")


def save_csv(summary: pd.DataFrame):
    path = OUT_TAB / "16_overhead_maxacc.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(summary: pd.DataFrame):
    display = {
        "model": "Model", "solution": "Solution", "percentage": "Pct.",
        "niid": "NIID", "max_acc": r"Max acc.", "round_at_max": "Peak round",
        "seconds_at_max": "Overhead s at peak", "final_acc": "Final acc.",
    }
    cols = [c for c in display if c in summary.columns]
    grp_cols = [c for c in ["model", "niid", "percentage"] if c in summary.columns]
    best_idx = (summary.groupby(grp_cols, dropna=False)["max_acc"].idxmax()
                if "max_acc" in summary.columns and grp_cols else [])
    best_set = set(best_idx) if hasattr(best_idx, "__iter__") else set()

    def _fmt(row, col, is_best):
        v = row[col]
        if pd.isna(v):
            return "—"
        if col == "percentage":
            return f"{v:.0%}"
        if col in ("max_acc", "final_acc"):
            s = f"{v:.3f}"
        elif isinstance(v, float):
            s = f"{v:.1f}" if v >= 10 else f"{v:.2f}"
        else:
            s = str(v)
        if col == "max_acc" and is_best:
            s = r"\textbf{" + s + "}"
        return s

    header = " & ".join(display[c] for c in cols) + r" \\"
    ncols  = len(cols)
    lines  = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Max accuracy, peak round and communication overhead per technique}",
        r"\label{tab:overhead_maxacc} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_pct = None
    for idx, row in summary.iterrows():
        pct = row.get("percentage")
        if prev_pct is not None and pct != prev_pct:
            lines.append(r"\midrule")
        prev_pct = pct
        lines.append(" & ".join(_fmt(row, c, idx in best_set) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "16_overhead_maxacc.tex"
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
    cumulative = compute_cumulative_overhead(train)
    data = merge_accuracy(cumulative, server_eval)

    # Techniques only — drop plaintext everywhere.
    data = data[data["solution"] != "plaintext"].copy()

    if data.empty:
        print("No technique data after merge (only plaintext present?).")
        return

    summary = build_summary(data)
    print("Summary — max accuracy, peak round and overhead (s) per technique:")
    print(summary[["model", "solution", "percentage", "niid",
                   "max_acc", "round_at_max", "seconds_at_max", "final_acc"]].to_string(index=False))
    print()

    save_csv(summary)
    save_latex(summary)

    for pct in sorted(data["percentage"].dropna().unique()):
        fig = plot_for_percentage(data, pct, models, niids)
        if fig:
            tag = f"{int(round(pct * 100)):03d}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG / f"16_overhead_maxacc_p{tag}.{ext}",
                            dpi=150, bbox_inches="tight")
            print(f"Saved 16_overhead_maxacc_p{tag}.png/pdf")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
