"""
15i — Cumulative bytes vs accuracy: one individual figure per (model, dataset, niid, pct).

Same data and layout as study 15 (15_bytes_vs_maxacc.py), but instead of a combined grid
with all models/niids in one figure, each (model, dataset, niid, percentage) combination
is saved as its own figure (two panels: accuracy curve + max-accuracy bar).

Output: output/figures/15_bytes_maxacc_individual/{model}_{dataset}_{tech}_niid{n}_p{pct}.{png,pdf}
        (one file per (model, dataset, niid, pct) — all techniques overlaid in that file)
No tables generated (use 15_bytes_vs_maxacc.py for tables).
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "15_bytes_maxacc_individual"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

PACKING_SOLS = {"sweepfl", "sw", "rw", "fedphe", "fedphe-scout", "sweepfl-scout",
                "topkrr", "srr"}

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

DIMS      = ["model", "dataset", "solution", "niid", "percentage"]
SEED_DIMS = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _sol_color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


def _sol_marker(sol: str) -> str:
    return SOL_MARKERS.get(sol, "o")


def _tok(model, dataset, niid, pct=None) -> str:
    """Build filename token from (model, dataset, niid[, pct])."""
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    parts = [str(model), str(dataset), niid_s]
    if pct is not None:
        parts.append(f"p{int(round(pct * 100)):03d}")
    return "_".join(parts)


# ── Byte estimation ───────────────────────────────────────────────────────────

def _est_bytes_per_client(row) -> float:
    sol = str(row.get("solution", ""))
    if sol in PACKING_SOLS:
        packs = float(row.get("packs_sent") or 0)
        pack_size = float(row.get("pack_size") or 0)
        return packs * pack_size * 4
    else:
        n = float(row.get("n_params") or 0)
        return n * 4


def compute_cumulative_bytes(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    df["bytes_client"] = df.apply(_est_bytes_per_client, axis=1)

    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["bytes_client"]
        .sum()
        .reset_index()
        .rename(columns={"bytes_client": "bytes_round"})
    )

    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_mb"] = (
        per_round.groupby(seed_group, dropna=False)["bytes_round"]
        .cumsum() / 1e6
    )
    return per_round


def merge_accuracy(cumulative: pd.DataFrame, server_eval: pd.DataFrame) -> pd.DataFrame:
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


# ── Peak helpers ──────────────────────────────────────────────────────────────

def _peak_row(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("round")
    return group.loc[group["mean_acc"].idxmax()]


def _max_acc(group: pd.DataFrame) -> float:
    return float(group["mean_acc"].max())


def _peak_round(group: pd.DataFrame) -> float:
    return float(_peak_row(group)["round"])


def _cost_at_peak(group: pd.DataFrame) -> float:
    return float(_peak_row(group)["cumulative_mb"])


def _clip_at_peak(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("round")
    if group.empty:
        return group
    return group[group["round"] <= _peak_round(group)]


def _ref_mb_per_round(sub: pd.DataFrame) -> float:
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_mb"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


def _draw_maxacc_bar(ax, stats: list):
    if not stats:
        ax.set_visible(False)
        return

    stats = sorted(stats, key=lambda t: t[1])
    sols   = [s[0] for s in stats]
    accs   = [s[1] * 100 for s in stats]
    rounds = [s[2] for s in stats]
    costs  = [s[3] for s in stats]
    colors = [_sol_color(s) for s in sols]

    y = np.arange(len(sols))
    bars = ax.barh(y, accs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    lo = max(0.0, (min(accs) // 5) * 5 - 5)
    hi = lo + (100 - lo) * 1.55
    ax.set_xlim(lo, hi)
    off = (hi - lo) * 0.012
    for bar, acc, rnd, cost in zip(bars, accs, rounds, costs):
        ax.text(
            bar.get_width() + off,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%  R{int(rnd)}  {cost:.1f}MB",
            va="center", ha="left", fontsize=7.5, fontweight="bold", color="#222222",
        )

    ax.axvline(100, color="#bbbbbb", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(sols, fontsize=8)
    ax.set_xlabel("Max accuracy (%)", fontsize=8)
    ax.set_title("Max accuracy reached\n(label: acc · peak round · MB)", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(combo: pd.DataFrame, model: str, dataset: str,
              niid, pct: float) -> plt.Figure | None:
    """Two-panel figure (curve + bar) for a single (model, dataset, niid, pct) cell."""
    if combo.empty:
        return None

    fig, (ax, ax_bar) = plt.subplots(
        1, 2,
        figsize=(16, 5.5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    ax_r = ax.twinx()

    solutions = sorted(combo["solution"].dropna().unique().tolist())
    ref_bpr   = _ref_mb_per_round(combo)

    stats    = []
    max_mb   = 0.0
    best_acc = 0.0
    best_sol = None

    for sol in solutions:
        sg = combo[combo["solution"] == sol].sort_values("round")
        if sg.empty:
            continue

        clipped = _clip_at_peak(sg)
        color   = _sol_color(sol)

        ax.plot(
            clipped["cumulative_mb"],
            clipped["mean_acc"] * 100,
            color=color, linestyle="-", linewidth=2.2,
            label=sol, zorder=3,
        )
        ax_r.plot(
            clipped["cumulative_mb"], clipped["round"],
            color=color, linewidth=0.8, alpha=0.0, zorder=1,
        )

        n_pts = len(clipped)
        step  = max(1, n_pts // 4)
        for _, rr in clipped.iloc[::step].iterrows():
            ax.annotate(
                f"R{int(rr['round'])}",
                xy=(rr["cumulative_mb"], rr["mean_acc"] * 100),
                fontsize=6, color=color, alpha=0.75,
                xytext=(3, 2), textcoords="offset points",
            )

        peak = clipped.iloc[-1]
        ax.plot(
            peak["cumulative_mb"], peak["mean_acc"] * 100,
            marker="*", color=color, markersize=13,
            markeredgecolor="white", markeredgewidth=0.5,
            zorder=5, linestyle="",
        )

        s_max   = _max_acc(sg)
        s_round = _peak_round(sg)
        s_cost  = _cost_at_peak(sg)
        stats.append((sol, s_max, s_round, s_cost))
        max_mb = max(max_mb, clipped["cumulative_mb"].max())
        if s_max > best_acc:
            best_acc, best_sol = s_max, sol

    if best_sol is not None:
        ax.axhline(
            best_acc * 100, color="#444444", linestyle="--",
            linewidth=1.4, alpha=0.8,
            label=f"best: {best_sol} {best_acc:.1%}",
        )

    if ref_bpr > 0 and max_mb > 0:
        max_r_est = max_mb / ref_bpr
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
    ax.set_title(f"{model} / {dataset}{niid_lbl}  —  p = {pct_lbl}", fontsize=10)
    ax.set_xlabel("Cumulative upload (MB, raw float32 pre-encryption)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    _draw_maxacc_bar(ax_bar, stats)

    fig.suptitle(
        f"Cumulative Data vs Accuracy (techniques only)  |  "
        f"{model} / {dataset}  niid={niid}  p={pct_lbl}\n"
        f"(★ = each technique's peak  |  bytes = raw float32 model data)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


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

    dims        = load_data.discover_dims(train)
    models      = dims.get("model", [])
    datasets    = dims.get("dataset", [])
    niids       = dims.get("niid", [None])
    percentages = [p for p in dims.get("percentage", []) if p is not None]
    print(f"Models     : {models}")
    print(f"Datasets   : {datasets}")
    print(f"Solutions  : {dims.get('solution', [])}")
    print(f"Percentages: {percentages}")
    print(f"NIIDs      : {niids}")
    print()

    print("Computing cumulative bytes...")
    cumulative = compute_cumulative_bytes(train)
    data = merge_accuracy(cumulative, server_eval)
    data = data[data["solution"] != "plaintext"].copy()

    if data.empty:
        print("No technique data after merge.")
        return

    n_saved = 0
    for pct in sorted(percentages):
        for model in models:
            for dataset in datasets:
                for niid in niids:
                    f = data.copy()
                    if "percentage" in f.columns:
                        f = f[f["percentage"] == pct]
                    if "model" in f.columns:
                        f = f[f["model"] == model]
                    if "dataset" in f.columns:
                        f = f[f["dataset"] == dataset]
                    if niid is not None and "niid" in f.columns:
                        f = f[f["niid"] == niid]
                    if f.empty:
                        continue

                    fig = plot_cell(f, model, dataset, niid, pct)
                    if fig is None:
                        continue

                    fname = _tok(model, dataset, niid, pct)
                    for ext in ("png", "pdf"):
                        fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                    dpi=150, bbox_inches="tight", pad_inches=0.02)
                    print(f"Saved 15_bytes_maxacc_individual/{fname}.png/pdf")
                    plt.close(fig)
                    n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
