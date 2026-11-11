"""
12i — Cumulative bytes transmitted vs accuracy: one individual figure per (model, dataset, niid, pct).

Same data as study 12 (12_bytes_vs_accuracy.py), but instead of one combined figure per
percentage with all (model × niid) rows, each (model, dataset, niid, percentage) combination
is saved as its own 2-panel figure (curve left + rounds-to-80% bar right).

Output: output/figures/12_bytes_acc_individual/{model}_{dataset}_niid{n}_p{pct}.{png,pdf}
        output/figures/12_bytes_acc_individual/{model}_{dataset}_niid{n}_plaintext.{png,pdf}
No tables generated (use 12_bytes_vs_accuracy.py for tables).
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "12_bytes_acc_individual"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

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


def _tok(model, dataset, niid, pct=None) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    if pct is None:
        return "_".join([str(model), str(dataset), niid_s, "plaintext"])
    return "_".join([str(model), str(dataset), niid_s, f"p{int(round(pct * 100)):03d}"])


def _sol_color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


# ── Byte estimation ───────────────────────────────────────────────────────────

def _est_bytes_per_client(row) -> float:
    sol = str(row.get("solution", ""))
    if sol in PACKING_SOLS:
        return float(row.get("packs_sent") or 0) * float(row.get("pack_size") or 0) * 4
    return float(row.get("n_params") or 0) * 4


def compute_cumulative_bytes(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    df["bytes_client"] = df.apply(_est_bytes_per_client, axis=1)
    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["bytes_client"]
        .sum().reset_index().rename(columns={"bytes_client": "bytes_round"})
    )
    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_mb"] = (
        per_round.groupby(seed_group, dropna=False)["bytes_round"].cumsum() / 1e6
    )
    return per_round


def merge_accuracy(cumulative: pd.DataFrame, server_eval: pd.DataFrame) -> pd.DataFrame:
    join_keys = [c for c in SEED_DIMS + ["round"]
                 if c in cumulative.columns and c in server_eval.columns]
    se = server_eval[
        [*join_keys, *(c for c in ["mean_acc"] if c in server_eval.columns)]
    ].copy()
    for c in join_keys:
        if cumulative[c].dtype != se[c].dtype:
            se[c] = pd.to_numeric(se[c], errors="ignore")
    merged = cumulative.merge(se, on=join_keys, how="inner")
    avg_group = [c for c in DIMS + ["round"] if c in merged.columns]
    return (
        merged.groupby(avg_group, dropna=False)
        .agg(cumulative_mb=("cumulative_mb", "mean"), mean_acc=("mean_acc", "mean"))
        .reset_index()
    )


def _clip_at_target(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("round")
    above = group[group["mean_acc"] >= TARGET_ACC]
    return group if above.empty else group[group["round"] <= above["round"].min()]


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


def _ref_mb_per_round(sub: pd.DataFrame) -> float:
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_mb"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


def _sorted_solutions(sols):
    other = sorted(s for s in sols if s != "plaintext")
    return other + (["plaintext"] if "plaintext" in sols else [])


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(combo: pd.DataFrame, model: str, dataset: str,
              niid, pct) -> plt.Figure:
    """2-panel figure for one (model, dataset, niid, pct): curve left + bar right."""
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 5.5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    ax_r = ax.twinx()

    solutions = _sorted_solutions(combo["solution"].dropna().unique().tolist())
    ref_bpr   = _ref_mb_per_round(combo)
    rounds_to_80: dict = {}
    mb_to_80:     dict = {}
    max_mb = 0.0

    for sol in solutions:
        sg = combo[combo["solution"] == sol].sort_values("round")
        if sg.empty:
            continue
        clipped = _clip_at_target(sg)
        color   = _sol_color(sol)
        ls      = "--" if sol == "plaintext" else "-"
        lw      = 1.8  if sol == "plaintext" else 2.2

        ax.plot(clipped["cumulative_mb"], clipped["mean_acc"] * 100,
                color=color, linestyle=ls, linewidth=lw, label=sol, zorder=3)
        ax_r.plot(clipped["cumulative_mb"], clipped["round"],
                  color=color, linestyle=ls, linewidth=0.8, alpha=0.0, zorder=1)

        n_pts = len(clipped)
        step = max(1, n_pts // 4)
        for _, rr in clipped.iloc[::step].iterrows():
            ax.annotate(
                f"R{int(rr['round'])}",
                xy=(rr["cumulative_mb"], rr["mean_acc"] * 100),
                fontsize=6, color=color, alpha=0.75,
                xytext=(3, 2), textcoords="offset points",
            )

        last = clipped.iloc[-1]
        hit  = last["mean_acc"] >= TARGET_ACC
        ax.plot(last["cumulative_mb"], last["mean_acc"] * 100,
                marker="*" if hit else "x", color=color,
                markersize=11 if hit else 8, zorder=5, linestyle="")

        max_mb = max(max_mb, clipped["cumulative_mb"].max())
        rounds_to_80[sol] = _rounds_to_target(sg)
        mb_to_80[sol]     = _mb_to_target(sg)

    ax.axhline(TARGET_ACC * 100, color="gray", linestyle=":", linewidth=1.2,
               alpha=0.7, label=f"{TARGET_ACC:.0%} threshold")

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

    pct_lbl  = f"{pct:.0%}" if pct is not None else "baseline"
    niid_lbl = f"niid={niid}" if niid is not None else "niid=NA"
    ax.set_title(f"{model} / {dataset}  |  {niid_lbl}  —  p = {pct_lbl}", fontsize=10)
    ax.set_xlabel("Cumulative upload (MB, raw float32 pre-encryption)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    valid = [(s, rounds_to_80[s], mb_to_80[s])
             for s in solutions if not np.isnan(rounds_to_80.get(s, float("nan")))]
    if valid:
        bar_sols = [v[0] for v in valid]
        bar_vals = [v[1] for v in valid]
        bar_mbs  = [v[2] for v in valid]
        bars = ax_bar.barh(bar_sols, bar_vals,
                           color=[_sol_color(s) for s in bar_sols],
                           alpha=0.85, edgecolor="white", linewidth=0.5)
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

    never = [s for s in solutions if np.isnan(rounds_to_80.get(s, float("nan")))]
    if never:
        ax_bar.text(0.5, -0.18, "Did not reach 80%: " + ", ".join(never),
                    transform=ax_bar.transAxes, fontsize=7, color="red",
                    ha="center", va="top")

    ax_bar.set_xlabel("Rounds to 80% acc.")
    ax_bar.set_title("Rounds to 80%", fontsize=9)
    ax_bar.grid(True, axis="x", alpha=0.3)
    ax_bar.invert_yaxis()

    pct_str = f"p = {pct:.0%}" if pct is not None else "plaintext baseline"
    fig.suptitle(
        f"Cumulative Data Transferred vs Accuracy  |  {pct_str}\n"
        f"(★ = reached {TARGET_ACC:.0%}  |  bytes = raw float32 model data)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def _select_combo(data: pd.DataFrame, model: str, dataset: str, niid, pct):
    """Filter data to one (model, dataset, niid) cell at the given pct."""
    if pct is None:
        sub = data[data["percentage"].isna()].copy()
    else:
        pct_mask = data["percentage"].notna() & (data["percentage"] == pct)
        pt_mask  = data["percentage"].isna() & (data["solution"] == "plaintext")
        sub = data[pct_mask | pt_mask].copy()
    if "model"   in sub.columns: sub = sub[sub["model"]   == model]
    if "dataset" in sub.columns: sub = sub[sub["dataset"] == dataset]
    if niid is not None and "niid" in sub.columns:
        sub = sub[sub["niid"] == niid]
    return sub


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
    datasets    = dims.get("dataset", [])
    niids       = dims.get("niid", [None])
    percentages = dims.get("percentage", [])
    print(f"Models     : {models}")
    print(f"Datasets   : {datasets}")
    print(f"Percentages: {percentages}")
    print(f"NIIDs      : {niids}")
    print()

    print("Computing cumulative bytes...")
    cumulative = compute_cumulative_bytes(train)
    data       = merge_accuracy(cumulative, server_eval)

    if data.empty:
        print("No data after merge.")
        return

    n_saved = 0

    for pct in sorted(data["percentage"].dropna().unique()):
        for model in models:
            for dataset in datasets:
                for niid in niids:
                    combo = _select_combo(data, model, dataset, niid, pct)
                    if combo.empty:
                        continue
                    fig = plot_cell(combo, model, dataset, niid, pct)
                    fname = _tok(model, dataset, niid, pct)
                    for ext in ("png", "pdf"):
                        fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                    dpi=150, bbox_inches="tight", pad_inches=0.02)
                    print(f"Saved 12_bytes_acc_individual/{fname}.png/pdf")
                    plt.close(fig)
                    n_saved += 1

    for model in models:
        for dataset in datasets:
            for niid in niids:
                combo = _select_combo(data, model, dataset, niid, None)
                if combo.empty:
                    continue
                fig = plot_cell(combo, model, dataset, niid, None)
                fname = _tok(model, dataset, niid, None)
                for ext in ("png", "pdf"):
                    fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                dpi=150, bbox_inches="tight", pad_inches=0.02)
                print(f"Saved 12_bytes_acc_individual/{fname}.png/pdf")
                plt.close(fig)
                n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 12i: individual bytes vs accuracy figures")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
