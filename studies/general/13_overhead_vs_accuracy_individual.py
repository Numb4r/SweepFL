"""
13i — Communication time overhead vs accuracy: one individual figure per (model, dataset, niid, pct).

Same data as study 13 (13_overhead_vs_accuracy.py), but instead of one combined figure per
percentage with all (model × niid) rows, each (model, dataset, niid, percentage) combination
is saved as its own 2-panel figure (curve left + stacked bar right).

Output: output/figures/13_overhead_acc_individual/{model}_{dataset}_niid{n}_p{pct}.{png,pdf}
        output/figures/13_overhead_acc_individual/{model}_{dataset}_niid{n}_plaintext.{png,pdf}
No tables generated (use 13_overhead_vs_accuracy.py for tables).
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "13_overhead_acc_individual"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

TARGET_ACC = 0.80

PACKING_SOLS = {"nvnl", "sw", "rw", "fedphe", "fedphe-scout", "nvnl-scout", "topkrr", "srr"}

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

COMM_COLS = [
    "pack_time_s",
    "mask_select_time_s",
    "encrypt_time_s",
    "serialize_time_s",
    "decrypt_time_s",
]

COMM_OP_GROUPS = [
    ("Pack+Mask",  ["pack_time_s", "mask_select_time_s"]),
    ("Encrypt",    ["encrypt_time_s"]),
    ("Serialize",  ["serialize_time_s"]),
    ("Decrypt",    ["decrypt_time_s"]),
]

BAR_COLORS = [plt.colormaps["Set2"](i) for i in range(len(COMM_OP_GROUPS))]

DIMS      = ["model", "dataset", "solution", "niid", "percentage"]
SEED_DIMS = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _tok(model, dataset, niid, pct=None) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    if pct is None:
        return "_".join([str(model), str(dataset), niid_s, "plaintext"])
    return "_".join([str(model), str(dataset), niid_s, f"p{int(round(pct * 100)):03d}"])


def _sol_color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


# ── Cumulative overhead computation ──────────────────────────────────────────

def compute_cumulative_overhead(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    present = [c for c in COMM_COLS if c in df.columns]
    df["overhead_s"] = df[present].fillna(0.0).sum(axis=1)
    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["overhead_s"]
        .sum().reset_index().rename(columns={"overhead_s": "overhead_round"})
    )
    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_overhead_s"] = (
        per_round.groupby(seed_group, dropna=False)["overhead_round"].cumsum()
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
        .agg(
            cumulative_overhead_s=("cumulative_overhead_s", "mean"),
            mean_acc=("mean_acc", "mean"),
        )
        .reset_index()
    )


def compute_comm_overhead_medians(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    present_cols = [c for g in COMM_OP_GROUPS for c in g[1] if c in df.columns]
    if not present_cols:
        return pd.DataFrame()
    seed_round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = df.groupby(seed_round_group, dropna=False)[present_cols].sum().reset_index()
    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    medians_seed = per_round.groupby(seed_group, dropna=False)[present_cols].median().reset_index()
    dim_group = [c for c in DIMS if c in medians_seed.columns]
    return medians_seed.groupby(dim_group, dropna=False)[present_cols].mean().reset_index()


def _clip_at_target(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("round")
    above = group[group["mean_acc"] >= TARGET_ACC]
    return group if above.empty else group[group["round"] <= above["round"].min()]


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
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_overhead_s"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


def _sorted_solutions(sols):
    other = sorted(s for s in sols if s != "plaintext")
    return other + (["plaintext"] if "plaintext" in sols else [])


def _draw_stacked_bar(ax, solutions, overhead_medians: pd.DataFrame,
                      model: str, dataset: str, niid):
    present_cols = [c for g in COMM_OP_GROUPS for c in g[1]]
    bar_sols = []
    bar_data = {label: [] for label, _ in COMM_OP_GROUPS}

    for sol in solutions:
        filt = overhead_medians["solution"] == sol
        if "model"   in overhead_medians.columns: filt &= overhead_medians["model"]   == model
        if "dataset" in overhead_medians.columns: filt &= overhead_medians["dataset"] == dataset
        if "niid"    in overhead_medians.columns and niid is not None:
            filt &= overhead_medians["niid"] == niid
        row = overhead_medians[filt]
        if row.empty:
            continue
        bar_sols.append(sol)
        for label, cols in COMM_OP_GROUPS:
            val = sum(float(row[c].iloc[0]) if c in row.columns else 0.0 for c in cols)
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

    for yi in range(len(bar_sols)):
        total = bottoms[yi]
        ax.text(total * 1.02, yi, f"{total:.2f}s",
                va="center", ha="left", fontsize=7, color="#333333")
    ax.set_xlim(right=bottoms.max() * 1.35 if bottoms.max() > 0 else 1)


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(combo: pd.DataFrame, overhead_medians: pd.DataFrame,
              model: str, dataset: str, niid, pct) -> plt.Figure:
    """2-panel figure for one (model, dataset, niid, pct): curve left + stacked bar right."""
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 5.5),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    ax_r = ax.twinx()

    solutions = _sorted_solutions(combo["solution"].dropna().unique().tolist())
    ref_spr   = _ref_s_per_round(combo)
    max_s     = 0.0

    for sol in solutions:
        sg = combo[combo["solution"] == sol].sort_values("round")
        if sg.empty:
            continue
        clipped = _clip_at_target(sg)
        color   = _sol_color(sol)
        ls      = "--" if sol == "plaintext" else "-"
        lw      = 1.8  if sol == "plaintext" else 2.2

        ax.plot(clipped["cumulative_overhead_s"], clipped["mean_acc"] * 100,
                color=color, linestyle=ls, linewidth=lw, label=sol, zorder=3)
        ax_r.plot(clipped["cumulative_overhead_s"], clipped["round"],
                  color=color, linestyle=ls, linewidth=0.8, alpha=0.0, zorder=1)

        n_pts = len(clipped)
        step = max(1, n_pts // 4)
        for _, rr in clipped.iloc[::step].iterrows():
            ax.annotate(
                f"R{int(rr['round'])}",
                xy=(rr["cumulative_overhead_s"], rr["mean_acc"] * 100),
                fontsize=6, color=color, alpha=0.75,
                xytext=(3, 2), textcoords="offset points",
            )

        last = clipped.iloc[-1]
        hit  = last["mean_acc"] >= TARGET_ACC
        ax.plot(last["cumulative_overhead_s"], last["mean_acc"] * 100,
                marker="*" if hit else "x", color=color,
                markersize=11 if hit else 8, zorder=5, linestyle="")

        max_s = max(max_s, clipped["cumulative_overhead_s"].max())

    ax.axhline(TARGET_ACC * 100, color="gray", linestyle=":", linewidth=1.2,
               alpha=0.7, label=f"{TARGET_ACC:.0%} threshold")

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
    niid_lbl = f"niid={niid}" if niid is not None else "niid=NA"
    ax.set_title(f"{model} / {dataset}  |  {niid_lbl}  —  p = {pct_lbl}", fontsize=10)
    ax.set_xlabel("Cumulative communication overhead (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    _draw_stacked_bar(ax_bar, solutions, overhead_medians, model, dataset, niid)

    pct_str = f"p = {pct:.0%}" if pct is not None else "plaintext baseline"
    fig.suptitle(
        f"Communication Overhead vs Accuracy  |  {pct_str}\n"
        f"(★ = reached {TARGET_ACC:.0%}  |  overhead = pack+mask+encrypt+serialize+decrypt)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def _select_combo(data: pd.DataFrame, model: str, dataset: str, niid, pct):
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

    print("Computing cumulative communication overhead...")
    cumulative       = compute_cumulative_overhead(train)
    data             = merge_accuracy(cumulative, server_eval)
    overhead_medians = compute_comm_overhead_medians(train)

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
                    fig = plot_cell(combo, overhead_medians, model, dataset, niid, pct)
                    fname = _tok(model, dataset, niid, pct)
                    for ext in ("png", "pdf"):
                        fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                    dpi=150, bbox_inches="tight", pad_inches=0.02)
                    print(f"Saved 13_overhead_acc_individual/{fname}.png/pdf")
                    plt.close(fig)
                    n_saved += 1

    for model in models:
        for dataset in datasets:
            for niid in niids:
                combo = _select_combo(data, model, dataset, niid, None)
                if combo.empty:
                    continue
                fig = plot_cell(combo, overhead_medians, model, dataset, niid, None)
                fname = _tok(model, dataset, niid, None)
                for ext in ("png", "pdf"):
                    fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                dpi=150, bbox_inches="tight", pad_inches=0.02)
                print(f"Saved 13_overhead_acc_individual/{fname}.png/pdf")
                plt.close(fig)
                n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 13i: individual overhead vs accuracy figures")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
