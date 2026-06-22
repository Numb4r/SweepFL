"""
14i — Cumulative neurons transmitted vs accuracy: one individual figure per (model, dataset, niid, pct).

Same data as study 14 (14_neurons_vs_accuracy.py), but instead of one combined figure per
percentage with all (model × niid) rows, each (model, dataset, niid, percentage) combination
is saved as its own 2-panel figure (curve left + neurons/round bar right).

Output: output/figures/14_neurons_acc_individual/{model}_{dataset}_niid{n}_p{pct}.{png,pdf}
        output/figures/14_neurons_acc_individual/{model}_{dataset}_niid{n}_plaintext.{png,pdf}
No tables generated (use 14_neurons_vs_accuracy.py for tables).
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "14_neurons_acc_individual"
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

DIMS      = ["model", "dataset", "solution", "niid", "percentage"]
SEED_DIMS = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _tok(model, dataset, niid, pct=None) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    if pct is None:
        return "_".join([str(model), str(dataset), niid_s, "plaintext"])
    return "_".join([str(model), str(dataset), niid_s, f"p{int(round(pct * 100)):03d}"])


def _sol_color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


def _fmt_millions(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


# ── Neuron estimation ─────────────────────────────────────────────────────────

def _neurons_per_client(row) -> float:
    sol = str(row.get("solution", ""))
    if sol in PACKING_SOLS:
        return float(row.get("packs_sent") or 0) * float(row.get("pack_size") or 0)
    return float(row.get("n_params") or 0)


def compute_cumulative_neurons(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    df["neurons_client"] = df.apply(_neurons_per_client, axis=1)
    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["neurons_client"]
        .sum().reset_index().rename(columns={"neurons_client": "neurons_round"})
    )
    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_neurons"] = (
        per_round.groupby(seed_group, dropna=False)["neurons_round"].cumsum()
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
            cumulative_neurons=("cumulative_neurons", "mean"),
            mean_acc=("mean_acc", "mean"),
        )
        .reset_index()
    )


def compute_neurons_per_round_medians(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    df["neurons_client"] = df.apply(_neurons_per_client, axis=1)
    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["neurons_client"]
        .sum().reset_index().rename(columns={"neurons_client": "neurons_round"})
    )
    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    medians_seed = (
        per_round.groupby(seed_group, dropna=False)["neurons_round"]
        .median().reset_index().rename(columns={"neurons_round": "neurons_median"})
    )
    dim_group = [c for c in DIMS if c in medians_seed.columns]
    return medians_seed.groupby(dim_group, dropna=False)["neurons_median"].mean().reset_index()


def _clip_at_target(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("round")
    above = group[group["mean_acc"] >= TARGET_ACC]
    return group if above.empty else group[group["round"] <= above["round"].min()]


def _rounds_to_target(group: pd.DataFrame) -> float:
    above = group[group["mean_acc"] >= TARGET_ACC]
    return float(above["round"].min()) if not above.empty else float("nan")


def _ref_neurons_per_round(sub: pd.DataFrame) -> float:
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_neurons"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


def _sorted_solutions(sols):
    other = sorted(s for s in sols if s != "plaintext")
    return other + (["plaintext"] if "plaintext" in sols else [])


def _draw_neurons_bar(ax, solutions, neuron_medians: pd.DataFrame,
                      model: str, dataset: str, niid):
    bar_sols = []
    bar_vals = []

    for sol in solutions:
        filt = neuron_medians["solution"] == sol
        if "model"   in neuron_medians.columns: filt &= neuron_medians["model"]   == model
        if "dataset" in neuron_medians.columns: filt &= neuron_medians["dataset"] == dataset
        if "niid"    in neuron_medians.columns and niid is not None:
            filt &= neuron_medians["niid"] == niid
        row = neuron_medians[filt]
        if row.empty:
            continue
        bar_sols.append(sol)
        bar_vals.append(float(row["neurons_median"].iloc[0]))

    if not bar_sols:
        ax.set_visible(False)
        return

    bars = ax.barh(bar_sols, bar_vals,
                   color=[_sol_color(s) for s in bar_sols],
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    max_val = max(bar_vals) if bar_vals else 1
    for bar, val in zip(bars, bar_vals):
        ax.text(val + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                _fmt_millions(val), va="center", ha="left",
                fontsize=8, fontweight="bold")

    ax.set_xlim(right=max_val * 1.45)
    ax.set_xlabel("Neurons/round (median)", fontsize=8)
    ax.set_title("Neurons per round", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_millions(x)))


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(combo: pd.DataFrame, neuron_medians: pd.DataFrame,
              model: str, dataset: str, niid, pct) -> plt.Figure:
    """2-panel figure for one (model, dataset, niid, pct): curve left + neurons bar right."""
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 5.5),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    ax_r = ax.twinx()

    solutions = _sorted_solutions(combo["solution"].dropna().unique().tolist())
    ref_npr   = _ref_neurons_per_round(combo)
    max_n     = 0.0

    for sol in solutions:
        sg = combo[combo["solution"] == sol].sort_values("round")
        if sg.empty:
            continue
        clipped = _clip_at_target(sg)
        color   = _sol_color(sol)
        ls      = "--" if sol == "plaintext" else "-"
        lw      = 1.8  if sol == "plaintext" else 2.2

        ax.plot(clipped["cumulative_neurons"] / 1e6, clipped["mean_acc"] * 100,
                color=color, linestyle=ls, linewidth=lw, label=sol, zorder=3)
        ax_r.plot(clipped["cumulative_neurons"] / 1e6, clipped["round"],
                  color=color, linestyle=ls, linewidth=0.8, alpha=0.0, zorder=1)

        n_pts = len(clipped)
        step = max(1, n_pts // 4)
        for _, rr in clipped.iloc[::step].iterrows():
            ax.annotate(
                f"R{int(rr['round'])}",
                xy=(rr["cumulative_neurons"] / 1e6, rr["mean_acc"] * 100),
                fontsize=6, color=color, alpha=0.75,
                xytext=(3, 2), textcoords="offset points",
            )

        last = clipped.iloc[-1]
        hit  = last["mean_acc"] >= TARGET_ACC
        ax.plot(last["cumulative_neurons"] / 1e6, last["mean_acc"] * 100,
                marker="*" if hit else "x", color=color,
                markersize=11 if hit else 8, zorder=5, linestyle="")

        max_n = max(max_n, clipped["cumulative_neurons"].max() / 1e6)

    ax.axhline(TARGET_ACC * 100, color="gray", linestyle=":", linewidth=1.2,
               alpha=0.7, label=f"{TARGET_ACC:.0%} threshold")

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
    niid_lbl = f"niid={niid}" if niid is not None else "niid=NA"
    ax.set_title(f"{model} / {dataset}  |  {niid_lbl}  —  p = {pct_lbl}", fontsize=10)
    ax.set_xlabel("Cumulative neurons transmitted (M)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    _draw_neurons_bar(ax_bar, solutions, neuron_medians, model, dataset, niid)

    pct_str = f"p = {pct:.0%}" if pct is not None else "plaintext baseline"
    fig.suptitle(
        f"Cumulative Neurons Transmitted vs Accuracy  |  {pct_str}\n"
        f"(★ = reached {TARGET_ACC:.0%}  |  neurons = packs_sent×pack_size for packing, n_params otherwise)",
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

    print("Computing cumulative neurons...")
    cumulative     = compute_cumulative_neurons(train)
    data           = merge_accuracy(cumulative, server_eval)
    neuron_medians = compute_neurons_per_round_medians(train)

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
                    fig = plot_cell(combo, neuron_medians, model, dataset, niid, pct)
                    fname = _tok(model, dataset, niid, pct)
                    for ext in ("png", "pdf"):
                        fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                    dpi=150, bbox_inches="tight", pad_inches=0.02)
                    print(f"Saved 14_neurons_acc_individual/{fname}.png/pdf")
                    plt.close(fig)
                    n_saved += 1

    for model in models:
        for dataset in datasets:
            for niid in niids:
                combo = _select_combo(data, model, dataset, niid, None)
                if combo.empty:
                    continue
                fig = plot_cell(combo, neuron_medians, model, dataset, niid, None)
                fname = _tok(model, dataset, niid, None)
                for ext in ("png", "pdf"):
                    fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                dpi=150, bbox_inches="tight", pad_inches=0.02)
                print(f"Saved 14_neurons_acc_individual/{fname}.png/pdf")
                plt.close(fig)
                n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 14i: individual neurons vs accuracy figures")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
