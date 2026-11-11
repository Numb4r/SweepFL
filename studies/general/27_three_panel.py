"""
27 — Three-panel summary: one figure per (model, dataset, niid, percentage).

Recreates the dissertation chapter-5 "three" figures with the new experiment data.
Each figure has 3 side-by-side panels comparing the packing techniques at one
sparsification percentage against every full-model baseline present in the logs
(plaintext, ckks, bfv, batchcrypt — percentage=None folders):

  P1  Accuracy (%) vs FL round          — line per solution
  P2  Mean upload payload per round (MB) — bar per solution
  P3  Mean per-round processing time (s) — stacked bar: train + encrypt + decrypt

Payload note (see docs/log_schema.md "Payload bytes caveat"): for packing
solutions `payload_bytes` only records mask metadata, so P2 uses a consistent
estimated wire size per solution instead:
  packing techniques : packs_sent * (2 * poly * 8)          [ciphertext per pack]
  plaintext          : n_params * 4                          [float32]
  ckks / bfv (full)  : ceil(n_params / (poly/2)) * (2*poly*8)
  batchcrypt         : ceil(n_params / 50) * (2 * n_length / 8)  [Paillier batches]

Baselines that were not run yet are simply absent; re-run this script after the
ckks/bfv/batchcrypt experiments finish to regenerate the figures with them.

Output: output/figures/27_three_panel/{model}_{dataset}_niid{n}_p{pct}.{png,pdf}
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

spec = importlib.util.spec_from_file_location("capitulo5_style", HERE / "capitulo5_style.py")
cap5_style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap5_style)
cap5_style.apply_style()

OUT_BASE = HERE / "output" / "figures" / "27_three_panel"

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
SOL_LABELS = {
    "plaintext":  "FedAvg",
    "sweepfl":       "SweepFL",
    "fedphe":     "FedPHE",
    "sw":         "SW",
    "rw":         "RW",
    "topkrr":     "TopkRR",
    "srr":        "SRR",
    "ckks":       "CKKS",
    "bfv":        "BFV",
    "batchcrypt": "BatchCrypt",
}
# Fixed bar order: full-model baselines first, then the packing techniques
SOL_ORDER = ["plaintext", "ckks", "bfv", "batchcrypt",
             "sweepfl", "fedphe", "sw", "rw", "topkrr", "srr"]

PACKING_SOLS = {"sweepfl", "fedphe", "sw", "rw", "topkrr", "srr",
                "sweepfl-scout", "fedphe-scout"}
PAILLIER_N_LENGTH = 2048   # bits (N_LENGTH used by the batchcrypt experiments)
PAILLIER_BATCH    = 50     # values packed per Paillier ciphertext (batchcrypt.py)

TIME_OPS = [("train_time_s", "Treino"),
            ("encrypt_time_s", "Cifragem"),
            ("decrypt_time_s", "Decifragem")]

CELL_DIMS = ["model", "dataset", "niid"]
AVG_DIMS  = ["model", "dataset", "solution", "niid", "percentage", "round"]


def _sol_order(sols):
    known = [s for s in SOL_ORDER if s in sols]
    return known + sorted(s for s in sols if s not in SOL_ORDER)


def _pct_label(pct) -> str:
    return f"p{int(round(pct * 100)):03d}"


def _cell_mask(df, model, dataset, niid, pct):
    """Rows for the packing techniques at `pct` plus every full-model baseline."""
    pct_mask = df["percentage"].notna() & (df["percentage"] == pct)
    full_mask = df["percentage"].isna()
    m = pct_mask | full_mask
    m &= df["model"] == model
    m &= df["dataset"] == dataset
    m &= df["niid"] == niid
    return m


def build_acc(server_eval: pd.DataFrame) -> pd.DataFrame:
    """mean_acc averaged over seeds/reruns per (config..., round)."""
    for col in ["round", "niid", "seed", "mean_acc"]:
        if col in server_eval.columns:
            server_eval[col] = pd.to_numeric(server_eval[col], errors="coerce")
    avg_cols = [c for c in AVG_DIMS if c in server_eval.columns]
    return (
        server_eval.groupby(avg_cols, dropna=False)["mean_acc"]
        .mean()
        .reset_index()
    )


def _est_wire_bytes(row) -> float:
    """Estimated upload bytes per client per round (see module docstring)."""
    sol      = str(row.get("solution", ""))
    measured = float(row.get("payload_bytes") or 0)
    n_params = float(row.get("n_params") or 0)
    poly     = float(row.get("poly_modulus_degree") or 0)
    ct_bytes = 2 * poly * 8  # one ciphertext of the chapter-4 lib (minimal arm)

    if sol in PACKING_SOLS:
        return float(row.get("packs_sent") or 0) * ct_bytes
    if sol == "plaintext":
        return measured if measured > 0 else n_params * 4
    if sol == "batchcrypt":
        if measured > 10_000:
            return measured
        return np.ceil(n_params / PAILLIER_BATCH) * (2 * PAILLIER_N_LENGTH / 8)
    if sol in ("ckks", "bfv"):
        if measured > 10_000:
            return measured
        if poly > 0:
            return np.ceil(n_params / (poly / 2)) * ct_bytes
    return measured


def build_train_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Per-config means over all rounds/clients: payload MB + per-op times."""
    cols = ["payload_bytes", "n_params", "packs_sent", "poly_modulus_degree"] \
        + [c for c, _ in TIME_OPS]
    for col in cols + ["niid"]:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], errors="coerce")
    train["wire_bytes"] = train.apply(_est_wire_bytes, axis=1)
    group_cols = [c for c in ["model", "dataset", "solution", "niid", "percentage"]
                  if c in train.columns]
    present = ["wire_bytes"] + [c for c, _ in TIME_OPS if c in train.columns]
    stats = train.groupby(group_cols, dropna=False)[present].mean().reset_index()
    stats["payload_mb"] = stats["wire_bytes"] / 1e6
    return stats


def _panel_accuracy(ax, acc, model, dataset, niid, pct):
    sub = acc[_cell_mask(acc, model, dataset, niid, pct)]
    if sub.empty:
        return False
    for sol in _sol_order(sub["solution"].dropna().unique().tolist()):
        sg = sub[sub["solution"] == sol].sort_values("round")
        if sg.empty:
            continue
        is_baseline = sg["percentage"].isna().all()
        ax.plot(
            sg["round"], sg["mean_acc"] * 100,
            color=SOL_COLORS.get(sol, "#888888"),
            linestyle="--" if is_baseline else "-",
            linewidth=1.5 if is_baseline else 2.0,
            marker=SOL_MARKERS.get(sol, "o") if not is_baseline else None,
            markersize=4, markevery=max(1, sg["round"].nunique() // 10),
            label=SOL_LABELS.get(sol, sol),
            zorder=2 if is_baseline else 3,
        )
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    return True


def _bar_axes(ax, sols):
    ax.set_xticks(range(len(sols)))
    ax.set_xticklabels([SOL_LABELS.get(s, s) for s in sols], rotation=30, ha="right")
    ax.grid(True, axis="y")


def _panel_payload(ax, stats, model, dataset, niid, pct):
    sub = stats[_cell_mask(stats, model, dataset, niid, pct)]
    sub = sub[sub.get("payload_mb").notna()] if "payload_mb" in sub.columns else sub.iloc[0:0]
    if sub.empty:
        ax.set_axis_off()
        return
    sols = _sol_order(sub["solution"].dropna().unique().tolist())
    vals = [sub[sub["solution"] == s]["payload_mb"].mean() for s in sols]
    ax.bar(range(len(sols)), vals,
           color=[SOL_COLORS.get(s, "#888888") for s in sols])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=16)
    ax.set_ylabel("Payload / rodada (MB)")
    _bar_axes(ax, sols)


def _panel_overhead(ax, stats, model, dataset, niid, pct):
    sub = stats[_cell_mask(stats, model, dataset, niid, pct)]
    if sub.empty:
        ax.set_axis_off()
        return
    sols = _sol_order(sub["solution"].dropna().unique().tolist())
    op_colors = plt.cm.get_cmap("Set2", len(TIME_OPS))
    bottom = np.zeros(len(sols))
    for k, (col, label) in enumerate(TIME_OPS):
        if col not in sub.columns:
            continue
        vals = np.array([sub[sub["solution"] == s][col].mean() for s in sols])
        vals = np.nan_to_num(vals)
        ax.bar(range(len(sols)), vals, bottom=bottom,
               color=op_colors(k), label=label)
        bottom += vals
    for i, v in enumerate(bottom):
        ax.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=16)
    ax.set_ylabel("Tempo / rodada (s)")
    _bar_axes(ax, sols)


def plot_cell(acc, stats, model, dataset, niid, pct):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5))
    if not _panel_accuracy(axes[0], acc, model, dataset, niid, pct):
        plt.close(fig)
        return None
    _panel_payload(axes[1], stats, model, dataset, niid, pct)
    _panel_overhead(axes[2], stats, model, dataset, niid, pct)

    sol_handles, sol_labels = axes[0].get_legend_handles_labels()
    time_handles, time_labels = axes[2].get_legend_handles_labels()

    fig.tight_layout(rect=[0, 0.22, 1, 0.94])
    fig.legend(sol_handles, sol_labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.1), ncol=len(sol_labels),
               frameon=True)
    if time_handles:
        fig.legend(time_handles, time_labels, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=len(time_labels),
                   frameon=True)
    return fig


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    server_eval, train = dfs["server_eval"], dfs["train"]
    if server_eval.empty or train.empty:
        print("No data found.")
        return

    acc = build_acc(server_eval)
    stats = build_train_stats(train)

    dims = load_data.discover_dims(acc)
    percentages = [p for p in dims.get("percentage", []) if p is not None]
    print(f"Models      : {dims.get('model', [])}")
    print(f"Datasets    : {dims.get('dataset', [])}")
    print(f"NIIDs       : {dims.get('niid', [])}")
    print(f"Percentages : {percentages}")
    baselines = sorted(acc[acc['percentage'].isna()]['solution'].unique())
    print(f"Full-model baselines present: {baselines}")
    print()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for model in dims.get("model", []):
        for dataset in dims.get("dataset", []):
            for niid in dims.get("niid", []):
                for pct in percentages:
                    fig = plot_cell(acc, stats, model, dataset, niid, pct)
                    if fig is None:
                        continue
                    fname = f"{model}_{dataset}_niid{niid:g}_{_pct_label(pct)}"
                    for ext in ("png", "pdf"):
                        fig.savefig(OUT_BASE / f"{fname}.{ext}",
                                    dpi=300, bbox_inches="tight", pad_inches=0.25)
                    print(f"  Saved {fname}.png/pdf")
                    plt.close(fig)
                    n_saved += 1

    print(f"\nTotal figures saved: {n_saved}")
    print(f"Output: {OUT_BASE.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 27: 3-panel figure (accuracy, payload, overhead) per config")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
