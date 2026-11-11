"""
27 — Three-panel summary: one figure per (model, dataset, niid, percentage).

Recreates the dissertation chapter-5 "three" figures with the new experiment data.
Each figure has 3 side-by-side panels comparing the packing techniques at one
sparsification percentage against every full-model baseline present in the logs
(plaintext, ckks, bfv, batchcrypt — percentage=None folders):

  P1  Accuracy (%) vs FL round          — line per solution
  P2  Mean upload payload per round (MB) — bar per solution
  P3  Mean per-round processing time (s) — stacked bar: train + encrypt + decrypt

Upload accounting: one rule for every bar, bytes on the wire = number of
ciphertexts times ciphertext size. `payload_bytes` is never used, because the
three arms log three different things there (mask metadata for the packing
techniques, serialized TenSEAL output for the references, and sys.getsizeof of a
Python list for plaintext), and mixing them made the old figure disagree with
every row of the paper's full-model table.

  packing techniques : packs_sent * (2 * poly * ell * 8)   [ell = 1, FHEdSoma]
  ckks (full)        : ceil(n_params / (poly/2)) * (2 * poly * 1 * 8)  [ell = 1, as tab:fullmodel]
  bfv (full)         : ceil(n_params / poly) * (2 * poly * (ell - 1) * 8)
                       [BFV batches poly slots per ciphertext, and SEAL keeps the last
                        prime of the chain as the special prime, outside the ciphertext;
                        checked against TenSEAL 0.3.15 on 2026-10-03]
  batchcrypt         : ceil(n_params / 50) * (2 * n_length / 8)  [Paillier]
  plaintext          : n_params * 4                        [float32]

packs_sent comes from mask.csv, not train.csv: client_logs.py truncates with
int() where the mask uses ceil(), logging 6 for a client that sent 7.

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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("load_data", HERE / "load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)

import paper_style as ps  # noqa: E402

spec = importlib.util.spec_from_file_location("capitulo5_style", HERE / "capitulo5_style.py")
cap5_style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap5_style)
# The paper includes these at 0.99\textwidth = 7.15 in. Authoring at 24 in and
# letting LaTeX scale by 0.30 put the bar annotations at 4.8 pt on the page,
# below what Elsevier prints legibly, so they are drawn at final size instead.
cap5_style.apply_style()
plt.rcParams.update({
    "mathtext.fontset": "dejavuserif",
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
})
FIGSIZE = (7.2, 2.5)
ANNOT_FS = 6.0

OUT_BASE = HERE / "output" / "figures" / "27_three_panel"

SOL_LABELS = ps.LABELS
# Fixed bar order: full-model baselines first, then the policies (paper_style)
SOL_ORDER = ps.ORDER

PACKING_SOLS = {"nvnl", "fedphe", "sw", "rw", "topkrr", "srr",
                "nvnl-scout", "fedphe-scout"}
PAILLIER_N_LENGTH = 2048   # bits (N_LENGTH used by the batchcrypt experiments)
PAILLIER_BATCH    = 50     # values packed per Paillier ciphertext (batchcrypt.py)

TIME_OPS = [("train_time_s", "Training"),
            ("encrypt_time_s", "Encryption"),
            ("decrypt_time_s", "Decryption")]

CELL_DIMS = ["model", "dataset", "niid"]
GROUP_KEYS = ["model", "dataset", "solution", "niid", "percentage"]
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


def _ell(row) -> int:
    """Primes in the modulus chain, from the ';'-separated coeff_mod_bits column."""
    raw = str(row.get("coeff_mod_bits") or "").replace(",", ";")
    return max(1, len([p for p in raw.split(";") if p.strip() not in ("", "nan")]))


def _wire_bytes(row) -> float:
    """Upload bytes per client per round under the single accounting above."""
    sol      = str(row.get("solution", ""))
    n_params = float(row.get("n_params") or 0)
    poly     = float(row.get("poly_modulus_degree") or 0)

    if sol == "plaintext":
        return n_params * 4
    if sol == "batchcrypt":
        return np.ceil(n_params / PAILLIER_BATCH) * (2 * PAILLIER_N_LENGTH / 8)
    if poly <= 0:
        return 0.0
    if sol == "bfv":
        # poly slots per ciphertext; the special prime is not part of the ciphertext
        return np.ceil(n_params / poly) * (2 * poly * max(1, _ell(row) - 1) * 8)
    ct_bytes = 2 * poly * (1 if sol == "ckks" else _ell(row)) * 8
    if sol in PACKING_SOLS:
        return float(row.get("packs_sent") or 0) * ct_bytes
    return np.ceil(n_params / (poly / 2)) * ct_bytes


def _mask_packs(masks: pd.DataFrame) -> pd.DataFrame:
    """Mean packets sent per config, read from mask.csv.

    Round 1 is dropped: it ran with 2 clients where every other round ran with 9.
    """
    if masks is None or masks.empty:
        return pd.DataFrame()
    sub = masks[masks["round"] > 1]
    keys = [c for c in GROUP_KEYS if c in sub.columns]
    return (sub.groupby(keys, dropna=False)["packs_sent"]
               .mean().reset_index().rename(columns={"packs_sent": "mask_packs"}))


def build_train_stats(train: pd.DataFrame, masks: pd.DataFrame = None) -> pd.DataFrame:
    """Per-config means over all rounds/clients: upload MB + per-op times."""
    cols = ["n_params", "packs_sent", "poly_modulus_degree"] + [c for c, _ in TIME_OPS]
    for col in cols + ["niid"]:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], errors="coerce")

    keys = [c for c in GROUP_KEYS if c in train.columns]
    packs = _mask_packs(masks)
    if not packs.empty:
        train = train.merge(packs, on=keys, how="left")
        hit = train["mask_packs"].notna()
        train.loc[hit, "packs_sent"] = train.loc[hit, "mask_packs"]

    train["wire_bytes"] = train.apply(_wire_bytes, axis=1)
    present = ["wire_bytes"] + [c for c, _ in TIME_OPS if c in train.columns]
    stats = train.groupby(keys, dropna=False)[present].mean().reset_index()
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
        st = ps.line_style(sol, rounds=True)
        ax.plot(sg["round"], sg["mean_acc"] * 100,
                **st)
    ps.apply_axes(ax)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    return True


def _bar_axes(ax, sols):
    ax.set_xticks(range(len(sols)))
    ax.set_xticklabels([SOL_LABELS.get(s, s) for s in sols], rotation=45, ha="right")
    ps.apply_axes(ax)


def _panel_payload(ax, stats, model, dataset, niid, pct):
    sub = stats[_cell_mask(stats, model, dataset, niid, pct)]
    sub = sub[sub.get("payload_mb").notna()] if "payload_mb" in sub.columns else sub.iloc[0:0]
    if sub.empty:
        ax.set_axis_off()
        return
    sols = _sol_order(sub["solution"].dropna().unique().tolist())
    vals = [sub[sub["solution"] == s]["payload_mb"].mean() for s in sols]
    for i, (sol, v) in enumerate(zip(sols, vals)):
        st = ps.bar_style(sol)
        st.pop("label")
        ax.bar(i, v, **st)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom",
                    fontsize=ANNOT_FS, rotation=90, xytext=(0, 2),
                    textcoords="offset points")
    ax.set_ylabel("Upload per client\nper round (MB)")
    ax.set_ylim(top=max(vals) * 1.28)
    _bar_axes(ax, sols)


def _panel_overhead(ax, stats, model, dataset, niid, pct):
    sub = stats[_cell_mask(stats, model, dataset, niid, pct)]
    if sub.empty:
        ax.set_axis_off()
        return
    sols = _sol_order(sub["solution"].dropna().unique().tolist())
    bottom = np.zeros(len(sols))
    for col, _ in TIME_OPS:
        if col not in sub.columns:
            continue
        vals = np.nan_to_num(np.array([sub[sub["solution"] == s][col].mean() for s in sols]))
        for i, (sol, v) in enumerate(zip(sols, vals)):
            base = ps.COLORS[sol]
            face = ps.tint(base) if col == "train_time_s" else base
            hatch = "////" if col == "decrypt_time_s" else ""
            ax.bar(i, v, bottom=bottom[i], color=face, hatch=hatch, edgecolor="white", linewidth=0.6)
        bottom += vals
    for i, v in enumerate(bottom):
        ax.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom",
                    fontsize=ANNOT_FS, rotation=90, xytext=(0, 2),
                    textcoords="offset points")
    ax.set_ylabel("Time per round (s)")
    ax.set_ylim(top=float(bottom.max()) * 1.28)
    _bar_axes(ax, sols)


def plot_cell(acc, stats, model, dataset, niid, pct):
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    if not _panel_accuracy(axes[0], acc, model, dataset, niid, pct):
        plt.close(fig)
        return None
    _panel_payload(axes[1], stats, model, dataset, niid, pct)
    _panel_overhead(axes[2], stats, model, dataset, niid, pct)

    sol_handles, sol_labels = axes[0].get_legend_handles_labels()
    time_handles = [Patch(facecolor=ps.TIME_SWATCH["training"], edgecolor="white", linewidth=0.6),
                    Patch(facecolor=ps.TIME_SWATCH["encryption"], edgecolor="white", linewidth=0.6),
                    Patch(facecolor=ps.TIME_SWATCH["decryption"], edgecolor="white", linewidth=0.6,
                          hatch="////")]
    time_labels = [label for _, label in TIME_OPS]

    fig.tight_layout()
    fig.legend(sol_handles, sol_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.055), ncol=len(sol_labels),
               frameon=False, handlelength=1.6)
    if time_handles:
        fig.legend(time_handles, time_labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.155), ncol=len(time_labels),
                   frameon=False, handlelength=1.6)
    return fig


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train = dfs["train"]
    # newest run of each config, like tab:fullmodel
    server_eval = dfs["server_eval"]
    if server_eval.empty or train.empty:
        print("No data found.")
        return

    acc = build_acc(server_eval)
    stats = build_train_stats(train, load_data.load_masks(logs_dir))

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
                                    dpi=300, bbox_inches="tight", pad_inches=0.02)
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
