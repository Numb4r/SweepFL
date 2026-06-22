"""
21 — Layer transmission frequency (horizontal bar chart).

Answers: over all rounds, how often was each model layer transmitted (i.e., had at least
one of its covering packs included in the server's union for that round)?

Metric: a layer is "sent" in a round if any pack that overlaps its flat-parameter range
appears in the server-side union of selected packs for that round.
send_pct[layer] = (rounds where layer was sent) / total_rounds × 100%

Layers are Keras layers (not individual weight tensors). Each Keras layer corresponds to
one or more consecutive weight tensors from model.get_weights():
  - Dense / Conv2D with bias    → kernel (ND) + bias (1D)        = 2 tensors
  - Conv2D / DepthwiseConv + BN → kernel (ND) + γ,β,μ,σ² (4×1D) = 5 tensors
  - LayerNorm / standalone norm → γ + β (2×1D)                   = 2 tensors
  - ViT attention projection    → weight (3D) + bias (2D)         = 2 tensors

Grouping is inferred from tensor shapes (no TF required).

Source data: mask.csv (via load_data.load_masks) + one model snapshot (.npz) per config.
pack_size from train.csv.

Layout: 1 figure per (model, technique), grid rows=percentage × cols=niid.
        Each subplot = horizontal bar chart (X = % rounds sent, Y = Keras layer).
        Bar annotation: number of packs covering that layer ("N pk").

Output: output/figures/21_layer_frequency/{model}_{technique}.{png,pdf}
Tables: output/tables/21_layer_frequency_summary.{csv,tex}
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
OUT_FIG_STUDY = OUT_FIG / "21_layer_frequency"
OUT_TAB = HERE / "output" / "tables"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _pct_colors(percentages: list) -> dict:
    cm = plt.colormaps.get_cmap("viridis")
    n = max(len(percentages) - 1, 1)
    return {p: cm(i / n) for i, p in enumerate(sorted(percentages))}


# ── Tensor → Keras-layer grouping (shape-based, no TF) ───────────────────────

def _group_tensors(shapes: list) -> list:
    """Group consecutive weight tensors into Keras layers by shape heuristics.

    Rules (applied in priority order per group start):
    1. ND kernel + 4× same 1D  → Conv/Dense + BatchNorm   (5 tensors)
    2. ND kernel + 1× 1D bias  → Conv/Dense + bias         (2 tensors)
    3. 3D tensor + 2D matching last 2 dims → attention proj + bias (ViT)
    4. 1D + same 1D            → LayerNorm or norm pair    (2 tensors)
    5. Singleton
    """
    groups = []
    i = 0
    n = len(shapes)

    while i < n:
        s = shapes[i]
        nd = len(s)

        if nd >= 2:
            j = i + 1
            out_ch = s[-1]
            # depthwise conv: (H,W,in,1) — output channels = in_channels
            out_ch_dw = s[-2] if nd == 4 and s[-1] == 1 else None

            # Rule 1: kernel + BatchNorm (4 same-shape 1D tensors)
            if j + 3 < n:
                bn_s = shapes[j]
                if (len(bn_s) == 1 and
                        all(shapes[j + k] == bn_s for k in range(4)) and
                        bn_s[0] in {out_ch, out_ch_dw}):
                    groups.append(list(range(i, j + 4)))
                    i = j + 4
                    continue

            # Rule 2: kernel + bias (single 1D matching output size)
            if (j < n and len(shapes[j]) == 1 and
                    shapes[j][0] in {out_ch, out_ch_dw}):
                groups.append([i, j])
                i = j + 1
                continue

            # Rule 3: 3D attention weight (embed,heads,head_dim) + 2D bias (heads,head_dim)
            if nd == 3 and j < n and len(shapes[j]) == 2 and shapes[j] == s[1:]:
                groups.append([i, j])
                i = j + 1
                continue

            # Standalone ND tensor (no matching follower)
            groups.append([i])
            i += 1

        elif nd == 1:
            # Rule 4: 1D pair of identical shape (LayerNorm γ+β, BN pair, etc.)
            if i + 1 < n and len(shapes[i + 1]) == 1 and shapes[i + 1] == s:
                groups.append([i, i + 1])
                i += 2
            else:
                groups.append([i])
                i += 1

        else:
            groups.append([i])
            i += 1

    return groups


def _label_groups(groups: list, shapes: list) -> list:
    """Assign human-readable names to tensor groups (e.g. 'Conv2D_1', 'Dense_2')."""
    counters: dict = {}
    labels = []

    for g in groups:
        s = shapes[g[0]]
        nd = len(s)
        n_t = len(g)

        if nd >= 2:
            if nd == 4 and s[-1] == 1:
                base = "DW_Conv"
            elif nd == 4:
                base = "Conv2D"
            elif nd == 3 and n_t == 2:
                base = "Attn"
            elif nd == 3:
                base = "Conv3D"
            else:  # nd == 2
                base = "Dense"

            if n_t >= 5:
                base += "+BN"
        elif nd == 1:
            base = "Norm" if n_t == 2 else "Bias"
        else:
            base = "Tensor"

        counters[base] = counters.get(base, 0) + 1
        labels.append(f"{base}_{counters[base]}")

    return labels


# ── Snapshot loading ──────────────────────────────────────────────────────────

def _find_snapshot(model_name: str, solution: str, logs_dir=None):
    """Find any .npz snapshot for this (model, solution) combo. Returns (path, pack_size)."""
    root = load_data._resolve_logs_dir(logs_dir)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        meta = load_data._parse_folder(folder.name)
        if meta is None:
            continue
        if meta["model"] != model_name or meta["solution"] != solution.lower():
            continue
        model_dir = folder / "model"
        if not model_dir.exists():
            continue
        npz_files = sorted(
            model_dir.glob("round_*.npz"),
            key=lambda p: int(p.stem.replace("round_", ""))
        )
        if not npz_files:
            continue
        pack_size = _get_pack_size(folder)
        if pack_size is None:
            continue
        return npz_files[-1], pack_size
    return None, None


def _get_pack_size(folder: Path) -> int | None:
    csv = folder / "train.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv, nrows=5)
        if "pack_size" in df.columns:
            val = pd.to_numeric(df["pack_size"], errors="coerce").dropna()
            if not val.empty:
                return int(val.iloc[0])
        if "poly_modulus_degree" in df.columns:
            val = pd.to_numeric(df["poly_modulus_degree"], errors="coerce").dropna()
            if not val.empty:
                return int(val.iloc[0]) // 2
    except Exception:
        pass
    return None


def _load_layer_info(npz_path: Path, pack_size: int):
    """Load snapshot and return Keras-layer groups with their pack coverage.

    Returns:
        groups      : list of list[int]  — tensor indices per Keras layer
        labels      : list of str        — human-readable name per group
        group_packs : list of set        — pack indices that cover the group
    """
    data = np.load(npz_path, allow_pickle=False)
    files = sorted(data.files, key=lambda x: int(x.replace("arr_", "")))
    shapes = [data[f].shape for f in files]
    sizes = [int(np.prod(s)) for s in shapes]
    offsets = np.cumsum([0] + sizes)

    groups = _group_tensors(shapes)
    labels = _label_groups(groups, shapes)

    # Pack set per group = union of pack ranges of all member tensors
    group_packs = []
    for g in groups:
        pset = set()
        for idx in g:
            if sizes[idx] == 0:
                continue
            sp = int(offsets[idx]) // pack_size
            ep = (int(offsets[idx + 1]) - 1) // pack_size
            pset.update(range(sp, ep + 1))
        group_packs.append(pset)

    return groups, labels, group_packs


# ── Core: layer send-count per (config+seed) ─────────────────────────────────

def _layer_send_counts(group_df: pd.DataFrame, group_packs: list) -> dict:
    """Count rounds where each Keras layer had at least one pack transmitted."""
    counts = [0] * len(group_packs)
    n_rounds = 0

    for rnd, rnd_df in group_df.sort_values("round").groupby("round", sort=False):
        server_union = set()
        for sel in rnd_df["selected"]:
            server_union.update(sel)
        for i, pset in enumerate(group_packs):
            if pset & server_union:
                counts[i] += 1
        n_rounds += 1

    return {"counts": counts, "n_rounds": n_rounds}


def build_freq_df(masks: pd.DataFrame,
                  layer_info_cache: dict,
                  logs_dir=None) -> pd.DataFrame:
    gcols = [c for c in GROUP if c in masks.columns]
    parts = []

    for key, grp in masks.groupby(gcols, dropna=False):
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        model_name = meta.get("model", "")
        solution   = meta.get("solution", "")

        cache_key = (model_name, solution)
        if cache_key not in layer_info_cache:
            npz_path, pack_size = _find_snapshot(model_name, solution, logs_dir)
            if npz_path is None:
                print(f"[warn] No snapshot for ({model_name}, {solution}) — skipping.")
                layer_info_cache[cache_key] = None
            else:
                groups, labels, group_packs = _load_layer_info(npz_path, pack_size)
                layer_info_cache[cache_key] = (groups, labels, group_packs)

        info = layer_info_cache[cache_key]
        if info is None:
            continue
        groups, labels, group_packs = info

        result = _layer_send_counts(grp, group_packs)
        n_rounds = result["n_rounds"]
        if n_rounds == 0:
            continue

        for i, cnt in enumerate(result["counts"]):
            npc = len(group_packs[i])
            raw_pct = cnt / n_rounds * 100.0
            parts.append({
                **meta,
                "layer_idx": i,
                "layer_label": labels[i],
                "n_packs_covering": npc,
                "send_count": cnt,
                "send_pct": raw_pct,
                "normalized_send_pct": raw_pct / max(npc, 1),
                "n_rounds": n_rounds,
            })

    if not parts:
        return pd.DataFrame()
    return pd.DataFrame(parts)


def _avg_over_seeds(freq_df: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in ["model", "dataset", "solution", "niid", "percentage",
                          "layer_idx", "layer_label", "n_packs_covering"]
             if c in freq_df.columns]
    return (
        freq_df.groupby(gcols, dropna=False)[["send_count", "send_pct", "normalized_send_pct", "n_rounds"]]
        .mean()
        .reset_index()
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_model_technique(freq_avg: pd.DataFrame, model: str, technique: str,
                         niids: list, percentages: list,
                         pct_color_map: dict) -> plt.Figure:
    sub = freq_avg.copy()
    if "model" in sub.columns:
        sub = sub[sub["model"] == model]
    if "solution" in sub.columns:
        sub = sub[sub["solution"] == technique]

    sorted_pcts = sorted(percentages)
    n_rows = max(len(sorted_pcts), 1)
    n_cols = max(len(niids), 1)

    n_layers = sub["layer_idx"].nunique() if not sub.empty else 10
    PER_LAYER_IN, MIN_ROW_IN, MAX_ROW_IN = 0.30, 2.5, 16.0
    desired = n_layers * PER_LAYER_IN
    row_height = min(max(MIN_ROW_IN, desired), MAX_ROW_IN)
    in_per_layer = row_height / max(n_layers, 1)
    ytick_fs = max(4, min(8, in_per_layer * 22))
    annot_fs = max(3, ytick_fs - 1)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, row_height * n_rows),
                             squeeze=False)
    fig.suptitle(f"Layer transmission frequency — {model} / {technique}", fontsize=13)

    for r, pct in enumerate(sorted_pcts):
        color = pct_color_map.get(pct, "steelblue")
        for c, niid in enumerate(niids):
            ax = axes[r][c]
            s = sub.copy()
            if "percentage" in s.columns:
                s = s[s["percentage"] == pct]
            if niid is not None and "niid" in s.columns:
                s = s[s["niid"] == niid]
            s = s.sort_values("layer_idx")

            if not s.empty:
                labels = s["layer_label"].tolist()
                y_pos = np.arange(len(s))
                bars = ax.barh(y_pos, s["send_pct"].values, color=color, alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=ytick_fs)
                ax.set_xlim(0, 105)
                ax.invert_yaxis()

                # annotate: number of packs covering this layer
                if "n_packs_covering" in s.columns:
                    for bar, npc in zip(bars, s["n_packs_covering"].values):
                        ax.text(bar.get_width() * 1.01 + 0.05,
                                bar.get_y() + bar.get_height() / 2,
                                f"{int(npc)} pk",
                                va="center", fontsize=annot_fs, color="gray")

            ax.set_title(f"{pct:.0%}  |  niid={niid}", fontsize=9)
            ax.grid(True, axis="x", alpha=0.3)

            if c == 0:
                ax.set_ylabel("Layer", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("% rounds sent", fontsize=8)

    fig.tight_layout()
    return fig


# ── Tables ────────────────────────────────────────────────────────────────────

def save_csv(freq_avg: pd.DataFrame):
    cols = [c for c in ["model", "dataset", "solution", "niid", "percentage",
                         "layer_idx", "layer_label", "n_packs_covering",
                         "send_count", "send_pct", "normalized_send_pct"] if c in freq_avg.columns]
    out = freq_avg[cols].copy()
    for c in out.select_dtypes(include=float).columns:
        out[c] = out[c].round(3)
    path = OUT_TAB / "21_layer_frequency_summary.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(freq_avg: pd.DataFrame):
    display_cols = {
        "model": "Model", "solution": "Solution", "niid": "NIID",
        "percentage": "Pct.", "layer_label": "Layer",
        "n_packs_covering": "Packs", "send_pct": "Send \\% (rounds)",
    }
    cols = [c for c in display_cols if c in freq_avg.columns]

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        if isinstance(val, float):
            return f"{val:.1f}"
        return str(val)

    header = " & ".join(display_cols.get(c, c) for c in cols) + r" \\"
    ncols = len(cols)
    lines = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Layer transmission frequency (Study 21)}",
        r"\label{tab:layer_frequency} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_key = None
    sort_cols = [c for c in ["solution", "model", "niid", "percentage", "layer_idx"]
                 if c in freq_avg.columns]
    for _, row in freq_avg.sort_values(sort_cols).iterrows():
        cur_key = (row.get("solution"), row.get("model"))
        if prev_key is not None and cur_key != prev_key:
            lines.append(r"\midrule")
        prev_key = cur_key
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "21_layer_frequency_summary.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    print("Loading mask data...")
    try:
        masks = load_data.load_masks(logs_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if masks.empty:
        print("No mask data found.")
        return
    masks = masks[masks["percentage"].notna()].copy()
    if masks.empty:
        print("No packing-technique mask data found.")
        return

    dims = load_data.discover_dims(masks)
    models = dims.get("model", [])
    solutions = dims.get("solution", [])
    niids = dims.get("niid", [])
    percentages = sorted(masks["percentage"].dropna().unique())
    print(f"Models     : {models}")
    print(f"Solutions  : {solutions}")
    print(f"Percentages: {percentages}")
    print(f"Mask rows  : {len(masks):,}")
    print()

    print("Computing layer send frequencies...")
    layer_info_cache = {}
    freq_df = build_freq_df(masks, layer_info_cache, logs_dir)
    if freq_df.empty:
        print("No layer frequency data computed (snapshots missing?).")
        return

    freq_avg = _avg_over_seeds(freq_df)
    pct_color_map = _pct_colors(percentages)

    # Print grouping info per model/solution
    for (m, sol), info in layer_info_cache.items():
        if info is None:
            continue
        groups, labels, group_packs = info
        print(f"  {m}/{sol}: {len(groups)} Keras layers "
              f"(from {sum(len(g) for g in groups)} weight tensors)")

    print()

    n_saved = 0
    for model in models:
        for technique in solutions:
            cs = freq_avg.copy()
            if "model" in cs.columns:
                cs = cs[cs["model"] == model]
            if "solution" in cs.columns:
                cs = cs[cs["solution"] == technique]
            if cs.empty:
                continue
            fig = plot_model_technique(freq_avg, model, technique, niids,
                                       percentages, pct_color_map)
            fname = f"{model}_{technique}"
            for ext in ("png", "pdf"):
                fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
            print(f"Saved 21_layer_frequency/{fname}.png/pdf")
            plt.close(fig)
            n_saved += 1
    print(f"Total figures: {n_saved}")
    print()
    save_csv(freq_avg)
    save_latex(freq_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 21: Keras-layer transmission frequency per round")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
