"""
22i — Layer transmission frequency heatmap: one individual figure per (model, dataset, technique, niid).

Same data as study 22 (22_layer_heatmap.py), but instead of one figure per (model, technique)
with multiple niid subplots side by side, each (model, dataset, technique, niid) combination
is saved as its own figure (2D heatmap: Y=layer, X=sparsification percentage, color=normalized_send_pct).

Output: output/figures/22_layer_heatmap_individual/{model}_{dataset}_{tech}_niid{n}.{png,pdf}
No tables generated (use 22_layer_heatmap.py for tables).
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "22_layer_heatmap_individual"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _tok(model, dataset, technique, niid) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid}"
    return "_".join([str(model), str(dataset), str(technique), niid_s])


# ── Tensor → Keras-layer grouping (identical to studies 21/22) ───────────────

def _group_tensors(shapes: list) -> list:
    groups = []
    i = 0
    n = len(shapes)
    while i < n:
        s = shapes[i]
        nd = len(s)
        if nd >= 2:
            j = i + 1
            out_ch = s[-1]
            out_ch_dw = s[-2] if nd == 4 and s[-1] == 1 else None
            if j + 3 < n:
                bn_s = shapes[j]
                if (len(bn_s) == 1 and
                        all(shapes[j + k] == bn_s for k in range(4)) and
                        bn_s[0] in {out_ch, out_ch_dw}):
                    groups.append(list(range(i, j + 4)))
                    i = j + 4
                    continue
            if j < n and len(shapes[j]) == 1 and shapes[j][0] in {out_ch, out_ch_dw}:
                groups.append([i, j])
                i = j + 1
                continue
            if nd == 3 and j < n and len(shapes[j]) == 2 and shapes[j] == s[1:]:
                groups.append([i, j])
                i = j + 1
                continue
            groups.append([i])
            i += 1
        elif nd == 1:
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
            else:
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


# ── Snapshot loading (identical to studies 21/22) ────────────────────────────

def _find_snapshot(model_name: str, solution: str, logs_dir=None):
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


def _get_pack_size(folder: Path):
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
    data = np.load(npz_path, allow_pickle=False)
    files = sorted(data.files, key=lambda x: int(x.replace("arr_", "")))
    shapes = [data[f].shape for f in files]
    sizes = [int(np.prod(s)) for s in shapes]
    offsets = np.cumsum([0] + sizes)
    groups = _group_tensors(shapes)
    labels = _label_groups(groups, shapes)
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


# ── Core: layer send-count ────────────────────────────────────────────────────

def _layer_send_counts(group_df: pd.DataFrame, group_packs: list) -> dict:
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


def build_freq_df(masks: pd.DataFrame, layer_info_cache: dict,
                  logs_dir=None) -> pd.DataFrame:
    gcols = [c for c in GROUP if c in masks.columns]
    parts = []
    for key, grp in masks.groupby(gcols, dropna=False):
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        model_name = meta.get("model", "")
        solution   = meta.get("solution", "")
        cache_key  = (model_name, solution)
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
        freq_df.groupby(gcols, dropna=False)[
            ["send_count", "send_pct", "normalized_send_pct", "n_rounds"]
        ]
        .mean()
        .reset_index()
    )


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(sub: pd.DataFrame, model: str, dataset: str,
              technique: str, niid, percentages: list) -> plt.Figure:
    """2D heatmap (Y=layer, X=pct) for one (model, dataset, technique, niid)."""
    sorted_pcts  = sorted(percentages)
    n_pcts       = len(sorted_pcts)

    layer_order = (
        sub.drop_duplicates("layer_idx")
        .sort_values("layer_idx")["layer_label"]
        .tolist()
    ) if not sub.empty else []
    n_layers = len(layer_order)

    fig_h = max(4.0, n_layers * 0.30)
    fig_w = max(3.5, n_pcts * 0.7 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat = np.full((n_layers, n_pcts), np.nan)
    for j, pct in enumerate(sorted_pcts):
        s = sub.copy()
        if "percentage" in s.columns:
            s = s[s["percentage"] == pct]
        s = s.sort_values("layer_idx")
        if not s.empty:
            for _, row in s.iterrows():
                mat[int(row["layer_idx"]), j] = row["normalized_send_pct"]

    vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 100.0

    im = ax.imshow(
        mat, aspect="auto", cmap="viridis",
        vmin=0, vmax=vmax,
        extent=[-0.5, n_pcts - 0.5, n_layers - 0.5, -0.5],
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label="norm. send % (rounds/pack)", pad=0.02)

    ax.set_xticks(range(n_pcts))
    ax.set_xticklabels([f"{p:.0%}" for p in sorted_pcts], fontsize=8)
    ax.set_xlabel("Sparsification", fontsize=9)

    ytick_fs = max(4, min(8, int(180 / max(n_layers, 1))))
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_order, fontsize=ytick_fs)
    ax.set_ylabel("Layer", fontsize=9)

    niid_lbl = f"niid={niid}" if niid is not None else "niid=NA"
    ax.set_title(
        f"Layer heatmap — {model} / {dataset} / {technique}  |  {niid_lbl}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


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

    dims        = load_data.discover_dims(masks)
    models      = dims.get("model", [])
    datasets    = dims.get("dataset", [])
    solutions   = dims.get("solution", [])
    niids       = dims.get("niid", [])
    percentages = sorted(masks["percentage"].dropna().unique())
    print(f"Models     : {models}")
    print(f"Datasets   : {datasets}")
    print(f"Solutions  : {solutions}")
    print(f"Percentages: {percentages}")
    print()

    print("Computing layer send frequencies...")
    layer_info_cache: dict = {}
    freq_df = build_freq_df(masks, layer_info_cache, logs_dir)
    if freq_df.empty:
        print("No layer frequency data computed (snapshots missing?).")
        return

    freq_avg = _avg_over_seeds(freq_df)

    n_saved = 0
    for model in models:
        for dataset in datasets:
            for technique in solutions:
                for niid in niids:
                    s = freq_avg.copy()
                    if "model" in s.columns:
                        s = s[s["model"] == model]
                    if "dataset" in s.columns:
                        s = s[s["dataset"] == dataset]
                    if "solution" in s.columns:
                        s = s[s["solution"] == technique]
                    if niid is not None and "niid" in s.columns:
                        s = s[s["niid"] == niid]
                    if s.empty:
                        continue

                    fig = plot_cell(s, model, dataset, technique, niid, percentages)
                    fname = _tok(model, dataset, technique, niid)
                    for ext in ("png", "pdf"):
                        fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                    dpi=150, bbox_inches="tight", pad_inches=0.02)
                    print(f"Saved 22_layer_heatmap_individual/{fname}.png/pdf")
                    plt.close(fig)
                    n_saved += 1

    print(f"Total figures: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 22i: individual layer transmission frequency heatmap figures")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
