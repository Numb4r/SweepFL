"""
22pt — Frequência de envio por camada (PT-BR), uma figura por
(model, dataset, technique, niid, percentage[, client_fraction]).

Variante traduzida e redesenhada do estudo 22 (22_layer_heatmap_individual.py):
separa também por percentage (o heatmap 2D camada×porcentagem vira uma faixa
de 1 linha por camada na porcentagem escolhida — mesmo formato visual do
estudo 20/pack_frequency_pt), com nomes de solução corretos via
solution_style.py e separação também por client_fraction (tema client_frac);
no tema sparsification_percentage (client_fraction=1.0) o rótulo é omitido.

Métrica: normalized_send_pct = send_pct / n_packs_covering (elimina o viés
de camadas maiores cobrirem mais pacotes) — unifica a métrica usada pelas
duas versões antigas do estudo 22, que divergiam entre si.

Nota: assim como o estudo 20, depende de mask.csv (só técnicas de
empacotamento) — os baselines full-model (FedAvg, CKKS, BFV, BatchCrypt)
nunca aparecem aqui, comportamento esperado.

Saída: output/figures/22_layer_heatmap_pt/{model}_{dataset}_{tech}_niid{n}_p{pct}[_cf{f}].{png,pdf}
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


load_data = _load("load_data")
sstyle = _load("solution_style")
cap5_style = _load("capitulo5_style")

OUT_FIG_STUDY = HERE / "output" / "figures" / "capitulo5_pt" / "22_layer_heatmap_pt"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "client_fraction", "seed"]

# v1: faixa compacta de hoje (mesmo figsize/fontsize de sempre).
# v2: estilo grande do Cap. 5 (título/eixos/legenda no padrão dos demais
#     estudos), figura redimensionada para caber as fontes maiores.
VARIANTS = ("v1", "v2")


def _tok(model, dataset, technique, niid, pct, cf, variant) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid:g}"
    return (f"{model}_{dataset}_{technique}_{niid_s}"
            f"_p{int(round(pct * 100)):03d}{sstyle.cf_suffix(cf)}_{variant}")


# ── Tensor → camada Keras (idêntico aos estudos 21/22) ───────────────────────

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


# ── Snapshot loading (idêntico aos estudos 21/22) ────────────────────────────

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


# ── Core: layer send-count (idêntico ao estudo 22) ───────────────────────────

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


def build_freq_df(masks: pd.DataFrame, layer_info_cache: dict, logs_dir=None) -> pd.DataFrame:
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
                print(f"[aviso] Sem snapshot para ({model_name}, {solution}) — pulando.")
                layer_info_cache[cache_key] = None
            else:
                layer_info_cache[cache_key] = _load_layer_info(npz_path, pack_size)
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
                **meta, "layer_idx": i, "layer_label": labels[i],
                "n_packs_covering": npc, "send_count": cnt, "send_pct": raw_pct,
                "normalized_send_pct": raw_pct / max(npc, 1), "n_rounds": n_rounds,
            })
    if not parts:
        return pd.DataFrame()
    return pd.DataFrame(parts)


def _avg_over_seeds(freq_df: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in ["model", "dataset", "solution", "niid", "percentage",
                          "client_fraction", "layer_idx", "layer_label", "n_packs_covering"]
             if c in freq_df.columns]
    return (
        freq_df.groupby(gcols, dropna=False)[
            ["send_count", "send_pct", "normalized_send_pct", "n_rounds"]
        ].mean().reset_index()
    )


# ── Individual cell plot (heatmap-strip, formato do estudo 20) ──────────────

def plot_cell(s: pd.DataFrame, model, dataset, technique, niid, pct, cf, variant="v1"):
    """variant="v1": faixa compacta de hoje. variant="v2": estilo grande do Cap. 5."""
    s = s.sort_values("layer_idx")
    layer_order = s["layer_label"].tolist()
    n_layers = len(layer_order)
    if n_layers == 0:
        return None

    if variant == "v2":
        cap5_style.apply_style(compact=False)
        fig_w = max(18.0, n_layers * 1.6)
        fig, ax = plt.subplots(figsize=(fig_w, 6.0))
        cbar_label_fs, cbar_tick_fs = 24, 20
        tick_fs = 20
        xlabel_fs, title_fs = None, 26
    else:
        cap5_style.apply_style(compact=True)
        fig_w = max(6.0, n_layers * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, 1.8))
        cbar_label_fs, cbar_tick_fs = 6.5, 6
        tick_fs = max(5, min(9, int(140 / max(n_layers, 1))))
        xlabel_fs, title_fs = 9, 9

    ax.grid(False)

    data = s["normalized_send_pct"].values.reshape(1, -1)
    vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else 100.0
    im = ax.imshow(
        data, aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
        extent=[0, n_layers, 0, 1],
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% norm. envio\n(rodadas/pacote)", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    ax.set_xticks(np.arange(n_layers) + 0.5)
    ax.set_xticklabels(layer_order, rotation=60, ha="right", fontsize=tick_fs)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, n_layers)
    if xlabel_fs is not None:
        ax.set_xlabel("Camada", fontsize=xlabel_fs)
    else:
        ax.set_xlabel("Camada")

    pct_lbl  = f"{pct:.0%}"
    niid_lbl = f"niid={niid:g}" if niid is not None else "niid=NA"
    ax.set_title(
        f"Frequência de envio por camada — {model} / {dataset} / {sstyle.label(technique)}  "
        f"|  {pct_lbl}  |  {niid_lbl}{sstyle.cf_title(cf)}",
        fontsize=title_fs,
    )
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    print("Carregando dados de máscara...")
    try:
        masks = load_data.load_masks(logs_dir)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return

    if masks.empty:
        print("Sem dados de máscara.")
        return
    masks = masks[masks["percentage"].notna()].copy()
    if masks.empty:
        print("Sem dados de máscara de técnicas de empacotamento.")
        return
    if "client_fraction" not in masks.columns:
        masks["client_fraction"] = 1.0
    masks["client_fraction"] = masks["client_fraction"].fillna(1.0)

    dims        = load_data.discover_dims(masks)
    models      = dims.get("model", [])
    datasets    = dims.get("dataset", [])
    solutions   = dims.get("solution", [])
    niids       = dims.get("niid", [])
    percentages = sorted(masks["percentage"].dropna().unique())
    cfs         = sorted(masks["client_fraction"].dropna().unique().tolist()) or [1.0]
    print(f"Modelos      : {models}")
    print(f"Datasets     : {datasets}")
    print(f"Soluções     : {solutions}")
    print(f"Porcentagens : {percentages}")
    print(f"Frações cli. : {cfs}")
    print()

    print("Calculando frequência de envio por camada...")
    layer_info_cache: dict = {}
    freq_df = build_freq_df(masks, layer_info_cache, logs_dir)
    if freq_df.empty:
        print("Nenhum dado de frequência por camada calculado (snapshots ausentes?).")
        return

    freq_avg = _avg_over_seeds(freq_df)

    n_saved = 0
    for model in models:
        for dataset in datasets:
            for technique in solutions:
                for niid in niids:
                    for pct in percentages:
                        for cf in cfs:
                            s = freq_avg.copy()
                            if "model" in s.columns:
                                s = s[s["model"] == model]
                            if "dataset" in s.columns:
                                s = s[s["dataset"] == dataset]
                            if "solution" in s.columns:
                                s = s[s["solution"] == technique]
                            if niid is not None and "niid" in s.columns:
                                s = s[s["niid"] == niid]
                            if "percentage" in s.columns:
                                s = s[s["percentage"] == pct]
                            if "client_fraction" in s.columns:
                                s = s[s["client_fraction"] == cf]
                            if s.empty:
                                continue

                            for variant in VARIANTS:
                                fig = plot_cell(s, model, dataset, technique, niid, pct, cf, variant)
                                if fig is None:
                                    continue
                                fname = _tok(model, dataset, technique, niid, pct, cf, variant)
                                for ext in ("png", "pdf"):
                                    fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                                dpi=300, bbox_inches="tight", pad_inches=0.08)
                                print(f"Salvo 22_layer_heatmap_pt/{fname}.png/pdf")
                                plt.close(fig)
                                n_saved += 1

    print(f"Total de figuras: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estudo 22pt: frequência de envio por camada, PT-BR e separado por config")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
