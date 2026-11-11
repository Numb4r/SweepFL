"""
20pt — Frequência de transmissão de pacotes (PT-BR), uma figura por
(model, dataset, technique, niid, percentage[, client_fraction]).

Variante traduzida do estudo 20 (20_pack_frequency_individual.py): mesma
granularidade total já existente, com nomes de solução corretos (SweepFL, SW,
FedPHE, RW) via solution_style.py e separação também por client_fraction
(tema client_frac); no tema sparsification_percentage (client_fraction=1.0)
o rótulo é omitido.

Nota: este estudo usa mask.csv, gerado apenas pelas técnicas de
empacotamento (SweepFL, SW, RW, FedPHE, TopK-RR, SRR). Os baselines full-model
(FedAvg, CKKS, BFV, BatchCrypt) nunca produzem mask.csv, então nunca
aparecerão aqui — comportamento esperado, não uma limitação a corrigir.

Saída: output/figures/20_pack_frequency_pt/{model}_{dataset}_{tech}_niid{n}_p{pct}[_cf{f}].{png,pdf}
"""

import argparse
import importlib.util
import sys
from collections import defaultdict
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "capitulo5_pt" / "20_pack_frequency_pt"
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


# ── Core: pack send-count (idêntico ao estudo 20) ────────────────────────────

def _pack_send_counts(group_df: pd.DataFrame) -> dict:
    n_packs = int(group_df["n_packs"].iloc[0])
    counts = defaultdict(int)
    n_rounds = 0

    for rnd, rnd_df in group_df.sort_values("round").groupby("round", sort=False):
        server_union = set()
        for sel in rnd_df["selected"]:
            server_union.update(sel)
        for idx in server_union:
            counts[idx] += 1
        n_rounds += 1

    return {"counts": dict(counts), "n_rounds": n_rounds, "n_packs": n_packs}


def build_freq_df(masks: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in GROUP if c in masks.columns]
    parts = []
    for key, grp in masks.groupby(gcols, dropna=False):
        result = _pack_send_counts(grp)
        n_packs, n_rounds = result["n_packs"], result["n_rounds"]
        if n_rounds == 0:
            continue
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        for pack_idx in range(n_packs):
            cnt = result["counts"].get(pack_idx, 0)
            parts.append({**meta, "pack_idx": pack_idx, "send_count": cnt,
                           "send_pct": cnt / n_rounds * 100.0, "n_rounds": n_rounds})
    if not parts:
        return pd.DataFrame()
    return pd.DataFrame(parts)


def _avg_over_seeds(freq_df: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in ["model", "dataset", "solution", "niid", "percentage",
                          "client_fraction", "pack_idx"] if c in freq_df.columns]
    return (
        freq_df.groupby(gcols, dropna=False)[["send_count", "send_pct", "n_rounds"]]
        .mean().reset_index()
    )


# ── Individual cell plot ───────────────────────────────────────────────────────

def plot_cell(s: pd.DataFrame, model, dataset, technique, niid, pct, cf, variant="v1"):
    """Faixa de heatmap (1 linha) para um (model, dataset, technique, niid, pct, cf).

    variant="v1": faixa compacta de hoje. variant="v2": estilo grande do Cap. 5.
    """
    if variant == "v2":
        cap5_style.apply_style(compact=False)
        figsize = (24, 5)
        xlabel_fs, cbar_label_fs, cbar_tick_fs = None, 24, 20
    else:
        cap5_style.apply_style(compact=True)
        figsize = (8, 1.6)
        xlabel_fs, cbar_label_fs, cbar_tick_fs = 15, 13, 12

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    ax.set_yticks([])
    if not s.empty:
        n_packs = int(s["pack_idx"].max()) + 1
        data = s.sort_values("pack_idx")["send_pct"].values.reshape(1, -1)
        im = ax.imshow(
            data, aspect="auto", cmap="viridis", vmin=0, vmax=100,
            extent=[0, n_packs, 0, 1],
        )
        ax.set_xlim(0, n_packs)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("% de rodadas\nenviado", fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

    ax.set_ylim(0, 1)
    if xlabel_fs is not None:
        ax.set_xlabel("Índice do pacote", fontsize=xlabel_fs)
        ax.tick_params(axis="x", labelsize=13)
    else:
        ax.set_xlabel("Índice do pacote")

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

    print("Calculando frequência de envio dos pacotes...")
    freq_df = build_freq_df(masks)
    if freq_df.empty:
        print("Nenhum dado de frequência calculado.")
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
                                fname = _tok(model, dataset, technique, niid, pct, cf, variant)
                                for ext in ("png", "pdf"):
                                    fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                                dpi=300, bbox_inches="tight", pad_inches=0.08)
                                print(f"Salvo 20_pack_frequency_pt/{fname}.png/pdf")
                                plt.close(fig)
                                n_saved += 1

    print(f"Total de figuras: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estudo 20pt: frequência de envio de pacotes, PT-BR e separado por config")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
