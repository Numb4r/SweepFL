"""
30 — Downlink do artigo (SPINDLE-FL, periódico): Tabela `downlink5` e Figura `downlink-resnet20`.

O download é derivado das máscaras (`mask.csv`): a cada rodada o servidor devolve a
união dos pacotes enviados pelos clientes da rodada, e cada cliente participante recebe
essa união. O volume é contado em bytes nominais de cifra, na mesma unidade do upload
da Tabela `fullmodel`:

    download_bytes_rodada = |união| × CT_BYTES × n_clientes_na_rodada
    CT_BYTES = 2 · N · 8 = 131 072 B   (uma cifra CKKS de um primo, N = 8192)

Versões anteriores usavam `union_cov_pct/100 × n_params × 4 B` (equivalente float32 em
claro), cerca de 8× menor. Os fatores de overhead e as razões entre políticas não mudam.

Warm-up: descarta as rodadas 1..int(0.05·T) (rodadas 1–10 com T = 200), como na tabela.

Saídas (em output/):
  figures/30_downlink_paper/downlink-resnet20.{pdf,png}
  tables/30_downlink_paper.csv   (fator por p e download a p = 30%, por par e política)

Uso: python 30_downlink_paper.py --logs-dir ../../logs/sparsification_percentage
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
s26 = _load("26_union_coverage_downlink_pt")  # build_series_df

OUT_FIG = HERE / "output" / "figures" / "30_downlink_paper"
OUT_TAB = HERE / "output" / "tables"

NIID = 1.0            # pasta niid1 = Dirichlet alpha = 1
PCT_DOWNLOAD = 0.3    # coluna Download da tabela e painel (b)/(c)
WARMUP_FRAC = 0.05
MODELS = ["lenet5", "dnn_tiny", "vit_tiny", "resnet20", "mobilenetv2"]
POLICIES = ["sweepfl", "sw", "rw", "fedphe"]
LABELS = {"sweepfl": "Round Robin", "sw": "Sliding Window", "rw": "Random", "fedphe": "Top-$k$"}
COLORS = {"sweepfl": "#1f63a8", "sw": "#56a6d6", "rw": "#e08a1e", "fedphe": "#b2182b"}

RC = {
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linestyle": ":", "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
}


def ct_bytes(poly_modulus_degree: int) -> int:
    return 2 * int(poly_modulus_degree) * 8


def build_series(masks: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    m = masks[(masks["niid"] == NIID) & masks["percentage"].notna()].copy()
    m["client_fraction"] = 1.0
    keys = [c for c in s26.GROUP if c in m.columns] + ["round", "cid"]
    m = m.drop_duplicates(subset=keys, keep="first")

    ser = s26.build_series_df(m)
    n_packs = m.groupby("model")["n_packs"].first()
    poly = pd.to_numeric(train.get("poly_modulus_degree"), errors="coerce")
    poly = int(poly.dropna().iloc[0]) if poly is not None and poly.notna().any() else 8192

    ser["n_packs"] = ser["model"].map(n_packs)
    ser["union_packs"] = (ser["union_cov_pct"] / 100.0 * ser["n_packs"]).round().astype(int)
    ser["download_bytes_round"] = ser["union_packs"] * ct_bytes(poly) * ser["n_clients_round"]

    max_round = ser.groupby([c for c in s26.GROUP if c in ser.columns])["round"].transform("max")
    ser = ser[ser["round"] > (max_round * WARMUP_FRAC).astype(int)].copy()

    grp = [c for c in s26.GROUP if c in ser.columns]
    ser = ser.sort_values(grp + ["round"])
    ser["cumulative_gb"] = ser.groupby(grp)["download_bytes_round"].cumsum() / 1e9
    return ser


def summary_table(ser: pd.DataFrame) -> pd.DataFrame:
    g = ser.groupby(["model", "solution", "percentage", "seed"])
    per_seed = pd.DataFrame({
        "factor": g["union_cov_pct"].mean() / g["client_cov_pct"].mean(),
        "download_gb": g["download_bytes_round"].sum() / 1e9,
    }).reset_index()
    return per_seed.groupby(["model", "solution", "percentage"])[["factor", "download_gb"]].mean().reset_index()


def plot_resnet20(ser: pd.DataFrame, summ: pd.DataFrame, server_eval: pd.DataFrame, out: Path):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    model = "resnet20"

    # (a) fator de overhead por p
    ax = axes[0]
    pcts = sorted(summ["percentage"].unique())
    x = np.arange(len(pcts))
    w = 0.2
    for i, pol in enumerate(POLICIES):
        vals = [summ.query("model == @model and solution == @pol and percentage == @p")["factor"].mean()
                for p in pcts]
        ax.bar(x + (i - 1.5) * w, vals, w, color=COLORS[pol], label=LABELS[pol])
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.7)
    ax.set_xticks(x, [f"{int(round(p * 100))}" for p in pcts])
    ax.set_xlabel("transmitted fraction $p$ (%)")
    ax.set_ylabel("overhead factor")
    ax.set_ylim(0, 2.0)
    ax.set_title("(a) overhead factor")

    sub = ser[(ser["model"] == model) & (np.isclose(ser["percentage"], PCT_DOWNLOAD))]

    # (b) tamanho da união por rodada a p = 30%
    ax = axes[1]
    for pol in POLICIES:
        s = sub[sub["solution"] == pol].groupby("round")["union_cov_pct"].mean()
        ax.plot(s.index, s.values, color=COLORS[pol], linewidth=0.9)
    n_packs = int(sub["n_packs"].iloc[0])
    k = int(np.ceil(PCT_DOWNLOAD * n_packs))
    ax.axhline(100.0 * k / n_packs, color="#555555", linestyle="--", linewidth=0.7)
    ax.set_xlabel("round")
    ax.set_ylabel("union size (% of packets)")
    ax.set_title(f"(b) union size, $p = {int(PCT_DOWNLOAD * 100)}\\%$")

    # (c) acurácia vs download acumulado (GB de cifra)
    ax = axes[2]
    se = server_eval.copy()
    se = se[(se["model"] == model) & (se["niid"] == NIID) & np.isclose(se["percentage"].astype(float), PCT_DOWNLOAD)]
    for pol in POLICIES:
        s = sub[sub["solution"] == pol][["seed", "round", "cumulative_gb"]]
        a = se[se["solution"] == pol][["seed", "round", "mean_acc"]]
        mrg = s.merge(a, on=["seed", "round"]).groupby("round")[["cumulative_gb", "mean_acc"]].mean()
        ax.plot(mrg["cumulative_gb"], mrg["mean_acc"] * 100, color=COLORS[pol], linewidth=0.9)
    ax.set_xlabel("cumulative download (GB)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("(c) cost of the downlink")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"downlink-resnet20.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(logs_dir):
    dfs = load_data.load_all(logs_dir)
    masks = load_data.load_masks(logs_dir)
    ser = build_series(masks, dfs["train"])
    summ = summary_table(ser)

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT_TAB / "30_downlink_paper.csv", index=False)

    fac = summ.pivot_table(index=["model", "solution"], columns="percentage", values="factor")
    dl = summ[np.isclose(summ["percentage"], PCT_DOWNLOAD)].set_index(["model", "solution"])["download_gb"]
    table = fac.round(2)
    table[f"download_gb_p{int(PCT_DOWNLOAD * 100)}"] = dl.round(2)
    table = table.reindex([(m, p) for m in MODELS for p in POLICIES if (m, p) in table.index])
    print(table.to_string())

    plot_resnet20(ser, summ, dfs["server_eval"], OUT_FIG)
    print(f"Figura salva em {OUT_FIG / 'downlink-resnet20.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    main(parser.parse_args().logs_dir)
