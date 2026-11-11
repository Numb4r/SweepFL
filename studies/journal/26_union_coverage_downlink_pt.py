"""
26pt — Cobertura da união por rodada & inflação de downlink (PT-BR).

As duas figuras deste estudo têm granularidades diferentes de propósito:
  * curvas de cobertura da união: uma figura por configuração completa,
    incluindo a porcentagem (model, dataset, niid, percentage[, cf]) — igual
    aos estudos 15/16/20/22;
  * barras de inflação de downlink: uma única figura por
    (model, dataset, niid[, cf]) com todas as porcentagens agrupadas no eixo
    X (uma barra por técnica dentro de cada grupo de porcentagem), igual ao
    estudo 26 original.

Variante traduzida do estudo 26 (26_union_coverage_downlink.py):
  * corrige o bug do `dataset` ignorado no original (misturava datasets
    diferentes do mesmo modelo na mesma figura/tabela quando existiam);
  * nomes de solução corretos (NVNL, SW, FedPHE, RW) via solution_style.py;
  * mantém a lógica de client_fraction já existente no original (fallback
    1.0, sufixo/rótulo só quando client_fraction != 1.0).

Nota: como o estudo 20/22, depende de mask.csv — os baselines full-model
(FedAvg, CKKS, BFV, BatchCrypt) nunca aparecem aqui, comportamento esperado.

Saída:
  output/figures/26_union_coverage_curves_pt/{model}_{dataset}_niid{n}_p{pct}[_cf{f}].{png,pdf}
  output/figures/26_downlink_inflation_pt/{model}_{dataset}_niid{n}[_cf{f}].{png,pdf}
  output/tables/26_union_coverage_downlink_summary_pt.{csv,tex}
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
cap5_style.apply_style()

OUT_FIG = HERE / "output" / "figures" / "capitulo5_pt"
OUT_FIG_CURVES = OUT_FIG / "26_union_coverage_curves_pt"
OUT_FIG_BARS   = OUT_FIG / "26_downlink_inflation_pt"
OUT_TAB = HERE / "output" / "tables"
OUT_FIG_CURVES.mkdir(parents=True, exist_ok=True)
OUT_FIG_BARS.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

GROUP = ["model", "dataset", "solution", "niid", "percentage", "client_fraction", "seed"]
GROUP_NO_SEED = ["model", "dataset", "solution", "niid", "percentage", "client_fraction"]

WARMUP_FRAC = 0.05  # descarta os primeiros 5% das rodadas (transiente inicial)


def _tok(model, dataset, niid, cf) -> str:
    """Nome de arquivo para as barras de inflação (combinam porcentagens)."""
    niid_s = "niidNA" if niid is None else f"niid{niid:g}"
    return f"{model}_{dataset}_{niid_s}{sstyle.cf_suffix(cf)}"


def _tok_pct(model, dataset, niid, pct, cf) -> str:
    """Nome de arquivo para as curvas (uma figura por porcentagem)."""
    niid_s = "niidNA" if niid is None else f"niid{niid:g}"
    return f"{model}_{dataset}_{niid_s}_p{int(round(pct * 100)):03d}{sstyle.cf_suffix(cf)}"


# ── Core: série por rodada de cobertura união/cliente (idêntico ao original) ─

def _round_series(group_df: pd.DataFrame) -> list:
    n_packs = int(group_df["n_packs"].iloc[0])
    if n_packs <= 0:
        return []

    records = []
    for rnd, rnd_df in group_df.groupby("round", sort=False):
        sets = [set(s) for s in rnd_df["selected"]]
        if not sets:
            continue
        union = set().union(*sets)
        client_covs = [len(s) / n_packs for s in sets]
        records.append({
            "round": int(rnd),
            "union_cov_pct": len(union) / n_packs * 100.0,
            "client_cov_pct": float(np.mean(client_covs)) * 100.0,
            "n_clients_round": len(sets),
        })
    return records


def build_series_df(masks: pd.DataFrame) -> pd.DataFrame:
    parts = []
    gcols = [c for c in GROUP if c in masks.columns]
    for key, grp in masks.groupby(gcols, dropna=False):
        series = _round_series(grp)
        if not series:
            continue
        df = pd.DataFrame(series)
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        for k, v in meta.items():
            df[k] = v
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _avg_over_seeds(series_df: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in GROUP_NO_SEED + ["round"] if c in series_df.columns]
    return (
        series_df.groupby(gcols, dropna=False)[["union_cov_pct", "client_cov_pct", "n_clients_round"]]
        .mean().reset_index()
    )


def build_summary(series_df: pd.DataFrame) -> pd.DataFrame:
    if series_df.empty:
        return pd.DataFrame()

    rows = []
    gcols = [c for c in GROUP if c in series_df.columns]
    for key, grp in series_df.groupby(gcols, dropna=False):
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        max_round = grp["round"].max()
        start = max(int(max_round * WARMUP_FRAC), 1)
        warm = grp[grp["round"] >= start]
        if warm.empty:
            warm = grp
        rows.append({
            **meta,
            "client_cov_pct": warm["client_cov_pct"].mean(),
            "union_cov_pct": warm["union_cov_pct"].mean(),
            "clients_per_round": warm["n_clients_round"].mean(),
        })

    per_seed = pd.DataFrame(rows)
    gcols_no_seed = [c for c in GROUP_NO_SEED if c in per_seed.columns]
    summary = (
        per_seed.groupby(gcols_no_seed, dropna=False)[
            ["client_cov_pct", "union_cov_pct", "clients_per_round"]
        ].mean().reset_index()
    )
    summary["inflation"] = np.where(
        summary["client_cov_pct"] > 0,
        summary["union_cov_pct"] / summary["client_cov_pct"],
        np.nan,
    )
    return summary.sort_values(["model", "dataset", "client_fraction", "niid", "percentage", "solution"])


# ── Figura 1: curvas de cobertura da união (1 painel por config completa) ────

def plot_curves_cell(cov_avg: pd.DataFrame, model, dataset, niid, pct, cf):
    sub = cov_avg[
        (cov_avg["model"] == model) & (cov_avg["dataset"] == dataset)
        & (cov_avg["niid"] == niid) & (cov_avg["percentage"] == pct)
        & (cov_avg["client_fraction"] == cf)
    ]
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 8.5))
    solutions = sstyle.sorted_solutions(sub["solution"].dropna().unique().tolist())
    for sol in solutions:
        ss = sub[sub["solution"] == sol].sort_values("round")
        if ss.empty:
            continue
        ax.plot(ss["round"], ss["union_cov_pct"], linewidth=1.8,
                color=sstyle.color(sol), label=sstyle.label(sol))

    ax.axhline(pct * 100, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="upload por cliente")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Cobertura da união (%)")

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), frameon=True)
    return fig


# ── Figura 2: barras de inflação de downlink (grupos por porcentagem) ───────

def plot_inflation_cell(summary: pd.DataFrame, model, dataset, niid, cf, percentages):
    sub = summary[
        (summary["model"] == model) & (summary["dataset"] == dataset)
        & (summary["niid"] == niid) & (summary["client_fraction"] == cf)
    ]
    if sub.empty:
        return None

    sorted_pcts = [p for p in sorted(percentages) if not sub[sub["percentage"] == p].empty]
    if not sorted_pcts:
        return None
    solutions = sstyle.sorted_solutions(sub["solution"].dropna().unique().tolist())
    n = len(solutions)
    width = 0.8 / max(n, 1)
    x = np.arange(len(sorted_pcts))

    fig, ax = plt.subplots(figsize=((1.8 * len(sorted_pcts) + 2) * 1.3, 7.5))
    for i, sol in enumerate(solutions):
        vals = []
        for pct in sorted_pcts:
            row = sub[(sub["solution"] == sol) & (sub["percentage"] == pct)]
            vals.append(row["inflation"].iloc[0] if not row.empty else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=sstyle.label(sol), color=sstyle.color(sol))

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label="sem inflação (downlink=upload)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0%}" for p in sorted_pcts])
    ax.set_xlabel("Porcentagem de esparsificação")
    ax.set_ylabel("Inflação de downlink")
    ax.grid(True, axis="y")

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), frameon=True)
    return fig


# ── Tabelas ───────────────────────────────────────────────────────────────────

def save_csv(summary: pd.DataFrame, multi_cf: bool):
    cols = ["model", "dataset", "client_fraction", "niid", "percentage", "solution",
            "client_cov_pct", "union_cov_pct", "inflation", "clients_per_round"]
    if not multi_cf:
        cols = [c for c in cols if c != "client_fraction"]
    out = summary[[c for c in cols if c in summary.columns]].copy()
    for c in out.select_dtypes(include=float).columns:
        out[c] = out[c].round(3)
    path = OUT_TAB / "26_union_coverage_downlink_summary_pt.csv"
    out.to_csv(path, index=False)
    print(f"Salvo {path}")


def save_latex(summary: pd.DataFrame, multi_cf: bool):
    display_cols = {
        "model": "Modelo", "dataset": "Base", "client_fraction": "Fração cli.",
        "niid": "NIID", "percentage": "Pct.", "solution": "Solução",
        "client_cov_pct": "Upload (%)", "union_cov_pct": "Downlink (%)",
        "inflation": "Inflação (x)", "clients_per_round": "Clientes/rodada",
    }
    if not multi_cf:
        display_cols.pop("client_fraction")
    cols = [c for c in display_cols if c in summary.columns]

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        if col == "solution":
            return sstyle.label(str(val))
        if isinstance(val, float):
            return f"{val:.1f}" if col == "clients_per_round" else f"{val:.2f}"
        return str(val)

    header = " & ".join(display_cols[c] for c in cols) + r" \\"
    ncols = len(cols)
    lines = [
        r"\begin{longtable}{" + "l" * ncols + "}",
        r"\caption{Síntese de cobertura da união e inflação de downlink (Estudo 26)}",
        r"\label{tab:union_coverage_downlink_pt} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncols) + r"}{r}{\textit{continua\ldots}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    prev_model = None
    for _, row in summary.iterrows():
        model = row.get("model", "")
        if prev_model is not None and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model
        lines.append(" & ".join(_fmt(row, c) for c in cols) + r" \\")
    lines.append(r"\end{longtable}")

    path = OUT_TAB / "26_union_coverage_downlink_summary_pt.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Salvo {path}")


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
        print("Sem dados de máscara de técnicas de empacotamento (tudo era plaintext).")
        return
    if "client_fraction" not in masks.columns:
        masks["client_fraction"] = 1.0
    masks["client_fraction"] = masks["client_fraction"].fillna(1.0)

    dedup_keys = [c for c in GROUP if c in masks.columns] + ["round", "cid"]
    before = len(masks)
    masks = masks.drop_duplicates(subset=dedup_keys, keep="first")
    dropped = before - len(masks)
    if dropped:
        print(f"Nota: descartadas {dropped:,} linhas duplicadas (config, round, cid) "
              f"de reruns com a mesma seed.")

    dims = load_data.discover_dims(masks)
    models      = dims.get("model", [])
    datasets    = dims.get("dataset", [])
    solutions   = dims.get("solution", [])
    niids       = dims.get("niid", [])
    percentages = sorted(masks["percentage"].dropna().unique())
    cf_values   = sorted(masks["client_fraction"].dropna().unique())
    multi_cf    = len(cf_values) > 1
    print(f"Modelos      : {models}")
    print(f"Datasets     : {datasets}")
    print(f"Soluções     : {solutions}")
    print(f"Porcentagens : {percentages}")
    print(f"NIIDs        : {niids}")
    print(f"Frações cli. : {cf_values}  (separando por cf: {multi_cf})")
    print(f"Linhas mask  : {len(masks):,}")
    print()

    print("Calculando séries de cobertura união/cliente por rodada...")
    series_df = build_series_df(masks)
    if series_df.empty:
        print("Nenhum dado de cobertura calculado.")
        return

    cov_avg = _avg_over_seeds(series_df)
    summary = build_summary(series_df)

    n_curve_figs, n_bar_figs = 0, 0
    for model in models:
        for dataset in datasets:
            for niid in niids:
                for cf in cf_values:
                    for pct in percentages:
                        fig = plot_curves_cell(cov_avg, model, dataset, niid, pct, cf)
                        if fig is not None:
                            fname = _tok_pct(model, dataset, niid, pct, cf)
                            for ext in ("png", "pdf"):
                                fig.savefig(OUT_FIG_CURVES / f"{fname}.{ext}",
                                            dpi=300, bbox_inches="tight", pad_inches=0.25)
                            print(f"Salvo 26_union_coverage_curves_pt/{fname}.png/pdf")
                            plt.close(fig)
                            n_curve_figs += 1

                    fig2 = plot_inflation_cell(summary, model, dataset, niid, cf, percentages)
                    if fig2 is not None:
                        fname = _tok(model, dataset, niid, cf)
                        for ext in ("png", "pdf"):
                            fig2.savefig(OUT_FIG_BARS / f"{fname}.{ext}",
                                         dpi=300, bbox_inches="tight", pad_inches=0.25)
                        print(f"Salvo 26_downlink_inflation_pt/{fname}.png/pdf")
                        plt.close(fig2)
                        n_bar_figs += 1

    print(f"Total de figuras de curva : {n_curve_figs}")
    print(f"Total de figuras de barra : {n_bar_figs}")

    print()
    save_csv(summary, multi_cf)
    save_latex(summary, multi_cf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estudo 26pt: cobertura da união e inflação de downlink, PT-BR e separado por config")
    parser.add_argument("--logs-dir", default=None,
                        help="Caminho para logs/<experiment_name>")
    args = parser.parse_args()
    main(args.logs_dir)
