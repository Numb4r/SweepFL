"""
28pt — Download acumulado vs acurácia, experimento completo (PT-BR), uma
figura por (model, dataset, niid, percentage[, client_fraction]).

Complementa os Estudos 15/16 (upload/overhead acumulado) com o lado do
*downlink*: usa a mesma cobertura da união por rodada já calculada no
Estudo 26 (`26_union_coverage_downlink_pt.py::build_series_df`, importada
aqui, não duplicada) para estimar quantos bytes o servidor precisa
reagregar e retransmitir a cada rodada — e não apenas o que cada cliente
enviou.

Diferença importante em relação ao Estudo 15: aqui a curva **não é
recortada no pico de acurácia** de cada solução — mostra o experimento
inteiro (todas as rodadas registradas) para poder ser correlacionado
diretamente com as séries completas do Estudo 26. O painel lateral resume
a **acurácia final** (última rodada) de cada solução, não a acurácia
máxima transitória.

Fórmula (documentada no Estudo 26): a cada rodada, o servidor reagrega a
**união** dos pacotes recebidos de todos os clientes participantes e
retransmite esse blob para cada um deles. Logo:

    download_bytes_rodada = |união| × CT_BYTES × n_clientes_na_rodada
    CT_BYTES = 2 · poly_modulus_degree · 8   (131 072 B com N = 8192)

`union_cov_pct` e `n_clientes_na_rodada` vêm do Estudo 26 (mask.csv);
|união| = round(union_cov_pct/100 × n_packs). O volume é contado em bytes
nominais de cifra, na mesma unidade do upload dos pacotes. Até 2026-09-30
este estudo usava `union_cov_pct/100 × n_params × 4` (equivalente float32
em claro), cerca de 8× menor; as razões entre técnicas não mudam.
Essa é a extensão "somada entre clientes" do *upload* do Estudo 15 para o
lado do *download* — com essa escolha, a razão
`download_acumulado / upload_acumulado` ao longo do treino converge para o
mesmo fator `inflation` já calculado no Estudo 26, permitindo correlacionar
os dois estudos diretamente.

Só se aplica às técnicas de *packing* (`sstyle.PACKING_SOLS`), já que só
elas geram `mask.csv` (mesmo recorte já adotado nos Estudos 15/16 e
20/22/26) — **exceto RW**, excluído aqui deliberadamente: a comparação
proposta neste estudo deve ser entre métodos de seleção que não sejam
inteiramente aleatórios (SweepFL, FedPHE, SW seguem alguma estrutura
determinística/adaptativa; RW sorteia os pacotes de forma independente a
cada rodada). Justificativa completa a ser desenvolvida no texto da
dissertação; aqui fica só a decisão de escopo. Portanto:
`STUDY_SOLS = sstyle.PACKING_SOLS - {"rw"}`.

Saída: output/figures/capitulo5_pt/28_download_maxacc_pt/{model}_{dataset}_niid{n}_p{pct}[_cf{f}].{png,pdf}
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
s15 = _load("15_bytes_vs_maxacc_pt")       # merge_accuracy, _cell_mask, _tok, _ref_mb_per_round
s26 = _load("26_union_coverage_downlink_pt")  # build_series_df (cobertura da união)

# RW excluído: a comparação deste estudo deve ficar restrita a métodos de
# seleção não inteiramente aleatórios (ver docstring do módulo).
STUDY_SOLS = sstyle.PACKING_SOLS - {"rw"}

OUT_FIG_STUDY = HERE / "output" / "figures" / "capitulo5_pt" / "28_download_maxacc_pt"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)


# ── Final-round helpers (experimento completo, não pico) ────────────────────

def _final_row(group):
    return group.sort_values("round").iloc[-1]


def _final_acc(group):
    return float(_final_row(group)["mean_acc"])


def _final_round(group):
    return float(_final_row(group)["round"])


def _cost_at_final(group):
    return float(_final_row(group)["cumulative_mb"])


def _draw_finalacc_bar(ax, stats: list):
    if not stats:
        ax.set_visible(False)
        return

    stats = sorted(stats, key=lambda t: t[1])
    sols   = [s[0] for s in stats]
    accs   = [s[1] * 100 for s in stats]
    rounds = [s[2] for s in stats]
    costs  = [s[3] for s in stats]
    colors = [sstyle.color(s) for s in sols]

    y = np.arange(len(sols))
    bars = ax.barh(y, accs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    lo = max(0.0, (min(accs) // 5) * 5 - 5)
    hi = lo + (100 - lo) * 1.55
    ax.set_xlim(lo, hi)
    off = (hi - lo) * 0.012
    for bar, acc, rnd, cost in zip(bars, accs, rounds, costs):
        ax.text(
            bar.get_width() + off, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%  R{int(rnd)}  {cost:.1f}MB",
            va="center", ha="left", fontsize=15, fontweight="bold", color="#222222",
        )

    ax.axvline(100, color="#bbbbbb", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_yticks([])
    ax.set_xlabel("Acurácia final (%)")
    ax.grid(True, axis="x")


# ── Download estimation (reaproveita a cobertura da união do Estudo 26) ─────

def compute_cumulative_download(masks: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    m = masks[masks["percentage"].notna()].copy()
    if "client_fraction" not in m.columns:
        m["client_fraction"] = 1.0
    m["client_fraction"] = m["client_fraction"].fillna(1.0)

    dedup_keys = [c for c in s26.GROUP if c in m.columns] + ["round", "cid"]
    m = m.drop_duplicates(subset=dedup_keys, keep="first")

    series_df = s26.build_series_df(m)
    if series_df.empty:
        return series_df

    n_packs = (
        m.groupby(["model", "dataset"], dropna=False)["n_packs"]
        .first().reset_index().rename(columns={"n_packs": "n_packs_model"})
    )
    series_df = series_df.merge(n_packs, on=["model", "dataset"], how="left")
    poly = pd.to_numeric(train.get("poly_modulus_degree"), errors="coerce")
    poly = int(poly.dropna().iloc[0]) if poly is not None and poly.notna().any() else 8192
    ct_bytes = 2 * poly * 8
    union_packs = (series_df["union_cov_pct"] / 100.0 * series_df["n_packs_model"]).round()
    series_df["download_bytes_round"] = union_packs * ct_bytes * series_df["n_clients_round"]

    seed_group = [c for c in s26.GROUP if c in series_df.columns]
    series_df = series_df.sort_values(seed_group + ["round"])
    series_df["cumulative_mb"] = (
        series_df.groupby(seed_group, dropna=False)["download_bytes_round"].cumsum() / 1e6
    )
    return series_df


# ── Individual cell plot (mesmo layout do Estudo 15, eixo de custo trocado) ──

def plot_cell(combo: pd.DataFrame, model, dataset, niid, pct, cf):
    if combo.empty:
        return None

    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(24, 9), gridspec_kw={"width_ratios": [3, 1]},
    )
    ax_r = ax.twinx()

    solutions = sstyle.sorted_solutions(combo["solution"].dropna().unique().tolist())
    ref_bpr   = s15._ref_mb_per_round(combo)

    stats, max_mb, best_acc, best_sol = [], 0.0, 0.0, None

    for sol in solutions:
        sg = combo[combo["solution"] == sol].sort_values("round")
        if sg.empty:
            continue

        color = sstyle.color(sol)

        ax.plot(
            sg["cumulative_mb"], sg["mean_acc"] * 100,
            color=color, linewidth=2.2, label=sstyle.label(sol), zorder=3,
        )
        ax_r.plot(sg["cumulative_mb"], sg["round"],
                   color=color, linewidth=0.8, alpha=0.0, zorder=1)

        n_pts = len(sg)
        step  = max(1, n_pts // 4)
        for _, rr in sg.iloc[::step].iterrows():
            ax.annotate(
                f"R{int(rr['round'])}", xy=(rr["cumulative_mb"], rr["mean_acc"] * 100),
                fontsize=15, color=color, alpha=0.75,
                xytext=(3, 2), textcoords="offset points",
            )

        final = sg.iloc[-1]
        ax.plot(
            final["cumulative_mb"], final["mean_acc"] * 100,
            marker="*", color=color, markersize=13,
            markeredgecolor="white", markeredgewidth=0.5, zorder=5, linestyle="",
        )

        s_final_acc = _final_acc(sg)
        s_final_round = _final_round(sg)
        s_final_cost = _cost_at_final(sg)
        stats.append((sol, s_final_acc, s_final_round, s_final_cost))
        max_mb = max(max_mb, sg["cumulative_mb"].max())
        if s_final_acc > best_acc:
            best_acc, best_sol = s_final_acc, sol

    if best_sol is not None:
        ax.axhline(
            best_acc * 100, color="#444444", linestyle="--", linewidth=1.4, alpha=0.8,
            label=f"melhor (final): {sstyle.label(best_sol)} {best_acc:.1%}",
        )

    if ref_bpr > 0 and max_mb > 0:
        max_r_est = max_mb / ref_bpr
        ax_r.set_ylim(0, max_r_est * 1.05)
        ax_r.set_ylabel("Rodada (aprox.)", color="#555555", fontsize=18)
        ax_r.tick_params(axis="y", labelcolor="#555555", labelsize=15)
        nice = [r for r in [10, 50, 100, 150, 200] if r <= max_r_est * 1.05]
        ax_r.set_yticks(nice)
    else:
        ax_r.set_yticks([])

    ax.set_xlabel("Download acumulado (MB, união de pacotes × clientes na rodada)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)

    _draw_finalacc_bar(ax_bar, stats)

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), frameon=True)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train, server_eval = dfs["train"], dfs["server_eval"]
    if train.empty or server_eval.empty:
        print("Sem dados de train/server_eval.")
        return

    print("Carregando dados de máscara...")
    try:
        masks = load_data.load_masks(logs_dir)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return
    if masks.empty:
        print("Sem dados de máscara.")
        return

    dims        = load_data.discover_dims(train)
    models      = dims.get("model", [])
    datasets    = dims.get("dataset", [])
    niids       = dims.get("niid", [None])
    percentages = [p for p in dims.get("percentage", []) if p is not None]
    print(f"Modelos      : {models}")
    print(f"Datasets     : {datasets}")
    print(f"Soluções     : {dims.get('solution', [])}")
    print(f"Porcentagens : {percentages}")
    print(f"NIIDs        : {niids}")
    print()

    print("Calculando download acumulado (cobertura da união, Estudo 26)...")
    cumulative = compute_cumulative_download(masks, train)
    if cumulative.empty:
        print("Nenhum dado de cobertura calculado.")
        return
    data = s15.merge_accuracy(cumulative, server_eval)
    data = data[data["solution"].isin(STUDY_SOLS)]  # packing, exceto RW (ver docstring do módulo)
    if data.empty:
        print("Sem dados após o merge.")
        return

    cfs = sorted(data["client_fraction"].dropna().unique().tolist()) or [1.0]

    n_saved = 0
    for pct in sorted(percentages):
        for model in models:
            for dataset in datasets:
                for niid in niids:
                    for cf in cfs:
                        sub = data[s15._cell_mask(data, model, dataset, niid, pct, cf)]
                        if sub.empty:
                            continue
                        fig = plot_cell(sub, model, dataset, niid, pct, cf)
                        if fig is None:
                            continue
                        fname = s15._tok(model, dataset, niid, pct, cf)
                        for ext in ("png", "pdf"):
                            fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                        dpi=300, bbox_inches="tight", pad_inches=0.25)
                        print(f"Salvo 28_download_maxacc_pt/{fname}.png/pdf")
                        plt.close(fig)
                        n_saved += 1

    print(f"Total de figuras: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
