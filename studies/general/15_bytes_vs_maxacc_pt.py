"""
15pt — Dados acumulados vs acurácia máxima (PT-BR), uma figura por
(model, dataset, niid, percentage[, client_fraction]).

Variante traduzida e separada do estudo 15 (15_bytes_vs_maxacc_individual.py):
  * nomes de solução corretos (NVNL, SW, FedPHE, RW, BatchCrypt, BFV) via
    solution_style.py, em vez das strings cruas do CSV;
  * inclui os baselines full-model (percentage=None: CKKS, BFV, BatchCrypt)
    na mesma figura, ao lado das técnicas de esparsificação — quando os
    dados do BatchCrypt existirem nos logs eles aparecem automaticamente,
    sem alterar este script; o FedAvg (plaintext) é excluído desta
    comparação por pedido do autor;
  * separa também por client_fraction (tema client_frac); no tema
    sparsification_percentage (client_fraction=1.0) o rótulo é omitido.

Métrica de bytes (dado bruto float32 pré-cifragem):
  técnicas de empacotamento (nvnl, sw, rw, fedphe, topkrr, srr): packs_sent × pack_size × 4
  demais (FedAvg, CKKS, BFV, BatchCrypt):                        n_params × 4
Total por rodada = soma entre clientes.

Saída: output/figures/15_bytes_maxacc_pt/{model}_{dataset}_niid{n}_p{pct}[_cf{f}].{png,pdf}
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

OUT_FIG_STUDY = HERE / "output" / "figures" / "capitulo5_pt" / "15_bytes_maxacc_pt"
OUT_FIG_STUDY.mkdir(parents=True, exist_ok=True)

DIMS      = ["model", "dataset", "solution", "niid", "percentage", "client_fraction"]
SEED_DIMS = DIMS + ["seed"]


def _tok(model, dataset, niid, pct, cf) -> str:
    niid_s = "niidNA" if niid is None else f"niid{niid:g}"
    return f"{model}_{dataset}_{niid_s}_p{int(round(pct * 100)):03d}{sstyle.cf_suffix(cf)}"


# ── Byte estimation ───────────────────────────────────────────────────────────

def _est_bytes_per_client(row) -> float:
    """Dado bruto float32 por cliente por rodada (pré-cifragem)."""
    sol = str(row.get("solution", ""))
    if sol in sstyle.PACKING_SOLS:
        packs = float(row.get("packs_sent") or 0)
        pack_size = float(row.get("pack_size") or 0)
        return packs * pack_size * 4
    n = float(row.get("n_params") or 0)
    return n * 4


def compute_cumulative_bytes(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    if "client_fraction" not in df.columns:
        df["client_fraction"] = 1.0
    df["client_fraction"] = df["client_fraction"].fillna(1.0)
    df["bytes_client"] = df.apply(_est_bytes_per_client, axis=1)

    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["bytes_client"]
        .sum().reset_index().rename(columns={"bytes_client": "bytes_round"})
    )

    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_mb"] = (
        per_round.groupby(seed_group, dropna=False)["bytes_round"].cumsum() / 1e6
    )
    return per_round


def merge_accuracy(cumulative: pd.DataFrame, server_eval: pd.DataFrame) -> pd.DataFrame:
    if "client_fraction" not in server_eval.columns:
        server_eval = server_eval.copy()
        server_eval["client_fraction"] = 1.0
    server_eval["client_fraction"] = server_eval["client_fraction"].fillna(1.0)

    join_keys = [c for c in SEED_DIMS + ["round"]
                 if c in cumulative.columns and c in server_eval.columns]
    se = server_eval[[*join_keys, *(c for c in ["mean_acc"] if c in server_eval.columns)]].copy()
    merged = cumulative.merge(se, on=join_keys, how="inner")

    avg_group = [c for c in DIMS + ["round"] if c in merged.columns]
    return (
        merged.groupby(avg_group, dropna=False)
        .agg(cumulative_mb=("cumulative_mb", "mean"), mean_acc=("mean_acc", "mean"))
        .reset_index()
    )


# ── Peak helpers ──────────────────────────────────────────────────────────────

def _peak_row(group):
    return group.sort_values("round").loc[group["mean_acc"].idxmax()]


def _max_acc(group):
    return float(group["mean_acc"].max())


def _peak_round(group):
    return float(_peak_row(group)["round"])


def _cost_at_peak(group):
    return float(_peak_row(group)["cumulative_mb"])


def _clip_at_peak(group):
    group = group.sort_values("round")
    if group.empty:
        return group
    return group[group["round"] <= _peak_round(group)]


def _ref_mb_per_round(sub: pd.DataFrame) -> float:
    vals = []
    for _, grp in sub.groupby([c for c in DIMS if c in sub.columns], dropna=False):
        grp = grp.sort_values("round")
        if len(grp) >= 2 and grp["round"].max() > 0:
            vals.append(grp["cumulative_mb"].iloc[-1] / grp["round"].iloc[-1])
    return float(np.median(vals)) if vals else 1.0


def _draw_maxacc_bar(ax, stats: list):
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
            va="center", ha="left", fontsize=11, fontweight="bold", color="#222222",
        )

    ax.axvline(100, color="#bbbbbb", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_yticks([])
    ax.set_xlabel("Acurácia máxima (%)", fontsize=15)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(True, axis="x", alpha=0.3)


# ── Individual cell plot ───────────────────────────────────────────────────────

def _cell_mask(data, model, dataset, niid, pct, cf):
    m = (data["percentage"] == pct) | data["percentage"].isna()
    m &= data["model"] == model
    m &= data["dataset"] == dataset
    m &= data["niid"] == niid
    m &= data["client_fraction"] == cf
    return m


def plot_cell(combo: pd.DataFrame, model, dataset, niid, pct, cf):
    if combo.empty:
        return None

    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [3, 1]},
    )
    ax_r = ax.twinx()

    solutions = sstyle.sorted_solutions(combo["solution"].dropna().unique().tolist())
    ref_bpr   = _ref_mb_per_round(combo)

    stats, max_mb, best_acc, best_sol = [], 0.0, 0.0, None

    for sol in solutions:
        sg = combo[combo["solution"] == sol].sort_values("round")
        if sg.empty:
            continue

        clipped = _clip_at_peak(sg)
        color   = sstyle.color(sol)
        is_full = sol in sstyle.FULL_MODEL_SOLS

        ax.plot(
            clipped["cumulative_mb"], clipped["mean_acc"] * 100,
            color=color, linestyle="--" if is_full else "-",
            linewidth=1.6 if is_full else 2.2,
            label=sstyle.label(sol), zorder=2 if is_full else 3,
        )
        ax_r.plot(clipped["cumulative_mb"], clipped["round"],
                   color=color, linewidth=0.8, alpha=0.0, zorder=1)

        n_pts = len(clipped)
        step  = max(1, n_pts // 4)
        for _, rr in clipped.iloc[::step].iterrows():
            ax.annotate(
                f"R{int(rr['round'])}", xy=(rr["cumulative_mb"], rr["mean_acc"] * 100),
                fontsize=11, color=color, alpha=0.75,
                xytext=(3, 2), textcoords="offset points",
            )

        peak = clipped.iloc[-1]
        ax.plot(
            peak["cumulative_mb"], peak["mean_acc"] * 100,
            marker="*", color=color, markersize=13,
            markeredgecolor="white", markeredgewidth=0.5, zorder=5, linestyle="",
        )

        s_max, s_round, s_cost = _max_acc(sg), _peak_round(sg), _cost_at_peak(sg)
        stats.append((sol, s_max, s_round, s_cost))
        max_mb = max(max_mb, clipped["cumulative_mb"].max())
        if s_max > best_acc:
            best_acc, best_sol = s_max, sol

    if best_sol is not None:
        ax.axhline(
            best_acc * 100, color="#444444", linestyle="--", linewidth=1.4, alpha=0.8,
            label=f"melhor: {sstyle.label(best_sol)} {best_acc:.1%}",
        )

    if ref_bpr > 0 and max_mb > 0:
        max_r_est = max_mb / ref_bpr
        ax_r.set_ylim(0, max_r_est * 1.05)
        ax_r.set_ylabel("Rodada (aprox.)", fontsize=14, color="#555555")
        ax_r.tick_params(axis="y", labelsize=13, labelcolor="#555555")
        nice = [r for r in [1, 2, 5, 10, 20, 30, 50, 100, 150, 200] if r <= max_r_est * 1.05]
        ax_r.set_yticks(nice)
    else:
        ax_r.set_yticks([])

    ax.set_xlabel("Upload acumulado (MB, float32 bruto pré-cifragem)", fontsize=15)
    ax.set_ylabel("Acurácia (%)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)

    _draw_maxacc_bar(ax_bar, stats)

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), fontsize=13, frameon=True)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    train, server_eval = dfs["train"], dfs["server_eval"]

    if train.empty or server_eval.empty:
        print("Sem dados de train/server_eval.")
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

    print("Calculando bytes acumulados...")
    cumulative = compute_cumulative_bytes(train)
    data = merge_accuracy(cumulative, server_eval)
    data = data[data["solution"].isin(sstyle.PACKING_SOLS)]  # só técnicas de packing (NVNL/FedPHE/SW/RW)
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
                        sub = data[_cell_mask(data, model, dataset, niid, pct, cf)]
                        if sub.empty:
                            continue
                        fig = plot_cell(sub, model, dataset, niid, pct, cf)
                        if fig is None:
                            continue
                        fname = _tok(model, dataset, niid, pct, cf)
                        for ext in ("png", "pdf"):
                            fig.savefig(OUT_FIG_STUDY / f"{fname}.{ext}",
                                        dpi=150, bbox_inches="tight", pad_inches=0.25)
                        print(f"Salvo 15_bytes_maxacc_pt/{fname}.png/pdf")
                        plt.close(fig)
                        n_saved += 1

    print(f"Total de figuras: {n_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
