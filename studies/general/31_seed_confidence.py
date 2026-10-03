"""
31 — Seed-confidence study: do the technique rankings hold across seeds 42/43/44?

MobileNetV2 · CIFAR10 · α=1 · p ∈ {0.1, 0.7}; techniques fedphe (top-k), nvnl (round-robin),
sw, rw; plaintext (seed 42) as reference. Metric: mean server accuracy over the last 10 rounds
(also best-round accuracy). Statistics per cell (n=3): mean, std, coefficient of variation,
95% CI (Student t, 2 d.o.f. → 4.303·s/√n), min–max. Pairwise table: gap between means and
whether the min–max ranges are disjoint (a conservative, n=3-appropriate separation test).

Output (studies/seed_confidence/): tables/31_*.csv|md, figures/31_*.png|pdf
  python3.10 studies/general/31_seed_confidence.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import solution_style as st
from seed_conf_common import OUT, TECHS, find_runs

LAST_N, T975 = 10, {1: 12.706, 2: 4.303}


def main():
    runs = find_runs()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    curves, rows = {}, []
    for r in runs:
        e = pd.read_csv(r["path"] / "server_eval.csv").sort_values("round")
        curves[(r["sol"], r["pct"], r["seed"])] = e.set_index("round")["mean_acc"]
        rows.append(dict(sol=r["sol"], pct=r["pct"], seed=r["seed"], source=r["source"], post_fix=r["post_fix"],
                         final=e["mean_acc"].tail(LAST_N).mean(), best=e["mean_acc"].max(),
                         best_round=int(e.loc[e["mean_acc"].idxmax(), "round"]), r200=e["mean_acc"].iloc[-1],
                         last_loss=e["mean_loss"].tail(LAST_N).mean()))
    runs_df = pd.DataFrame(rows).sort_values(["pct", "sol", "seed"])
    runs_df.to_csv(OUT / "tables/31_runs.csv", index=False)

    base = runs_df[runs_df.sol == "plaintext"]
    rows = []
    for (pct, sol), g in runs_df[runs_df.sol != "plaintext"].groupby(["pct", "sol"]):
        n, s = len(g), g["final"].std(ddof=1) if len(g) > 1 else np.nan
        ci = T975.get(n - 1, 1.96) * s / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(p=pct, technique=st.label(sol), sol=sol, n=n, seeds=" ".join(map(str, g.seed)),
                         mean=g.final.mean(), std=s, cv_pct=100 * s / g.final.mean(), ci95=ci,
                         ci_low=g.final.mean() - ci, ci_high=g.final.mean() + ci,
                         min=g.final.min(), max=g.final.max(), best_mean=g.best.mean(),
                         loss_mean=g.last_loss.mean()))
    tab = pd.DataFrame(rows)
    tab["o"] = tab.sol.map({s: i for i, s in enumerate(TECHS)})
    tab = tab.sort_values(["p", "o"]).drop(columns="o")
    tab.to_csv(OUT / "tables/31_confidence.csv", index=False)

    # pairwise separation
    pr = []
    for pct, g in tab.groupby("p"):
        for i, a in g.iterrows():
            for j, b in g.iterrows():
                if a.sol >= b.sol:
                    continue
                gap = a["mean"] - b["mean"]
                disjoint = a["min"] > b["max"] or b["min"] > a["max"]
                ci_sep = a.ci_low > b.ci_high or b.ci_low > a.ci_high
                pr.append(dict(p=pct, A=a.technique, B=b.technique, gap_pp=100 * gap,
                               ranges_disjoint=disjoint, ci95_disjoint=ci_sep))
    pair = pd.DataFrame(pr)
    pair.to_csv(OUT / "tables/31_pairwise.csv", index=False)

    with open(OUT / "tables/31_confidence.md", "w") as f:
        f.write("| p | Técnica | n | seeds | média (últ.10 r.) | desvio | CV % | IC95% | mín–máx | melhor rodada (média) | perda |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in tab.iterrows():
            f.write(f"| {r.p:g} | {r.technique} | {r.n} | {r.seeds} | {r['mean']:.4f} | {r['std']:.4f} | {r.cv_pct:.1f} | "
                    f"[{r.ci_low:.3f}, {r.ci_high:.3f}] | {r['min']:.3f}–{r['max']:.3f} | {r.best_mean:.3f} | {r.loss_mean:.2f} |\n")
        f.write("\n**Por seed**\n\n| p | Técnica | seed | final | melhor (rodada) | r200 | pós-fix |\n|---|---|---|---|---|---|---|\n")
        for _, r in runs_df.iterrows():
            f.write(f"| {r.pct:g} | {st.label(r.sol)} | {r.seed} | {r.final:.4f} | {r.best:.3f} ({r.best_round}) | {r.r200:.3f} | {'sim' if r.post_fix else 'não'} |\n")
        f.write("\n**Separação entre técnicas**\n\n| p | A | B | Δ média (p.p.) | faixas mín–máx disjuntas | ICs 95% disjuntos |\n|---|---|---|---|---|---|\n")
        for _, r in pair.iterrows():
            f.write(f"| {r.p:g} | {r.A} | {r.B} | {r.gap_pp:+.1f} | {'sim' if r.ranges_disjoint else 'não'} | {'sim' if r.ci95_disjoint else 'não'} |\n")

    pcts = sorted(tab.p.unique())
    # Fig A: bars + points per p
    fig, axs = plt.subplots(1, len(pcts), figsize=(5 * len(pcts), 3.8), sharey=True)
    for ax, pct in zip(np.atleast_1d(axs), pcts):
        t = tab[tab.p == pct]
        x = np.arange(len(t))
        ax.bar(x, t["mean"], yerr=t.ci95, capsize=5, color=[st.color(s) for s in t.sol], alpha=.85)
        for xi, s in zip(x, t.sol):
            g = runs_df[(runs_df.pct == pct) & (runs_df.sol == s)]
            ax.scatter([xi] * len(g), g.final, c="k", s=14, zorder=3)
            for _, r in g.iterrows():
                ax.annotate(str(r.seed), (xi, r.final), xytext=(5, 0), textcoords="offset points", fontsize=6)
        if len(base):
            ax.axhline(base.final.mean(), color=st.color("plaintext"), ls="--", label="FedAvg plaintext (seed 42)")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xticks(x, [st.label(s) for s in t.sol])
        ax.set_title(f"p={pct:g}", fontsize=10)
    np.atleast_1d(axs)[0].set_ylabel("acurácia final (média últimas 10 rodadas)")
    fig.suptitle("MobileNetV2 · CIFAR10 · α=1 — média, IC95% (barras) e pontos por seed", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figures/31_bars_confidence.{ext}", dpi=200)
    plt.close(fig)

    # Fig B: curves, every seed thin + mean thick
    fig, axs = plt.subplots(1, len(pcts), figsize=(6 * len(pcts), 4), sharey=True)
    for ax, pct in zip(np.atleast_1d(axs), pcts):
        for sol in TECHS:
            ks = [k for k in curves if k[0] == sol and k[1] == pct]
            if not ks:
                continue
            df = pd.concat([curves[k] for k in ks], axis=1)
            for k in ks:
                ax.plot(curves[k].index, curves[k].rolling(5, min_periods=1).mean(), color=st.color(sol), lw=.6, alpha=.5)
            m = df.mean(axis=1).rolling(5, min_periods=1).mean()
            ax.plot(m.index, m, color=st.color(sol), lw=1.8, label=st.label(sol))
        pk = [k for k in curves if k[0] == "plaintext"]
        if pk:
            ax.plot(curves[pk[0]].index, curves[pk[0]].rolling(5, min_periods=1).mean(), "k--", lw=1, label="FedAvg (seed 42)")
        ax.set_title(f"p={pct:g} (fino = seed, grosso = média; suavização 5 r.)", fontsize=9)
        ax.set_xlabel("rodada")
        ax.legend(fontsize=7)
    np.atleast_1d(axs)[0].set_ylabel("acurácia do servidor")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figures/31_curves_seeds.{ext}", dpi=200)
    plt.close(fig)

    # Fig C: CV
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    w = .2
    for i, sol in enumerate(TECHS):
        t = tab[tab.sol == sol].sort_values("p")
        ax.bar(np.arange(len(t)) + (i - 1.5) * w, t.cv_pct, w, color=st.color(sol), label=st.label(sol))
    ax.set_xticks(range(len(pcts)), [f"p={p:g}" for p in pcts])
    ax.set_ylabel("coeficiente de variação entre seeds (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figures/31_cv.{ext}", dpi=200)
    plt.close(fig)

    print((OUT / "tables/31_confidence.md").read_text())


if __name__ == "__main__":
    main()
