"""
29 — Seed variance of accuracy for the packing techniques (seed_experiment_v2).

Answers: are the accuracy differences between techniques larger than the run-to-run
spread across seeds?

Sources (α=1 only, the one condition where seed 42 is comparable; see docs/seed_experiment.md)
  --seed-dir   logs/seed_experiment_v2      seeds 43/44
  --base-dir   logs/sparsification_percentage  seed 42 + plaintext (FedAvg) baseline
  --fix-dir    logs/range_fix               seed 42 reruns after the CKKS range fix; they
                                            replace the base-dir cell with the same key
Runs whose last 10 rounds contain NaN (the pre-fix overflow signature) are dropped and listed.

Metric: final accuracy = mean server `mean_acc` over the last 10 rounds.

Output (default studies/seed_experiment_v2/):
  figures/29_seed_variance/bars_{model}_p{pct}.{png,pdf}   mean ± std per technique, FedAvg line
  figures/29_seed_variance/curves_{model}_p{pct}.{png,pdf} mean ± std band per round, FedAvg curve
  tables/29_seed_variance.{csv,tex}  model, p, technique, n_seeds, seeds, mean, std, var, min, max, final_r200

  python3 studies/general/29_seed_variance.py [--seed-dir ...] [--percentages 0.1 0.7] [--out ...]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import load_data
import solution_style as st

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
KEY = ["model", "solution", "percentage", "seed"]
LAST_N = 10


def load_seval(path: Path, tag: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[aviso] {path} não existe — pulando")
        return pd.DataFrame()
    df = load_data.load_all(path)["server_eval"]
    df["source"] = tag
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", default=REPO / "logs/seed_experiment_v2", type=Path)
    ap.add_argument("--base-dir", default=REPO / "logs/sparsification_percentage", type=Path)
    ap.add_argument("--fix-dir", default=REPO / "logs/range_fix", type=Path)
    ap.add_argument("--percentages", nargs="+", type=float, default=[0.1, 0.7])
    ap.add_argument("--niid", type=float, default=1.0)
    ap.add_argument("--out", default=REPO / "studies/seed_experiment_v2", type=Path)
    a = ap.parse_args()

    df = pd.concat([load_seval(a.base_dir, "base"), load_seval(a.fix_dir, "fix"),
                    load_seval(a.seed_dir, "seed")], ignore_index=True)
    df = df[df["niid"] == a.niid].copy()
    df = df[df["model"].isin(df.loc[df["source"] == "seed", "model"].unique())]  # models with extra seeds
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")

    # plaintext baseline (seed 42), before restricting to packing solutions
    fedavg = df[df["solution"] == "plaintext"]
    df = df[df["solution"].isin(st.PACKING_SOLS)
            & df["percentage"].round(3).isin([round(p, 3) for p in a.percentages])]
    # range_fix replaces the old seed-42 cell with the same key
    fixed = set(map(tuple, df[df["source"] == "fix"][KEY].drop_duplicates().values))
    df = df[~((df["source"] == "base") & df[KEY].apply(tuple, axis=1).isin(fixed))]

    runs = []
    for k, g in df.sort_values("round").groupby(KEY):
        tail = g.tail(LAST_N)
        runs.append(dict(zip(KEY, k), source=g["source"].iloc[0], last_round=int(g["round"].max()),
                         broken=bool(tail["mean_loss"].isna().any()),
                         final=tail["mean_acc"].mean(), final_r200=g["mean_acc"].iloc[-1]))
    runs = pd.DataFrame(runs)
    if runs.empty:
        raise SystemExit("nenhuma run encontrada nas pastas informadas")
    broken = runs[runs["broken"]]
    if len(broken):
        print(f"[descartadas] {len(broken)} runs com NaN nas últimas {LAST_N} rodadas:")
        print(broken[KEY + ["source"]].to_string(index=False))
    ok = runs[~runs["broken"]]

    tab = (ok.groupby(["model", "percentage", "solution"])
             .agg(n_seeds=("seed", "size"), seeds=("seed", lambda s: " ".join(map(str, sorted(s)))),
                  mean=("final", "mean"), std=("final", "std"), var=("final", "var"),
                  min=("final", "min"), max=("final", "max"), final_r200=("final_r200", "mean"))
             .reset_index())
    tab["order"] = tab["solution"].map({s: i for i, s in enumerate(st.SOL_ORDER)})
    tab = tab.sort_values(["model", "percentage", "order"]).drop(columns="order")

    fig_dir = a.out / "figures" / "29_seed_variance"
    tab_dir = a.out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(tab_dir / "29_seed_variance.csv", index=False)
    tab.assign(solution=tab["solution"].map(st.label)).to_latex(
        tab_dir / "29_seed_variance.tex", index=False, float_format="%.4f")

    for (model, pct), t in tab.groupby(["model", "percentage"]):
        base = fedavg[fedavg["model"] == model].groupby("round", as_index=False)["mean_acc"].mean()
        base_acc = base.tail(LAST_N)["mean_acc"].mean() if len(base) else None
        name = f"{model}_p{pct:g}"

        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(t))
        ax.bar(x, t["mean"], yerr=t["std"].fillna(0), capsize=4,
               color=[st.color(s) for s in t["solution"]])
        for xi, (_, r) in zip(x, t.iterrows()):
            pts = ok[(ok["model"] == model) & (ok["percentage"] == pct) & (ok["solution"] == r["solution"])]
            ax.scatter([xi] * len(pts), pts["final"], color="k", s=10, zorder=3)
        if base_acc is not None:
            ax.axhline(base_acc, color=st.color("plaintext"), ls="--", label=f"{st.label('plaintext')} (seed 42)")
            ax.legend(fontsize=8)
        ax.set_xticks(x, [f"{st.label(s)}\nn={n}" for s, n in zip(t["solution"], t["n_seeds"])])
        ax.set_ylabel(f"acurácia final (média últimas {LAST_N} rodadas)")
        ax.set_title(f"{model} · p={pct:g} · α={a.niid:g} — média ± desvio entre seeds", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"bars_{name}.{ext}", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        sel = df[(df["model"] == model) & (df["percentage"] == pct)]
        sel = sel.merge(ok[KEY], on=KEY)  # only runs kept above
        for sol in t["solution"]:
            c = sel[sel["solution"] == sol].groupby("round")["mean_acc"].agg(["mean", "std"])
            ax.plot(c.index, c["mean"], color=st.color(sol), label=st.label(sol), lw=1.2)
            ax.fill_between(c.index, c["mean"] - c["std"].fillna(0), c["mean"] + c["std"].fillna(0),
                            color=st.color(sol), alpha=0.2, lw=0)
        if len(base):
            ax.plot(base["round"], base["mean_acc"], color=st.color("plaintext"), ls="--",
                    lw=1, label=f"{st.label('plaintext')} (seed 42)")
        ax.set_xlabel("rodada")
        ax.set_ylabel("acurácia (servidor)")
        ax.set_title(f"{model} · p={pct:g} · α={a.niid:g} — média ± desvio entre seeds", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"curves_{name}.{ext}", dpi=200)
        plt.close(fig)

    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Output: {a.out.resolve()}")


if __name__ == "__main__":
    main()
