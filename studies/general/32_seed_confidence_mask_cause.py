"""
32 — Why FedPHE (top-k) at p=0.1 is poor and seed-sensitive: mask diagnosis (companion to 31).

Evidence produced, MobileNetV2 · CIFAR10 · α=1 · p=0.1, seeds 42/43/44, fedphe vs nvnl/sw/rw:
  1. Coverage: distinct packs ever sent out of n_packs, and cumulative-unique-packs curve per round.
  2. Rank–frequency of packs (how concentrated the selection is).
  3. Mechanism: fedphe ranks packs by the L2 norm of the *weights* (client/literature/fedphe.py:
     get_topk_mask on `packed_parameters`), not of the update. We rebuild per-pack norms from
     model/round_100.npz and show norm vs. selection frequency, and what layer kinds the
     always-sent packs hold (BN moving_variance/gamma vs conv kernels).
  4. Layer-kind coverage: share of each parameter kind (conv/dense kernel, BN gamma/beta/mean/var,
     bias) that sits in packs that were ever sent.
  5. Seed sensitivity: packs sent by the best seed (43) but not by the others, vs accuracy.

Output: studies/seed_confidence/{tables,figures}/32_*
  python3.10 studies/general/32_seed_confidence_mask_cause.py
"""
import collections
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import solution_style as st
from seed_conf_common import OUT, TECHS, find_runs

CHUNK = 4096
KINDS = ["kernel", "bn_gamma", "bn_beta", "bn_mean", "bn_var", "bias"]


def read_masks(path):
    """-> {round: {pack,...}} union over clients, and Counter of pack selections."""
    byr, cnt = collections.defaultdict(set), collections.Counter()
    with open(path / "mask.csv") as f:
        for r in csv.DictReader(f):
            ps = [int(p) for p in r["selected_packs"].split(";") if p != ""]
            byr[int(r["round"])].update(ps)
            cnt.update(ps)
    return byr, cnt, int(r["n_packs"])


def layer_kinds(arrays):
    """Per-array kind from get_weights() order: ndim>=2 → kernel; runs of 1-D arrays are BN
    quadruples (gamma, beta, moving_mean, moving_variance); a trailing lone 1-D = bias."""
    kinds, i = [], 0
    while i < len(arrays):
        if arrays[i].ndim >= 2:
            kinds.append("kernel"); i += 1; continue
        j = i
        while j < len(arrays) and arrays[j].ndim == 1:
            j += 1
        run = j - i
        if run % 4 == 0:
            kinds += [KINDS[1 + k % 4] for k in range(run)]
        else:
            kinds += ["bn_gamma", "bn_beta", "bn_mean", "bn_var"] * (run // 4) + ["bias"] * (run % 4)
        i = j
    return kinds


def pack_kind_table(arrays, n_packs):
    """fraction[pack][kind] of elements per pack."""
    kinds = layer_kinds(arrays)
    frac = np.zeros((n_packs, len(KINDS)))
    pos = 0
    for a, k in zip(arrays, kinds):
        ki = KINDS.index(k)
        for p in range(pos // CHUNK, (pos + a.size - 1) // CHUNK + 1):
            lo, hi = max(pos, p * CHUNK), min(pos + a.size, (p + 1) * CHUNK)
            frac[p, ki] += hi - lo
        pos += a.size
    return frac, kinds


def load_snapshot(path, rnd=100):
    z = np.load(path / f"model/round_{rnd}.npz")
    return [z[f"arr_{i}"].astype(np.float32) for i in range(len(z.files))]


def main():
    runs = [r for r in find_runs(pcts=(0.1,)) if r["sol"] != "plaintext"]
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    acc = {}
    info = {}
    for r in runs:
        byr, cnt, n = read_masks(r["path"])
        e = pd.read_csv(r["path"] / "server_eval.csv")
        acc[(r["sol"], r["seed"])] = e["mean_acc"].tail(10).mean()
        cum, seen = [], set()
        for k in sorted(byr):
            seen |= byr[k]
            cum.append(len(seen))
        tot = sum(cnt.values())
        info[(r["sol"], r["seed"])] = dict(path=r["path"], cnt=cnt, n=n, cum=cum, ever=len(cnt),
                                           top_share=sum(c for _, c in cnt.most_common(18)) / tot,
                                           always=sorted(p for p, c in cnt.items() if c == max(cnt.values())
                                                         and c >= 0.99 * (len(byr) * 9)),
                                           per_round=int(np.mean([len(v) for v in byr.values()])))

    # ---- Table 1: coverage
    rows = [dict(technique=st.label(s), seed=sd, acc_final=acc[(s, sd)], n_packs=v["n"], packs_ever_sent=v["ever"],
                 ever_pct=100 * v["ever"] / v["n"], packs_sent_every_round_by_everyone=len(v["always"]),
                 share_of_sends_in_top18=v["top_share"])
            for (s, sd), v in sorted(info.items(), key=lambda kv: (TECHS.index(kv[0][0]), kv[0][1]))]
    t1 = pd.DataFrame(rows)
    t1.to_csv(OUT / "tables/32_coverage.csv", index=False)

    # ---- Fig 1: cumulative unique packs
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for (s, sd), v in info.items():
        axs[0].plot(range(1, len(v["cum"]) + 1), v["cum"], color=st.color(s), ls=["-", "--", ":"][sd - 42],
                    label=f"{st.label(s)} s{sd}")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("rodada (log)")
    axs[0].set_ylabel("packs distintos já enviados (de 176)")
    axs[0].set_title("Cobertura acumulada (p=0.1: 18 packs/rodada)", fontsize=9)
    axs[0].legend(fontsize=6, ncol=2)
    for (s, sd), v in info.items():
        f = np.array(sorted([v["cnt"].get(p, 0) for p in range(v["n"])], reverse=True)) / max(v["cnt"].values())
        axs[1].plot(f, color=st.color(s), ls=["-", "--", ":"][sd - 42])
    axs[1].set_xlabel("pack (ordenado por frequência)")
    axs[1].set_ylabel("frequência relativa de envio")
    axs[1].set_title("Concentração da seleção (rank–frequência)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figures/32_coverage.{ext}", dpi=200)
    plt.close(fig)

    # ---- Mechanism: fedphe norm vs frequency + kinds
    krows, nrm_rows, diff_rows = [], [], []
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, sd in zip(axs, (42, 43, 44)):
        v = info.get(("fedphe", sd))
        if v is None:
            continue
        try:
            arrays = load_snapshot(v["path"])
        except FileNotFoundError:
            ax.set_title(f"seed {sd}: sem snapshot"); continue
        flat = np.concatenate([a.ravel() for a in arrays])
        n = v["n"]
        flat = np.pad(flat, (0, n * CHUNK - flat.size))
        norms = np.linalg.norm(flat.reshape(n, CHUNK), axis=1)
        freq = np.array([v["cnt"].get(p, 0) for p in range(n)]) / max(v["cnt"].values())
        frac, kinds = pack_kind_table(arrays, n)
        dom = np.array(KINDS)[frac.argmax(axis=1)]
        for k in KINDS:
            m = dom == k
            if m.any():
                ax.scatter(norms[m], freq[m], s=14, label=k, alpha=.8)
        ax.set_xscale("log")
        ax.set_xlabel("‖pack‖₂ dos pesos (round 100, escala log)")
        ax.set_title(f"FedPHE seed {sd}: norma × frequência de envio", fontsize=9)
        order = np.argsort(-norms)[:18]
        ever = {p for p, c in v["cnt"].items() if c > 0}
        nrm_rows.append(dict(seed=sd, overlap_top18_norm_vs_sent_top18=len(set(order) & set(
            p for p, _ in v["cnt"].most_common(18))), ever_sent=len(ever),
            spearman_norm_freq=pd.Series(norms).corr(pd.Series(freq), method="spearman")))
        # kind coverage
        tot = frac.sum(axis=0)
        cov = frac[sorted(ever)].sum(axis=0)
        for i, k in enumerate(KINDS):
            krows.append(dict(seed=sd, kind=k, params=int(tot[i]), params_in_sent_packs=int(cov[i]),
                              coverage_pct=100 * cov[i] / tot[i] if tot[i] else np.nan))
        # what the always-sent packs hold
        always_frac = frac[v["always"]].sum(axis=0)
        for i, k in enumerate(KINDS):
            krows[-len(KINDS) + i]["share_of_always_sent_pct"] = 100 * always_frac[i] / max(always_frac.sum(), 1)
    axs[0].set_ylabel("frequência relativa de envio")
    axs[-1].legend(fontsize=6, title="tipo dominante", title_fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figures/32_norm_vs_frequency.{ext}", dpi=200)
    plt.close(fig)
    pd.DataFrame(nrm_rows).to_csv(OUT / "tables/32_norm_vs_selection.csv", index=False)
    kt = pd.DataFrame(krows)
    kt.to_csv(OUT / "tables/32_kind_coverage.csv", index=False)

    # ---- Fig: kind coverage bars
    if len(kt):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        w = .25
        for i, sd in enumerate(sorted(kt.seed.unique())):
            g = kt[kt.seed == sd].set_index("kind").reindex(KINDS)
            ax.bar(np.arange(len(KINDS)) + (i - 1) * w, g.coverage_pct, w, label=f"seed {sd}")
        ax.set_xticks(range(len(KINDS)), KINDS, rotation=20)
        ax.set_ylabel("% dos parâmetros do tipo em packs já enviados")
        ax.set_title("FedPHE p=0.1: cobertura por tipo de parâmetro", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"figures/32_kind_coverage.{ext}", dpi=200)
        plt.close(fig)

    # ---- Magnitude per parameter kind (why the L2 ranking is dominated by BN moving_variance)
    v = info.get(("fedphe", 43)) or next(iter(info.values()))
    arrays = load_snapshot(v["path"])
    byk = collections.defaultdict(list)
    for a, k in zip(arrays, layer_kinds(arrays)):
        byk[k].append(np.abs(a).ravel())
    mag = pd.DataFrame([dict(kind=k, n_params=sum(x.size for x in byk[k]),
                             mean_abs=np.concatenate(byk[k]).mean(), max_abs=np.concatenate(byk[k]).max(),
                             l2_total=np.linalg.norm(np.concatenate(byk[k]))) for k in KINDS if k in byk])
    mag.to_csv(OUT / "tables/32_kind_magnitude.csv", index=False)

    # ---- Client train accuracy vs server accuracy (local fit vs. synchronised model)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, sol in zip(axs, ("fedphe", "nvnl")):
        for sd, ls in zip((42, 43, 44), ("-", "--", ":")):
            r = next((x for x in runs if x["sol"] == sol and x["seed"] == sd), None)
            if r is None:
                continue
            t = pd.read_csv(r["path"] / "train.csv").groupby("round").train_acc.mean()
            e = pd.read_csv(r["path"] / "server_eval.csv").set_index("round").mean_acc
            ax.plot(t.index, t.rolling(5, min_periods=1).mean(), color="gray", ls=ls, lw=1)
            ax.plot(e.index, e.rolling(5, min_periods=1).mean(), color=st.color(sol), ls=ls, lw=1.4, label=f"seed {sd}")
        ax.set_title(f"{st.label(sol)} p=0.1 (cinza = acc. de treino dos clientes; cor = servidor)", fontsize=8)
        ax.set_xlabel("rodada")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("acurácia")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figures/32_train_vs_server.{ext}", dpi=200)
    plt.close(fig)
    print(mag.to_string(index=False))

    # ---- Seed sensitivity: packs only in seed 43
    sets = {sd: {p for p, c in info[("fedphe", sd)]["cnt"].items() if c > 0} for sd in (42, 43, 44) if ("fedphe", sd) in info}
    cnts = {sd: info[("fedphe", sd)]["cnt"] for sd in sets}
    rows = []
    for sd, s in sets.items():
        other = set().union(*[sets[o] for o in sets if o != sd])
        rows.append(dict(seed=sd, acc_final=acc[("fedphe", sd)], packs_ever=len(s), exclusive_packs=len(s - other),
                         exclusive_list=" ".join(map(str, sorted(s - other))),
                         sends_in_exclusive=sum(cnts[sd][p] for p in s - other)))
    allp = set().union(*sets.values())
    core = set.intersection(*sets.values())
    pd.DataFrame(rows).to_csv(OUT / "tables/32_seed_sensitivity.csv", index=False)
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(pd.DataFrame(nrm_rows).to_string(index=False))
    print(kt.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(pd.DataFrame(rows).to_string(index=False))
    print("union", len(allp), "core(all seeds)", len(core))


if __name__ == "__main__":
    main()
