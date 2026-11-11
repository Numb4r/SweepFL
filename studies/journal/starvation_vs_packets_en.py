#!/usr/bin/env python3
"""Starvation against packet count, English labels, for the FGCS paper.

A packet is *starved* when it never enters the server-side union of the received
masks over the whole run. Two panels, both computed from mask.csv alone:

  (a) starved fraction against the packet count P, at p = 10%, four policies,
      Dirichlet alpha = 1 (solid) and the IID reference (dashed). The niid0.1
      folders ran with IidPartitioner, so they are the IID curve. Round Robin,
      Sliding Window and Random sit on zero for every pair; Top-k rises
      with P.
  (b) starved fraction against the transmitted fraction p, Top-k on each of the
      five pairs at alpha = 1, which locates the effect at the restrictive end of
      the budget range. On MobileNetV2 at p = 70% and 90% the masks come from
      logs/range_fix, as in accuracy_vs_budget_en.py; the count is zero there in
      both sources.

Round 1 is dropped throughout: it ran with 2 clients where every other round ran
with 9, so its union is not comparable.

Usage:  python3 starvation_vs_packets_en.py
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_data as ld  # noqa: E402
import paper_style as ps  # noqa: E402

_LOGS_DEFAULTS = [os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "logs", "sparsification_percentage")]
LOGS = os.environ.get("LOGS") or next((p for p in _LOGS_DEFAULTS if os.path.isdir(p)), _LOGS_DEFAULTS[0])
RANGE_FIX = os.path.join(os.path.dirname(LOGS), "range_fix")
RANGE_FIX_CELLS = {("mobilenetv2", "fedphe", "1", pct) for pct in ("0.7", "0.9")}
OUT = os.environ.get("FIG_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "figures")
os.makedirs(OUT, exist_ok=True)

# draw order: Round Robin (largest marker) first, so Sliding Window and Random overlay it at zero
POLICIES = [(k, ps.LABELS[k]) for k in ("nvnl", "sw", "rw", "fedphe")]
ZORDER = {"nvnl": 3, "sw": 4, "rw": 5, "fedphe": 6}
SIZE = {"nvnl": 7, "sw": 4.5, "rw": 4.5, "fedphe": ps.MARKERSIZE}
# ordered by packet count, which is the x axis of panel (a)
PAIRS = [("lenet5", "FASHION_MNIST", "LeNet-5"),
         ("dnn_tiny", "FASHION_MNIST", "DNN"),
         ("vit_tiny", "tanganke_gtsrb", "ViT-Tiny"),
         ("resnet20", "CIFAR10", "ResNet-20"),
         ("mobilenetv2", "CIFAR10", "MobileNetV2")]
BUDGETS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
FIRST_ROUND = 2   # round 1 ran with 2 clients instead of 9


MAIN, REF = "1", "0.1"   # Dirichlet alpha = 1; the niid0.1 folders ran IID


def starved(model: str, dataset: str, policy: str, niid: str, pct: str
            ) -> Optional[Tuple[int, int]]:
    """(packets never sent, total packets) over the run, or None when absent or collapsed."""
    root = RANGE_FIX if (model, policy, niid, pct) in RANGE_FIX_CELLS else LOGS
    runs = ld.config_runs(root, f"{model}_{dataset}_{policy}_",
                          [f"_niid{niid}_", f"_percentage{pct}_"])
    if not runs:
        return None
    # multi-run configs are Round Robin reruns, whose masks coincide
    n_packs, masks = ld.run_masks(runs[0], first_round=FIRST_ROUND)
    if policy == "fedphe":
        # the ranking is meaningless once the aggregate carries non-finite weights
        _, last_clean, collapsed = ld.last_clean_acc(runs[0])
        if collapsed:
            masks = {t: m for t, m in masks.items() if t <= last_clean}
    seen = set().union(*(m for ms in masks.values() for m in ms))
    return n_packs - len(seen), n_packs


def build() -> None:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)

    # ── (a) starved fraction vs packet count, p = 10% ───────────────────────
    xs = [starved(m, d, "nvnl", MAIN, "0.1")[1] for m, d, _ in PAIRS]
    for policy, _ in POLICIES:
        for niid, part in ((MAIN, "non-iid"), (REF, "iid")):
            ys = []
            for model, dataset, _ in PAIRS:
                r = starved(model, dataset, policy, niid, "0.1")
                ys.append(100.0 * r[0] / r[1] if r else np.nan)
            st = ps.partition_style({**ps.line_style(policy), "markersize": SIZE[policy],
                                     "zorder": ZORDER[policy]}, part)
            if niid != MAIN:
                st.pop("label")
            axa.plot(xs, ys, **st)
    # LeNet-5's one starved Top-k packet is the zero-padded last packet (266 real
    # weights of 4096 slots, 0.43% of the model): kept in the count, drawn hollow
    for niid in (MAIN, REF):
        r = starved(PAIRS[0][0], PAIRS[0][1], "fedphe", niid, "0.1")
        axa.plot([xs[0]], [100.0 * r[0] / r[1]], ps.MARKERS["fedphe"], markerfacecolor="white",
                 markeredgecolor=ps.COLORS["fedphe"], markersize=ps.MARKERSIZE,
                 markeredgewidth=1.1, zorder=7)
    axa.annotate("padding packet\n(0.43% of the model)", xy=(xs[0], 6.25),
                 xytext=(xs[0] * 1.45, 11), fontsize=6, color=ps.TEXT_SECONDARY,
                 arrowprops=dict(arrowstyle="-", color=ps.TEXT_SECONDARY, lw=0.5))
    axa.set_xscale("log")
    axa.set_xticks(xs)
    axa.set_xticklabels([str(x) for x in xs], fontsize=7)
    axa.minorticks_off()
    axa.tick_params(axis="y", labelsize=7)
    axa.set_xlabel("packet count $P$", fontsize=7.5)
    axa.set_ylabel("starved packets (% of $P$)", fontsize=7.5)
    axa.set_ylim(-4, 100)
    axa.set_title("(a) starved packets vs. packet count, $p$ = 10%", fontsize=8)

    # ── (b) starved fraction vs budget, Top-k, alpha = 1 ────────────────────
    xb = [float(b) * 100 for b in BUDGETS]
    pair_key = {"dnn_tiny": "dnn", "vit_tiny": "vit"}
    for model, dataset, label in PAIRS:
        key = pair_key.get(model, model)
        ys = []
        for pct in BUDGETS:
            r = starved(model, dataset, "fedphe", MAIN, pct)
            ys.append(100.0 * r[0] / r[1] if r else np.nan)
        axb.plot(xb, ys, "-", color=ps.FEDPHE_RAMP[key], linewidth=ps.LINEWIDTHS["fedphe"],
                 marker=ps.PAIR_MARKERS[key], markersize=ps.MARKERSIZE, label=label)
    # same padding packet: LeNet-5 stays at 1/16 at every budget
    lw = [100.0 * r[0] / r[1] for r in (starved("lenet5", "FASHION_MNIST", "fedphe", MAIN, b)
                                         for b in BUDGETS)]
    axb.plot(xb, lw, ps.PAIR_MARKERS["lenet5"], markerfacecolor="white",
             markeredgecolor=ps.FEDPHE_RAMP["lenet5"], markersize=ps.MARKERSIZE,
             markeredgewidth=1.1, zorder=5)
    axb.set_xticks(xb)
    axb.set_xticklabels([f"{int(v)}" for v in xb], fontsize=7)
    axb.tick_params(axis="y", labelsize=7)
    axb.set_xlabel("transmitted fraction $p$ (%)", fontsize=7.5)
    axb.set_ylabel("starved packets (% of $P$)", fontsize=7.5)
    axb.set_ylim(-4, 100)
    axb.set_title("(b) Top-$k$ (FedPHE) only, by budget", fontsize=8)
    ps.legend(axb, loc="upper right", fontsize=6.2, handlelength=1.4, labelspacing=0.25)

    pol_leg = ps.legend(axa, loc="upper left", fontsize=6.6, handlelength=1.4, labelspacing=0.25)
    axa.add_artist(pol_leg)
    # solid/filled vs dotted/hollow carries the heterogeneity level, which only Top-k resolves
    g = ps.TEXT_SECONDARY
    cue = [plt.Line2D([], [], color=g, linestyle="-", marker="o", markersize=4),
           plt.Line2D([], [], color=g, linestyle=":", marker="o", markersize=4, markerfacecolor="white")]
    ps.legend(axa, cue, [r"Dirichlet $\alpha = 0.1$", "IID"], loc="lower right",
              fontsize=6.6, handlelength=1.6, labelspacing=0.25)
    for ax in (axa, axb):
        ps.apply_axes(ax)

    path = os.path.join(OUT, "starvation-vs-packets.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "savefig.dpi": 300})
    build()
