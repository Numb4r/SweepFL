# Reconstructed from the bytecode (journal-nvnl/figures/scripts/__pycache__/clientfrac_en.cpython-310.pyc)
# on 2026-11-11. The original source was lost. Data logic (run/accuracy/factor, panels, axes, limits) follows the
# bytecode; colors, markers, labels and axis styling come from paper_style.py (the bytecode predates it and
# hard-coded an older palette).
"""Partial participation sweep (60 clients), English labels.

Four columns, one per (pair, budget): ResNet-20 and MobileNetV2 at p = 10% and 50%.
Top row: final accuracy against the sampled fraction f_c, four policies, with
plaintext FedAvg under the same f_c dashed. Bottom row: downlink overhead factor
(union over one client's k packets, rounds 1..10 excluded as warm-up, as in
Table downlink5).

The figure publishes the twelve configurations the text of Section 8.4 relies on.

Dirichlet alpha = 1 (folders niid1). Usage:  python3 clientfrac_en.py
"""
from __future__ import annotations

import collections
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_data as ld  # noqa: E402
import paper_style as ps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.environ.get("LOGS") or os.path.join(HERE, "..", "..", "logs", "client_frac")
OUT = os.environ.get("FIG_OUT") or os.path.join(HERE, "output", "figures")
os.makedirs(OUT, exist_ok=True)

POLICIES = ["nvnl", "sw", "rw", "fedphe"]
PANELS = [("resnet20", "ResNet-20", 67, "0.1"), ("resnet20", "ResNet-20", 67, "0.5"),
          ("mobilenetv2", "MobileNetV2", 176, "0.1"), ("mobilenetv2", "MobileNetV2", 176, "0.5")]
FRACTIONS = ["0.25", "0.5", "0.75"]
NIID = "1"
WARMUP = 10


def run(model: str, sol: str, pct: str, cf: str):
    runs = ld.config_runs(LOGS, f"{model}_CIFAR10_{sol}_",
                          [f"_niid{NIID}_", f"_percentage{pct}_", f"_cf{cf}_"])
    return runs[0] if runs else None


def accuracy(model: str, sol: str, pct: str, cf: str) -> Optional[float]:
    r = run(model, sol, pct, cf)
    a = ld.last_clean_acc(r)[0] if r is not None else None
    return None if a is None else 100.0 * a


def factor(model: str, sol: str, pct: str, cf: str) -> Optional[float]:
    """Mean per-round union of the clients' packets over one client's k (rounds > WARMUP)."""
    r = run(model, sol, pct, cf)
    m = ld.read_run(r, "mask.csv") if r is not None else None
    if m is None:
        return None
    k = int(m["packs_sent"].iloc[0])
    union = collections.defaultdict(set)
    for t, sp in zip(m["round"], m["selected_packs"].astype(str)):
        if t > WARMUP:
            union[t] |= {int(i) for i in sp.split(";") if i and i != "nan"}
    return sum(len(u) for u in union.values()) / len(union) / k


def _nan(vals):
    return [np.nan if v is None else v for v in vals]


def build() -> None:
    fig, axes = plt.subplots(2, len(PANELS), figsize=(7.2, 3.6), constrained_layout=True,
                             sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0]})
    xs = [float(f) for f in FRACTIONS]
    for j, (model, label, packs, pct) in enumerate(PANELS):
        top, bot = axes[0, j], axes[1, j]
        ref = [accuracy(model, "plaintext", "-1", cf) for cf in FRACTIONS]
        top.plot(xs, _nan(ref), **{**ps.line_style("plaintext"), "marker": None, "label": None})
        for sol in POLICIES:
            acc = [accuracy(model, sol, pct, cf) for cf in FRACTIONS]
            fac = [factor(model, sol, pct, cf) for cf in FRACTIONS]
            style = {**ps.line_style(sol), "label": None}
            top.plot(xs, _nan(acc), **style)
            bot.plot(xs, _nan(fac), **style)
        top.set_title(f"{label}, $P$ = {packs}\n$p = {int(float(pct) * 100)}\\%$", fontsize=7)
        top.set_ylim(0, 85)
        bot.set_ylim(0.5, 9)
        bot.set_xticks(xs)
        bot.set_xticklabels(["0.25", "0.50", "0.75"], fontsize=6.8)
        for ax in (top, bot):
            ps.apply_axes(ax)
            ax.tick_params(labelsize=6.8)
            ax.set_xlim(0.18, 0.82)
        if j:
            top.set_yticklabels([])
            bot.set_yticklabels([])
    axes[0, 0].set_ylabel("final accuracy (%)", fontsize=7)
    axes[1, 0].set_ylabel("overhead factor", fontsize=7)
    fig.supxlabel("sampled client fraction $f_c$ ($N_c = 60$)", fontsize=7.5)

    keys = POLICIES  # legend order of the original figure
    handles = [plt.Line2D([], [], **{**ps.line_style(k), "label": None}) for k in keys]
    handles.append(plt.Line2D([], [], **{**ps.line_style("plaintext"), "marker": None, "label": None}))
    labels = [ps.LABELS[k] for k in keys] + ["plaintext FedAvg"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=5, fontsize=6.8,
               frameon=False, handlelength=1.8)
    path = os.path.join(OUT, "clientfrac.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "axes.grid": True,
                         "grid.linestyle": ":", "grid.alpha": 0.6, "axes.axisbelow": True,
                         "savefig.dpi": 300})
    build()
