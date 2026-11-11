#!/usr/bin/env python3
"""Final accuracy against the transmitted fraction, English labels, FGCS paper.

One panel per model-dataset pair, ordered by packet count, four selection
policies plus the plaintext FedAvg reference as a dashed horizontal line. The
figure carries the result that the boundary between structural and
gradient-guided selection discriminates where the task is hard and attenuates
where it is not.

Accuracy is the last round of each run, averaged over the runs of a
configuration, which is the basis of the full-model table in Section 7. Reruns
of one configuration are separate observations and both count, so this reads
every folder rather than the deduplicated set.

Dirichlet alpha = 1 (folders niid1; the niid0.1 folders ran IID). In five
MobileNetV2 runs the aggregate corrupts the model partway through: every client
drops to chance with a non-finite evaluation loss while local training keeps
improving (all four policies at p = 90%, Top-k at p = 70%). For those,
ld.last_clean_acc reads the accuracy at the last round before the first
corrupted aggregate, which the caption of the figure states.

Round Robin, Sliding Window and Top-k on MobileNetV2 at p = 70 and 90% are read
instead from logs/range_fix, which reran those cells with the MAGG_D_REF
correction and complete all 200 rounds. Random keeps the main-sweep runs, since
range_fix starts round 1 with all nine clients and that changes the Random
policy (correcoes/14).

Usage:  python3 accuracy_vs_budget_en.py
"""
from __future__ import annotations

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

_LOGS_DEFAULTS = [os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "logs", "sparsification_percentage")]
LOGS = os.environ.get("LOGS") or next((p for p in _LOGS_DEFAULTS if os.path.isdir(p)), _LOGS_DEFAULTS[0])
OUT = os.environ.get("FIG_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "figures")
os.makedirs(OUT, exist_ok=True)

POLICIES = ["fedphe", "rw", "sweepfl", "sw"]
PAIRS = [("lenet5", "FASHION_MNIST", "LeNet-5", 16),
         ("dnn_tiny", "FASHION_MNIST", "DNN", 28),
         ("vit_tiny", "tanganke_gtsrb", "ViT-Tiny", 52),
         ("resnet20", "CIFAR10", "ResNet-20", 67),
         ("mobilenetv2", "CIFAR10", "MobileNetV2", 176)]
BUDGETS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
NIID = "1"   # Dirichlet alpha = 1; the niid0.1 folders ran IID
RANGE_FIX = os.path.join(os.path.dirname(LOGS), "range_fix")
RANGE_FIX_CELLS = {("mobilenetv2", pol, pct) for pol in ("sweepfl", "sw", "fedphe") for pct in ("0.7", "0.9")}


def final_accuracy(model: str, dataset: str, policy: str, pct: str) -> Optional[float]:
    """Mean over runs of the accuracy ld.last_clean_acc reads, in percent."""
    root = RANGE_FIX if (model, policy, pct) in RANGE_FIX_CELLS else LOGS
    runs = ld.config_runs(root, f"{model}_{dataset}_{policy}_", [f"_niid{NIID}_", f"_percentage{pct}_"])
    per_run = [a for a, _, _ in (ld.last_clean_acc(r) for r in runs) if a is not None]
    return 100.0 * sum(per_run) / len(per_run) if per_run else None


def build() -> None:
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(7.2, 1.95),
                             sharey=False, constrained_layout=True)
    xs = [float(b) * 100 for b in BUDGETS]

    for ax, (model, dataset, label, packs) in zip(axes, PAIRS):
        ref = final_accuracy(model, dataset, "plaintext", "-1")
        if ref is not None:
            ax.axhline(ref, color=ps.COLORS["plaintext"], linestyle=ps.LINESTYLES["plaintext"],
                       linewidth=ps.LINEWIDTHS["plaintext"])
        for policy in POLICIES:
            ys = [final_accuracy(model, dataset, policy, b) for b in BUDGETS]
            ys = [np.nan if v is None else v for v in ys]
            ax.plot(xs, ys, **ps.line_style(policy))
        ps.apply_axes(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(v)}" for v in xs], fontsize=6.8)
        ax.tick_params(axis="y", labelsize=6.8)
        # 0-100 left the lower half of every panel empty
        ax.set_ylim(20, 95)
        ax.set_title(f"{label}\n$P$ = {packs}", fontsize=7)

    axes[0].set_ylabel("final accuracy (%)", fontsize=7)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.supxlabel("transmitted fraction $p$ (%)", fontsize=7.5)

    keys = ["plaintext"] + POLICIES
    handles = [plt.Line2D([], [], **ps.line_style(k)) for k in keys]
    ps.legend(fig, handles, [ps.LABELS[k] for k in keys], loc="lower center",
              bbox_to_anchor=(0.5, -0.13), ncol=5, fontsize=6.8, handlelength=1.6)

    path = os.path.join(OUT, "accuracy-vs-budget.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "savefig.dpi": 300})
    build()
