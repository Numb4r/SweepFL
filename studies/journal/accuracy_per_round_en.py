#!/usr/bin/env python3
"""Accuracy per round on all five pairs, English labels, for the FGCS paper.

One panel per model-dataset pair, ordered by packet count. Full-model references
are dashed, partial-upload policies at p = 10% are solid. The figure exists to
show that the decay of BFV and BatchCrypt is confined to ResNet-20: on the other
four pairs every reference tracks plaintext FedAvg to round 200.

Accuracy is the server-side mean over clients, averaged over the runs of a
configuration. Reruns of one configuration are separate observations and both
count, which is the basis of the full-model table in Section 7.

Dirichlet alpha = 1 (folders niid1; the niid0.1 folders ran IID), p = 10%.

Usage:  python3 accuracy_per_round_en.py
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

# dashed: full-model references.  solid: partial upload at p = 10%.
REFERENCES = [(k, "-1") for k in ("plaintext", "ckks", "bfv", "batchcrypt")]
POLICIES = [(k, "0.1") for k in ("fedphe", "rw", "nvnl", "sw")]
PAIRS = [("lenet5", "FASHION_MNIST", "LeNet-5", 16),
         ("dnn_tiny", "FASHION_MNIST", "DNN", 28),
         ("vit_tiny", "tanganke_gtsrb", "ViT-Tiny", 52),
         ("resnet20", "CIFAR10", "ResNet-20", 67),
         ("mobilenetv2", "CIFAR10", "MobileNetV2", 176)]
NIID = "1"


def curve(model: str, dataset: str, sol: str, pct: str) -> Optional[np.ndarray]:
    """Per-round accuracy in percent, averaged over the runs of the config."""
    runs = ld.config_runs(LOGS, f"{model}_{dataset}_{sol}_", [f"_niid{NIID}_", f"_percentage{pct}_"])
    evs = [ld.read_run(r, "server_eval.csv") for r in runs]
    evs = [ev.set_index("round")["mean_acc"] for ev in evs if ev is not None]
    if not evs:
        return None
    rounds = sorted(set.intersection(*(set(ev.index) for ev in evs)))
    return np.mean([ev.loc[rounds].to_numpy() for ev in evs], axis=0) * 100.0


def build() -> None:
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(7.2, 2.0),
                             constrained_layout=True)

    for ax, (model, dataset, label, packs) in zip(axes, PAIRS):
        for sol, pct in REFERENCES + POLICIES:
            y = curve(model, dataset, sol, pct)
            if y is not None:
                ax.plot(range(1, len(y) + 1), y, **ps.line_style(sol, rounds=True))
        ps.apply_axes(ax)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 100, 200])
        ax.tick_params(labelsize=6.8)
        ax.set_title(f"{label}\n$P$ = {packs}", fontsize=7)

    axes[0].set_ylabel("accuracy (%)", fontsize=7)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.supxlabel("communication round", fontsize=7.5)

    handles = [plt.Line2D([], [], **ps.line_style(k)) for k, _ in REFERENCES + POLICIES]
    ps.legend(fig, handles, [ps.LABELS[k] for k, _ in REFERENCES + POLICIES], loc="lower center",
              bbox_to_anchor=(0.5, -0.20), ncol=4, fontsize=6.6, handlelength=1.8)

    path = os.path.join(OUT, "accuracy-per-round.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "savefig.dpi": 300})
    build()
