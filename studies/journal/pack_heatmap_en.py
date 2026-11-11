#!/usr/bin/env python3
"""Per-packet send frequency heatmap, English labels, for the FGCS paper.

Reads mask.csv directly from the run logs. Needs no model snapshots, so it is
unaffected by the deleted round_*.npz files.

A cell is one (policy, packet index) pair; its value is the fraction of rounds in
which the server-side union of the received masks contained that packet. Zero is
drawn in a distinct colour, because the argument turns on the difference between
"never" and "rarely", which a sequential map renders almost identically.

Usage:  python3 pack_heatmap_en.py
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

LOGS = os.environ.get("LOGS") or os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "logs", "sparsification_percentage")
OUT = os.environ.get("FIG_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "figures")
os.makedirs(OUT, exist_ok=True)

POLICIES = [("nvnl", "Round Robin"), ("sw", "Sliding Window"),
            ("rw", "Random"), ("fedphe", "Top-$k$ (FedPHE)")]
PAIRS = [("lenet5", "FASHION_MNIST", "LeNet-5"),
         ("dnn_tiny", "FASHION_MNIST", "DNN"),
         ("vit_tiny", "tanganke_gtsrb", "ViT-Tiny"),
         ("resnet20", "CIFAR10", "ResNet-20"),
         ("mobilenetv2", "CIFAR10", "MobileNetV2")]
NIID, PCT = "1", "0.1"          # Dirichlet alpha = 1 (niid0.1 folders ran IID), p = 10%
FIRST_ROUND = 2                 # round 1 ran with 2 clients instead of 9


def _run_dirs(model: str, dataset: str, policy: str) -> list[str]:
    prefix = f"{model}_{dataset}_{policy}_"
    out = []
    for d in glob.glob(os.path.join(LOGS, "*")):
        b = os.path.basename(d)
        if b.startswith(prefix) and f"_niid{NIID}_" in b and f"_percentage{PCT}_" in b:
            if os.path.exists(os.path.join(d, "mask.csv")):
                out.append(d)
    return out


def send_frequency(model: str, dataset: str, policy: str
                   ) -> Optional[Tuple[np.ndarray, int]]:
    """Return (per-packet fraction of rounds in the union, round count)."""
    union: Dict[int, set] = defaultdict(set)
    n_packs = None
    for d in _run_dirs(model, dataset, policy):
        with open(os.path.join(d, "mask.csv"), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if int(row["round"]) < FIRST_ROUND:
                    continue
                n_packs = int(row["n_packs"])
                raw = row.get("selected_packs") or ""
                if raw:
                    union[int(row["round"])].update(
                        int(x) for x in raw.split(";") if x != "")
    if not union or n_packs is None:
        return None
    rounds = len(union)
    counts = np.zeros(n_packs)
    for packets in union.values():
        for p in packets:
            if 0 <= p < n_packs:
                counts[p] += 1
    return counts / rounds, rounds


def build() -> None:
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(7.2, 5.6),
                             constrained_layout=True)
    # sequential map for what is sent, one reserved colour for what never is
    base = plt.get_cmap("viridis")(np.linspace(0.12, 1.0, 255))
    cmap = ListedColormap(np.vstack([np.array([0.93, 0.93, 0.93, 1.0]), base]))
    cmap.set_under("#ededed")

    for ax, (model, dataset, label) in zip(np.atleast_1d(axes), PAIRS):
        rows, ticks = [], []
        for key, name in POLICIES:
            res = send_frequency(model, dataset, key)
            if res is None:
                continue
            freq, _ = res
            never = int((freq == 0).sum())
            rows.append(freq * 100.0)
            # one line per row: five stacked panels leave no room for two
            ticks.append(f"{name} ({never} starved)" if never else name)
        if not rows:
            continue
        grid = np.vstack(rows)
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0.001, vmax=100,
                       interpolation="nearest")
        ax.set_yticks(range(len(ticks)))
        ax.set_yticklabels(ticks, fontsize=6.5)
        ax.set_xlabel(f"packet index, {label} ({grid.shape[1]} packets)",
                      fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
        cb.set_label("rounds sent (%)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)

    for ext in ("pdf",):
        path = os.path.join(OUT, f"pack-heatmap-p10.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    # serif, to match the body font and the other figures of the paper
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "savefig.dpi": 300})
    build()
