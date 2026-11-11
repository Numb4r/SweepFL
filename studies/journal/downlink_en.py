#!/usr/bin/env python3
"""Downlink figures for the FGCS paper, English labels.

Three panels, all computed from mask.csv, train.csv and server_eval.csv:

  A  overhead factor by transmitted fraction (union size / per-client upload)
  B  union size per round
  C  cumulative download in GB against accuracy

Download is derived from the masks and counted in nominal ciphertext bytes, the
unit of the upload in tab:fullmodel: each round the server returns the union of
the round's masks to every participating client, so a round costs
|union| x CT_BYTES x clients, with CT_BYTES = 2 * N * 8 = 131 072 B (one
single-prime CKKS ciphertext, N = 8192). Before 2026-09-30 panel C counted
4 B per parameter (float32 plaintext), about 8x less; the factors do not change.

Folders niid1. The values match tab:downlink5: at p=30% on ResNet-20 1.00x for
Round Robin and Sliding Window, 1.24x for Top-k, 1.68x for Random; download
4.71 / 5.85 / 7.90 GB over rounds 11-200.

Usage:  python3 downlink_en.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

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

POLICIES = ["fedphe", "rw", "nvnl", "sw"]
BUDGETS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
MODEL, DATASET, NIID = "resnet20", "CIFAR10", "1"
WARMUP = 0.05
CT_BYTES = 2 * 8192 * 8   # one CKKS ciphertext, N = 8192, single prime


def _runs(policy: str, pct: str):
    return ld.config_runs(LOGS, f"{MODEL}_{DATASET}_{policy}_",
                          [f"_niid{NIID}_", f"_percentage{pct}_"])


def series(policy: str, pct: str) -> Optional[Dict]:
    """Per-round union size and per-client upload, as fractions of the model."""
    runs = _runs(policy, pct)
    if not runs:
        return None
    # the only multi-run configs here are Round Robin reruns, whose masks coincide
    n_packs, masks = ld.run_masks(runs[0], first_round=2)
    if not masks:
        return None
    n_params = int(ld.read_run(runs[0], "train.csv")["n_params"].iloc[0])
    rounds = sorted(masks)[int(len(masks) * WARMUP):]
    return {
        "rounds": rounds,
        "union": np.array([len(set().union(*masks[r])) / n_packs for r in rounds]),
        "client": np.array([np.mean([len(m) for m in masks[r]]) / n_packs for r in rounds]),
        "n_params": n_params,
        "n_packs": n_packs,
    }


def accuracy_and_clients(policy: str, pct: str) -> Tuple[Optional[np.ndarray],
                                                         Optional[np.ndarray]]:
    """Per-round accuracy, mean over the runs of the configuration, and client count."""
    evs = [ld.read_run(r, "server_eval.csv") for r in _runs(policy, pct)]
    evs = [ev.set_index("round").loc[1:200] for ev in evs if ev is not None and len(ev) >= 200]
    if not evs:
        return None, None
    acc = np.mean([ev["mean_acc"].to_numpy() for ev in evs], axis=0)
    return acc, evs[0]["n_clients"].to_numpy()


def build() -> None:
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.2, 2.35),
                                        constrained_layout=True)

    # ---- A: overhead factor by budget -----------------------------------
    width = 0.2
    xs = np.arange(len(BUDGETS))
    for i, key in enumerate(POLICIES):
        vals = []
        for pct in BUDGETS:
            s = series(key, pct)
            vals.append(float(s["union"].mean() / s["client"].mean()) if s else np.nan)
        axa.bar(xs + (i - 1.5) * width, vals, width, **ps.bar_style(key))
    axa.axhline(1.0, color=ps.TEXT_SECONDARY, lw=0.8, ls="--", zorder=0)
    axa.set_xticks(xs)
    axa.set_xticklabels([f"{int(float(b)*100)}" for b in BUDGETS], fontsize=7)
    axa.set_xlabel("transmitted fraction $p$ (%)", fontsize=7.5)
    axa.set_ylabel("overhead factor", fontsize=7.5)
    axa.tick_params(labelsize=7)
    axa.set_title("(a) overhead factor", fontsize=8)

    # ---- B: union size per round -----------------------------------------
    for key in POLICIES:
        s = series(key, "0.3")
        if s:
            axb.plot(s["rounds"], s["union"] * 100, **ps.line_style(key, rounds=True))
    s0 = series("nvnl", "0.3")
    if s0:
        axb.axhline(float(s0["client"].mean() * 100), color=ps.TEXT_SECONDARY, lw=0.8, ls="--")
    axb.set_xlabel("round", fontsize=7.5)
    axb.set_ylabel("union size (% of packets)", fontsize=7.5)
    axb.tick_params(labelsize=7)
    axb.set_title("(b) union size, $p$ = 30%", fontsize=8)

    # ---- C: cumulative download against accuracy -------------------------
    for key in POLICIES:
        s = series(key, "0.3")
        acc, ncli = accuracy_and_clients(key, "0.3")
        if s is None or acc is None:
            continue
        n = min(len(s["rounds"]), len(acc))
        idx = np.array(s["rounds"][:n]) - 1
        union_packs = np.rint(s["union"][:n] * s["n_packs"])
        per_round = union_packs * CT_BYTES * ncli[idx]
        axc.plot(np.cumsum(per_round) / 1e9, acc[idx] * 100, **ps.line_style(key, rounds=True))
    axc.set_xlabel("cumulative downlink (GB)", fontsize=7.5)
    axc.set_ylabel("accuracy (%)", fontsize=7.5)
    axc.tick_params(labelsize=7)
    axc.set_title("(c) cost of the downlink, $p$ = 30%", fontsize=8)

    for ax in (axa, axb, axc):
        ps.apply_axes(ax)
    handles = [plt.Line2D([], [], **ps.line_style(k)) for k in POLICIES]
    ps.legend(fig, handles, [ps.LABELS[k] for k in POLICIES], loc="upper center",
              bbox_to_anchor=(0.5, 1.10), ncol=4, fontsize=7.5)

    path = os.path.join(OUT, "downlink-resnet20.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    # serif, to match the body font and the other figures of the paper
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "savefig.dpi": 300})
    build()
