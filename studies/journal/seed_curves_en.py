"""Seed repeats on MobileNetV2 (fig:seeds): per-round curves at p = 10% and round-200 points at p = 10%, 70%.

Seed 42 comes from logs/sparsification_percentage (p = 70% under RR, SW and Top-k from logs/range_fix,
as in paper_numbers.py); seeds 43 and 44 come from logs/seed_experiment_v2. Random has seed 42 only.
Accuracy per run is read as in ld.last_clean_acc. Also writes output/seed_curves.md with the numbers.

Usage:  python3 seed_curves_en.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_data as ld  # noqa: E402
import paper_style as ps  # noqa: E402

LOGS = Path(__file__).resolve().parents[2] / "logs"
SP, RANGE_FIX, SEEDS_V2 = LOGS / "sparsification_percentage", LOGS / "range_fix", LOGS / "seed_experiment_v2"
OUT = Path(os.environ.get("FIG_OUT") or Path(__file__).resolve().parent / "output" / "figures")
OUT.mkdir(parents=True, exist_ok=True)
NUMBERS = Path(__file__).resolve().parent / "output" / "seed_curves.md"

MODEL, DATASET, NIID = "mobilenetv2", "CIFAR10", "1"
POLICIES = ["sweepfl", "sw", "fedphe"]          # RR, SW, Top-k: three seeds
SEEDS = [42, 43, 44]
BUDGETS = ["0.1", "0.7"]
LAST_N = 10
ROUNDS = 200


def runs_of(sol: str, pct: str, seed: int):
    """Runs of one cell and seed; the one config_runs returns per config (newest complete)."""
    if seed == 42:
        root = RANGE_FIX if (pct == "0.7" and sol in POLICIES) else SP
    else:
        root = SEEDS_V2
    toks = [f"_niid{NIID}_", f"_percentage{pct}_", f"_seed{seed}_"]
    found = ld.config_runs(root, f"{MODEL}_{DATASET}_{sol}_", toks)
    if len(found) != 1:
        print(f"[warn] {sol} p={pct} seed={seed}: {len(found)} runs in {root.name}", file=sys.stderr)
    return found[-1] if found else None


def clean_series(run) -> np.ndarray:
    """Per-round server accuracy (%), cut before the first non-finite loss (ld.last_clean_acc rule)."""
    ev = ld.read_run(run, "server_eval.csv")
    bad = ev["mean_loss"].isna()
    if bad.any():
        ev = ev[ev["round"] < int(ev.loc[bad, "round"].min())]
    return 100.0 * ev.set_index("round")["mean_acc"].to_numpy()


def cell(sol: str, pct: str, seeds=SEEDS) -> dict:
    """Per-seed round-200 accuracy, last-ten mean and curves of one policy at one budget."""
    out = {"final": {}, "last10": {}, "curve": {}}
    for s in seeds:
        run = runs_of(sol, pct, s)
        if run is None:
            continue
        acc, _, _ = ld.last_clean_acc(run, ROUNDS)
        if acc is None:
            continue
        ser = clean_series(run)
        out["final"][s] = 100.0 * acc
        out["last10"][s] = float(ser[-LAST_N:].mean())
        out["curve"][s] = ser
    return out


def summary(vals: dict) -> tuple[float, float, float, float]:
    v = list(vals.values())
    return mean(v), (stdev(v) if len(v) > 1 else float("nan")), min(v), max(v)


def overlap(a: dict, b: dict) -> bool:
    return max(min(a.values()), min(b.values())) <= min(max(a.values()), max(b.values()))


def numbers(data: dict) -> list[str]:
    lines = ["## MobileNetV2, alpha=1, seeds 42/43/44: round 200 (mean ± sample std [min, max]; per seed)"]
    for pct in BUDGETS:
        lines.append(f"\n### p = {float(pct) * 100:.0f}%\n")
        lines.append("| policy | round 200 | per seed | last 10 rounds | per seed |")
        lines.append("|---|---|---|---|---|")
        for sol in POLICIES + ["rw"]:
            d = data[(sol, pct)]
            m, sd, lo, hi = summary(d["final"])
            m10, sd10, lo10, hi10 = summary(d["last10"])
            sd_txt = f" ± {sd:.1f}" if sd == sd else ""
            sd10_txt = f" ± {sd10:.1f}" if sd10 == sd10 else ""
            lines.append(f"| {ps.LABELS[sol]} | {m:.1f}{sd_txt} [{lo:.1f}, {hi:.1f}] | "
                         f"{', '.join(f'{s}: {v:.1f}' for s, v in d['final'].items())} | "
                         f"{m10:.1f}{sd10_txt} [{lo10:.1f}, {hi10:.1f}] | "
                         f"{', '.join(f'{s}: {v:.1f}' for s, v in d['last10'].items())} |")
        rr, sw = data[("sweepfl", pct)], data[("sw", pct)]
        lines.append("")
        for key, label in (("final", "round 200"), (f"last10", f"mean of last {LAST_N} rounds")):
            diff = mean(rr[key].values()) - mean(sw[key].values())
            state = "ranges overlap" if overlap(rr[key], sw[key]) else "ranges disjoint"
            lines.append(f"- RR - SW at {label}: {diff:+.1f} points in the mean, {state} "
                         f"(RR [{min(rr[key].values()):.1f}, {max(rr[key].values()):.1f}], "
                         f"SW [{min(sw[key].values()):.1f}, {max(sw[key].values()):.1f}])")
    return lines


def panel_curves(ax, data: dict) -> None:
    for sol in POLICIES:
        curves = list(data[(sol, "0.1")]["curve"].values())
        n = min(len(c) for c in curves)
        arr = np.stack([c[:n] for c in curves])
        x = np.arange(1, n + 1)
        ax.fill_between(x, arr.min(0), arr.max(0), color=ps.COLORS[sol], alpha=0.2, linewidth=0,
                        zorder=ps.ZORDER[sol] - 0.5)
        ax.plot(x, arr.mean(0), **ps.line_style(sol, rounds=True))
    rw = data[("rw", "0.1")]["curve"].get(42)
    if rw is not None:
        ax.plot(np.arange(1, len(rw) + 1), rw, **ps.line_style("rw", rounds=True))
    ps.apply_axes(ax)
    ax.set_xlim(0, ROUNDS)
    ax.set_ylim(0, 90)
    ax.set_xlabel("communication round", fontsize=7)
    ax.set_ylabel("accuracy (%)", fontsize=7)
    ax.tick_params(labelsize=6.8)
    ax.set_title("(a) $p$ = 10%", fontsize=7)


def panel_final(ax, data: dict, fedavg: float | None) -> None:
    sols = POLICIES + ["rw"]
    width = 0.18
    for gi, pct in enumerate(BUDGETS):
        for si, sol in enumerate(sols):
            x = gi + (si - (len(sols) - 1) / 2) * (width + 0.03)
            vals = list(data[(sol, pct)]["final"].values())
            ax.bar(x, mean(vals), width, color=ps.tint(ps.COLORS[sol], 0.72), edgecolor=ps.COLORS[sol],
                   linewidth=0.6, zorder=2)
            jitter = np.linspace(-width / 3, width / 3, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(x + np.asarray(jitter), vals, s=16, color=ps.COLORS[sol], edgecolor="black",
                       linewidth=0.5, zorder=4)
    if fedavg is not None:
        ax.axhline(fedavg, color=ps.COLORS["plaintext"], linestyle=ps.DASHES["plaintext"],
                   linewidth=ps.LINEWIDTHS["plaintext"], zorder=3)
    ps.apply_axes(ax)
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels([f"{float(b) * 100:.0f}" for b in BUDGETS], fontsize=6.8)
    ax.set_xlim(-0.55, len(BUDGETS) - 0.45)
    ax.set_ylim(0, 90)
    ax.set_xlabel("transmitted fraction $p$ (%)", fontsize=7)
    ax.set_ylabel("accuracy at round 200 (%)", fontsize=7)
    ax.tick_params(labelsize=6.8)
    ax.set_title("(b) round 200", fontsize=7)


def build() -> None:
    data = {(sol, pct): cell(sol, pct, SEEDS if sol in POLICIES else [42])
            for sol in POLICIES + ["rw"] for pct in BUDGETS}
    fed_run = ld.config_runs(SP, f"{MODEL}_{DATASET}_plaintext_", [f"_niid{NIID}_", "_percentage-1_", "_seed42_"])
    fedavg = None
    if fed_run:
        acc, _, _ = ld.last_clean_acc(fed_run[-1], ROUNDS)
        fedavg = None if acc is None else 100.0 * acc

    lines = numbers(data)
    if fedavg is not None:
        lines.append(f"\nPlaintext FedAvg, seed 42, round 200: {fedavg:.1f}")
    NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.4, 1]},
                                     constrained_layout=True)
    panel_curves(ax_a, data)
    panel_final(ax_b, data, fedavg)

    keys = ["sweepfl", "sw", "fedphe", "rw"]
    handles = [plt.Line2D([], [], **ps.line_style(k)) for k in keys]
    labels = [ps.LABELS[k] for k in keys]
    if fedavg is not None:
        handles.append(plt.Line2D([], [], color=ps.COLORS["plaintext"], linestyle=ps.DASHES["plaintext"],
                                  linewidth=ps.LINEWIDTHS["plaintext"]))
        labels.append("FedAvg (seed 42)")
    ps.legend(fig, handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=5,
              fontsize=6.8, handlelength=1.6)

    path = OUT / "seed-curves.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}\nwrote {NUMBERS}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "savefig.dpi": 300})
    build()
