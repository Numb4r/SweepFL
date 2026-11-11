#!/usr/bin/env python3
"""Every number the manuscript cites, recomputed from the run logs.

Main condition is Dirichlet alpha = 1 (folders `niid1`). The `niid0.1` folders of
the sparsification and client-fraction sweeps ran with IidPartitioner (bug fixed
in keys@7c5984b) and are reported as the IID reference. The IID columns must
reproduce the numbers the manuscript carried before the switch; that is the check
on this pipeline.

Writes output/paper_numbers.md.   Usage:  python3 paper_numbers.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import load_data as ld  # noqa: E402

LOGS = Path(__file__).resolve().parents[2] / "logs"
SP, CF = LOGS / "sparsification_percentage", LOGS / "client_frac"
OUT = HERE / "output" / "paper_numbers.md"

PAIRS = [("lenet5", "FASHION_MNIST", "LeNet-5", 16), ("dnn_tiny", "FASHION_MNIST", "DNN", 28),
         ("vit_tiny", "tanganke_gtsrb", "ViT-Tiny", 52), ("resnet20", "CIFAR10", "ResNet-20", 67),
         ("mobilenetv2", "CIFAR10", "MobileNetV2", 176)]
CIFAR = [p for p in PAIRS if p[1] == "CIFAR10"]
POLS = [("sweepfl", "RR"), ("sw", "SW"), ("rw", "RW"), ("fedphe", "Top-k")]
FULL = [("plaintext", "FedAvg"), ("ckks", "CKKS"), ("bfv", "BFV"), ("batchcrypt", "BatchC.")]
BUDGETS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
FRACS = ["0.25", "0.5", "0.75"]
CONDS = [("1", "alpha=1"), ("0.1", "IID")]
WARMUP = 0.05
lines: list[str] = []


def out(s: str = "") -> None:
    lines.append(s)


RANGE_FIX = LOGS / "range_fix"
USE_RANGE_FIX = True


def runs(root, model, dataset, sol, niid, pct, cf=None):
    """Runs of one configuration.

    MobileNetV2 at p = 70 and 90% under RR, SW and Top-k is read from logs/range_fix,
    the reruns with the MAGG_D_REF divisor that complete all 200 rounds and replace the
    five main-sweep runs whose aggregate corrupted the model (same rule as
    accuracy_vs_budget_en.py). Random keeps the main-sweep runs.
    """
    if (USE_RANGE_FIX and root == SP and model == "mobilenetv2" and niid == "1"
            and pct in ("0.7", "0.9") and sol in ("sweepfl", "sw", "fedphe")):
        root = RANGE_FIX
    toks = [f"_niid{niid}_", f"_percentage{pct}_"] + ([f"_cf{cf}_"] if cf else [])
    return ld.config_runs(root, f"{model}_{dataset}_{sol}_", toks)


final_acc = ld.final_acc


def accs(root, *key, cf=None):
    """Accuracy per run under the rule of ld.last_clean_acc, with the round it was read at."""
    return [ld.last_clean_acc(r) for r in runs(root, *key, cf=cf)]


def macc(values):
    """Mean over the runs that produced a value; None when none did."""
    ok = [a for a, _, _ in values if a is not None]
    return mean(ok) if ok else None


def collapse_round(run):
    """Last round before the aggregate corrupted the model, or None when it never did."""
    _, t, collapsed = ld.last_clean_acc(run)
    return t if collapsed else None


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    return v if isinstance(v, str) else f"{v:.{nd}f}"


def downlink(run, pct_download=False, until=None):
    """(overhead factor, starved packets, download MB) of one run, rounds >= 2.

    `until` caps the window at the last round before a corrupted aggregate, for the runs
    whose selection then ranks non-finite weights.
    """
    n_packs, masks = ld.run_masks(run, first_round=2)
    if until is not None:
        masks = {t: m for t, m in masks.items() if t <= until}
    if not masks:
        return None
    rounds = sorted(masks)
    seen = set().union(*(m for ms in masks.values() for m in ms))
    kept = rounds[int(len(rounds) * WARMUP):]
    union = np.array([len(set().union(*masks[r])) for r in kept], float)
    client = np.array([np.mean([len(m) for m in masks[r]]) for r in kept])
    factor = union.mean() / client.mean()
    mb = None
    if pct_download:
        n_params = int(ld.read_run(run, "train.csv")["n_params"].iloc[0])
        ev = ld.read_run(run, "server_eval.csv").set_index("round")["n_clients"]
        mb = sum(len(set().union(*masks[r])) / n_packs * n_params * 4 * ev.get(r, 0)
                 for r in kept) / 1e6
    return factor, n_packs - len(seen), mb


def distinct_masks(run):
    _, masks = ld.run_masks(run, first_round=2)
    return mean(len({frozenset(m) for m in ms}) for ms in masks.values()) if masks else None


def round_time(run):
    tr = ld.read_run(run, "train.csv")
    return float((tr["train_time_s"] + tr["encrypt_time_s"] + tr["decrypt_time_s"]).mean())


# --------------------------------------------------------------------------- tables
def cell(v) -> str:
    """Mean accuracy; the per-run list only when the cell averages several runs."""
    return fmt(macc(v)) + (f" {[fmt(a) for a, _, _ in v]}" if len(v) > 1 else "")


def table_fullmodel():
    out("## tab:fullmodel (final accuracy; newest run of each config; [per-run] shown only when a cell has several runs)")
    for niid, cname in CONDS:
        out(f"\n### {cname}\n")
        out("| pair | FedAvg | CKKS | BFV | BatchC. | RR p30 | RR p10 |")
        out("|---|---|---|---|---|---|---|")
        for m, d, name, _ in PAIRS:
            cells = []
            for sol, _ in FULL:
                v = accs(SP, m, d, sol, niid, "-1")
                cells.append(cell(v))
            for pct in ("0.3", "0.1"):
                v = accs(SP, m, d, "sweepfl", niid, pct)
                cells.append(cell(v))
            out(f"| {name} | " + " | ".join(cells) + " |")


def table_policies():
    out("\n## Final accuracy by policy and budget (tab/text of sec:boundary, fig:accbudget)")
    for niid, cname in CONDS:
        out(f"\n### {cname}\n")
        out("| pair | policy | " + " | ".join(f"p{b}" for b in BUDGETS) + " |")
        out("|---|---|" + "---|" * len(BUDGETS))
        for m, d, name, _ in PAIRS:
            for sol, lab in POLS:
                cells = []
                for b in BUDGETS:
                    v = accs(SP, m, d, sol, niid, b)
                    mark = min((t for _, t, c in v if c and t), default=None)
                    cells.append(fmt(macc(v)) + (f" (r{mark})" if mark else ""))
                out(f"| {name} | {lab} | " + " | ".join(cells) + " |")
        out("\nSpread among the four policies at p=10% (points):")
        for m, d, name, _ in PAIRS:
            v = [macc(accs(SP, m, d, s, niid, "0.1")) for s, _ in POLS]
            v = [x for x in v if isinstance(x, float)]
            out(f"- {name}: {100 * (max(v) - min(v)):.1f}")


def table_downlink():
    out("\n## tab:downlink5 (overhead factor, warm-up 5%, rounds>=2) + starvation")
    for niid, cname in CONDS:
        out(f"\n### {cname}\n")
        out("| pair | policy | " + " | ".join(f"p{b}" for b in BUDGETS) + " | Download p30 (MB) |")
        out("|---|---|" + "---|" * (len(BUDGETS) + 1))
        starv = {}
        for m, d, name, P in PAIRS:
            for sol, lab in POLS:
                cells, mb = [], None
                for b in BUDGETS:
                    rs = runs(SP, m, d, sol, niid, b)
                    if not rs:
                        cells.append("n/a"); continue
                    # Top-k ranks the weights, so its masks stop meaning anything once the
                    # aggregate is corrupted; the other three draw masks that do not.
                    res = [downlink(r, pct_download=(b == "0.3"),
                                    until=collapse_round(r) if sol == "fedphe" else None)
                           for r in rs]
                    res = [x for x in res if x]
                    cells.append(f"{mean(x[0] for x in res):.2f}")
                    starv[(name, lab, b)] = mean(x[1] for x in res)
                    if b == "0.3":
                        mb = mean(x[2] for x in res)
                out(f"| {name} | {lab} | " + " | ".join(cells) + f" | {fmt(mb, 0)} |")
        out(f"\nStarved packets over rounds 2-200 ({cname}):\n")
        out("| pair | P | policy | " + " | ".join(f"p{b}" for b in BUDGETS) + " |")
        out("|---|---|---|" + "---|" * len(BUDGETS))
        for m, d, name, P in PAIRS:
            for _, lab in POLS:
                out(f"| {name} | {P} | {lab} | " + " | ".join(
                    fmt(starv.get((name, lab, b)), 1) for b in BUDGETS) + " |")


def table_clientfrac():
    out("\n## tab:clientfrac (60 clients) + starvation + RW distinct masks")
    for niid, cname in CONDS:
        out(f"\n### {cname}\n")
        out("| pair | p | policy | factor " + " ".join(FRACS) + " | acc " + " ".join(FRACS)
            + " | starved | RW distinct masks/round |")
        out("|---|---|---|---|---|---|---|")
        for m, d, name, _ in CIFAR:
            for pct in ("0.1", "0.5"):
                for sol, lab in POLS:
                    f, a, s, dm = [], [], [], []
                    for cf in FRACS:
                        rs = runs(CF, m, d, sol, niid, pct, cf=cf)
                        res = [downlink(r) for r in rs]
                        res = [x for x in res if x]
                        f.append(f"{mean(x[0] for x in res):.2f}" if res else "n/a")
                        s.append(fmt(mean(x[1] for x in res), 0) if res else "n/a")
                        a.append(fmt(macc([ld.last_clean_acc(r) for r in rs])))
                        if sol == "rw":
                            dm.append(fmt(mean(distinct_masks(r) for r in rs), 2) if rs else "n/a")
                    out(f"| {name} | {pct} | {lab} | {' / '.join(f)} | {' / '.join(a)} | "
                        f"{' / '.join(s)} | {' / '.join(dm)} |")
        for m, d, name, _ in CIFAR:
            acc = macc(accs(SP, m, d, "plaintext", niid, "-1"))
            out(f"- FedAvg 9 clients {name}: {fmt(acc)}")


def collapsed_runs():
    out("\n## Runs whose aggregate corrupted the model (alpha=1)")
    out("| sweep | pair | policy | p | first NaN | adopted: last round before it | last finite round | round 200 |")
    out("|---|---|---|---|---|---|---|---|")
    n_cf = 0
    for m, d, name, _ in PAIRS:
        for sol, lab in POLS:
            for b in BUDGETS:
                for r in runs(SP, m, d, sol, "1", b):
                    ev = ld.read_run(r, "server_eval.csv")
                    if ev is None or not ev["mean_loss"].isna().any():
                        continue
                    bad = ev["mean_loss"].isna()
                    first = int(ev.loc[bad, "round"].min())
                    acc, t, _ = ld.last_clean_acc(r)
                    good = ev[~bad]
                    tg = int(good["round"].max())
                    ag = float(good.loc[good["round"] == tg, "mean_acc"].iloc[0])
                    a200 = float(ev.loc[ev["round"] == 200, "mean_acc"].iloc[0])
                    out(f"| nine-client | {name} | {lab} | {b} | r{first} | {acc:.3f} (r{t}) "
                        f"| {ag:.3f} (r{tg}) | {a200:.3f} |")
    for m, d, name, _ in CIFAR:
        for sol, lab in POLS:
            for pct in ("0.1", "0.5"):
                for cf in FRACS:
                    for r in runs(CF, m, d, sol, "1", pct, cf=cf):
                        ev = ld.read_run(r, "server_eval.csv")
                        if ev is not None and ev["mean_loss"].isna().any():
                            n_cf += 1
                            out(f"| client-fraction | {name} | {lab} | {pct} | cf {cf} | | | |")
    out(f"\nSixty-client sweep: {n_cf} run(s) with a non-finite evaluation loss.")


def rw_nine_clients():
    out("\n## RW, nine-client sweep: distinct masks per round (rounds>=2)")
    for niid, cname in CONDS:
        vals = []
        for m, d, name, _ in PAIRS:
            for b in BUDGETS:
                for r in runs(SP, m, d, "rw", niid, b):
                    vals.append(distinct_masks(r))
        out(f"- {cname}: mean {mean(vals):.3f}, min {min(vals):.2f}, max {max(vals):.2f} over {len(vals)} runs")


def latency():
    out("\n## Round time (train + encrypt + decrypt, s), p=10% arms and full-model arms")
    for niid, cname in CONDS:
        out(f"\n### {cname}\n")
        for m, d, name, _ in PAIRS:
            t = {}
            for sol, lab in FULL:
                rs = runs(SP, m, d, sol, niid, "-1")
                if rs:
                    t[lab] = mean(round_time(r) for r in rs)
            for sol, lab in POLS:
                rs = runs(SP, m, d, sol, niid, "0.1")
                if rs:
                    t[lab] = mean(round_time(r) for r in rs)
            bc = t.get("BatchC.")
            nxt = max(v for k, v in t.items() if k != "BatchC.")
            out(f"- {name}: BatchCrypt {fmt(bc, 1)} s, next slowest {nxt:.1f} s, "
                f"ratio {bc / nxt:.1f}" if bc else f"- {name}: no BatchCrypt run")


def mask_replay():
    out("\n## Server-side mask check replayed offline (RR/SW, both sweeps, both partitions)")
    total = bad = nruns = 0
    for root in (SP, CF):
        for f in sorted(root.iterdir()):
            mm = re.search(r"_(sweepfl|sw)_none_.*_percentage([0-9.]+)_", f.name)
            if not mm or not (f / "mask.csv").exists():
                continue
            nruns += 1
            pol, p = mm.group(1), float(mm.group(2))
            n_packs, masks = ld.run_masks([f])
            k = math.ceil(n_packs * p)
            stride = k if pol == "sweepfl" else math.ceil(k * 0.5)
            for t, ms in masks.items():
                want = {(t * stride + i) % n_packs for i in range(k)}
                total += len(ms)
                bad += sum(m != want for m in ms)
    out(f"- {nruns} runs, {total} uploads, {bad} mismatches against o = t*stride mod P")


def exposure():
    """Share of transmitted packets whose participation set is a proper subset of the round's clients.

    For round t and packet j in the union, S_j is the set of clients that sent j; the packet is
    exposed when 0 < |S_j| < n (n clients aggregated that round). Rounds >= 2, first complete run.
    """
    out("\n## Exposure to the participation term (alpha=1, nine clients, rounds>=2, seed 42)")
    out("share of transmitted packets with |S_j| < n, and mean |S_j|/n")
    out("\n| pair | policy | p0.1 | p0.3 | p0.5 | p0.7 | p0.9 |\n|---|---|---|---|---|---|---|")
    for m, d, name, _ in PAIRS:
        for sol, lab in POLS:
            cells = []
            for b in BUDGETS:
                rs = runs(SP, m, d, sol, "1", b)
                if not rs:
                    cells.append("-")
                    continue
                _, masks = ld.run_masks(rs[0], first_round=2)
                ex = tot = 0
                frac = []
                for ms in masks.values():
                    n = len(ms)
                    cnt = {}
                    for sset in ms:
                        for j in sset:
                            cnt[j] = cnt.get(j, 0) + 1
                    for c in cnt.values():
                        tot += 1
                        ex += c < n
                        frac.append(c / n)
                cells.append(f"{ex / tot:.2f} / {sum(frac) / len(frac):.2f}")
            out(f"| {name} | {lab} | " + " | ".join(cells) + " |")


def round_one():
    out("\n## Round 1 participation (nine-client sweep)")
    from collections import Counter
    c = Counter()
    for f in SP.iterdir():
        agg = ld._read_csv_safe(f / "server_agg.csv")
        if agg is not None and len(agg) and int(agg["round"].iloc[0]) == 1:
            packing = any(f"_{s}_none_" in f.name for s in ("sweepfl", "sw", "rw", "fedphe"))
            c[(int(agg["n_clients"].iloc[0]), "packing" if packing else "full model")] += 1
    out(f"- clients aggregated in round 1, by arm: {dict(c)}")


if __name__ == "__main__":
    for step in (table_fullmodel, table_policies, table_downlink, table_clientfrac,
                 collapsed_runs, rw_nine_clients, latency, mask_replay, round_one, exposure):
        step()
        print(f"done {step.__name__}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
