"""Shared helpers for the seed-confidence study (scripts 31 and 32).

Finds, for MobileNetV2/CIFAR10/α=1, one finished (200-round) run per (solution, p, seed)
across logs/seed_experiment_v2 (seeds 43/44), logs/range_fix (seed 42, post-fix) and
logs/sparsification_percentage (seed 42, plaintext baseline; pre-fix for p=0.1).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "studies/seed_confidence"
SOURCES = [("range_fix", 3), ("seed_experiment_v2", 2), ("sparsification_percentage", 1)]  # (dir, priority)
RX = re.compile(r"^(?P<model>[^_]+)_(?P<ds>[^_]+)_(?P<sol>[^_]+)_none_(?P<rounds>\d+)rounds_(?P<nc>\d+)clients_"
                r"seed(?P<seed>\d+)_niid(?P<niid>[\d.]+)_percentage(?P<pct>-?[\d.]+)_.*_(?P<ts>[\d-]+_[\d-]+)$")
TECHS = ["fedphe", "sweepfl", "sw", "rw"]


def find_runs(model="mobilenetv2", ds="CIFAR10", niid=1.0, pcts=(0.1, 0.7), rounds=200):
    best = {}
    for src, prio in SOURCES:
        root = REPO / "logs" / src
        if not root.exists():
            continue
        for d in root.iterdir():
            m = RX.match(d.name)
            if not m or m["model"] != model or m["ds"] != ds or float(m["niid"]) != niid or int(m["rounds"]) != rounds:
                continue
            sol, seed, pct = m["sol"], int(m["seed"]), float(m["pct"])
            if sol == "plaintext":
                pct = -1.0
            elif round(pct, 3) not in pcts or sol not in TECHS:
                continue
            try:
                done = json.load(open(d / "checkpoint/meta.json"))["round"] >= rounds
            except Exception:
                done = False
            if not done or not (d / "server_eval.csv").exists():
                continue
            key = (sol, round(pct, 3), seed)
            rank = (prio, m["ts"])
            if key not in best or rank > best[key][0]:
                head = open(d / "train.csv").readline()
                best[key] = (rank, dict(sol=sol, pct=pct, seed=seed, path=d, source=src,
                                        post_fix="magg_renorm" in head))
    return [v[1] for v in best.values()]
