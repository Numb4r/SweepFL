#!/usr/bin/env python3
"""Pendências da Seção 8, lidas dos mesmos logs e com as mesmas regras do paper_numbers.py.

A. Acurácia final de cada política em cada orçamento, nos cinco pares (D10: Round Robin
   contra Top-k no LeNet-5 e no DNN; quem é o melhor e o pior a p = 10%).
B. Tempo por rodada de cada braço e a razão do BatchCrypt sobre o próximo mais lento.
C. Pacotes starved do Top-k a p = 10% nos pares CIFAR-10, com nove clientes e com
   15, 30 e 45 clientes por rodada no sweep de sessenta.

Uso:  cd figures/scripts && python3 check_pendencias.py
      python3 check_pendencias.py --logs /outro/caminho/logs   (se os logs mudaram de lugar)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import paper_numbers as pn  # noqa: E402

MAIN = "1"  # pasta niid1: a condição principal do artigo (Dirichlet alpha = 0.1)


def pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def accuracy_block():
    print("\n== A. Acurácia final por política e orçamento (%, média das corridas) ==")
    for m, d, name, P in pn.PAIRS:
        print(f"\n{name} (P = {P})")
        table = {}
        for sol, lab in pn.POLS:
            cells = []
            for b in pn.BUDGETS:
                v = pn.accs(pn.SP, m, d, sol, MAIN, b)
                vals = [a for a, _, _ in v if a is not None]
                early = [t for _, t, c in v if c]
                table[(lab, b)] = mean(vals) if vals else None
                cell = f"{pct(table[(lab, b)])} (n={len(vals)}"
                if len(vals) > 1:
                    cell += ": " + ", ".join(f"{100 * a:.1f}" for a in vals)
                cell += ")"
                if early:
                    cell += f" LIDA ANTES DA RODADA 200 {early}"
                cells.append(cell)
            print(f"  {lab:6s} " + " | ".join(f"p{b} {c}" for b, c in zip(pn.BUDGETS, cells)))
        diffs = []
        for b in pn.BUDGETS:
            rr, tk = table[("RR", b)], table[("Top-k", b)]
            diffs.append("n/a" if rr is None or tk is None else f"{100 * (rr - tk):+.1f}")
        print("  RR - Top-k: " + " | ".join(f"p{b} {x}" for b, x in zip(pn.BUDGETS, diffs)))
        at10 = {lab: table[(lab, "0.1")] for _, lab in pn.POLS if table[(lab, "0.1")] is not None}
        if at10:
            best, worst = max(at10, key=at10.get), min(at10, key=at10.get)
            print(f"  p0.1: melhor {best}, pior {worst}, spread {100 * (at10[best] - at10[worst]):.1f}")


def latency_block():
    print("\n== B. Tempo por rodada (treino + cifração + decifração, s): full-model e p = 10% ==")
    for m, d, name, _ in pn.PAIRS:
        t = {}
        for sol, lab in pn.FULL:
            rs = pn.runs(pn.SP, m, d, sol, MAIN, "-1")
            if rs:
                t[lab] = mean(pn.round_time(r) for r in rs)
        for sol, lab in pn.POLS:
            rs = pn.runs(pn.SP, m, d, sol, MAIN, "0.1")
            if rs:
                t[lab] = mean(pn.round_time(r) for r in rs)
        bc = t.pop("BatchC.", None)
        if bc is None or not t:
            print(f"  {name}: sem corrida de BatchCrypt")
            continue
        nxt = max(t, key=t.get)
        print(f"  {name}: BatchCrypt {bc:.2f} s, próximo mais lento {nxt} {t[nxt]:.2f} s, "
              f"razão {bc / t[nxt]:.2f}")
        print("     demais: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(t.items(), key=lambda kv: -kv[1])))


def starved_block():
    print("\n== C. Pacotes starved do Top-k a p = 10% (rodadas >= 2), por corrida ==")
    for m, d, name, P in pn.CIFAR:
        vals = []
        for r in pn.runs(pn.SP, m, d, "fedphe", MAIN, "0.1"):
            x = pn.downlink(r, until=pn.collapse_round(r))
            if x:
                vals.append(x[1])
        print(f"  {name} (P = {P}), nove clientes: {vals}")
        for cf, n in zip(pn.FRACS, (15, 30, 45)):
            vals = []
            for r in pn.runs(pn.CF, m, d, "fedphe", MAIN, "0.1", cf=cf):
                x = pn.downlink(r)
                if x:
                    vals.append(x[1])
            print(f"  {name} (P = {P}), sessenta clientes, f_c = {cf} ({n} por rodada): {vals}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", help="pasta que contém sparsification_percentage, client_frac e range_fix")
    args = ap.parse_args()
    if args.logs:
        root = Path(args.logs)
        pn.SP, pn.CF, pn.RANGE_FIX = root / "sparsification_percentage", root / "client_frac", root / "range_fix"
    for path in (pn.SP, pn.CF):
        if not path.is_dir():
            sys.exit(f"pasta de logs não encontrada: {path} (use --logs)")
    accuracy_block()
    latency_block()
    starved_block()