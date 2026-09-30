#!/usr/bin/env python3
"""Offline CKKS single-prime range probe on saved aggregate snapshots (no FL, no GPU).

For every model/round_<r>.npz (cid-0 copy of the decrypted aggregate) it rebuilds the
plaintext the server would decrypt next round, Σ_i D_i·w_i ≈ D_total·w, packs it exactly
like the clients, and reports the worst pack's slot ratio and coefficient ratio against
q/(2·scale). Wrap happens when the coefficient ratio reaches 1.
--verify-round R also encrypts that round's worst pack as 9 client shares with the real
ckks_python_simplify library, sums homomorphically and decrypts, to show the wrap.

  python3 experiments/tests/ckks_range_probe.py <run_folder> [<run_folder> ...] \
      [--d-total 50000] [--prescale 1] [--verify-round 103]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "client"))
from client_logs import ckks_coeffs  # noqa: E402

N, BITS, SCALE_BITS = 8192, 62, 40


def load(path):
    z = np.load(path)
    return np.concatenate([z[f"arr_{i}"].astype(np.float64).ravel() for i in range(len(z.files))])


def packs_of(flat, chunk=N // 2):
    return [flat[i:i + chunk] for i in range(0, flat.size, chunk)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--d-total", type=float, default=50000.0)
    ap.add_argument("--prescale", type=float, default=1.0, help="divide D_total·w by this (fix under test)")
    ap.add_argument("--verify-round", type=int, default=0)
    a = ap.parse_args()
    bound = 2.0 ** (BITS - SCALE_BITS - 1)  # q/(2Δ) up to q < 2^62

    for run in a.runs:
        print(f"\n== {os.path.basename(run.rstrip('/'))}")
        print(f"{'round':>5} {'slot_ratio':>10} {'coef_ratio':>10} {'pack':>5} {'min_w':>9}")
        files = sorted(glob.glob(f"{run}/model/round_*.npz"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
        for f in files:
            r = int(re.findall(r"\d+", os.path.basename(f))[0])
            w = load(f)
            ps = packs_of(w * a.d_total / a.prescale)
            coef = np.array([np.abs(ckks_coeffs(p, N)).max() for p in ps])
            slot = np.array([np.abs(p).max() for p in ps])
            k = int(coef.argmax())
            flag = "  <-- WRAP" if coef[k] >= bound else ""
            print(f"{r:>5} {slot.max() / bound:>10.3f} {coef[k] / bound:>10.3f} {k:>5} {w.min():>9.2f}{flag}")
            if r == a.verify_round:
                verify(ps[k], bound)


def verify(pack, bound, clients=9):
    from ckks_python_simplify import CKKSContext, CKKSTensor
    ctx = CKKSContext(N, BITS, SCALE_BITS)
    for g in (1.0, 1.05):  # as-is, and 5% larger (≈ one more round of drift)
        target = pack * g
        agg = CKKSTensor(ctx, list(target / clients))
        for _ in range(clients - 1):
            agg += CKKSTensor(ctx, list(target / clients))
        err = np.abs(np.asarray(agg.decrypt())[:pack.size] - target)
        print(f"   verify x{g}: 9 encrypted shares summed+decrypted -> max|err| = {err.max():.4g} "
              f"(q/Δ = {2 * bound:.4g}); after /D_total ≈ {err.max() / 50000:.3g} per weight")


if __name__ == "__main__":
    main()
