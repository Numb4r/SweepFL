"""range.csv instrumentation (client_logs.write_range_logs) + numpy encoder replica."""
import csv
import math
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "client"))
from client_logs import ckks_coeffs, write_range_logs  # noqa: E402

N = 8192


def test_replica_matches_library_encoder():
    ckks_python = pytest.importorskip("ckks_python")
    q = ckks_python.generate_prime(62, N)
    scale = 2 ** 40
    z = np.random.default_rng(0).normal(0, 50, N // 2)
    lib = np.array(ckks_python.Encoder(N, q, scale).encode(list(z)).coeffs, dtype=np.int64)
    lib = np.where(lib > q // 2, lib - q, lib)  # centre
    ours = np.round(ckks_coeffs(z, N) * scale).astype(np.int64)
    assert np.max(np.abs(lib - ours)) <= 2


def test_constant_pack_concentrates_in_a0():
    # A pack of all-v encodes to the constant polynomial v: this is why an all-positive
    # moving_variance block drives the range, while one isolated outlier is diluted by N/2.
    a = ckks_coeffs(np.full(N // 2, 7.0), N)
    assert a[0] == pytest.approx(7.0) and np.max(np.abs(a[1:])) < 1e-9
    spike = np.zeros(N // 2); spike[3] = 1000.0
    assert np.max(np.abs(ckks_coeffs(spike, N))) == pytest.approx(2 * 1000.0 / N, rel=1e-6)


def test_range_csv_written_and_finite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = types.SimpleNamespace(N=N, q=(1 << 62) - 57, scale=2 ** 40)
    D = 5000
    weights = [np.random.default_rng(1).normal(0, 1, (100, 60)), np.full(3000, 200.0)]
    flat = np.concatenate([w.ravel() for w in weights]) * D
    packs = [flat[i:i + N // 2] for i in range(0, flat.size, N // 2)]
    self = types.SimpleNamespace(context=ctx, x_train=np.zeros(D), cid=0, solution="nvnl",
                                 seed=42, experiment_name="t", log_folder="exp")
    write_range_logs(self, 1, weights, packs, [1, 1, 0])
    rows = list(csv.DictReader(open("logs/exp/range.csv")))
    assert len(rows) == 1
    r = rows[0]
    for col in ("max_abs_w", "max_abs_wd", "slot_ratio", "coef_max", "coef_ratio", "bound"):
        assert math.isfinite(float(r[col])), col
    assert float(r["max_abs_w"]) == pytest.approx(200.0)
    assert float(r["bound"]) == pytest.approx(2 ** 21, rel=1e-6)
    assert int(r["argmax_pack"]) == 1  # the 200·D block starts inside pack 1
    assert len(r["per_pack_coef"].split(";")) == 3


def test_non_simplify_context_is_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    self = types.SimpleNamespace(context=object(), log_folder="exp")
    write_range_logs(self, 1, [np.ones(3)], [np.ones(3)], [1])
    assert not os.path.exists("logs/exp/range.csv")


@pytest.mark.parametrize("d_ref, wraps", [(1, True), (1024, False)])
def test_prescale_prevents_wrap_real_ckks(d_ref, wraps):
    """BN-like pack (mean 45) from 9 clients with ΣD=50 000: Σ D_i·w wraps q/(2Δ)=2^21
    without the pre-scale; with MAGG_D_REF it decodes to FedAvg exactly."""
    simplify = pytest.importorskip("ckks_python_simplify")
    ctx = simplify.CKKSContext(N, 62, 40)
    w, D = np.full(N // 2, 45.0), [50000 / 9] * 9
    agg = None
    for d in D:
        ct = simplify.CKKSTensor(ctx, list(w * d / d_ref))
        agg = ct if agg is None else agg + ct
    out = np.asarray(agg.decrypt())[:w.size] / (sum(D) / d_ref)
    assert bool(np.max(np.abs(out - w)) > 1.0) == wraps
    if not wraps:
        assert np.max(np.abs(out - w)) < 1e-6
