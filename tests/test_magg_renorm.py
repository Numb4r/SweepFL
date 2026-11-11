"""
tests/test_magg_renorm.py
─────────────────────────
Oracle tests for the per-pack renormalization fix (MAGG_RENORM=1).

Mathematical protocol (pure numpy, no CKKS / Docker required):

  Client i pre-weights:   scaled[k] = D_i * w_i[k]       ← MAGG_RENORM=1
                          scaled[k] = w_i[k] / N          ← MAGG_RENORM=0 (legacy)

  Server (pure-sum path): agg_pack[k] = Σ_{S_k} scaled_i[k]
                          M_agg[k]    = Σ_{S_k} D_i

  Client renorm:          result[k]   = agg_pack[k] / M_agg[k]  ← MAGG_RENORM=1
                          result[k]   = agg_pack[k]              ← MAGG_RENORM=0 (no renorm)

  Ground-truth (oracle):  expected[k] = Σ_{S_k} D_i * w_i[k] / Σ_{S_k} D_i   (FedAvg)

Tests:
  T1  full-coverage sanity: all clients, all packs → corrected = oracle = simple wt. avg.
  T2  fedphe scenario: divergent (data-dependent) masks →
        MAGG_RENORM=1 matches oracle; MAGG_RENORM=0 shows quantified bias.
  T3  sweepfl control: identical round-robin masks (all clients same window) →
        MAGG_RENORM=1 ≈ MAGG_RENORM=0, both match oracle (renorm is no-op).
  T4  single-contributor pack: corrected result equals that client's weights exactly.
  T5  bias grows with sparsity: lower ζ → larger legacy error; corrected stays ~0.
"""

import math
import os

import numpy as np
import pytest


# ─────────────────────────── Pure-numpy protocol helpers ─────────────────────

def _make_packs(weights_2d):
    """Return list of rows from a (K, B) array — i.e. the K packs."""
    return [weights_2d[k] for k in range(weights_2d.shape[0])]


def _topk_mask_l2(weights_2d, zeta):
    """Pack-level top-k by L2 norm (spec-correct). Returns {0,1} int8 array."""
    K = weights_2d.shape[0]
    topk = max(1, math.ceil(K * zeta))
    scores = np.array([np.linalg.norm(weights_2d[k]) for k in range(K)])
    idx = np.argsort(scores)[::-1][:topk]
    mask = np.zeros(K, dtype=np.int8)
    mask[idx] = 1
    return mask


def _rr_mask(round_idx, K, zeta):
    """Round-robin mask — deterministic, identical for all clients."""
    size = max(1, math.ceil(K * zeta))
    mask = np.zeros(K, dtype=np.int8)
    start = (round_idx * size) % K
    for i in range(size):
        mask[(start + i) % K] = 1
    return mask


def _server_aggregate_pure(clients, total_examples):
    """
    Replicate server-side aggregate_packs_and_mask in pure-sum mode.

    clients : list of (scaled_weights_2d, mask, D_i)
              scaled_weights_2d shape (K, B)
    Returns  : (agg_packs[K], M_agg[K])
               agg_packs[k] is None when no client contributed to pack k.
    """
    K = clients[0][0].shape[0]
    agg_packs = [None] * K
    M_agg = np.zeros(K)

    for scaled_w, mask, D_i in clients:
        for k in range(K):
            if mask[k] == 0:
                continue
            M_agg[k] += D_i                          # Σ_{S_k} D_i
            contrib = scaled_w[k]                    # D_i·w_i[k] or w_i[k]/N
            if agg_packs[k] is None:
                agg_packs[k] = contrib.copy()
            else:
                agg_packs[k] = agg_packs[k] + contrib

    return agg_packs, M_agg


def _client_renorm(agg_packs, M_agg, magg_renorm):
    """
    Replicate he_packs_to_model renormalization gate.

    magg_renorm=1 : divide each active pack by M_agg[k]  (corrected)
    magg_renorm=0 : no division                           (legacy)
    """
    result = []
    for k, pack in enumerate(agg_packs):
        if pack is None:
            result.append(None)
        elif M_agg[k] > 0 and magg_renorm:
            result.append(pack / M_agg[k])
        else:
            result.append(pack)
    return result


def _oracle(weights_list, Ds, masks):
    """
    Ground-truth per-pack FedAvg for arbitrary contributor sets.

    weights_list : list of (K, B) arrays
    Ds           : list of dataset sizes
    masks        : list of {0,1} int8 arrays length K
    Returns      : list of (B,) arrays, one per pack (None if no contributor).
    """
    K = weights_list[0].shape[0]
    result = []
    for k in range(K):
        contributors = [
            (weights_list[i][k], Ds[i])
            for i in range(len(Ds))
            if masks[i][k] == 1
        ]
        if not contributors:
            result.append(None)
            continue
        total_D = sum(d for _, d in contributors)
        result.append(sum(w * d for w, d in contributors) / total_D)
    return result


# ─────────────────────────── T1 — full coverage round-trip ───────────────────

def test_T1_full_coverage_matches_oracle():
    """All packs covered by all clients: corrected == oracle (weighted average)."""
    rng = np.random.default_rng(42)
    n_clients, B, K = 3, 8, 4
    Ds = [100, 200, 150]
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]
    masks = [np.ones(K, dtype=np.int8)] * n_clients

    # Corrected: D_i pre-scale
    clients = [(weights[i] * Ds[i], masks[i], Ds[i]) for i in range(n_clients)]
    agg, M_agg = _server_aggregate_pure(clients, total)
    result = _client_renorm(agg, M_agg, magg_renorm=1)
    oracle = _oracle(weights, Ds, masks)

    for k in range(K):
        np.testing.assert_allclose(result[k], oracle[k], rtol=1e-10,
                                   err_msg=f"T1 FAIL pack {k}")


# ─────────────────────────── T2 — fedphe: divergent masks ────────────────────

def test_T2_fedphe_corrected_matches_oracle():
    """MAGG_RENORM=1: corrected path matches FedAvg oracle for sparse packs."""
    rng = np.random.default_rng(7)
    n_clients, B, K = 3, 16, 6
    Ds = [80, 200, 120]
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]
    # Deliberately divergent masks (clients pick different subsets of packs)
    masks = [
        np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),
        np.array([0, 0, 1, 1, 1, 0], dtype=np.int8),
        np.array([1, 0, 0, 1, 0, 1], dtype=np.int8),
    ]

    clients = [(weights[i] * Ds[i], masks[i], Ds[i]) for i in range(n_clients)]
    agg, M_agg = _server_aggregate_pure(clients, total)
    result = _client_renorm(agg, M_agg, magg_renorm=1)
    oracle = _oracle(weights, Ds, masks)

    for k in range(K):
        if oracle[k] is None:
            continue
        np.testing.assert_allclose(result[k], oracle[k], rtol=1e-10,
                                   err_msg=f"T2 FAIL corrected at pack {k}")


def test_T2_fedphe_legacy_biased():
    """
    MAGG_RENORM=0 (legacy): /num_clients pre-scale without renorm is biased.
    Specifically, sparse packs (not all clients contribute) are underweighted.
    """
    rng = np.random.default_rng(7)
    n_clients, B, K = 3, 16, 6
    Ds = [80, 200, 120]
    N = n_clients
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]
    masks = [
        np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),
        np.array([0, 0, 1, 1, 1, 0], dtype=np.int8),
        np.array([1, 0, 0, 1, 0, 1], dtype=np.int8),
    ]

    # Legacy: /N pre-scale, no renorm
    clients_legacy = [(weights[i] / N, masks[i], Ds[i]) for i in range(n_clients)]
    agg_l, M_agg_l = _server_aggregate_pure(clients_legacy, total)
    result_legacy = _client_renorm(agg_l, M_agg_l, magg_renorm=0)
    oracle = _oracle(weights, Ds, masks)

    biased_packs = 0
    for k in range(K):
        if oracle[k] is None:
            continue
        err = np.mean(np.abs(result_legacy[k] - oracle[k]))
        if err > 1e-6:
            biased_packs += 1

    assert biased_packs > 0, (
        "Legacy path should produce bias on at least one sparse pack "
        "(pack with fewer than N contributing clients)."
    )


def test_T2_bias_is_corrected():
    """Corrected path has < 1e-9 error; legacy path has much larger error on same data."""
    rng = np.random.default_rng(7)
    n_clients, B, K = 3, 16, 6
    Ds = [80, 200, 120]
    N = n_clients
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]
    masks = [
        np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),
        np.array([0, 0, 1, 1, 1, 0], dtype=np.int8),
        np.array([1, 0, 0, 1, 0, 1], dtype=np.int8),
    ]
    oracle = _oracle(weights, Ds, masks)

    # Corrected
    agg_c, M_agg_c = _server_aggregate_pure(
        [(weights[i] * Ds[i], masks[i], Ds[i]) for i in range(n_clients)], total
    )
    result_c = _client_renorm(agg_c, M_agg_c, magg_renorm=1)

    # Legacy
    agg_l, M_agg_l = _server_aggregate_pure(
        [(weights[i] / N, masks[i], Ds[i]) for i in range(n_clients)], total
    )
    result_l = _client_renorm(agg_l, M_agg_l, magg_renorm=0)

    errs_c, errs_l = [], []
    for k in range(K):
        if oracle[k] is None:
            continue
        errs_c.append(np.mean(np.abs(result_c[k] - oracle[k])))
        errs_l.append(np.mean(np.abs(result_l[k] - oracle[k])))

    mean_err_c = np.mean(errs_c)
    mean_err_l = np.mean(errs_l)

    assert mean_err_c < 1e-9, f"Corrected mean error {mean_err_c:.2e} should be ~0"
    assert mean_err_l > mean_err_c * 100, (
        f"Legacy error {mean_err_l:.4f} should be >> corrected {mean_err_c:.2e}"
    )


# ─────────────────────────── T3 — sweepfl control: identical RR masks ───────────

def test_T3_sweepfl_identical_masks_both_paths_match_oracle():
    """
    With identical RR masks (all clients same window) AND equal dataset sizes:
      - M_agg[k] = total for every active pack k (full coverage, no sparsity bias).
      - Corrected and legacy give the same result (data-weighted = simple average).
      - The sparsity-induced bias that affects fedphe does NOT apply to sweepfl.

    Note: with unequal Ds the corrected path STILL gives the correct FedAvg, while
    legacy gives a simple average — that's a separate (data-weighting) correction
    also addressed by the fix, tested in test_T3_sweepfl_unequal_ds_corrected_wins.
    """
    rng = np.random.default_rng(13)
    n_clients, B, K, zeta = 4, 8, 6, 0.5
    # Equal Ds: data-weighted FedAvg == simple average → both paths agree
    Ds = [100, 100, 100, 100]
    N = n_clients
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]
    # All clients share the exact same RR mask
    shared_mask = _rr_mask(round_idx=2, K=K, zeta=zeta)
    masks = [shared_mask.copy() for _ in range(n_clients)]
    oracle = _oracle(weights, Ds, masks)

    # Corrected
    agg_c, M_agg_c = _server_aggregate_pure(
        [(weights[i] * Ds[i], masks[i], Ds[i]) for i in range(n_clients)], total
    )
    result_c = _client_renorm(agg_c, M_agg_c, magg_renorm=1)

    # Legacy
    agg_l, M_agg_l = _server_aggregate_pure(
        [(weights[i] / N, masks[i], Ds[i]) for i in range(n_clients)], total
    )
    result_l = _client_renorm(agg_l, M_agg_l, magg_renorm=0)

    for k in range(K):
        if oracle[k] is None:
            continue
        np.testing.assert_allclose(
            result_c[k], oracle[k], rtol=1e-10,
            err_msg=f"T3 corrected FAIL at pack {k}",
        )
        np.testing.assert_allclose(
            result_l[k], oracle[k], rtol=1e-10,
            err_msg=f"T3 legacy FAIL at pack {k}",
        )
        # Corrected ≈ legacy (renorm is no-op when all clients contribute)
        np.testing.assert_allclose(
            result_c[k], result_l[k], rtol=1e-10,
            err_msg=f"T3 corrected ≠ legacy at pack {k} (should be equal for sweepfl)",
        )


def test_T3b_sweepfl_unequal_ds_corrected_wins():
    """
    sweepfl with unequal Ds (non-IID): identical masks mean no sparsity bias, but the
    legacy /N pre-scale gives a simple average instead of the FedAvg weighted average.
    Corrected path (D_i pre-scale + renorm by M_agg) always matches FedAvg oracle.
    """
    rng = np.random.default_rng(13)
    n_clients, B, K, zeta = 4, 8, 6, 0.5
    Ds = [50, 200, 80, 170]   # deliberately unequal (non-IID scenario)
    N = n_clients
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]
    shared_mask = _rr_mask(round_idx=2, K=K, zeta=zeta)
    masks = [shared_mask.copy() for _ in range(n_clients)]
    oracle = _oracle(weights, Ds, masks)

    # Corrected
    agg_c, M_agg_c = _server_aggregate_pure(
        [(weights[i] * Ds[i], masks[i], Ds[i]) for i in range(n_clients)], total
    )
    result_c = _client_renorm(agg_c, M_agg_c, magg_renorm=1)

    # Legacy
    agg_l, M_agg_l = _server_aggregate_pure(
        [(weights[i] / N, masks[i], Ds[i]) for i in range(n_clients)], total
    )
    result_l = _client_renorm(agg_l, M_agg_l, magg_renorm=0)

    errs_c, errs_l = [], []
    for k in range(K):
        if oracle[k] is None:
            continue
        errs_c.append(np.mean(np.abs(result_c[k] - oracle[k])))
        errs_l.append(np.mean(np.abs(result_l[k] - oracle[k])))

    assert np.mean(errs_c) < 1e-9, "Corrected must match FedAvg oracle for sweepfl too"
    assert np.mean(errs_l) > np.mean(errs_c), (
        "Legacy simple-average differs from FedAvg oracle when Ds are unequal"
    )


# ─────────────────────────── T4 — single-contributor pack ────────────────────

def test_T4_single_contributor_pack():
    """Pack with only one contributor: corrected result equals that client's weights."""
    rng = np.random.default_rng(99)
    n_clients, B, K = 3, 8, 4
    Ds = [100, 200, 150]
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]
    # Pack index 1 kept only by client 0
    masks = [
        np.array([1, 1, 0, 1], dtype=np.int8),
        np.array([1, 0, 1, 1], dtype=np.int8),
        np.array([1, 0, 1, 0], dtype=np.int8),
    ]

    clients = [(weights[i] * Ds[i], masks[i], Ds[i]) for i in range(n_clients)]
    agg, M_agg = _server_aggregate_pure(clients, total)
    result = _client_renorm(agg, M_agg, magg_renorm=1)

    # Only client 0 sent pack 1 → result should equal w0[1] exactly
    np.testing.assert_allclose(
        result[1], weights[0][1], rtol=1e-10,
        err_msg="T4: single-contributor pack result should equal that client's weights",
    )


# ─────────────────────────── T5 — bias grows with sparsity ───────────────────

@pytest.mark.parametrize("zeta", [0.9, 0.7, 0.5, 0.3, 0.1])
def test_T5_corrected_stays_near_zero_regardless_of_sparsity(zeta):
    """Corrected path error < 1e-9 for all sparsity levels."""
    rng = np.random.default_rng(21)
    n_clients, B, K = 5, 16, 10
    Ds = [100, 150, 80, 120, 90]
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]

    # Divergent random masks with zeta% coverage
    topk = max(1, math.ceil(K * zeta))
    rng2 = np.random.default_rng(42)
    masks = []
    for _ in range(n_clients):
        idx = rng2.choice(K, size=topk, replace=False)
        m = np.zeros(K, dtype=np.int8)
        m[idx] = 1
        masks.append(m)

    clients = [(weights[i] * Ds[i], masks[i], Ds[i]) for i in range(n_clients)]
    agg, M_agg = _server_aggregate_pure(clients, total)
    result = _client_renorm(agg, M_agg, magg_renorm=1)
    oracle = _oracle(weights, Ds, masks)

    errs = []
    for k in range(K):
        if oracle[k] is None:
            continue
        errs.append(np.mean(np.abs(result[k] - oracle[k])))

    mean_err = np.mean(errs) if errs else 0.0
    assert mean_err < 1e-9, (
        f"Corrected error at ζ={zeta}: {mean_err:.2e} (should be ~0)"
    )


def test_T5_legacy_bias_on_sparse_packs():
    """
    Legacy (MAGG_RENORM=0) error > corrected when masks are divergent (fedphe scenario).
    This quantifies the regression that the fix addresses.
    """
    rng = np.random.default_rng(21)
    n_clients, B, K, zeta = 5, 16, 10, 0.3
    Ds = [100, 150, 80, 120, 90]
    N = n_clients
    total = sum(Ds)

    weights = [rng.standard_normal((K, B)) for _ in range(n_clients)]
    topk = max(1, math.ceil(K * zeta))
    rng2 = np.random.default_rng(42)
    masks = []
    for _ in range(n_clients):
        idx = rng2.choice(K, size=topk, replace=False)
        m = np.zeros(K, dtype=np.int8)
        m[idx] = 1
        masks.append(m)

    oracle = _oracle(weights, Ds, masks)

    agg_c, M_agg_c = _server_aggregate_pure(
        [(weights[i] * Ds[i], masks[i], Ds[i]) for i in range(n_clients)], total
    )
    result_c = _client_renorm(agg_c, M_agg_c, magg_renorm=1)

    agg_l, M_agg_l = _server_aggregate_pure(
        [(weights[i] / N, masks[i], Ds[i]) for i in range(n_clients)], total
    )
    result_l = _client_renorm(agg_l, M_agg_l, magg_renorm=0)

    errs_c, errs_l = [], []
    for k in range(K):
        if oracle[k] is None:
            continue
        errs_c.append(np.mean(np.abs(result_c[k] - oracle[k])))
        errs_l.append(np.mean(np.abs(result_l[k] - oracle[k])))

    assert np.mean(errs_c) < 1e-9, "Corrected must be near-exact"
    assert np.mean(errs_l) > np.mean(errs_c), (
        "Legacy must be more biased than corrected on sparse/divergent packs"
    )
    print(
        f"\n  ζ={zeta}: corrected_err={np.mean(errs_c):.2e}  "
        f"legacy_err={np.mean(errs_l):.4f}  "
        f"bias_factor≈{np.mean(errs_l)/max(np.mean(errs_c),1e-15):.0f}×"
    )
