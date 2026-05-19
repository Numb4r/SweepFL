"""
Tests for the srr (Spread Round-Robin) mask-selection logic.

srr selects k = ceil(n * percentage) packs per round, evenly spaced across
the model with stride = n // k.  The offset advances by 1 each round and
wraps after stride rounds, guaranteeing full coverage in exactly stride rounds.

Example: n=28, percentage=2/28 → k=2, stride=14
  Round 0: {0, 14}
  Round 1: {1, 15}
  ...
  Round 13: {13, 27}
  Round 14: {0, 14}  (wraps back)

The tests cover:
  1. Count: ceil(n * pct) packs selected.
  2. Binary mask.
  3. Uniform spacing: selected packs are stride apart.
  4. Offset advances by 1 each round.
  5. Full coverage in exactly stride = n // k rounds.
  6. Wrap-around: round=stride identical to round=0.
  7. Disjoint consecutive windows (when k ≤ stride).
  8. Edge cases: n=0, k=0, k≥n, single pack, 100%.
"""

import math
import sys
import os

import numpy as np
import pytest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, "client"))

from literature.fedphe import get_spread_rr_mask


# ── Helpers ────────────────────────────────────────────────────────────────────

def _selected(mask):
    return [i for i, v in enumerate(mask) if v == 1]


def _n_selected(mask):
    return sum(1 for v in mask if v == 1)


def _make_packs(n: int):
    return [np.zeros(10, dtype=np.float32) for _ in range(n)]


# ── 1. Count ───────────────────────────────────────────────────────────────────

class TestCount:

    @pytest.mark.parametrize("n,pct", [(10, 0.3), (28, 2/28), (20, 0.5), (7, 0.1)])
    def test_selected_count_equals_ceil(self, n, pct):
        packs = _make_packs(n)
        mask = get_spread_rr_mask(0, packs, pct, percentage=True)
        assert _n_selected(mask) == math.ceil(n * pct)

    @pytest.mark.parametrize("pct", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_count_stable_across_rounds(self, pct):
        n = 20
        packs = _make_packs(n)
        expected = math.ceil(n * pct)
        for rnd in range(30):
            assert _n_selected(get_spread_rr_mask(rnd, packs, pct, percentage=True)) == expected

    def test_length_equals_n_packs(self):
        n = 15
        packs = _make_packs(n)
        mask = get_spread_rr_mask(3, packs, 0.2, percentage=True)
        assert len(mask) == n


# ── 2. Binary mask ─────────────────────────────────────────────────────────────

class TestBinaryMask:

    def test_values_are_zero_or_one(self):
        packs = _make_packs(10)
        mask = get_spread_rr_mask(0, packs, 0.3, percentage=True)
        for v in mask:
            assert v in (0, 1)


# ── 3. Uniform spacing ─────────────────────────────────────────────────────────

class TestUniformSpacing:

    def test_spacing_equals_stride_for_round_zero(self):
        """At round 0, selected packs must be exactly stride = n//k apart."""
        n, pct = 28, 2 / 28  # k=2, stride=14
        packs = _make_packs(n)
        k = math.ceil(n * pct)  # = 2
        stride = n // k          # = 14
        sel = _selected(get_spread_rr_mask(0, packs, pct, percentage=True))
        assert len(sel) == k
        for i in range(1, len(sel)):
            assert (sel[i] - sel[i - 1]) == stride

    @pytest.mark.parametrize("n,pct", [(20, 0.1), (30, 0.2), (12, 0.25)])
    def test_spacing_correct_for_various_configs(self, n, pct):
        packs = _make_packs(n)
        k = math.ceil(n * pct)
        stride = n // k
        sel = _selected(get_spread_rr_mask(0, packs, pct, percentage=True))
        for i in range(1, len(sel)):
            assert (sel[i] - sel[i - 1]) == stride


# ── 4. Offset advances by 1 each round ────────────────────────────────────────

class TestOffsetAdvance:

    def test_offset_increments_by_one(self):
        """Round r selects packs shifted by +1 compared to round r-1."""
        n, pct = 28, 2 / 28  # k=2, stride=14
        packs = _make_packs(n)
        sel0 = _selected(get_spread_rr_mask(0, packs, pct, percentage=True))
        sel1 = _selected(get_spread_rr_mask(1, packs, pct, percentage=True))
        # Each index should advance by 1
        for a, b in zip(sel0, sel1):
            assert b == a + 1

    def test_user_example_n28_k2(self):
        """Verify the exact user example: n=28, k=2."""
        n = 28
        k = 2
        pct = k / n
        packs = _make_packs(n)
        assert _selected(get_spread_rr_mask(0, packs, pct, percentage=True)) == [0, 14]
        assert _selected(get_spread_rr_mask(1, packs, pct, percentage=True)) == [1, 15]
        assert _selected(get_spread_rr_mask(13, packs, pct, percentage=True)) == [13, 27]


# ── 5. Full coverage in stride rounds ──────────────────────────────────────────

class TestFullCoverage:

    @pytest.mark.parametrize("n,pct", [
        (28, 2/28),   # k=2, stride=14, 14*2=28 ✓
        (20, 0.1),    # k=2, stride=10, 10*2=20 ✓
        (12, 0.25),   # k=3, stride=4,  4*3=12  ✓
        (10, 0.5),    # k=5, stride=2,  2*5=10  ✓
    ])
    def test_coverage_in_stride_rounds(self, n, pct):
        """Full coverage in stride rounds when n is divisible by stride (n%k==0)."""
        packs = _make_packs(n)
        k = math.ceil(n * pct)
        stride = n // k
        # Only test configs where stride*k == n (no remainder)
        assert stride * k == n, f"Precondition failed: stride({stride})*k({k})={stride*k} != n={n}"
        covered = set()
        for rnd in range(stride):
            covered.update(_selected(get_spread_rr_mask(rnd, packs, pct, percentage=True)))
        assert covered == set(range(n)), f"uncovered: {set(range(n)) - covered}"


# ── 6. Wrap-around ─────────────────────────────────────────────────────────────

class TestWrapAround:

    @pytest.mark.parametrize("n,pct", [(28, 2/28), (10, 0.3), (20, 0.2)])
    def test_round_stride_equals_round_zero(self, n, pct):
        packs = _make_packs(n)
        k = math.ceil(n * pct)
        stride = n // k
        mask0 = get_spread_rr_mask(0,      packs, pct, percentage=True)
        maskS = get_spread_rr_mask(stride, packs, pct, percentage=True)
        assert mask0 == maskS

    def test_multiple_cycles_consistent(self):
        n, pct = 10, 0.3
        packs = _make_packs(n)
        k = math.ceil(n * pct)
        stride = n // k
        for rnd in range(stride):
            m1 = get_spread_rr_mask(rnd,          packs, pct, percentage=True)
            m2 = get_spread_rr_mask(rnd + stride,  packs, pct, percentage=True)
            assert m1 == m2, f"cycle mismatch at round {rnd}"


# ── 7. Disjoint consecutive windows ────────────────────────────────────────────

class TestDisjointWindows:

    def test_consecutive_rounds_disjoint_when_k_lt_stride(self):
        """When k < stride, no pack appears in two consecutive rounds."""
        n, pct = 28, 2 / 28  # k=2, stride=14 → 2 < 14
        packs = _make_packs(n)
        for rnd in range(13):
            s0 = set(_selected(get_spread_rr_mask(rnd,     packs, pct, percentage=True)))
            s1 = set(_selected(get_spread_rr_mask(rnd + 1, packs, pct, percentage=True)))
            assert s0.isdisjoint(s1), f"overlap between round {rnd} and {rnd+1}: {s0 & s1}"

    @pytest.mark.parametrize("n,pct", [(20, 0.1), (30, 0.1)])
    def test_no_overlap_low_percentage(self, n, pct):
        packs = _make_packs(n)
        k = math.ceil(n * pct)
        stride = n // k
        if k < stride:
            for rnd in range(stride - 1):
                s0 = set(_selected(get_spread_rr_mask(rnd,     packs, pct, percentage=True)))
                s1 = set(_selected(get_spread_rr_mask(rnd + 1, packs, pct, percentage=True)))
                assert s0.isdisjoint(s1)


# ── 8. Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_packs_returns_empty(self):
        mask = get_spread_rr_mask(0, [], 0.3, percentage=True)
        assert mask == []

    def test_single_pack_returns_one(self):
        packs = _make_packs(1)
        mask = get_spread_rr_mask(0, packs, 1.0, percentage=True)
        assert mask == [1]

    def test_100_percent_selects_all(self):
        n = 8
        packs = _make_packs(n)
        mask = get_spread_rr_mask(0, packs, 1.0, percentage=True)
        assert _n_selected(mask) == n
        assert all(v == 1 for v in mask)

    def test_k_ge_n_selects_all(self):
        n = 5
        packs = _make_packs(n)
        # k = ceil(5 * 0.9) = 5 >= n
        mask = get_spread_rr_mask(0, packs, 0.9, percentage=True)
        assert _n_selected(mask) == n

    def test_absolute_size_window(self):
        """Test with percentage=False (absolute pack count)."""
        n = 20
        packs = _make_packs(n)
        k = 4
        mask = get_spread_rr_mask(0, packs, k, percentage=False)
        assert _n_selected(mask) == k
        stride = n // k
        sel = _selected(mask)
        for i in range(1, len(sel)):
            assert sel[i] - sel[i - 1] == stride

    def test_round_zero_starts_at_index_zero(self):
        """Round 0 must always include pack index 0."""
        for n in [10, 20, 28, 50]:
            pct = 0.1
            packs = _make_packs(n)
            sel = _selected(get_spread_rr_mask(0, packs, pct, percentage=True))
            assert 0 in sel, f"n={n}: round 0 did not include pack 0, got {sel}"
