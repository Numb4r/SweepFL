"""
Tests for the topkrr solution mask-selection logic.

topkrr combines two packing strategies:
  - rounds < rr_start_round  → top-k (highest magnitude packs)
  - rounds >= rr_start_round → round-robin (sliding window, no overlap)

The tests cover:
  1. Top-k phase: correct count, selects highest-magnitude packs, binary mask.
  2. Round-robin phase: correct count, non-overlapping window, wraps around.
  3. Phase transition: exact round boundary, RR offset starts at 0.
  4. Full-model coverage: after ceil(1/p) RR rounds every pack is visited.
  5. Edge cases: percentage 0→1, single pack, 100% selection.
"""

import math
import sys
import os

import numpy as np
import pytest

# Allow imports from client/
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, "client"))

from literature.fedphe import get_robin_round_mask, get_topk_mask


# ── Helpers ────────────────────────────────────────────────────────────────────

def _n_selected(mask) -> int:
    return int(sum(mask))


def _selected_indices(mask):
    return [i for i, v in enumerate(mask) if v == 1]


def _make_packs(n: int, chunk: int = 10) -> list:
    """Return n packs with values 0,1,...,n-1 so pack i has magnitude ~ i."""
    rng = np.random.default_rng(42)
    packs = []
    for i in range(n):
        # pack i has mean absolute value ≈ i  (larger i → more important)
        packs.append(rng.uniform(i - 0.1, i + 0.1, size=chunk).astype(np.float32))
    return packs


def _topkrr_mask(packs, percentage, current_round, rr_start_round):
    """Replicates the mask-selection branch from fit_topkrr."""
    if current_round < rr_start_round:
        return get_topk_mask(packs, percentage)
    else:
        rr_round = current_round - rr_start_round
        return get_robin_round_mask(
            round=rr_round, packs=packs,
            size_window=percentage, percentage=True,
        )


# ── 1. Top-k phase ─────────────────────────────────────────────────────────────

class TestTopKPhase:

    @pytest.mark.parametrize("n,pct", [(10, 0.3), (20, 0.5), (7, 0.1)])
    def test_selected_count_matches_ceil(self, n, pct):
        packs = _make_packs(n)
        mask = get_topk_mask(packs, pct)
        assert _n_selected(mask) == math.ceil(n * pct)

    def test_selects_highest_magnitude_packs(self):
        """Packs are ordered by magnitude 0..n-1; top-k must pick the last k."""
        n, pct = 10, 0.3
        packs = _make_packs(n)  # pack i has magnitude ≈ i
        k = math.ceil(n * pct)
        mask = get_topk_mask(packs, pct)
        selected = set(_selected_indices(mask))
        # The k packs with highest magnitude are indices n-k..n-1
        expected = set(range(n - k, n))
        assert selected == expected

    def test_mask_is_binary(self):
        packs = _make_packs(8)
        mask = get_topk_mask(packs, 0.5)
        for v in mask:
            assert v in (0, 1)

    def test_full_percentage_selects_all(self):
        n = 6
        packs = _make_packs(n)
        mask = get_topk_mask(packs, 1.0)
        assert _n_selected(mask) == n

    def test_mask_length_equals_n_packs(self):
        n = 12
        packs = _make_packs(n)
        mask = get_topk_mask(packs, 0.4)
        assert len(mask) == n

    def test_empty_packs_returns_empty(self):
        mask = get_topk_mask([], 0.5)
        assert mask == []

    def test_stateless_same_input_same_output(self):
        """Top-k does not depend on round number — same packs → same mask."""
        packs = _make_packs(10)
        m1 = get_topk_mask(packs, 0.3)
        m2 = get_topk_mask(packs, 0.3)
        assert m1 == m2


# ── 2. Round-robin phase ───────────────────────────────────────────────────────

class TestRoundRobinPhase:

    @pytest.mark.parametrize("n,pct", [(10, 0.3), (20, 0.1), (7, 0.5)])
    def test_selected_count_matches_ceil(self, n, pct):
        packs = _make_packs(n)
        mask = get_robin_round_mask(round=0, packs=packs, size_window=pct, percentage=True)
        assert _n_selected(mask) == math.ceil(n * pct)

    def test_mask_is_binary(self):
        packs = _make_packs(10)
        mask = get_robin_round_mask(round=0, packs=packs, size_window=0.3, percentage=True)
        for v in mask:
            assert v in (0, 1)

    def test_window_advances_each_round(self):
        """Consecutive RR rounds must select disjoint windows (no overlap)."""
        n, pct = 10, 0.3
        packs = _make_packs(n)
        size = math.ceil(n * pct)  # = 3

        mask0 = get_robin_round_mask(round=0, packs=packs, size_window=pct, percentage=True)
        mask1 = get_robin_round_mask(round=1, packs=packs, size_window=pct, percentage=True)

        sel0 = set(_selected_indices(mask0))
        sel1 = set(_selected_indices(mask1))
        # Stride = size_window packs → windows must not overlap
        assert sel0.isdisjoint(sel1), f"overlap between round 0 ({sel0}) and round 1 ({sel1})"

    def test_window_wraps_around(self):
        """After ceil(1/pct) rounds the window wraps back to the start."""
        n, pct = 10, 0.5  # 5 packs selected per round; 2 rounds → full cycle
        packs = _make_packs(n)
        n_rounds = math.ceil(1 / pct)  # = 2

        mask0 = get_robin_round_mask(round=0, packs=packs, size_window=pct, percentage=True)
        mask_wrap = get_robin_round_mask(round=n_rounds, packs=packs, size_window=pct, percentage=True)

        assert _selected_indices(mask0) == _selected_indices(mask_wrap)

    def test_full_coverage_after_ceil_1_over_pct_rounds(self):
        """Every pack must appear in at least one mask within ceil(1/pct) rounds."""
        n, pct = 12, 0.25  # 3 packs/round → 4 rounds for full coverage
        packs = _make_packs(n)
        n_rounds = math.ceil(1 / pct)

        covered = set()
        for rnd in range(n_rounds):
            mask = get_robin_round_mask(round=rnd, packs=packs, size_window=pct, percentage=True)
            covered.update(_selected_indices(mask))

        assert covered == set(range(n)), f"uncovered packs: {set(range(n)) - covered}"

    def test_mask_length_equals_n_packs(self):
        n = 15
        packs = _make_packs(n)
        mask = get_robin_round_mask(round=3, packs=packs, size_window=0.2, percentage=True)
        assert len(mask) == n


# ── 3. Phase transition ────────────────────────────────────────────────────────

class TestPhaseTransition:

    RR_START = 50
    N_PACKS  = 10
    PCT      = 0.3

    def _packs(self):
        return _make_packs(self.N_PACKS)

    def test_last_topk_round_uses_topk(self):
        """Round rr_start - 1 must use the top-k strategy (magnitude-based)."""
        packs = self._packs()
        rnd = self.RR_START - 1
        mask = _topkrr_mask(packs, self.PCT, rnd, self.RR_START)
        # top-k is deterministic by magnitude; verify it matches get_topk_mask directly
        expected = get_topk_mask(packs, self.PCT)
        assert list(mask) == list(expected)

    def test_first_rr_round_uses_rr_at_offset_zero(self):
        """Round rr_start must use round-robin with rr_round=0 (offset from rr_start)."""
        packs = self._packs()
        mask = _topkrr_mask(packs, self.PCT, self.RR_START, self.RR_START)
        expected = get_robin_round_mask(round=0, packs=packs, size_window=self.PCT, percentage=True)
        assert list(mask) == list(expected)

    def test_second_rr_round_uses_rr_at_offset_one(self):
        packs = self._packs()
        mask = _topkrr_mask(packs, self.PCT, self.RR_START + 1, self.RR_START)
        expected = get_robin_round_mask(round=1, packs=packs, size_window=self.PCT, percentage=True)
        assert list(mask) == list(expected)

    def test_topk_and_first_rr_may_differ(self):
        """top-k selects by magnitude; RR starts at index 0 — they are generally different."""
        packs = _make_packs(10)
        pct   = 0.3
        rr_start = 10

        mask_topk = _topkrr_mask(packs, pct, rr_start - 1, rr_start)
        mask_rr0  = _topkrr_mask(packs, pct, rr_start,     rr_start)

        # Both select the same number of packs but by different criteria
        assert _n_selected(mask_topk) == _n_selected(mask_rr0)
        # For our magnitude-ordered packs the strategies will pick different windows
        # (top-k picks the tail, RR picks the head at round 0)
        assert list(mask_topk) != list(mask_rr0)

    def test_both_phases_select_same_count(self):
        """Selected pack count must be ceil(n*pct) regardless of phase."""
        packs = _make_packs(10)
        pct, rr_start = 0.3, 20
        k = math.ceil(len(packs) * pct)

        for rnd in [0, rr_start - 1, rr_start, rr_start + 5]:
            mask = _topkrr_mask(packs, pct, rnd, rr_start)
            assert _n_selected(mask) == k, f"round={rnd}: got {_n_selected(mask)}, expected {k}"

    @pytest.mark.parametrize("rr_start", [1, 50, 200, 400])
    def test_transition_at_various_rr_start_values(self, rr_start):
        """Transition happens at exactly rr_start regardless of its value."""
        packs = _make_packs(8)
        pct = 0.25

        # Round just before: top-k
        m_before = _topkrr_mask(packs, pct, rr_start - 1, rr_start)
        expected_topk = get_topk_mask(packs, pct)
        assert list(m_before) == list(expected_topk)

        # Round exactly at: RR round 0
        m_at = _topkrr_mask(packs, pct, rr_start, rr_start)
        expected_rr0 = get_robin_round_mask(round=0, packs=packs, size_window=pct, percentage=True)
        assert list(m_at) == list(expected_rr0)


# ── 4. Coverage guarantee ──────────────────────────────────────────────────────

class TestCoverageGuarantee:

    def test_topk_covers_same_packs_every_round(self):
        """Top-k is stateless: same packs are selected every round (no coverage guarantee)."""
        packs = _make_packs(10)
        pct = 0.3
        masks = [get_topk_mask(packs, pct) for _ in range(20)]
        first = masks[0]
        for m in masks[1:]:
            assert list(m) == list(first)

    @pytest.mark.parametrize("n,pct", [
        (10, 0.1),
        (12, 0.25),
        (20, 0.5),
    ])
    def test_rr_full_coverage_in_ceil_1_over_pct_rounds(self, n, pct):
        """RR guarantees full model coverage in ceil(1/pct) rounds."""
        packs = _make_packs(n)
        n_rounds = math.ceil(1 / pct)
        covered = set()
        for rnd in range(n_rounds):
            covered.update(_selected_indices(
                get_robin_round_mask(round=rnd, packs=packs, size_window=pct, percentage=True)
            ))
        assert covered == set(range(n))

    def test_hybrid_rr_phase_covers_all_packs(self):
        """After the RR phase starts, full coverage is achieved within ceil(1/pct) extra rounds."""
        n, pct, rr_start = 12, 0.25, 50
        packs = _make_packs(n)
        n_rr_rounds = math.ceil(1 / pct)

        covered = set()
        for rnd in range(rr_start, rr_start + n_rr_rounds):
            mask = _topkrr_mask(packs, pct, rnd, rr_start)
            covered.update(_selected_indices(mask))

        assert covered == set(range(n))


# ── 5. Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_pack_topk(self):
        packs = _make_packs(1)
        mask = get_topk_mask(packs, 1.0)
        assert _n_selected(mask) == 1

    def test_single_pack_rr(self):
        packs = _make_packs(1)
        mask = get_robin_round_mask(round=0, packs=packs, size_window=1.0, percentage=True)
        assert _n_selected(mask) == 1

    def test_rr_start_round_zero(self):
        """rr_start=0 means the entire run uses round-robin."""
        packs = _make_packs(8)
        pct   = 0.25
        for rnd in range(10):
            mask = _topkrr_mask(packs, pct, rnd, rr_start_round=0)
            expected = get_robin_round_mask(round=rnd, packs=packs, size_window=pct, percentage=True)
            assert list(mask) == list(expected)

    def test_rr_start_very_large_means_always_topk(self):
        """rr_start > total rounds means the run is entirely top-k."""
        packs = _make_packs(10)
        pct   = 0.3
        rr_start = 10_000
        expected = get_topk_mask(packs, pct)
        for rnd in range(1, 401):
            mask = _topkrr_mask(packs, pct, rnd, rr_start)
            assert list(mask) == list(expected)

    def test_percentage_clamp_at_100(self):
        """percentage=1.0 must select all packs in both phases."""
        packs = _make_packs(8)
        n = len(packs)
        assert _n_selected(get_topk_mask(packs, 1.0)) == n
        assert _n_selected(
            get_robin_round_mask(round=0, packs=packs, size_window=1.0, percentage=True)
        ) == n

    @pytest.mark.parametrize("pct", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_topk_count_correct_for_all_percentages(self, pct):
        n = 20
        packs = _make_packs(n)
        mask = get_topk_mask(packs, pct)
        assert _n_selected(mask) == math.ceil(n * pct)

    @pytest.mark.parametrize("pct", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_rr_count_correct_for_all_percentages(self, pct):
        n = 20
        packs = _make_packs(n)
        mask = get_robin_round_mask(round=0, packs=packs, size_window=pct, percentage=True)
        assert _n_selected(mask) == math.ceil(n * pct)
