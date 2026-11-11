"""
Unit tests for the duplicate-folder handling in studies/general/load_data.py.

logs/ can hold two kinds of same-config folder pairs, and only one of them may be
concatenated:

  * resume fragments   — disjoint round ranges left behind when merge_resume.py
                         never ran. Their union is the real run: keep all.
  * duplicate reruns   — overlapping round ranges from running the same config
                         twice. Keep one, otherwise studies that sum a per-client
                         metric over a round report double the true cost.

No Docker, no cluster, no GPU required. Synthetic log trees only.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "studies" / "general"))

from load_data import count_experiments, load_all, select_folders

_CONFIG = (
    "resnet20_CIFAR10_{sol}_none_200rounds_9clients_seed42_"
    "niid0.1_percentage0.3_8192_-1_-1_60_1099511627776_None"
)


def _make_folder(root: Path, sol: str, timestamp: str, rounds, clients=2) -> Path:
    """Create one experiment folder with a train.csv covering `rounds`."""
    folder = root / f"{_CONFIG.format(sol=sol)}_{timestamp}"
    folder.mkdir(parents=True)
    rows = [
        {"experiment_name": "t", "round": r, "cid": c,
         "payload_bytes": 100, "n_params": 274042, "n_packs": 67,
         "packs_sent": 20, "pack_size": 4096, "train_time_s": 1.0}
        for r in rounds for c in range(clients)
    ]
    pd.DataFrame(rows).to_csv(folder / "train.csv", index=False)
    return folder


def _names(folders):
    return sorted(f.name for f in folders)


# ── select_folders ─────────────────────────────────────────────────────────────

def test_single_folder_is_kept(tmp_path):
    a = _make_folder(tmp_path, "sweepfl", "2026-06-01_10-00-00", range(1, 201))
    keep, skip = select_folders([a])
    assert _names(keep) == [a.name]
    assert skip == []


def test_duplicate_reruns_keep_only_one(tmp_path):
    """Two complete 1-200 runs of the same config: one must be dropped."""
    a = _make_folder(tmp_path, "sweepfl", "2026-06-01_10-00-00", range(1, 201))
    b = _make_folder(tmp_path, "sweepfl", "2026-06-09_10-00-00", range(1, 201))
    keep, skip = select_folders([a, b])
    assert len(keep) == 1
    assert len(skip) == 1
    # newest wins the tie
    assert _names(keep) == [b.name]


def test_resume_fragments_are_both_kept(tmp_path):
    """Disjoint 1-124 + 125-200 is a resume chain, not a duplicate."""
    a = _make_folder(tmp_path, "sw", "2026-06-09_06-00-00", range(1, 125))
    b = _make_folder(tmp_path, "sw", "2026-06-10_10-00-00", range(125, 201))
    keep, skip = select_folders([a, b])
    assert _names(keep) == _names([a, b])
    assert skip == []


def test_abandoned_attempt_loses_to_complete_resume_chain(tmp_path):
    """A long dead-end run must not shadow a shorter pair that covers 1-200.

    Mirrors resnet20/sweepfl/p=0.9/niid1 in logs/sparsification_percentage: an
    abandoned 1-123 attempt next to a 1-107 + 108-200 chain. Longest-first greedy
    picks the dead end and loses rounds 124-200.
    """
    dead = _make_folder(tmp_path, "sweepfl", "2026-06-02_14-00-00", range(1, 124))
    head = _make_folder(tmp_path, "sweepfl", "2026-06-10_01-00-00", range(1, 108))
    tail = _make_folder(tmp_path, "sweepfl", "2026-06-10_10-00-00", range(108, 201))
    keep, skip = select_folders([dead, head, tail])
    assert _names(keep) == _names([head, tail])
    assert _names(skip) == [dead.name]


def test_folder_without_train_csv_is_kept(tmp_path):
    """Empty shells (logs/client_frac) carry no rounds and must not be dropped."""
    a = _make_folder(tmp_path, "fedphe", "2026-06-17_10-00-00", range(1, 201))
    shell = tmp_path / f"{_CONFIG.format(sol='fedphe')}_2026-06-13_10-00-00"
    shell.mkdir()
    keep, skip = select_folders([a, shell])
    assert _names(keep) == _names([a, shell])
    assert skip == []


def test_distinct_configs_never_compete(tmp_path):
    a = _make_folder(tmp_path, "sweepfl", "2026-06-01_10-00-00", range(1, 201))
    b = _make_folder(tmp_path, "sw", "2026-06-01_11-00-00", range(1, 201))
    keep, skip = select_folders([a, b])
    assert _names(keep) == _names([a, b])
    assert skip == []


# ── load_all / count_experiments ───────────────────────────────────────────────

def test_load_all_does_not_double_count_reruns(tmp_path):
    """The bug this guards: 2 folders x 2 clients gave 4 rows per round."""
    _make_folder(tmp_path, "sweepfl", "2026-06-01_10-00-00", range(1, 11), clients=2)
    _make_folder(tmp_path, "sweepfl", "2026-06-09_10-00-00", range(1, 11), clients=2)
    train = load_all(str(tmp_path))["train"]
    assert train.groupby("round").size().max() == 2
    assert train["round"].nunique() == 10


def test_load_all_unions_resume_fragments(tmp_path):
    _make_folder(tmp_path, "sw", "2026-06-09_06-00-00", range(1, 7), clients=2)
    _make_folder(tmp_path, "sw", "2026-06-10_10-00-00", range(7, 11), clients=2)
    train = load_all(str(tmp_path))["train"]
    assert train.groupby("round").size().max() == 2
    assert sorted(train["round"].unique()) == list(range(1, 11))


def test_count_experiments_excludes_duplicates(tmp_path):
    _make_folder(tmp_path, "sweepfl", "2026-06-01_10-00-00", range(1, 11))
    _make_folder(tmp_path, "sweepfl", "2026-06-09_10-00-00", range(1, 11))
    _make_folder(tmp_path, "sw", "2026-06-01_11-00-00", range(1, 11))
    assert count_experiments(str(tmp_path)) == 2
