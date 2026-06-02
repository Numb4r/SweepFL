#!/usr/bin/env python3
"""
experiments/merge_resume.py
Merge original + resumed experiment log folders in-place.

When a SLURM job is interrupted and resumed, the resumed experiment writes its
rounds to a NEW folder.  This script merges all CSV logs, model snapshots, and
checkpoint from every resumed folder back into the ORIGINAL folder so the full
experiment is in one place.  Resumed folders are moved to <parent>/archive/.

Usage:
    python3 experiments/merge_resume.py [--logs-dir PATH] [--dry-run] [--yes]

Options:
    --dry-run   Show what would be merged without making any changes
    --yes/--auto  Skip interactive confirmation (use inside SLURM jobs)
    --logs-dir  Root logs directory (default: logs/)

Examples:
    # Interactive — asks y/n before each merge
    python3 experiments/merge_resume.py

    # Dry-run: see what would happen
    python3 experiments/merge_resume.py --dry-run

    # Non-interactive (used automatically by generated resume.sh)
    python3 experiments/merge_resume.py --yes

    # Custom logs directory
    python3 experiments/merge_resume.py --logs-dir /path/to/logs
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# CSV files that exist in every experiment folder
CSV_FILES = ["train.csv", "evaluate.csv", "server_agg.csv", "server_eval.csv"]

# Columns to sort by when merging each CSV
_SORT_COLS: dict[str, list[str]] = {
    "train.csv": ["round", "cid"],
    "evaluate.csv": ["round", "cid"],
    "server_agg.csv": ["round"],
    "server_eval.csv": ["round"],
}

# Columns that uniquely identify a row (used for dedup)
_DEDUP_COLS: dict[str, list[str]] = {
    "train.csv": ["round", "cid"],
    "evaluate.csv": ["round", "cid"],
    "server_agg.csv": ["round"],
    "server_eval.csv": ["round"],
}


def _info(msg: str) -> None:
    print(msg, file=sys.stderr)


def _out(msg: str) -> None:
    print(msg)


# ─── JSON helpers ────────────────────────────────────────────────────────────


def _load_details(folder: Path) -> dict | None:
    path = folder / "experiment_details.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        _info(f"  [WARN] Cannot read {path}: {exc}")
        return None


def _get_checkpoint_round(folder: Path) -> int | None:
    meta = folder / "checkpoint" / "meta.json"
    if not meta.exists():
        return None
    try:
        with open(meta) as fh:
            d = json.load(fh)
        return int(d.get("round", 0))
    except Exception:
        return None


def _is_complete(folder: Path, details: dict) -> bool:
    """True if checkpoint round >= NUM_ROUNDS (experiment finished all rounds)."""
    num_rounds = int(details.get("NUM_ROUNDS", 0))
    ckpt = _get_checkpoint_round(folder)
    return ckpt is not None and ckpt >= num_rounds


def _is_already_merged(original: Path) -> bool:
    """True if original already has a MERGED_FROM marker (was merged before)."""
    details = _load_details(original)
    return bool(details and details.get("MERGED_FROM"))


# ─── Chain detection ─────────────────────────────────────────────────────────


def _build_chain(start: Path, logs_dir: Path) -> list[Path]:
    """
    Follow RESUMED_FROM pointers back to the root original folder.
    Returns [original, ..., resume_n] ordered from oldest to newest.
    """
    chain: list[Path] = []
    current: Path | None = start
    visited: set[Path] = set()

    while current is not None:
        if current in visited:
            _info(f"  [WARN] Cycle detected in RESUMED_FROM chain at {current.name}")
            break
        visited.add(current)
        chain.insert(0, current)
        details = _load_details(current)
        if details is None:
            break
        resumed_from = details.get("RESUMED_FROM", "")
        if not resumed_from:
            break
        parent = logs_dir / resumed_from
        if not parent.is_dir():
            _info(f"  [WARN] RESUMED_FROM points to missing folder: {parent}")
            break
        current = parent

    return chain


def find_chains(logs_dir: Path) -> list[list[Path]]:
    """
    Return a list of merge-ready chains.
    Each chain is [original, resume1, resume2, ...] ordered oldest→newest.
    Only chains whose LAST folder is complete (all rounds done) are returned.
    Chains whose original already has MERGED_FROM are skipped (already merged).
    """
    # Map: original_path → chain  (None = conflicted, skip)
    chains_by_original: dict[Path, list[Path] | None] = {}

    for details_path in sorted(logs_dir.rglob("experiment_details.json")):
        folder = details_path.parent

        # Skip anything inside an archive/ directory
        if "archive" in folder.relative_to(logs_dir).parts:
            continue

        details = _load_details(folder)
        if details is None:
            continue

        resumed_from = details.get("RESUMED_FROM", "")
        if not resumed_from:
            continue  # Not a resumed experiment

        if not _is_complete(folder, details):
            continue  # This resume hasn't finished yet

        chain = _build_chain(folder, logs_dir)
        if len(chain) < 2:
            continue  # No chain found

        original = chain[0]

        if _is_already_merged(original):
            continue  # Already merged

        if original in chains_by_original:
            if chains_by_original[original] is not None:
                _info(
                    f"  [WARN] Multiple completed resume chains for the same original:\n"
                    f"         {original.name}\n"
                    f"         Skipping to avoid ambiguous merge."
                )
                chains_by_original[original] = None  # Mark conflicted
        else:
            chains_by_original[original] = chain

    return [c for c in chains_by_original.values() if c is not None]


# ─── Core merge logic ─────────────────────────────────────────────────────────


def _archive_copy(folder: Path, archive_dir: Path) -> Path:
    """Copy folder to archive_dir as a safety backup.  Returns destination path."""
    dest = archive_dir / folder.name
    if dest.exists():
        _info(f"    [WARN] Archive backup already exists, skipping copy: {dest.name}")
        return dest
    shutil.copytree(folder, dest)
    return dest


def _merge_csvs(chain: list[Path], original: Path) -> None:
    """Append resumed CSV rows into the original folder's CSV files."""
    resumed_folders = chain[1:]
    for csv_name in CSV_FILES:
        dfs: list[pd.DataFrame] = []

        orig_csv = original / csv_name
        if orig_csv.exists():
            try:
                dfs.append(pd.read_csv(orig_csv))
            except Exception as exc:
                _info(f"    [WARN] Could not read {orig_csv.name}: {exc}")

        for resume in resumed_folders:
            resume_csv = resume / csv_name
            if resume_csv.exists():
                try:
                    dfs.append(pd.read_csv(resume_csv))
                except Exception as exc:
                    _info(f"    [WARN] Could not read {resume_csv}: {exc}")

        if len(dfs) <= 1:
            continue  # Nothing new to merge for this CSV

        merged = pd.concat(dfs, ignore_index=True)

        sort_cols = [c for c in _SORT_COLS[csv_name] if c in merged.columns]
        dedup_cols = [c for c in _DEDUP_COLS[csv_name] if c in merged.columns]

        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)

        if dedup_cols:
            # keep="last" → resumed data wins if same round appears twice
            merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
            if sort_cols:
                merged = merged.sort_values(sort_cols).reset_index(drop=True)

        merged.to_csv(orig_csv, index=False)
        _info(f"    {csv_name}: {len(merged)} rows total")


def _merge_model_snapshots(chain: list[Path], original: Path) -> None:
    """Copy model/round_*.npz from resumed folders into the original/model/ dir."""
    dest_model = original / "model"
    for resume in chain[1:]:
        src_model = resume / "model"
        if not src_model.is_dir():
            continue
        dest_model.mkdir(parents=True, exist_ok=True)
        for npz in src_model.glob("round_*.npz"):
            dest = dest_model / npz.name
            if not dest.exists():
                shutil.copy2(npz, dest)


def _replace_checkpoint(chain: list[Path], original: Path) -> None:
    """Replace original's checkpoint with the one from the last folder in the chain."""
    last_ckpt = chain[-1] / "checkpoint"
    if not last_ckpt.is_dir():
        return
    dest_ckpt = original / "checkpoint"
    if dest_ckpt.exists():
        shutil.rmtree(dest_ckpt)
    shutil.copytree(last_ckpt, dest_ckpt)


def _update_details(chain: list[Path], original: Path, logs_dir: Path) -> None:
    """Add MERGED_FROM and MERGED_AT to original's experiment_details.json."""
    details = _load_details(original) or {}
    details["MERGED_FROM"] = [
        str(f.relative_to(logs_dir)) for f in chain[1:]
    ]
    details["MERGED_AT"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(original / "experiment_details.json", "w") as fh:
        json.dump(details, fh, indent=2)


def merge_chain(chain: list[Path], *, dry_run: bool, logs_dir: Path) -> bool:
    """
    Merge all CSV/model/checkpoint data from chain[1:] into chain[0] (original).
    Moves chain[1:] to <parent>/archive/ afterwards.
    Returns True on success.
    """
    original = chain[0]
    resumed_folders = chain[1:]
    archive_dir = original.parent / "archive"

    if dry_run:
        _out(f"  Would merge {len(chain)} folder(s):")
        for f in chain:
            label = "[original]" if f == original else "[resumed ]"
            _out(f"    {label} {f.name}")
        _out(f"  Archive destination: {archive_dir.relative_to(logs_dir)}/")
        return True

    try:
        # ── Step 1: Safety backup — copy everything to archive before touching ──
        archive_dir.mkdir(parents=True, exist_ok=True)
        _info(f"  Backing up to archive/...")
        for folder in chain:
            _archive_copy(folder, archive_dir)

        # ── Step 2: Merge CSV files into original ────────────────────────────
        _info(f"  Merging CSV files...")
        _merge_csvs(chain, original)

        # ── Step 3: Copy model snapshots ─────────────────────────────────────
        _info(f"  Copying model snapshots...")
        _merge_model_snapshots(chain, original)

        # ── Step 4: Replace checkpoint with latest ────────────────────────────
        _info(f"  Updating checkpoint...")
        _replace_checkpoint(chain, original)

        # ── Step 5: Update experiment_details.json ────────────────────────────
        _update_details(chain, original, logs_dir)

        # ── Step 6: Remove INCOMPLETE.log from original ───────────────────────
        incomplete = original / "INCOMPLETE.log"
        if incomplete.exists():
            incomplete.unlink()

        # ── Step 7: Remove live resumed folder copies ─────────────────────────
        # (safe to delete: archive copies were made in Step 1)
        _info(f"  Removing resumed folder(s) from live location...")
        for resume in resumed_folders:
            shutil.rmtree(resume)

        return True

    except Exception as exc:
        _info(f"  [ERROR] Merge failed: {exc}")
        _info(f"  Backups are in {archive_dir}/ — original data has NOT been lost.")
        return False


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge original + resumed experiment folders in-place.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs-dir", default="logs", help="Root logs directory (default: logs/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without making any changes",
    )
    parser.add_argument(
        "--yes",
        "--auto",
        dest="yes",
        action="store_true",
        help="Skip interactive confirmation (use in SLURM/batch scripts)",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    if not logs_dir.is_dir():
        _info(f"[merge] Logs directory not found: {logs_dir}")
        return

    chains = find_chains(logs_dir)

    if not chains:
        _out("[merge] No completed resume chains found to merge.")
        return

    _out(f"\n[merge] Found {len(chains)} chain(s) ready to merge:\n")
    for i, chain in enumerate(chains, 1):
        original = chain[0]
        _out(f"  Chain {i}/{len(chains)}:")
        _out(f"    [original] {original.relative_to(logs_dir)}")
        for resume in chain[1:]:
            _out(f"    [resumed ] {resume.relative_to(logs_dir)}")
    _out("")

    if args.dry_run:
        _out("[merge] Dry-run mode — no changes made.\n")
        for chain in chains:
            merge_chain(chain, dry_run=True, logs_dir=logs_dir)
        return

    if not args.yes:
        try:
            ans = input("Proceed with merge? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _out("\n[merge] Aborted.")
            return
        if ans not in ("y", "yes"):
            _out("[merge] Aborted.")
            return

    success = 0
    for i, chain in enumerate(chains, 1):
        original = chain[0]
        _out(f"\n[merge] ({i}/{len(chains)}) {original.name}")
        if merge_chain(chain, dry_run=False, logs_dir=logs_dir):
            _out(f"  ✓ Merged successfully")
            success += 1
        else:
            _out(f"  ✗ Failed (see warnings above)")

    _out(f"\n[merge] {success}/{len(chains)} chain(s) merged successfully.")


if __name__ == "__main__":
    main()
