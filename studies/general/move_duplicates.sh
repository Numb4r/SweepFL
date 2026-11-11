#!/bin/bash
# Move duplicate and erroneous experiment folders to logs/sparsification_percentage/duplicates/
# Delegates to Python for reliable associative-array logic.
#
# Usage (run from studies/sparsification_percentage/):
#   bash move_duplicates.sh [--dry-run]

cd "$(dirname "$0")"
python3 - "$@" <<'PYEOF'
import sys, re, shutil
from pathlib import Path
from collections import defaultdict

dry_run = "--dry-run" in sys.argv
logs_dir = Path("../../logs/sparsification_percentage").resolve()
dest = logs_dir / "duplicates"

if not logs_dir.exists():
    print(f"ERROR: {logs_dir} not found.")
    sys.exit(1)

dest.mkdir(exist_ok=True)

TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")

# Group folders by config (name without trailing timestamp)
groups = defaultdict(list)
erroneous = []

for p in sorted(logs_dir.iterdir()):
    if not p.is_dir() or p.name == "duplicates":
        continue
    m = TS_RE.search(p.name)
    if not m:
        continue
    ts = m.group(1)
    config = p.name[: m.start()]

    # SweepFL uppercase with percentage=-1 → always erroneous
    if "_sweepfl_" in p.name and "percentage-1" in p.name:
        erroneous.append(p)
        continue

    groups[config].append((ts, p))

moved = 0

# Keep newest timestamp per config; move older ones
for config, entries in groups.items():
    if len(entries) <= 1:
        continue
    entries.sort(key=lambda x: x[0])  # oldest first
    to_move = entries[:-1]             # all except newest
    for ts, p in to_move:
        target = dest / p.name
        action = f"mv {p.name} -> duplicates/"
        if dry_run:
            print(f"[DRY-RUN] {action}")
        else:
            shutil.move(str(p), str(target))
            print(action)
        moved += 1

# Move erroneous SweepFL folders
for p in erroneous:
    target = dest / p.name
    action = f"mv {p.name} -> duplicates/  [SweepFL uppercase erroneous]"
    if dry_run:
        print(f"[DRY-RUN] {action}")
    else:
        shutil.move(str(p), str(target))
        print(action)
    moved += 1

remaining = sum(1 for p in logs_dir.iterdir() if p.is_dir() and p.name != "duplicates")
print(f"\nMoved {moved} folders to duplicates/")
print(f"Remaining in logs/sparsification_percentage: {remaining} folders")
PYEOF
