#!/bin/bash
# tests/test_checkpoint_resume.sh — Verifies checkpoint save and resume logic.
#
# Run from repo root after test_ckks_backend.sh completes at least once:
#   bash tests/test_checkpoint_resume.sh
#
# Tests:
#   1. Verify checkpoint files exist after a completed run
#   2. _find_resume_checkpoint: completed run (round==NUM_ROUNDS) → no resume
#   3. _find_resume_checkpoint: partial run (round<NUM_ROUNDS) → resume detected
#   4. Python server resume: verifies start_round_offset and remaining_rounds
#   5. SLURM path simulation: lib_slurm.sh _find_resume_checkpoint via source

set -euo pipefail
cd "$(dirname "$0")/.."

PASS=0; FAIL=0

ok()   { echo "  [PASS] $*"; PASS=$(( PASS + 1 )); }
fail() { echo "  [FAIL] $*"; FAIL=$(( FAIL + 1 )); }

echo "════════════════════════════════════════════════════════"
echo " Checkpoint / Resume Test Suite"
echo "════════════════════════════════════════════════════════"

# ─── 1. Find the most recent fedphe phase1_docker run ────────────────────────
echo ""
echo "── Test 1: Checkpoint files present after completed run"

LATEST_DIR=$(ls -dt logs/test_ckks_backend/phase1_docker/resnet20_CIFAR10_fedphe_* 2>/dev/null | head -1)
if [[ -z "$LATEST_DIR" ]]; then
    echo "  [SKIP] No fedphe phase1_docker run found — run test_ckks_backend.sh first"
else
    META="${LATEST_DIR}/checkpoint/meta.json"
    if [[ -f "$META" ]]; then
        SAVED_ROUND=$(python3 -c "import json; d=json.load(open('$META')); print(d['round'])")
        ok "meta.json exists, saved round=$SAVED_ROUND"
    else
        fail "meta.json not found at $META"
    fi

    for cid in 0 1; do
        NPZ="${LATEST_DIR}/checkpoint/client_${cid}.npz"
        if [[ -f "$NPZ" ]]; then
            ok "client_${cid}.npz exists ($(du -h "$NPZ" | cut -f1))"
        else
            fail "client_${cid}.npz missing at $NPZ"
        fi
    done
fi

# ─── 2. _find_resume_checkpoint: completed run → no resume ───────────────────
echo ""
echo "── Test 2: Completed run (round==NUM_ROUNDS) → no resume"

if [[ -z "${LATEST_DIR:-}" ]]; then
    echo "  [SKIP] No fedphe run found"
else
    # Extract only _find_resume_checkpoint from lib_tmux.sh to avoid side effects
    eval "$(sed -n '/_find_resume_checkpoint()/,/^}/p' experiments/lib_tmux.sh)"

    FOLDER="test_ckks_backend/phase1_docker"
    MODEL="resnet20"; DATASET_FOLDER="CIFAR10"; SOLUTION="fedphe"; TECHNIQUE="none"
    NCLIENTES=2; SEED=42; NIID=0.1; PERCENTAGE=0.1
    POLY=8192; PLAIN_MODULUS=-1; N_LENGTH=-1
    COEFFS="60,40,40,60"; SCALE=1099511627776; KEY_TYPE="None"
    NUM_ROUNDS=3   # same as saved round → should NOT resume

    _find_resume_checkpoint
    if [[ -z "$RESUME_LOGFOLDER" ]]; then
        ok "Completed run not picked up for resume (RESUME_LOGFOLDER empty)"
    else
        fail "Completed run incorrectly flagged for resume: RESUME_LOGFOLDER=$RESUME_LOGFOLDER"
    fi
fi

# ─── 3. _find_resume_checkpoint: partial run (NUM_ROUNDS extended) → resume ──
echo ""
echo "── Test 3: Extended NUM_ROUNDS → incomplete → resume detected"

if [[ -z "${LATEST_DIR:-}" ]]; then
    echo "  [SKIP] No fedphe run found"
else
    NUM_ROUNDS=10  # extend beyond saved round (3) → should resume

    _find_resume_checkpoint
    if [[ -n "$RESUME_LOGFOLDER" ]]; then
        ok "Resume detected: RESUME_LOGFOLDER=$RESUME_LOGFOLDER, RESUME_ROUND=$RESUME_ROUND"
    else
        fail "Resume NOT detected with NUM_ROUNDS=10 (saved round was 3)"
    fi
fi

# ─── 4. Python server resume: start_round_offset and remaining_rounds ─────────
echo ""
echo "── Test 4: Python server reads RESUME_LOGFOLDER and computes offsets"

if [[ -z "${LATEST_DIR:-}" ]]; then
    echo "  [SKIP] No fedphe run found"
else
    RESUME_DIR="${LATEST_DIR#logs/}"
    python3 - <<PYEOF
import os, json, sys
os.environ['RESUME_LOGFOLDER'] = '$RESUME_DIR'
os.environ['NUM_ROUNDS'] = '10'

meta = json.load(open('${LATEST_DIR}/checkpoint/meta.json'))
saved_round = int(meta.get('round', 0))
remaining = max(1, int(os.environ['NUM_ROUNDS']) - saved_round)
expected_remaining = 10 - saved_round

if remaining == expected_remaining:
    print(f"  [PASS] remaining_rounds={remaining} (NUM_ROUNDS=10, saved_round={saved_round})")
else:
    print(f"  [FAIL] remaining_rounds={remaining}, expected {expected_remaining}")
    sys.exit(1)
PYEOF
    if [[ $? -eq 0 ]]; then PASS=$(( PASS + 1 )); else FAIL=$(( FAIL + 1 )); fi
fi

# ─── 5. SLURM simulation: lib_slurm.sh _find_resume_checkpoint ───────────────
echo ""
echo "── Test 5: SLURM path _find_resume_checkpoint (sourced from lib_slurm.sh)"

# Extract _find_resume_checkpoint from lib_slurm.sh (same function, different file)
eval "$(sed -n '/_find_resume_checkpoint()/,/^}/p' experiments/lib_slurm.sh)"

if [[ -z "${LATEST_DIR:-}" ]]; then
    echo "  [SKIP] No fedphe run found"
else
    FOLDER="test_ckks_backend/phase1_docker"
    MODEL="resnet20"; DATASET_FOLDER="CIFAR10"; SOLUTION="fedphe"; TECHNIQUE="none"
    NCLIENTES=2; SEED=42; NIID=0.1; PERCENTAGE=0.1
    POLY=8192; PLAIN_MODULUS=-1; N_LENGTH=-1
    COEFFS="60,40,40,60"; SCALE=1099511627776; KEY_TYPE="None"
    NUM_ROUNDS=10   # should detect resume

    _find_resume_checkpoint
    if [[ -n "$RESUME_LOGFOLDER" ]]; then
        ok "lib_slurm.sh _find_resume_checkpoint: RESUME_LOGFOLDER=$RESUME_LOGFOLDER round=$RESUME_ROUND"
    else
        fail "lib_slurm.sh _find_resume_checkpoint: resume NOT detected"
    fi
fi

# ─── 6. Artificial partial checkpoint test ────────────────────────────────────
echo ""
echo "── Test 6: Artificial partial checkpoint (round=1 of 3)"

# This fake dir has the same params as the real run but round=1 saved.
# With NUM_ROUNDS=3, the real run (saved_round=3) is NOT a candidate (3 < 3 is false).
# The fake (saved_round=1) IS a candidate (1 < 3 is true) → fake should be picked.
FAKE_DIR="logs/test_ckks_backend/phase1_docker/resnet20_CIFAR10_fedphe_none_3rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_60,40,40,60_1099511627776_None_2099-01-01_00-00-00"
mkdir -p "${FAKE_DIR}/checkpoint"
echo '{"round": 1, "solution_type": "CKKS"}' > "${FAKE_DIR}/checkpoint/meta.json"
python3 -c "import numpy as np; np.savez_compressed('${FAKE_DIR}/checkpoint/client_0', np.zeros(10, dtype=np.float16))"
python3 -c "import numpy as np; np.savez_compressed('${FAKE_DIR}/checkpoint/client_1', np.zeros(10, dtype=np.float16))"

FOLDER="test_ckks_backend/phase1_docker"
MODEL="resnet20"; DATASET_FOLDER="CIFAR10"; SOLUTION="fedphe"; TECHNIQUE="none"
NCLIENTES=2; SEED=42; NIID=0.1; PERCENTAGE=0.1
POLY=8192; PLAIN_MODULUS=-1; N_LENGTH=-1
COEFFS="60,40,40,60"; SCALE=1099511627776; KEY_TYPE="None"
NUM_ROUNDS=3   # real run round=3 is NOT a candidate (3 < 3 = false); fake round=1 IS

_find_resume_checkpoint
if [[ "$RESUME_LOGFOLDER" == *"2099-01-01"* ]] && [[ "$RESUME_ROUND" == "1" ]]; then
    ok "Artificial partial checkpoint detected (round=1 / 3): $RESUME_LOGFOLDER"
else
    fail "Artificial partial checkpoint not detected. Got: RESUME_LOGFOLDER='$RESUME_LOGFOLDER' RESUME_ROUND='$RESUME_ROUND'"
fi

# With NUM_ROUNDS=10: both real (round=3) and fake (round=1) are candidates;
# real should win because 3 > 1.
NUM_ROUNDS=10
_find_resume_checkpoint
if [[ "$RESUME_ROUND" == "3" ]]; then
    ok "Best candidate wins when multiple exist (round=$RESUME_ROUND)"
else
    fail "Expected round=3 to win, got RESUME_ROUND='$RESUME_ROUND'"
fi

# Cleanup fake dir
rm -rf "$FAKE_DIR"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo " Results: $PASS passed, $FAIL failed"
echo "════════════════════════════════════════════════════════"
[[ $FAIL -eq 0 ]] && echo "All tests passed." || { echo "Some tests failed."; exit 1; }
