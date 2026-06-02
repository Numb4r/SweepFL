#!/bin/bash
# experiments/test_resume_e2e.sh — End-to-end checkpoint/resume smoke test.
#
# 1. Run plaintext + nvnl for 1 round  → creates checkpoint/meta.json (round=1)
# 2. Manually patch meta.json so round=1 (already 1, just confirming)
# 3. Re-run with NUM_ROUNDS=2 (same params, different timestamp)
#    → _find_resume_checkpoint detects round=1 < 2 → resumes
# 4. Verify:
#    - server_eval.csv of resumed run has round=2 (offset applied)
#    - experiment_details.json has RESUMED_FROM and RESUME_ROUND=1
#    - [RESUME] appears in server and client logs

set -uo pipefail
cd "$(dirname "$0")/.."

GPU_IDS=(0)
CLIENTS_PER_GPU=2

if [[ ! -S /run/nvidia-persistenced/socket ]]; then
    USE_GPU=false
fi

CKKS_CONFIGS=("ckks 8192 60,40,40,60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
CKKS_BACKEND="ckks_python_simplify"
SIMPLIFY_N=8192
SIMPLIFY_BITS=62
SIMPLIFY_SCALE_BITS=40

NIIDS=(0.1)
SEEDS=(42)
TECHNIQUE="none"
PERCENTAGES=(0.1)
MODEL_DATASET_PAIRS=("resnet20:CIFAR10")
EXPERIMENT_NAME=""
FOLDER="test_resume_e2e"
RUNTIME="docker"
SIF_DIR="."

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PASS=0; FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$(( PASS + 1 )); }
fail() { echo "  [FAIL] $*"; FAIL=$(( FAIL + 1 )); }

# ── Phase 0: plaintext 1 round → init weights ─────────────────────────────────
echo "▶ Phase 0: plaintext 1 round (seeds init_weights)"
SOLUTIONS=("plaintext")
NUM_ROUNDS=1
run_theme

# ── Phase 1: nvnl 1 round → create checkpoint ─────────────────────────────────
echo ""
echo "▶ Phase 1: nvnl 1 round (creates checkpoint)"
SOLUTIONS=("nvnl")
NUM_ROUNDS=1
run_theme

# Find the 1-round nvnl run
FIRST_DIR=$(ls -dt logs/test_resume_e2e/resnet20_CIFAR10_nvnl_none_1rounds_2clients_seed42_* 2>/dev/null | head -1)
if [[ -z "$FIRST_DIR" ]]; then
    echo "[FATAL] Phase 1 dir not found"; exit 1
fi
echo "First run dir: $FIRST_DIR"

# Verify checkpoint created
META="${FIRST_DIR}/checkpoint/meta.json"
if [[ -f "$META" ]]; then
    CKPT_ROUND=$(python3 -c "import json; print(json.load(open('$META'))['round'])")
    echo "  meta.json: round=$CKPT_ROUND"
else
    echo "[FATAL] meta.json not created"; exit 1
fi

for cid in 0 1; do
    NPZ="${FIRST_DIR}/checkpoint/client_${cid}.npz"
    [[ -f "$NPZ" ]] || { echo "[FATAL] client_${cid}.npz missing"; exit 1; }
done
echo "  Checkpoint files OK (round=1)"

# ── Phase 2: nvnl 2 rounds → should detect round=1 and resume ─────────────────
echo ""
echo "▶ Phase 2: nvnl 2 rounds (should resume from round 1)"
SOLUTIONS=("nvnl")
NUM_ROUNDS=2
run_theme

# Find the 2-round nvnl resumed run
SECOND_DIR=$(ls -dt logs/test_resume_e2e/resnet20_CIFAR10_nvnl_none_2rounds_2clients_seed42_* 2>/dev/null | head -1)
if [[ -z "$SECOND_DIR" ]]; then
    echo "[FATAL] Phase 2 dir not found"; exit 1
fi
echo "Resumed run dir: $SECOND_DIR"

# ── Verify experiment_details.json ────────────────────────────────────────────
echo ""
echo "── Checking experiment_details.json"
DETAILS="${SECOND_DIR}/experiment_details.json"
RESUMED_FROM=$(python3 -c "import json; d=json.load(open('$DETAILS')); print(d.get('RESUMED_FROM',''))")
RESUME_ROUND=$(python3 -c "import json; d=json.load(open('$DETAILS')); print(d.get('RESUME_ROUND',0))")

if [[ -n "$RESUMED_FROM" ]]; then
    ok "RESUMED_FROM set: $RESUMED_FROM"
else
    fail "RESUMED_FROM is empty in experiment_details.json"
fi

if [[ "$RESUME_ROUND" == "1" ]]; then
    ok "RESUME_ROUND=1 in experiment_details.json"
else
    fail "RESUME_ROUND expected 1, got: $RESUME_ROUND"
fi

# ── Verify server_eval.csv has round 2 (not round 1) ─────────────────────────
echo ""
echo "── Checking server_eval.csv round numbers"
EVAL_CSV="${SECOND_DIR}/server_eval.csv"
if [[ ! -f "$EVAL_CSV" ]]; then
    fail "server_eval.csv not found"
else
    ROUNDS_IN_CSV=$(tail -n +2 "$EVAL_CSV" | awk -F, '{print $2}' | tr '\n' ' ')
    echo "  Rounds in server_eval.csv: $ROUNDS_IN_CSV"
    if echo "$ROUNDS_IN_CSV" | grep -qw "2"; then
        ok "Round 2 present in server_eval.csv"
    else
        fail "Round 2 NOT found in server_eval.csv (got: $ROUNDS_IN_CSV)"
    fi
    if echo "$ROUNDS_IN_CSV" | grep -qw "1"; then
        fail "Round 1 found in resumed server_eval.csv (should start at 2)"
    else
        ok "Round 1 NOT in resumed CSV (offset applied correctly)"
    fi
fi

# ── Verify server_agg.csv round numbers ──────────────────────────────────────
echo ""
echo "── Checking server_agg.csv round numbers"
AGG_CSV="${SECOND_DIR}/server_agg.csv"
if [[ ! -f "$AGG_CSV" ]]; then
    fail "server_agg.csv not found"
else
    ROUNDS_IN_AGG=$(tail -n +2 "$AGG_CSV" | awk -F, '{print $2}' | tr '\n' ' ')
    echo "  Rounds in server_agg.csv: $ROUNDS_IN_AGG"
    if echo "$ROUNDS_IN_AGG" | grep -qw "2"; then
        ok "Round 2 present in server_agg.csv"
    else
        fail "Round 2 NOT in server_agg.csv"
    fi
fi

# ── Verify [RESUME] in server log ─────────────────────────────────────────────
echo ""
echo "── Checking server log for [RESUME]"
SERVER_LOG="${SECOND_DIR}/tmux_SERVER.log"
if [[ ! -f "$SERVER_LOG" ]]; then
    fail "tmux_SERVER.log not found"
else
    if grep -q "\[RESUME\]" "$SERVER_LOG"; then
        RESUME_LINE=$(grep "\[RESUME\]" "$SERVER_LOG" | head -1)
        ok "Server log has [RESUME]: $RESUME_LINE"
    else
        fail "[RESUME] not found in server log"
    fi
fi

# ── Verify [RESUME] in client logs ────────────────────────────────────────────
echo ""
echo "── Checking client logs for [RESUME]"
for cid in 0 1; do
    CLIENT_LOG="${SECOND_DIR}/tmux_client-${cid}.log"
    if [[ ! -f "$CLIENT_LOG" ]]; then
        fail "tmux_client-${cid}.log not found"
    elif grep -q "\[RESUME\]" "$CLIENT_LOG"; then
        RESUME_LINE=$(grep "\[RESUME\]" "$CLIENT_LOG" | head -1)
        ok "Client $cid [RESUME]: $RESUME_LINE"
    else
        fail "Client $cid: [RESUME] not found in log"
    fi
done

# ── Verify checkpoint updated after resume ────────────────────────────────────
echo ""
echo "── Checking resumed run created its own checkpoint"
META2="${SECOND_DIR}/checkpoint/meta.json"
if [[ -f "$META2" ]]; then
    NEW_ROUND=$(python3 -c "import json; print(json.load(open('$META2'))['round'])")
    if [[ "$NEW_ROUND" == "2" ]]; then
        ok "Resumed run checkpoint has round=2"
    else
        fail "Resumed run checkpoint has round=$NEW_ROUND (expected 2)"
    fi
else
    fail "Resumed run did not create checkpoint/meta.json"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo " Results: $PASS passed, $FAIL failed"
echo "════════════════════════════════════════════════════════"
[[ $FAIL -eq 0 ]] && echo "All tests passed." || { echo "Some tests FAILED."; exit 1; }
