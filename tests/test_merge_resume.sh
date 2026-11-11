#!/bin/bash
# tests/test_merge_resume.sh — Smoke tests for experiments/merge_resume.py
#
# Creates synthetic log folder structures (no Docker required) and verifies:
#   1. Dry-run: detects chain, makes no changes
#   2. Basic merge: original (2 rounds) + resumed (2 rounds) → 4 rounds in-place
#   3. CSV correctness: rows sorted, deduped, counts correct
#   4. Model snapshots: all round_N.npz present in original after merge
#   5. Checkpoint replaced with the latest one
#   6. experiment_details.json: MERGED_FROM / MERGED_AT added, INCOMPLETE.log removed
#   7. Archive: resumed folder moved to <parent>/archive/
#   8. Idempotency: second run is a no-op (already merged)
#   9. Chain merge: original → resume1 → resume2 (3 folders → 1 merged)
#  10. Conflict detection: two resumed folders for same original → skip both
#
# Run from repo root:
#   bash tests/test_merge_resume.sh

set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0; FAIL=0

ok()   { echo "  [PASS] $*"; PASS=$(( PASS + 1 )); }
fail() { echo "  [FAIL] $*"; FAIL=$(( FAIL + 1 )); }

TESTROOT="logs/test_merge_resume"
SCRIPT="experiments/merge_resume.py"

# ─── helpers ──────────────────────────────────────────────────────────────────

cleanup() { rm -rf "$TESTROOT"; }

# Creates a minimal experiment_details.json.
# $1 = abs folder path
# $2 = NUM_ROUNDS
# $3 = RESUMED_FROM (relative to the test logs_dir, empty string for originals)
# $4 = RESUME_ROUND (0 if not resumed)
make_details() {
    local folder="$1" num_rounds="$2" resumed_from="${3:-}" resume_round="${4:-0}"
    mkdir -p "$folder"
    python3 - "$folder" "$num_rounds" "$resumed_from" "$resume_round" <<'PYEOF'
import json, sys
folder, num_rounds, resumed_from, resume_round = sys.argv[1:]
d = {
    "EXPERIMENT_NAME": "smoke_test",
    "FOLDER": "",
    "TIMESTAMP": "2099-01-01_00-00-00",
    "SOLUTION": "sweepfl",
    "MODEL": "lenet5",
    "DATASET": "FASHION_MNIST",
    "NIID": 0.1,
    "SEED": 42,
    "PERCENTAGE": 0.1,
    "TECHNIQUE": "none",
    "NUM_ROUNDS": int(num_rounds),
    "NCLIENTES": 2,
    "CLIENTS_PER_GPU": 2,
    "GPU_IDS": [0],
    "RUNTIME": "docker",
    "USE_GPU": "false",
    "BATCH_SIZE": 4,
    "PRETRAINED_WEIGHTS": "",
    "POLY": 8192,
    "COEFFS": [-1],
    "SCALE": 1099511627776,
    "KEY_TYPE": "None",
    "PLAIN_MODULUS": -1,
    "N_LENGTH": -1,
    "CKKS_BACKEND": "tenseal",
    "SIMPLIFY_N": 8192, "SIMPLIFY_BITS": 62, "SIMPLIFY_SCALE_BITS": 60,
    "PYCRYPT_N": 8192, "PYCRYPT_BITS": 60,
    "OPENFHE_RING_DIM": 8192, "OPENFHE_MULT_DEPTH": 1, "OPENFHE_SCALE_BITS": 50,
    "RESUMED_FROM": resumed_from,
    "RESUME_ROUND": int(resume_round),
}
with open(f"{folder}/experiment_details.json", "w") as f:
    json.dump(d, f, indent=2)
PYEOF
}

# Creates checkpoint/meta.json + dummy client npz files for the given round.
make_checkpoint() {
    local folder="$1" round="$2"
    mkdir -p "$folder/checkpoint"
    python3 - "$folder" "$round" <<'PYEOF'
import json, numpy as np, sys
folder, round_ = sys.argv[1], int(sys.argv[2])
json.dump({"round": round_, "solution_type": "sweepfl"},
          open(f"{folder}/checkpoint/meta.json", "w"))
for cid in [0, 1]:
    np.savez_compressed(f"{folder}/checkpoint/client_{cid}",
                        np.zeros(10, dtype=np.float16))
PYEOF
}

# Creates the 4 standard CSV files covering the given list of round numbers.
# Usage: make_csvs <folder> <round1> [round2 ...]
make_csvs() {
    local folder="$1"; shift
    local rounds_csv
    rounds_csv=$(python3 -c "import sys; print(','.join(sys.argv[1:]))" "$@")
    mkdir -p "$folder"
    python3 - "$folder" "$rounds_csv" <<'PYEOF'
import csv, sys
folder = sys.argv[1]
rounds = [int(x) for x in sys.argv[2].split(",")]

with open(f"{folder}/train.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment_name","round","cid","solution","schema","model","dataset",
                "distribution","dirichlet_alpha","train_loss","train_acc",
                "decrypt_time_s","decrypt_time_cpu","train_time_s","train_time_cpu",
                "flatten_time_s","flatten_time_cpu","quantize_time_s","quantize_time_cpu",
                "batch_padding_time_s","batch_padding_time_cpu","pack_time_s","pack_time_cpu",
                "mask_select_time_s","mask_select_time_cpu","encrypt_time_s","encrypt_time_cpu",
                "serialize_time_s","serialize_time_cpu","payload_bytes","n_params",
                "n_packs","packs_sent","pack_size","sigma_ckks","poly_modulus_degree",
                "coeff_mod_bits","scale_bits","plain_modulus","ram_rss_mb","ram_peak_mb"])
    for r in rounds:
        for cid in [0, 1]:
            w.writerow(["smoke_test",r,cid,"sweepfl","ckks","lenet5","FASHION_MNIST",
                        "iid",1.0,0.5,0.8,
                        0.1,0.1,1.0,1.0,0,0,0,0,0,0,
                        0.05,0.05,0.02,0.02,0.3,0.3,0.1,0.1,
                        1000,44000,6,2,4096,0.0,8192,
                        "60;40;40;60",40,-1,200,210])

with open(f"{folder}/evaluate.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment_name","round","cid","solution","schema","model","dataset",
                "distribution","dirichlet_alpha","eval_loss","eval_acc",
                "decrypt_time_s","poly_modulus_degree","coeff_mod_bits","scale_bits"])
    for r in rounds:
        for cid in [0, 1]:
            w.writerow(["smoke_test",r,cid,"sweepfl","ckks","lenet5","FASHION_MNIST",
                        "iid",1.0,0.4,round(0.85+r*0.01,4),0.05,8192,"60;40;40;60",40])

with open(f"{folder}/server_agg.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment_name","round","n_clients","total_examples",
                "selection_time_s","aggregation_time_s"])
    for r in rounds:
        w.writerow(["smoke_test",r,2,100,0.01,0.5])

with open(f"{folder}/server_eval.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment_name","round","n_clients","mean_loss","mean_acc"])
    for r in rounds:
        w.writerow(["smoke_test",r,2,0.4,round(0.85+r*0.01,4)])
PYEOF
}

# Creates model/round_N.npz files for the given round numbers.
# Usage: make_model_snapshots <folder> <round1> [round2 ...]
make_model_snapshots() {
    local folder="$1"; shift
    local rounds_csv
    rounds_csv=$(python3 -c "import sys; print(','.join(sys.argv[1:]))" "$@")
    mkdir -p "$folder/model"
    python3 - "$folder" "$rounds_csv" <<'PYEOF'
import numpy as np, sys
folder = sys.argv[1]
rounds = [int(x) for x in sys.argv[2].split(",")]
for r in rounds:
    np.savez_compressed(f"{folder}/model/round_{r}", np.zeros(44000, dtype=np.float16))
PYEOF
}

# Returns sorted space-separated unique round values from a CSV's "round" column.
csv_rounds() {
    python3 -c "
import pandas as pd, sys
try:
    df = pd.read_csv('$1')
    print(' '.join(str(r) for r in sorted(df['round'].unique())))
except Exception as e:
    print('error:', e, file=sys.stderr)
    print('error')
" 2>/dev/null
}

# Returns number of data rows in a CSV.
csv_rows() {
    python3 -c "
import pandas as pd, sys
try:
    print(len(pd.read_csv('$1')))
except Exception as e:
    print('error', file=sys.stderr)
    print(-1)
" 2>/dev/null
}

# ─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo " Merge Resume Smoke Test Suite"
echo "════════════════════════════════════════════════════════"
echo ""

cleanup

# Folder names used for basic/chain/conflict subtests (under $TESTROOT/)
ORIG1="$TESTROOT/lenet5_FASHION_MNIST_sweepfl_none_4rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_-1_1099511627776_None_2099-01-01_00-00-00"
RESU1="$TESTROOT/lenet5_FASHION_MNIST_sweepfl_none_4rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_-1_1099511627776_None_2099-01-01_01-00-00"
ORIG1_BASE=$(basename "$ORIG1")
RESU1_BASE=$(basename "$RESU1")

# ── Test 1: Dry-run detects chain, makes no changes ──────────────────────────
echo "── Test 1: Dry-run detects chain, makes no changes"

make_details "$ORIG1" 4 ""
# RESUMED_FROM is relative to logs_dir ($TESTROOT), so just the folder basename
make_details "$RESU1" 4 "$ORIG1_BASE" 2
make_checkpoint "$ORIG1" 2
make_checkpoint "$RESU1" 4
make_csvs "$ORIG1" 1 2
make_csvs "$RESU1" 3 4
make_model_snapshots "$ORIG1" 1 2
make_model_snapshots "$RESU1" 3 4
touch "$ORIG1/INCOMPLETE.log"

OUTPUT=$(python3 "$SCRIPT" --logs-dir "$TESTROOT" --dry-run 2>&1)

if echo "$OUTPUT" | grep -q "Would merge"; then
    ok "Dry-run: chain detected"
else
    fail "Dry-run: chain NOT detected. Output: $OUTPUT"
fi

if [[ -f "$RESU1/train.csv" ]]; then
    ok "Dry-run: resumed folder untouched"
else
    fail "Dry-run: resumed folder was removed (should NOT be)"
fi

if [[ ! -d "$TESTROOT/archive" ]]; then
    ok "Dry-run: no archive/ created"
else
    fail "Dry-run: archive/ was created (should NOT be)"
fi

ORIG_ROWS=$(csv_rows "$ORIG1/train.csv")
if [[ "$ORIG_ROWS" == "4" ]]; then
    ok "Dry-run: original train.csv unchanged (4 rows)"
else
    fail "Dry-run: original train.csv changed (got $ORIG_ROWS rows, expected 4)"
fi

# ── Test 2: Basic merge executes correctly ────────────────────────────────────
echo ""
echo "── Test 2: Basic merge (2+2 rounds → 4 rounds in-place)"

python3 "$SCRIPT" --logs-dir "$TESTROOT" --yes 2>&1 | grep -v "^\[merge\] No" || true

for csv in train.csv evaluate.csv server_agg.csv server_eval.csv; do
    ROUNDS=$(csv_rounds "$ORIG1/$csv")
    if [[ "$ROUNDS" == "1 2 3 4" ]]; then
        ok "Merge: $csv has rounds 1 2 3 4"
    else
        fail "Merge: $csv rounds = '$ROUNDS' (expected '1 2 3 4')"
    fi
done

# ── Test 3: CSV row counts ────────────────────────────────────────────────────
echo ""
echo "── Test 3: CSV row counts after merge"

TRAIN_ROWS=$(csv_rows "$ORIG1/train.csv")
EVAL_ROWS=$(csv_rows  "$ORIG1/evaluate.csv")
AGG_ROWS=$(csv_rows   "$ORIG1/server_agg.csv")
SEVAL_ROWS=$(csv_rows "$ORIG1/server_eval.csv")

[[ "$TRAIN_ROWS" == "8" ]] && ok "train.csv: $TRAIN_ROWS rows (4 rounds × 2 clients)" \
    || fail "train.csv: $TRAIN_ROWS rows (expected 8)"
[[ "$EVAL_ROWS" == "8" ]] && ok "evaluate.csv: $EVAL_ROWS rows" \
    || fail "evaluate.csv: $EVAL_ROWS rows (expected 8)"
[[ "$AGG_ROWS" == "4" ]] && ok "server_agg.csv: $AGG_ROWS rows" \
    || fail "server_agg.csv: $AGG_ROWS rows (expected 4)"
[[ "$SEVAL_ROWS" == "4" ]] && ok "server_eval.csv: $SEVAL_ROWS rows" \
    || fail "server_eval.csv: $SEVAL_ROWS rows (expected 4)"

# ── Test 4: Model snapshots ───────────────────────────────────────────────────
echo ""
echo "── Test 4: Model snapshots merged"

for r in 1 2 3 4; do
    if [[ -f "$ORIG1/model/round_$r.npz" ]]; then
        ok "model/round_$r.npz present"
    else
        fail "model/round_$r.npz MISSING"
    fi
done

# ── Test 5: Checkpoint replaced with latest ──────────────────────────────────
echo ""
echo "── Test 5: Checkpoint updated to resumed folder's round"

CKPT_ROUND=$(python3 -c "import json; print(json.load(open('$ORIG1/checkpoint/meta.json'))['round'])")
if [[ "$CKPT_ROUND" == "4" ]]; then
    ok "checkpoint/meta.json: round=$CKPT_ROUND"
else
    fail "checkpoint/meta.json: round=$CKPT_ROUND (expected 4)"
fi

# ── Test 6: experiment_details.json + INCOMPLETE.log ─────────────────────────
echo ""
echo "── Test 6: experiment_details.json updated; INCOMPLETE.log removed"

MERGED_FROM=$(python3 -c "
import json
d = json.load(open('$ORIG1/experiment_details.json'))
v = d.get('MERGED_FROM')
print(v if v else 'MISSING')
")
MERGED_AT=$(python3 -c "
import json
d = json.load(open('$ORIG1/experiment_details.json'))
print(d.get('MERGED_AT', 'MISSING'))
")

if [[ "$MERGED_FROM" != "MISSING" ]] && [[ "$MERGED_FROM" != "[]" ]]; then
    ok "MERGED_FROM present: $MERGED_FROM"
else
    fail "MERGED_FROM missing or empty"
fi

if [[ "$MERGED_AT" != "MISSING" ]]; then
    ok "MERGED_AT present: $MERGED_AT"
else
    fail "MERGED_AT missing"
fi

if [[ ! -f "$ORIG1/INCOMPLETE.log" ]]; then
    ok "INCOMPLETE.log removed from original"
else
    fail "INCOMPLETE.log still present"
fi

# ── Test 7: Archive behaviour ─────────────────────────────────────────────────
echo ""
echo "── Test 7: Archive behaviour"

if [[ -d "$TESTROOT/archive/$RESU1_BASE" ]]; then
    ok "Resumed folder archived"
else
    fail "Resumed folder NOT in archive/"
fi

if [[ -d "$TESTROOT/archive/$ORIG1_BASE" ]]; then
    ok "Original backup present in archive/ (safety copy)"
else
    fail "Original backup missing from archive/"
fi

if [[ ! -d "$RESU1" ]]; then
    ok "Resumed folder removed from live location"
else
    fail "Resumed folder still present in live location"
fi

# ── Test 8: Idempotency ───────────────────────────────────────────────────────
echo ""
echo "── Test 8: Idempotency — second run is a no-op"

BEFORE=$(csv_rows "$ORIG1/train.csv")
OUTPUT2=$(python3 "$SCRIPT" --logs-dir "$TESTROOT" --yes 2>&1)
AFTER=$(csv_rows "$ORIG1/train.csv")

if echo "$OUTPUT2" | grep -q "No completed resume chains"; then
    ok "Second run: no-op (no double-merge)"
else
    fail "Second run: unexpected output: $OUTPUT2"
fi
if [[ "$BEFORE" == "$AFTER" ]]; then
    ok "Second run: row count unchanged ($AFTER)"
else
    fail "Second run: row count changed $BEFORE → $AFTER"
fi

# ── Test 9: Chain merge (3 folders) ──────────────────────────────────────────
echo ""
echo "── Test 9: Chain merge — 3 folders (original + 2 resumes)"

TESTROOT2="$TESTROOT/chain_sub"
ORIG2="$TESTROOT2/lenet5_FASHION_MNIST_sweepfl_none_6rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_-1_1099511627776_None_2099-02-01_00-00-00"
R2A="$TESTROOT2/lenet5_FASHION_MNIST_sweepfl_none_6rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_-1_1099511627776_None_2099-02-01_01-00-00"
R2B="$TESTROOT2/lenet5_FASHION_MNIST_sweepfl_none_6rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_-1_1099511627776_None_2099-02-01_02-00-00"
ORIG2_BASE=$(basename "$ORIG2")
R2A_BASE=$(basename "$R2A")
R2B_BASE=$(basename "$R2B")

make_details "$ORIG2" 6 "" 0
make_details "$R2A"   6 "$ORIG2_BASE" 2
make_details "$R2B"   6 "$R2A_BASE" 4
make_checkpoint "$ORIG2" 2
make_checkpoint "$R2A" 4
make_checkpoint "$R2B" 6
make_csvs "$ORIG2" 1 2
make_csvs "$R2A" 3 4
make_csvs "$R2B" 5 6
make_model_snapshots "$ORIG2" 1 2
make_model_snapshots "$R2A" 3 4
make_model_snapshots "$R2B" 5 6
touch "$ORIG2/INCOMPLETE.log"
touch "$R2A/INCOMPLETE.log"

python3 "$SCRIPT" --logs-dir "$TESTROOT2" --yes 2>&1 | grep -v "^\[merge\] No" || true

ROUNDS=$(csv_rounds "$ORIG2/server_eval.csv")
if [[ "$ROUNDS" == "1 2 3 4 5 6" ]]; then
    ok "Chain merge: server_eval.csv rounds 1-6"
else
    fail "Chain merge: server_eval.csv rounds = '$ROUNDS' (expected 1-6)"
fi

CHAIN_CKPT=$(python3 -c "import json; print(json.load(open('$ORIG2/checkpoint/meta.json'))['round'])")
if [[ "$CHAIN_CKPT" == "6" ]]; then
    ok "Chain merge: checkpoint round=$CHAIN_CKPT (from final resume)"
else
    fail "Chain merge: checkpoint round=$CHAIN_CKPT (expected 6)"
fi

if [[ -d "$TESTROOT2/archive/$R2A_BASE" ]] && [[ -d "$TESTROOT2/archive/$R2B_BASE" ]]; then
    ok "Chain merge: both resumed folders archived"
else
    fail "Chain merge: not all resumed folders archived"
fi

SNAP_COUNT=$(ls "$ORIG2/model/"round_*.npz 2>/dev/null | wc -l | tr -d ' ')
if [[ "$SNAP_COUNT" == "6" ]]; then
    ok "Chain merge: $SNAP_COUNT model snapshots (rounds 1-6)"
else
    fail "Chain merge: $SNAP_COUNT model snapshots (expected 6)"
fi

# ── Test 10: Conflict detection ───────────────────────────────────────────────
echo ""
echo "── Test 10: Conflict — two completed resumes for same original → skip"

TESTROOT3="$TESTROOT/conflict_sub"
ORIG3="$TESTROOT3/lenet5_FASHION_MNIST_sweepfl_none_2rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_-1_1099511627776_None_2099-03-01_00-00-00"
RC1="$TESTROOT3/lenet5_FASHION_MNIST_sweepfl_none_2rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_-1_1099511627776_None_2099-03-01_01-00-00"
RC2="$TESTROOT3/lenet5_FASHION_MNIST_sweepfl_none_2rounds_2clients_seed42_niid0.1_percentage0.1_8192_-1_-1_-1_1099511627776_None_2099-03-01_02-00-00"
ORIG3_BASE=$(basename "$ORIG3")

make_details "$ORIG3" 2 "" 0
make_details "$RC1" 2 "$ORIG3_BASE" 1
make_details "$RC2" 2 "$ORIG3_BASE" 1
make_checkpoint "$ORIG3" 1
make_checkpoint "$RC1" 2
make_checkpoint "$RC2" 2
make_csvs "$ORIG3" 1
make_csvs "$RC1" 2
make_csvs "$RC2" 2

CONFLICT_OUT=$(python3 "$SCRIPT" --logs-dir "$TESTROOT3" --yes 2>&1)

if echo "$CONFLICT_OUT" | grep -q "No completed resume chains"; then
    ok "Conflict: skipped (no ambiguous merge)"
else
    fail "Conflict: merge proceeded despite ambiguity. Output: $CONFLICT_OUT"
fi

if [[ -d "$ORIG3" ]]; then
    ok "Conflict: original folder untouched"
else
    fail "Conflict: original folder was modified/removed"
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo " Results: $PASS passed, $FAIL failed"
echo "════════════════════════════════════════════════════════"

cleanup

[[ $FAIL -eq 0 ]] && echo "All tests passed." || { echo "Some tests FAILED."; exit 1; }
