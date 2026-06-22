#!/bin/bash
# experiments/tests/test_client_frac.sh — Smoke-test for the CLIENT_FRACTION mechanic.
#
# Verifies that only round(NCLIENTS × CLIENT_FRACTION) clients are sampled per round.
#
# Config: 4 clients · 2 rounds · FashionMNIST · LeNet-5 · CLIENT_FRACTION=0.5
#         Expected: exactly 2 distinct CIDs per round (from pool of 0–3).
#
# Usage (from repo root):
#   bash experiments/tests/test_client_frac.sh

# ── GPU layout (4 clients, single slot) ──────────────────────────────────────
GPU_IDS=(0)
CLIENTS_PER_GPU=4           # NCLIENTES = 1 × 4 = 4
# CPU fallback when nvidia-persistenced socket is absent.
if [[ ! -S /run/nvidia-persistenced/socket ]]; then
    USE_GPU=false
    echo "[WARN] nvidia-persistenced socket not found — running clients in CPU mode"
fi

# ── HE parameters (not used by plaintext; required by key-gen call in run_theme) ──
CKKS_CONFIGS=("ckks 8192 60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
CKKS_BACKEND="ckks_python_simplify"
SIMPLIFY_N=8192
SIMPLIFY_BITS=62
SIMPLIFY_SCALE_BITS=40

# ── Experiment parameters ─────────────────────────────────────────────────────
FOLDER="test_client_frac"
EXPERIMENT_NAME=""
RUNTIME="docker"
SIF_DIR="."
NUM_ROUNDS=2
NIIDS=(0.1)
SEEDS=(42)
TECHNIQUE="none"
SOLUTIONS=('plaintext')
PERCENTAGES=(0.1)           # ignored for plaintext (run_theme forces -1 for non-packing)
MODEL_DATASET_PAIRS=("lenet5:FASHION_MNIST")
CLIENT_FRACTIONS=(0.5)      # → server gets CLIENT_FRACTION=0.5 → n_part = round(4×0.5) = 2

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  CLIENT_FRACTION Smoke Test                               ║"
echo "║  4 clients · CF=0.5 · 2 rounds · FASHION_MNIST · LeNet-5 ║"
echo "║  Expected: exactly 2 clients sampled per round            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "▶ Running plaintext with CLIENT_FRACTION=0.5 (Docker, CPU)"

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
run_theme

# ── Verification ──────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════ VERIFICATION ══════════════════════════"

LOGDIR="logs/test_client_frac"
CF_FOLDER=$(ls -1d "$LOGDIR"/lenet5_FASHION_MNIST_plaintext_*_cf0.5_* 2>/dev/null | sort | tail -1)
PASS=true

if [[ -z "$CF_FOLDER" ]]; then
    echo "  [FAIL] No experiment folder with cf0.5_ found under $LOGDIR"
    PASS=false
else
    echo "  Folder: $(basename "$CF_FOLDER")"
    echo ""

    # 1. Token cf0.5_ in folder name
    if [[ "$(basename "$CF_FOLDER")" == *"_cf0.5_"* ]]; then
        echo "  [OK]  Token _cf0.5_ present in folder name"
    else
        echo "  [FAIL] Token _cf0.5_ NOT found in folder name"
        PASS=false
    fi

    # 2. server_agg.csv: n_clients == 2 in all rounds
    if [[ -f "$CF_FOLDER/server_agg.csv" ]]; then
        bad_rows=$(tail -n +2 "$CF_FOLDER/server_agg.csv" | awk -F, '$3 != 2 {print NR": "$0}')
        if [[ -z "$bad_rows" ]]; then
            echo "  [OK]  server_agg.csv: n_clients = 2 in all rounds"
        else
            echo "  [FAIL] server_agg.csv: unexpected n_clients values:"
            echo "$bad_rows"
            PASS=false
        fi
    else
        echo "  [FAIL] server_agg.csv not found"
        PASS=false
    fi

    # 3. server_eval.csv: n_clients == 2 in all rounds
    if [[ -f "$CF_FOLDER/server_eval.csv" ]]; then
        bad_rows=$(tail -n +2 "$CF_FOLDER/server_eval.csv" | awk -F, '$3 != 2 {print NR": "$0}')
        if [[ -z "$bad_rows" ]]; then
            echo "  [OK]  server_eval.csv: n_clients = 2 in all rounds"
        else
            echo "  [FAIL] server_eval.csv: unexpected n_clients values:"
            echo "$bad_rows"
            PASS=false
        fi
    else
        echo "  [FAIL] server_eval.csv not found"
        PASS=false
    fi

    # 4. train.csv: exactly 2 distinct CIDs per round
    if [[ -f "$CF_FOLDER/train.csv" ]]; then
        bad_rounds=$(tail -n +2 "$CF_FOLDER/train.csv" | \
            awk -F, '{cids[$2][$3]=1} END{for(r in cids){c=0; for(x in cids[r])c++; if(c!=2)print "round "r": "c" CIDs"}}' | \
            sort -t' ' -k2 -n)
        if [[ -z "$bad_rounds" ]]; then
            echo "  [OK]  train.csv: exactly 2 distinct CIDs per round (both rounds)"
        else
            echo "  [FAIL] train.csv: wrong CID count:"
            echo "$bad_rounds"
            PASS=false
        fi
        echo "  --- CIDs round 1: $(tail -n +2 "$CF_FOLDER/train.csv" | awk -F, '$2==1{print $3}' | sort -n | tr '\n' ' ')"
        echo "  --- CIDs round 2: $(tail -n +2 "$CF_FOLDER/train.csv" | awk -F, '$2==2{print $3}' | sort -n | tr '\n' ' ')"
    else
        echo "  [FAIL] train.csv not found"
        PASS=false
    fi

    # 5. evaluate.csv: exactly 2 distinct CIDs per round
    if [[ -f "$CF_FOLDER/evaluate.csv" ]]; then
        bad_rounds=$(tail -n +2 "$CF_FOLDER/evaluate.csv" | \
            awk -F, '{cids[$2][$3]=1} END{for(r in cids){c=0; for(x in cids[r])c++; if(c!=2)print "round "r": "c" CIDs"}}' | \
            sort -t' ' -k2 -n)
        if [[ -z "$bad_rounds" ]]; then
            echo "  [OK]  evaluate.csv: exactly 2 distinct CIDs per round"
        else
            echo "  [FAIL] evaluate.csv: wrong CID count:"
            echo "$bad_rounds"
            PASS=false
        fi
    else
        echo "  [FAIL] evaluate.csv not found"
        PASS=false
    fi

    # 6. Flower log: "sampled" line
    SERVER_LOG="$CF_FOLDER/tmux_SERVER.log"
    if [[ -f "$SERVER_LOG" ]]; then
        sampled_lines=$(grep -iE "sampled|configure_fit" "$SERVER_LOG" || true)
        if [[ -n "$sampled_lines" ]]; then
            echo "  [OK]  Flower log has sampling lines:"
            echo "$sampled_lines" | sed 's/^/         /'
        else
            echo "  [INFO] No 'sampled'/'configure_fit' lines in server log (Flower may not emit them at this log level)"
        fi
    fi
fi

echo "═══════════════════════════════════════════════════════════════════"
if $PASS; then
    echo "✅ All checks PASSED — CLIENT_FRACTION=0.5 with 4 clients → 2 clients/round confirmed."
else
    echo "❌ Some checks FAILED — see details above."
    exit 1
fi
