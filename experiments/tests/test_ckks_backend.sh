#!/bin/bash
# experiments/test_ckks_backend.sh — Smoke-test for the ckks_python_simplify backend.
#
# Runs 5 experiments across 3 phases:
#   Phase 0 — plaintext   (Docker)        → seeds init_weights (required by HE solutions)
#   Phase 1 — sweepfl+fedphe (Docker)        → ckks_python_simplify backend
#   Phase 2 — sweepfl+fedphe (Singularity)   → ckks_python_simplify backend
#
# Config: 2 clients · 3 rounds · FashionMNIST · LeNet-5
#         ring_dim=8192, bits=60, scale_bits=40, percentage=0.1
#
# Usage (from repo root):
#   bash experiments/test_ckks_backend.sh

# ── GPU layout (2 clients on one GPU) ────────────────────────────────────────
GPU_IDS=(0)
CLIENTS_PER_GPU=2
# Set USE_GPU=false if nvidia-persistenced socket is unavailable (CPU fallback).
if [[ ! -S /run/nvidia-persistenced/socket ]]; then
    USE_GPU=false
    echo "[WARN] nvidia-persistenced socket not found — running clients in CPU mode"
fi

# ── HE parameters ────────────────────────────────────────────────────────────
CKKS_CONFIGS=("ckks 8192 60,40,40,60")
KEY_TYPES=('None')
SCALE_POWER=40          # SCALE = 2^40 (matches ckks_python_simplify scale_bits)
PLAIN_MODULUS=-1
N_LENGTH=-1
CKKS_BACKEND="ckks_python_simplify"
SIMPLIFY_N=8192
SIMPLIFY_BITS=62
SIMPLIFY_SCALE_BITS=40

# ── Experiment parameters ─────────────────────────────────────────────────────
NIIDS=(0.1)
SEEDS=(42)
TECHNIQUE="none"
NUM_ROUNDS=3
PERCENTAGES=(0.1)
# MODEL_DATASET_PAIRS=("lenet5:FASHION_MNIST")
MODEL_DATASET_PAIRS=("resnet20:CIFAR10")

EXPERIMENT_NAME=""      # auto-generates as {model}_{dataset}_{solution}

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

# ── Inter-phase cleanup ───────────────────────────────────────────────────────
# Stops any lingering FL containers and frees port 9999.
# Must be called between phases that use different runtimes (Docker → Singularity)
# to prevent "Address already in use" errors.
_cleanup_between_phases() {
    echo "── Cleanup: stopping FL containers, releasing port 9999..."
    docker rm -f fl_server-gpu 2>/dev/null || true
    for _c in $(seq 0 $(( ${NCLIENTES:-2} - 1 ))); do
        docker rm -f "client-$_c" 2>/dev/null || true
    done
    local _t=0
    while ss -tlnp 2>/dev/null | grep -q ':9999[^0-9]' && (( _t < 15 )); do
        echo "  waiting for port 9999 to be free... ($((++_t))/15)"
        sleep 1
    done
    # client/data/ is owned by root (Docker runs as root).  Singularity runs as
    # $USER and cannot create lock files in a root-owned directory.
    # Fix: use the Docker image itself (root inside container) to chown the whole
    # tree to the calling user, then Singularity can read/write freely.
    if [[ -d client/data ]]; then
        echo "── Fixing client/data ownership for Singularity..."
        docker run --rm \
            -v "$PWD/client/data:/data" \
            flhe:client \
            chown -R "$(id -u):$(id -g)" /data 2>/dev/null || true
    fi
    echo "── Cleanup done."
}

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  CKKS Backend Federation Test — ckks_python_simplify     ║"
echo "║  2 clients · 3 rounds · FashionMNIST · LeNet-5           ║"
echo "║  ring_dim=8192  bits=60  scale_bits=40  pct=0.1          ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── Phase 0: plaintext (Docker) — seed init_weights ──────────────────────────
echo ""
echo "▶ Phase 0: plaintext (Docker) — generating initial model weights"
FOLDER="test_ckks_backend/phase0_plaintext"
SOLUTIONS=("plaintext")
RUNTIME="docker"
SIF_DIR="."
run_theme
_cleanup_between_phases

# ── Phase 1: sweepfl + fedphe (Docker) ──────────────────────────────────────────
echo ""
echo "▶ Phase 1: sweepfl + fedphe (Docker / ckks_python_simplify)"
FOLDER="test_ckks_backend/phase1_docker"
SOLUTIONS=("sweepfl" "fedphe")
RUNTIME="docker"
SIF_DIR="."
run_theme
_cleanup_between_phases

# ── Phase 2: sweepfl + fedphe (Singularity) ─────────────────────────────────────
echo ""
echo "▶ Phase 2: sweepfl + fedphe (Singularity / ckks_python_simplify)"
FOLDER="test_ckks_backend/phase2_singularity"
SOLUTIONS=("sweepfl" "fedphe")
RUNTIME="singularity"
SIF_DIR="."
# run_theme

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════ SUMMARY ═══════════════════════════"
all_ok=true
for phase in phase0_plaintext phase1_docker phase2_singularity; do
    for dir in logs/test_ckks_backend/${phase}/*/; do
        [[ -d "$dir" ]] || continue
        exp=$(basename "$dir")
        train_csv="${dir}train.csv"
        # plaintext doesn't write train.csv, check server_eval.csv instead
        eval_csv="${dir}server_eval.csv"
        if [[ -f "$train_csv" ]] || [[ -f "$eval_csv" ]]; then
            rows=$(wc -l < "${train_csv:-$eval_csv}" 2>/dev/null || echo 0)
            echo "  [OK]  $phase / $exp  (${rows} rows)"
        else
            echo "  [FAIL] $phase / $exp  — no CSV found"
            all_ok=false
        fi
    done
done
echo "════════════════════════════════════════════════════════════════"
if $all_ok; then
    echo "All phases completed successfully."
else
    echo "Some experiments may have failed — check logs above."
    exit 1
fi
