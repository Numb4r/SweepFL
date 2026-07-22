#!/bin/bash
# experiments/stress_test_client_fedphe.sh — client-scale stress test for fedphe (top-k).
# Sweeps PERCENTAGES × NIIDS × MODEL_DATASET_PAIRS (4 experiments total) with 400 clients.

# ── Container runtime ─────────────────────────────────────────────
RUNTIME="${RUNTIME:-docker}"
SIF_DIR="${SIF_DIR:-.}"
SERVER_HOST="${SERVER_HOST:-}"

# ── GPU layout ────────────────────────────────────────────────────
GPU_IDS=(0 1 2 3)      # overridden on SLURM by SLURM_GPUS_ON_NODE
CLIENTS_PER_GPU=100    # 4 GPUs × 100 = 400 clients

# ── Experiment metadata ───────────────────────────────────────────
FOLDER="stress_test_client"
EXPERIMENT_NAME=""
NUM_ROUNDS=200
NIIDS=(0.1 1)
SEEDS=(42)
TECHNIQUE="none"

# ── Solutions ─────────────────────────────────────────────────────
SOLUTIONS=('fedphe')

# ── Sparsification percentages ────────────────────────────────────
PERCENTAGES=(0.1)

# ── Client participation fractions ───────────────────────────────
CLIENT_FRACTIONS=(1.0)

# ── Model ↔ Dataset pairs ─────────────────────────────────────────
MODEL_DATASET_PAIRS=(
    "resnet20:CIFAR10"
    "mobilenetv2:CIFAR10"
)

# ── HE parameters ─────────────────────────────────────────────────
CKKS_CONFIGS=("ckks 8192 60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
CKKS_BACKEND="ckks_python_simplify"
SIMPLIFY_N=8192
SIMPLIFY_BITS=62
SIMPLIFY_SCALE_BITS=40

# ── Per-model configuration ───────────────────────────────────────
model_config() {
    local model="$1"
    case "$model" in
        mobilenetv2) BATCH_SIZE=16 ;;
        *)           BATCH_SIZE=4  ;;
    esac
}

# ── Run ───────────────────────────────────────────────────────────
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
run_theme
