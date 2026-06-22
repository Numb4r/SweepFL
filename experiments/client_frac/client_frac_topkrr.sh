#!/bin/bash
# experiments/client_frac_topkrr.sh — client_fraction sweep for topkrr (top-k then round-robin).
# Sweeps CLIENT_FRACTIONS × PERCENTAGES × NIIDS × MODEL_DATASET_PAIRS.
# RR_START_ROUND=100: switches from top-k to round-robin at round 100 (of 200).

# ── Container runtime ─────────────────────────────────────────────
RUNTIME="${RUNTIME:-docker}"
SIF_DIR="${SIF_DIR:-.}"
SERVER_HOST="${SERVER_HOST:-}"

# ── GPU layout ────────────────────────────────────────────────────
GPU_IDS=(0 1 2)        # overridden on SLURM by SLURM_GPUS_ON_NODE
CLIENTS_PER_GPU=20     # 3 GPUs × 20 = 60 clients

# ── Experiment metadata ───────────────────────────────────────────
FOLDER="client_frac"
EXPERIMENT_NAME=""
NUM_ROUNDS=200
NIIDS=(0.1 1)
SEEDS=(42)
TECHNIQUE="none"
RR_START_ROUND=100     # switch top-k → round-robin at this round

# ── Solutions ─────────────────────────────────────────────────────
SOLUTIONS=('topkrr')

# ── Sparsification percentages ────────────────────────────────────
PERCENTAGES=(0.1 0.5)

# ── Client participation fractions ───────────────────────────────
CLIENT_FRACTIONS=(0.25 0.5 0.75)

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
