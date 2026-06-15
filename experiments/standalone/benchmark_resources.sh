#!/bin/bash
# experiments/benchmark_resources.sh — Quick resource benchmark in plaintext.
#
# 3 model/dataset pairs × 2 distributions (IID + non-IID) = 6 experiments.
# 2 clients, 5 rounds, Docker.
#
# Usage (from repo root):
#   bash experiments/benchmark_resources.sh

# ── Container runtime ─────────────────────────────────────────────
RUNTIME="docker"
SIF_DIR="."
SERVER_HOST=""
USE_GPU=false   # local machine: no nvidia-persistenced socket

# ── GPU layout — 2 clients on a single GPU ────────────────────────
GPU_IDS=(0)
CLIENTS_PER_GPU=2

# ── Experiment metadata ───────────────────────────────────────────
FOLDER="benchmark_resources"
EXPERIMENT_NAME=""
NUM_ROUNDS=5
NIIDS=(0.1 1.0)   # 0.1 = non-IID, 1.0 ≈ IID
SEEDS=(42)
TECHNIQUE="none"

# ── Solutions ─────────────────────────────────────────────────────
SOLUTIONS=('plaintext')
PERCENTAGES=(-1)   # plaintext ignores percentages

# ── Model ↔ Dataset pairs ─────────────────────────────────────────
MODEL_DATASET_PAIRS=(
    "dnn_tiny:FASHION_MNIST"
    "lenet5:FASHION_MNIST"
    "mobilenetv2:CIFAR10"
)

# ── Per-model batch size ──────────────────────────────────────────
model_config() {
    local model="$1"
    case "$model" in
        mobilenetv2) BATCH_SIZE=16 ;;
        *)           BATCH_SIZE=4  ;;
    esac
}

# ── HE parameters (unused by plaintext, required by generate_keys) ─
CKKS_CONFIGS=(
    "ckks 8192 60,40,40,60"
)
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
CKKS_BACKEND="tenseal"

# ── Run ───────────────────────────────────────────────────────────
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
run_theme
