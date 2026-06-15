#!/bin/bash
# Theme: Vary encrypted-packet percentage across sparsification techniques.
# Each model runs on its designated dataset (no cross-product).
#
# Techniques compared: nvnl, sw, rw, fedphe
# Percentage sweep:    10 % → 30 % → 50 % → 70 %

GPU_IDS=(7 8 9)
CLIENTS_PER_GPU=3

FOLDER="sparsification_percentage"
EXPERIMENT_NAME=""
NUM_ROUNDS=200
NIIDS=(0.1 1)
SEEDS=(42)
TECHNIQUE="none"

SOLUTIONS=('fedphe')
PERCENTAGES=(0.1 0.3 0.5 0.7 0.9)

# dnn_tiny and lenet5 share FashionMNIST; mobilenetv2 → CIFAR-10; vit_tiny → GTSRB
MODEL_DATASET_PAIRS=(
    "dnn_tiny:FASHION_MNIST"
    "lenet5:FASHION_MNIST"
    "resnet20:CIFAR10"
    "mobilenetv2:CIFAR10"
    "vit_tiny:tanganke/gtsrb"
)
# ── Per-model configuration (optional) ───────────────────────────
# Define model_config() to set per-model env vars before each experiment.
# Called automatically by run_theme with (model, dataset) arguments.
# Variables set here are reset to their defaults before each model.
# Currently supported: BATCH_SIZE (default 4)
#
model_config() {
    local model="$1" dataset="$2"
    case "$model" in
        mobilenetv2) BATCH_SIZE=16 ;;
        vit_tiny)    BATCH_SIZE=16 ;;
        *)           BATCH_SIZE=4  ;;
    esac
}

CKKS_CONFIGS=("ckks 8192 60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
# CKKS_BACKEND="tenseal"
# Optional backend-specific overrides (uncomment only what you need):
SIMPLIFY_N=8192; SIMPLIFY_BITS=62; SIMPLIFY_SCALE_BITS=40
PYCRYPT_N=8192; PYCRYPT_BITS=62
OPENFHE_RING_DIM=8192; OPENFHE_MULT_DEPTH=1; OPENFHE_SCALE_BITS=50
CKKS_BACKEND="ckks_python_simplify"









# ── Per-model configuration (optional) ───────────────────────────
# Define model_config() to set per-model env vars before each experiment.
# Called automatically by run_theme with (model, dataset) arguments.
# Variables set here are reset to their defaults before each model.
# Currently supported: BATCH_SIZE (default 4)
#
model_config() {
    local model="$1" dataset="$2"
    case "$model" in
        mobilenetv2) BATCH_SIZE=16 ;;
        vit_tiny)    BATCH_SIZE=16 ;;
        *)           BATCH_SIZE=4  ;;
    esac
}
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
run_theme
