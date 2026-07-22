#!/bin/bash
# Theme: Full-model encrypted baseline (CKKS, no sparsification) for the
# sparsification_percentage comparison figures.
#
# ckks is a non-packing solution: run_theme runs it once per model/niid/seed
# with percentage=-1 (PERCENTAGES is ignored).
#
# Baseline note: the full-model CKKS baseline uses the TenSEAL backend with the
# literature parameters [60,40,40,60], scale 2^40 (the chapter-4 lib is additive-
# only with 4096-slot ciphertexts and has no full-model single-vector path).

GPU_IDS=(7 8 9)
CLIENTS_PER_GPU=3

FOLDER="sparsification_percentage"
EXPERIMENT_NAME=""
NUM_ROUNDS=200
NIIDS=(0.1 1)
SEEDS=(42)
TECHNIQUE="none"

SOLUTIONS=('ckks')
PERCENTAGES=(0.1)   # ignored for non-packing solutions

# dnn_tiny and lenet5 share FashionMNIST; mobilenetv2/resnet20 → CIFAR-10; vit_tiny → GTSRB
MODEL_DATASET_PAIRS=(
    "dnn_tiny:FASHION_MNIST"
    "lenet5:FASHION_MNIST"
    "resnet20:CIFAR10"
    "mobilenetv2:CIFAR10"
    "vit_tiny:tanganke/gtsrb"
)

# ── Per-model configuration (optional) ───────────────────────────
model_config() {
    local model="$1" dataset="$2"
    case "$model" in
        mobilenetv2) BATCH_SIZE=16 ;;
        vit_tiny)    BATCH_SIZE=16 ;;
        *)           BATCH_SIZE=4  ;;
    esac
}

CKKS_CONFIGS=("ckks 8192 60,40,40,60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
SIMPLIFY_N=8192; SIMPLIFY_BITS=62; SIMPLIFY_SCALE_BITS=40
PYCRYPT_N=8192; PYCRYPT_BITS=62
OPENFHE_RING_DIM=8192; OPENFHE_MULT_DEPTH=1; OPENFHE_SCALE_BITS=50
CKKS_BACKEND="tenseal"

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
run_theme
