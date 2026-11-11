#!/bin/bash
# Theme: rerun of the MobileNetV2/CIFAR-10 niid1 p∈{0.7,0.9} cells of sparsification_percentage
# that collapsed (NaN eval, acc 0.100) from single-prime CKKS range overflow on the D_i-weighted
# server sum. Same config as the family; the only code difference is MAGG_D_REF pre-scale
# (client/client_utils.py) plus range.csv instrumentation. See docs/ckks_range_overflow.md.

GPU_IDS=(0 1 2)
CLIENTS_PER_GPU=3

FOLDER="range_fix"
EXPERIMENT_NAME=""
NUM_ROUNDS=200
NIIDS=(1)
SEEDS=(42)
TECHNIQUE="none"
CLIENT_FRACTION=1.0

SOLUTIONS=('sweepfl')
PERCENTAGES=(0.7 0.9)

MODEL_DATASET_PAIRS=("mobilenetv2:CIFAR10")

model_config() {
    local model="$1" dataset="$2"
    case "$model" in
        mobilenetv2) BATCH_SIZE=16 ;;
        *)           BATCH_SIZE=4  ;;
    esac
}

CKKS_CONFIGS=("ckks 8192 60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
SIMPLIFY_N=8192; SIMPLIFY_BITS=62; SIMPLIFY_SCALE_BITS=40
PYCRYPT_N=8192; PYCRYPT_BITS=62
OPENFHE_RING_DIM=8192; OPENFHE_MULT_DEPTH=1; OPENFHE_SCALE_BITS=50
CKKS_BACKEND="ckks_python_simplify"

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
run_theme
