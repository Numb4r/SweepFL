#!/bin/bash
# Theme: Vary encrypted-packet percentage across sparsification techniques.
# Each model runs on its designated dataset (no cross-product).
#
# Techniques compared: nvnl, sw, rw, fedphe
# Percentage sweep:    10 % → 30 % → 50 % → 70 %

GPU_IDS=(7)
CLIENTS_PER_GPU=3

FOLDER="sparsification_percentage"
EXPERIMENT_NAME=""
NUM_ROUNDS=50
NIIDS=(0.1)
SEEDS=(42)
TECHNIQUE="none"

SOLUTIONS=('nvnl' 'sw' 'rw' 'fedphe')
PERCENTAGES=(0.1 0.3 0.5 0.7)

# dnn_tiny and lenet5 share FashionMNIST; mobilenetv2 → CIFAR-10; vit_tiny → GTSRB
MODEL_DATASET_PAIRS=(
    "dnn_tiny:FASHION_MNIST"
    "lenet5:FASHION_MNIST"
    "mobilenetv2:CIFAR10"
    # "vit_tiny:tanganke/gtsrb"
)

CKKS_CONFIGS=("ckks 8192 60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
# CKKS_BACKEND="tenseal"
CKKS_BACKEND="ckks_python_simplify"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run_theme
