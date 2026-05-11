#!/bin/bash
# experiments/template.sh — Copy this file to create a new themed experiment batch.
#
# Usage (from repo root):
#   bash experiments/my_theme.sh
#
# The only required change is MODEL_DATASET_PAIRS — everything else has sane defaults.

# ── Container runtime ─────────────────────────────────────────────
RUNTIME="docker"       # docker | singularity
SIF_DIR="."            # path to .sif files (singularity only)
SERVER_HOST=""         # override server hostname (empty = auto-detect per runtime)

# ── GPU layout ────────────────────────────────────────────────────
GPU_IDS=(5 6 7)        # Physical GPU indices to use
CLIENTS_PER_GPU=3      # Clients per GPU

# ── Experiment metadata ───────────────────────────────────────────
FOLDER="my_theme"      # Subfolder inside logs/ — keep it short and descriptive
EXPERIMENT_NAME=""     # Leave empty to auto-generate as {model}_{dataset}_{solution}
NUM_ROUNDS=50
NIIDS=(0.1)            # Dirichlet α: 0.1 = non-IID, 1.0 ≈ IID
SEEDS=(42)
TECHNIQUE="none"       # topk | robin_round | slided_window | none

# ── Solutions ─────────────────────────────────────────────────────
# Packing solutions (support PERCENTAGES sweep): nvnl sw rw fedphe nvnl-scout fedphe-scout
# Non-packing solutions (PERCENTAGES ignored):   ckks bfv batchcrypt plaintext
SOLUTIONS=('nvnl' 'sw' 'rw' 'fedphe')

# Fraction of packs encrypted and transmitted (only for packing solutions).
PERCENTAGES=(0.1 0.3 0.5)

# ── Model ↔ Dataset pairs ─────────────────────────────────────────
# Format: "model_name:DATASET_NAME"
# Each pair runs independently — no cross-product between models and datasets.
#
# Available models : dnn | dnn_tiny | lenet5 | resnet20 | mobilenetv2 | vit_tiny | simple_cnn
# Available datasets: CIFAR10 | FASHION_MNIST | MNIST | AISNP/PathMNIST | tanganke/gtsrb
MODEL_DATASET_PAIRS=(
    "dnn_tiny:FASHION_MNIST"
    "lenet5:FASHION_MNIST"
    "mobilenetv2:CIFAR10"
    "vit_tiny:tanganke/gtsrb"
)

# ── HE parameters ─────────────────────────────────────────────────
CKKS_CONFIGS=(
    "ckks 8192 60,40,40,60"
    # "ckks 8192 30,60,30"     # incompatible with scalar mul — avoid
)
KEY_TYPES=('None')
SCALE_POWER=40         # SCALE = 2^SCALE_POWER
PLAIN_MODULUS=-1       # BFV plain modulus   (-1 if unused)
N_LENGTH=-1            # Paillier key length (-1 if unused)
CKKS_BACKEND="tenseal" # tenseal | ckks_python_simplify | ckks_python | openfhe

# Optional backend-specific overrides (uncomment only what you need):
# SIMPLIFY_N=8192; SIMPLIFY_BITS=62; SIMPLIFY_SCALE_BITS=60
# PYCRYPT_N=8192; PYCRYPT_BITS=60
# OPENFHE_RING_DIM=8192; OPENFHE_MULT_DEPTH=1; OPENFHE_SCALE_BITS=50

# Pre-trained weights are detected automatically from
# logs/models/pretrained/<model>/*/pretrained.weights.h5
# No configuration needed here.

# ── Run ───────────────────────────────────────────────────────────
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run_theme
