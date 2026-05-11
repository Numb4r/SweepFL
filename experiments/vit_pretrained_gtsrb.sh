#!/bin/bash
# Theme: ViT-Tiny pre-trained on CIFAR-100, fine-tuned via FL on GTSRB.
# Compares pre-trained vs. random-init baselines across sparsification techniques.
#
# Step 1 (run once):   make pretrain MODEL=vit_tiny DATASET=cifar100 EPOCHS=100 GPU=5
# Step 2 (this script): bash experiments/vit_pretrained_gtsrb.sh

GPU_IDS=(5 6 7)
CLIENTS_PER_GPU=3

FOLDER="vit_pretrained_gtsrb"
EXPERIMENT_NAME=""
NUM_ROUNDS=50
NIIDS=(0.1)
SEEDS=(42)
TECHNIQUE="none"

SOLUTIONS=('nvnl' 'sw' 'rw' 'fedphe' 'plaintext')
PERCENTAGES=(0.1 0.3 0.5)

# ViT-Tiny runs GTSRB (43-class traffic signs).
# Pre-trained weights are auto-detected from logs/models/pretrained/vit_tiny/.
MODEL_DATASET_PAIRS=(
    "vit_tiny:tanganke/gtsrb"
)

CKKS_CONFIGS=("ckks 8192 60,40,40,60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
CKKS_BACKEND="tenseal"

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run_theme
