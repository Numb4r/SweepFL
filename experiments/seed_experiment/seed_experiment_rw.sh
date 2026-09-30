#!/bin/bash
# Theme: seed_experiment — repeat the sparsification techniques across multiple
# seeds so accuracy differences can be separated from run-to-run variance.
# All existing logs use seed 42 only; this theme adds seeds 43 and 44.
#
# Technique: rw — pondering-random, weighted random draw with penalty/reward
# Percentage sweep: 10 % and 70 %
# Seed sweep:       43 → 44
#
# Config is identical to sparsification_percentage except FOLDER, SEEDS and the
# pair list, so results stay directly comparable to the existing seed-42 runs.
#
# v2 (2026-09-23): narrowed to α=1 (NIIDS=(1)), p ∈ {0.1, 0.7}, FOLDER=seed_experiment_v2.
# The Aug sweep (logs/seed_experiment/, NIIDS=(0.1 1), p 0.1–0.9, 160 runs) ran before the
# MAGG_D_REF fix and 23/160 runs collapsed to NaN. Only α=1 is comparable to seed 42
# (its niid0.1 folder is IID). See docs/seed_experiment.md.

GPU_IDS=(0 1 2)          # overridden from SLURM_GPUS_ON_NODE on the cluster
CLIENTS_PER_GPU=3        # → 9 clients

FOLDER="seed_experiment_v2"
EXPERIMENT_NAME=""
NUM_ROUNDS=200
NIIDS=(1)
SEEDS=(43 44)
TECHNIQUE="none"

SOLUTIONS=('rw')
PERCENTAGES=(0.1 0.7)

MODEL_DATASET_PAIRS=(
    "resnet20:CIFAR10"
    "mobilenetv2:CIFAR10"
)

model_config() {
    local model="$1" dataset="$2"
    case "$model" in
        mobilenetv2) BATCH_SIZE=16 ;;   # BatchNorm unstable at batch 4
        *)           BATCH_SIZE=4  ;;
    esac
}

CKKS_CONFIGS=("ckks 8192 60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
SIMPLIFY_N=8192; SIMPLIFY_BITS=62; SIMPLIFY_SCALE_BITS=40
CKKS_BACKEND="ckks_python_simplify"

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
run_theme
