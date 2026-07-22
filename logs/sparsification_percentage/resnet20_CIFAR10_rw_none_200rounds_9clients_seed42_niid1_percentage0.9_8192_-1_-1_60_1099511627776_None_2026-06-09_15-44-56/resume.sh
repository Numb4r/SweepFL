#!/bin/bash
# Auto-generated resume script
# Source folder : sparsification_percentage/resnet20_CIFAR10_rw_none_200rounds_9clients_seed42_niid1_percentage0.9_8192_-1_-1_60_1099511627776_None_2026-06-09_15-44-56
# Generated     : 2026-06-10 09:22:12
# Interrupted at: round 177 / 200
#
# Run locally : bash logs/sparsification_percentage/resnet20_CIFAR10_rw_none_200rounds_9clients_seed42_niid1_percentage0.9_8192_-1_-1_60_1099511627776_None_2026-06-09_15-44-56/resume.sh
# Run on SLURM: sbatch logs/sparsification_percentage/resnet20_CIFAR10_rw_none_200rounds_9clients_seed42_niid1_percentage0.9_8192_-1_-1_60_1099511627776_None_2026-06-09_15-44-56/resume_sbatch.sh

GPU_IDS=(0 1 2)
CLIENTS_PER_GPU=3

FOLDER="sparsification_percentage"
SOLUTIONS=('rw')
NIIDS=(1)
SEEDS=(42)
PERCENTAGES=(0.9)
NUM_ROUNDS=200
TECHNIQUE="none"
EXPERIMENT_NAME="resnet20_CIFAR10_rw"

MODEL_DATASET_PAIRS=("resnet20:CIFAR10")

CKKS_CONFIGS=("ckks 8192 60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1

CKKS_BACKEND="ckks_python_simplify"
SIMPLIFY_N=8192
SIMPLIFY_BITS=62
SIMPLIFY_SCALE_BITS=40
PYCRYPT_N=8192
PYCRYPT_BITS=62
OPENFHE_RING_DIM=8192
OPENFHE_MULT_DEPTH=1
OPENFHE_SCALE_BITS=50

BATCH_SIZE=4
RR_START_ROUND=200

# Locate repo root by walking up from this script's directory
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$_SCRIPT_DIR"
while [[ "$_REPO_ROOT" != "/" ]] && [[ ! -f "${_REPO_ROOT}/experiments/lib.sh" ]]; do
    _REPO_ROOT="$(dirname "$_REPO_ROOT")"
done
if [[ ! -f "${_REPO_ROOT}/experiments/lib.sh" ]]; then
    echo "ERROR: cannot find experiments/lib.sh from $_SCRIPT_DIR"; exit 1
fi
source "${_REPO_ROOT}/experiments/lib.sh"
run_theme

# ── Auto-merge: join this resumed experiment with the original folder ──────
_MERGE_SCRIPT="${_REPO_ROOT}/experiments/merge_resume.py"
if [[ -f "$_MERGE_SCRIPT" ]]; then
    echo ""
    echo "[merge] Merging resumed experiment into original folder..."
    python3 "$_MERGE_SCRIPT" --logs-dir "${_REPO_ROOT}/logs" --yes
fi
