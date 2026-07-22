#!/bin/bash
# Auto-generated sbatch — resume of interrupted experiment
# Source folder : sparsification_percentage/resnet20_CIFAR10_sw_none_200rounds_9clients_seed42_niid1_percentage0.9_8192_-1_-1_60_1099511627776_None_2026-06-09_06-33-44
# Generated     : 2026-06-10 09:22:12
# Interrupted at: round 124 / 200
#SBATCH --job-name=resume_sparsification_percentage_sw_resnet20
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=y218172@dac.unicamp.br
#SBATCH --mem=64G

export RUNTIME=singularity
export SIF_DIR="${SIF_DIR:-$HOME/keys}"
export USE_GPU=true

REPO="$HOME/keys"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }

_SBATCH_TRAP_REGISTERED=0
_sbatch_exit_hook() {
    local code=$?
    if declare -f _slurm_fail_notify &>/dev/null; then
        _slurm_fail_notify "$code"
    fi
}
trap '_sbatch_exit_hook' EXIT

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Partition: $SLURM_JOB_PARTITION  |  GPUs on node: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo ""

mkdir -p logs

EXPERIMENT_SCRIPT="logs/sparsification_percentage/resnet20_CIFAR10_sw_none_200rounds_9clients_seed42_niid1_percentage0.9_8192_-1_-1_60_1099511627776_None_2026-06-09_06-33-44/resume.sh"
bash "$EXPERIMENT_SCRIPT"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
