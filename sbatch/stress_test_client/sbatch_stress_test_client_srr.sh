#!/bin/bash
# ── SLURM job: client-scale stress test — srr (spread round-robin) ────────────
#
# 4 GPUs × 100 clients/GPU = 400 clients total.

#SBATCH --job-name=fl_stress_test_client_srr
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=y218172@dac.unicamp.br
#SBATCH --mem=0

# ── Guard: este é um job SLURM; submeta, não rode com bash ───────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: '$0' é um script SLURM — submeta com sbatch, não execute com bash." >&2
    echo "  Submeter:        sbatch $0" >&2
    echo "  Um job por par:  bash sbatch/submit_per_pair.sh $0" >&2
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
export RUNTIME=singularity
export SIF_DIR="${SIF_DIR:-$HOME/keys}"
export USE_GPU=true

# ── Pre-flight ────────────────────────────────────────────────────────────────
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

# ── Run experiment ────────────────────────────────────────────────────────────
EXPERIMENT_SCRIPT="experiments/stress_test_client/stress_test_client_srr.sh"

bash "$EXPERIMENT_SCRIPT"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
