#!/bin/bash
# ── SLURM job: seed_experiment sweep — sw ──────────────────────────────
#
# Repeats the sparsification sweep across seeds 43 and 44 so results can be
# reported as mean ± spread instead of single runs.
#
# Submit all pair×seed combinations as independent jobs (recommended):
#   bash sbatch/submit_per_pair.sh sbatch/seed_experiment/sbatch_seed_experiment_sw.sh
# Or run the whole sweep inside one job:
#   sbatch sbatch/seed_experiment/sbatch_seed_experiment_sw.sh
#
# Monitor: squeue -u $USER
#          tail -f logs/slurm_<jobid>.out
#
# GPU sizing: CLIENTS_PER_GPU=3, gres=gpu:3 → 9 clients total.
# Wall time:  10 experiments × 200 rounds per per-pair job (resnet20 /
#             mobilenetv2 only, no cheap pairs); checkpoint/resume covers
#             an overrun.

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=fl_seed_experiment_sw
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
EXPERIMENT_SCRIPT="experiments/seed_experiment/seed_experiment_sw.sh"

bash "$EXPERIMENT_SCRIPT"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
