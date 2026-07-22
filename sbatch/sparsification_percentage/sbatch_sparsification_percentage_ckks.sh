#!/bin/bash
# ── SLURM job template for Slidhe FL experiments ──────────────────────────────
#
# Usage:
#   cp sbatch/template.sh sbatch/my_experiment.sh
#   # edit partition, GPUs, EXPERIMENT_SCRIPT
#   sbatch sbatch/my_experiment.sh
#
# Before submitting:
#   1. Adjust #SBATCH directives (partition, GPUs, time) for your target machine.
#   2. Set EXPERIMENT_SCRIPT to the experiment you want to run.
#   3. Set SIF_DIR to where flhe_client.sif and flhe_server.sif are located.
#   4. The script auto-detects GPU_IDS from SLURM_GPUS_ON_NODE — no need to
#      change GPU_IDS inside your experiment script.
#
# GPU sizing:
#   Request exactly  |GPU_IDS| * CLIENTS_PER_GPU  GPUs.
#   Example: CLIENTS_PER_GPU=3, 3 GPUs requested → 9 clients total.
#   SLURM exposes allocated GPUs as 0, 1, 2 (virtual, not physical IDs).
#
# Monitoring:
#   squeue -u $USER                         # job status
#   tail -f logs/slurm_<jobid>.out          # live SLURM stdout
#   tail -f logs/<LOGFOLDER>/server.log     # server log
#   tail -f logs/<LOGFOLDER>/client-0.log   # per-client logs

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=fl_experiment_sparsification_percentage-ckks
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
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
export SIF_DIR="${SIF_DIR:-$HOME/keys}"   # directory containing .sif files
export USE_GPU=true

# Optional: override Telegram credentials if different from lib.sh defaults
# export TELEGRAM_TOKEN="..."
# export TELEGRAM_CHAT="..."

# ── Pre-flight ────────────────────────────────────────────────────────────────
REPO="$HOME/keys"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }

# Application-level failure notification (complements #SBATCH --mail-type above).
# Fires on any non-zero exit — including crashes, OOM inside the container, etc.
# Must be registered before sourcing lib.sh so _slurm_fail_notify is available.
_SBATCH_TRAP_REGISTERED=0
_sbatch_exit_hook() {
    local code=$?
    # _slurm_fail_notify is defined in lib_slurm.sh (sourced via experiment script)
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
# Change this to the experiment script you want to run.
EXPERIMENT_SCRIPT="experiments/sparsification_percentage/sparsification_percentage_ckks.sh"

bash "$EXPERIMENT_SCRIPT"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
