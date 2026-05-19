#!/bin/bash
# ── MobileNetV2 CIFAR-10 benchmark — SLURM job array ─────────────────────────
#
# Runs one batch size per array task in parallel (4 tasks total).
# Each task requests 1 GPU and trains for 200 epochs.
# Executes inside flhe_client.sif (Singularity) — no host Python deps needed.
#
# Submit:
#   sbatch benchmark/sbatch_mobilenetv2.sh
#
# Override SIF location if needed:
#   SIF_DIR=/other/path sbatch benchmark/sbatch_mobilenetv2.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f benchmark/results/slurm_<jobid>_<taskid>.out
#
# Results land in benchmark/results/:
#   batch_16.csv / batch_32.csv / batch_64.csv / batch_128.csv
#   summary.csv  (appended by each task on completion)

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=mobilenetv2_bench
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00:00
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=benchmark/results/slurm_%A_%a.out
#SBATCH --error=benchmark/results/slurm_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=y218172@dac.unicamp.br

# ── Batch size mapping ────────────────────────────────────────────────────────
BATCH_SIZES=(16 32 64 128)
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

# ── Setup ─────────────────────────────────────────────────────────────────────
REPO="$HOME/keys"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }

# Results dir must exist before SLURM writes the .out/.err files and before the
# container writes CSVs. Pre-create with open permissions so the container
# (which may run as a different UID) can write to it.
mkdir -p benchmark/results
chmod 777 benchmark/results

SIF_DIR="${SIF_DIR:-$REPO}"
SIF="$SIF_DIR/flhe_client.sif"

if [ ! -f "$SIF" ]; then
    echo "ERROR: Singularity image not found: $SIF"
    echo "       Build it with:  make singularity-from-docker   (or make singularity-configure)"
    exit 1
fi

echo "Job $SLURM_JOB_ID  array_task=$SLURM_ARRAY_TASK_ID  batch_size=$BS"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "SIF: $SIF"
echo "Started: $(date)"
echo ""

# ── Run inside Singularity ────────────────────────────────────────────────────
# --nv          : pass NVIDIA GPU into the container
# --bind $PWD:$PWD : expose the repo at its real path so relative imports work
singularity exec \
    --nv \
    --bind "$PWD:$PWD" \
    --env PYTHONUNBUFFERED=1 \
    --env TF_CPP_MIN_LOG_LEVEL=2 \
    --env TF_ENABLE_ONEDNN_OPTS=0 \
    "$SIF" \
    python3 "$PWD/benchmark/train_mobilenetv2_cifar10.py" --batch-size "$BS"

echo ""
echo "Finished: $(date)"
