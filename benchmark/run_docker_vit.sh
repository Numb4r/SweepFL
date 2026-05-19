#!/bin/bash
# Runs the ViT-Tiny GTSRB benchmark inside the flhe:client Docker container.
#
# Usage:
#   bash benchmark/run_docker_vit.sh [--batch-size N] [--gpu GPU_ID] [--no-gpu] [--dry-run]
#
# Examples:
#   bash benchmark/run_docker_vit.sh                     # all batch sizes, GPU 0
#   bash benchmark/run_docker_vit.sh --batch-size 32     # single batch size, GPU 0
#   bash benchmark/run_docker_vit.sh --gpu 1             # all batch sizes, GPU 1
#   bash benchmark/run_docker_vit.sh --no-gpu            # CPU mode (no GPU flag)
#
# The GTSRB dataset is downloaded on first run and cached in benchmark/data/
# so subsequent runs don't re-download it.

set -euo pipefail

GPU_ID=7
USE_GPU=true
BATCH_SIZE_ARG=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size) BATCH_SIZE_ARG="--batch-size $2"; shift 2 ;;
        --gpu)        GPU_ID="$2"; shift 2 ;;
        --no-gpu)     USE_GPU=false; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER_NAME="benchmark_vit_gtsrb"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

mkdir -p "$REPO_ROOT/benchmark/results_vit_gtsrb"
chmod 777 "$REPO_ROOT/benchmark/results_vit_gtsrb"

GPU_FLAG=""
if [[ "$USE_GPU" == "true" ]]; then
    GPU_FLAG="--gpus=device=$GPU_ID"
fi

CMD=(
    docker run --rm
    --name "$CONTAINER_NAME"
    $GPU_FLAG
    -v "$REPO_ROOT/benchmark:/benchmark"
    -v "$REPO_ROOT/client:/client"
    -e PYTHONUNBUFFERED=1
    -e HF_DATASETS_CACHE=/benchmark/data
    -e HF_HOME=/benchmark/data/hf_home
    -e HF_HUB_READ_TIMEOUT=300
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e TF_CPP_MIN_LOG_LEVEL=2
    -e TF_ENABLE_ONEDNN_OPTS=0
    flhe:client
    python /benchmark/train_vit_gtsrb.py $BATCH_SIZE_ARG
)

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] Command that would run:"
    echo "  ${CMD[*]}"
    exit 0
fi

echo "ViT-Tiny GTSRB benchmark — Docker (gpu=$GPU_ID, use_gpu=$USE_GPU)"
echo "Dataset cache → $REPO_ROOT/benchmark/data/"
echo "Results       → $REPO_ROOT/benchmark/results_vit_gtsrb/"
echo ""
"${CMD[@]}"
