#!/bin/bash
# Runs the MobileNetV2 CIFAR-10 benchmark inside the flhe:client Docker container.
#
# Usage:
#   bash benchmark/run_docker.sh [--batch-size N] [--gpu GPU_ID] [--no-gpu] [--dry-run]
#
# Examples:
#   bash benchmark/run_docker.sh                     # all batch sizes, GPU 0
#   bash benchmark/run_docker.sh --batch-size 32     # single batch size, GPU 0
#   bash benchmark/run_docker.sh --gpu 1             # all batch sizes, GPU 1
#   bash benchmark/run_docker.sh --no-gpu            # CPU mode (no GPU flag)

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
CONTAINER_NAME="benchmark_mobilenetv2"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

mkdir -p "$REPO_ROOT/benchmark/results"
chmod 777 "$REPO_ROOT/benchmark/results"

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
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e TF_CPP_MIN_LOG_LEVEL=2
    -e TF_ENABLE_ONEDNN_OPTS=0
    flhe:client
    python /benchmark/train_mobilenetv2_cifar10.py $BATCH_SIZE_ARG
)

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] Command that would run:"
    echo "  ${CMD[*]}"
    exit 0
fi

echo "MobileNetV2 CIFAR-10 benchmark — Docker (gpu=$GPU_ID, use_gpu=$USE_GPU)"
echo "Results → $REPO_ROOT/benchmark/results/"
echo ""
"${CMD[@]}"
