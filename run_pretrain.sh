#!/bin/bash
# Runs pretrain.py inside the flhe:client Docker container with GPU access.
# The client image already has TensorFlow + CUDA configured correctly.
#
# Usage:
#   ./run_pretrain.sh [--model vit_tiny] [--dataset cifar100] [--epochs 100]
#                     [--batch-size 128] [--lr 3e-4] [--gpu 5]
#
# All args are forwarded to pretrain.py. Defaults match the pretrain.py defaults.

set -euo pipefail

GPU_ID="${GPU_ID:-5}"   # which physical GPU to use inside Docker

# Parse --gpu from args if provided, otherwise keep GPU_ID default
ARGS=()
i=1
while [[ $i -le $# ]]; do
    arg="${!i}"
    if [[ "$arg" == "--gpu" ]]; then
        i=$(( i + 1 ))
        GPU_ID="${!i}"
    else
        ARGS+=("$arg")
    fi
    i=$(( i + 1 ))
done

echo "Pre-training on GPU $GPU_ID inside flhe:client container"
echo "Args: ${ARGS[*]:-<defaults>}"
echo ""

mkdir -p logs/models/pretrained

KERAS_CACHE="${HOME}/.keras"
mkdir -p "$KERAS_CACHE"

docker run --rm \
    --name "pretrain-$(date +%s)" \
    --gpus="device=$GPU_ID" \
    -v "$PWD/pretrain.py:/pretrain.py:ro" \
    -v "$PWD/client:/client:ro" \
    -v "$PWD/logs:/logs" \
    -v "$KERAS_CACHE:/root/.keras" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e TF_CPP_MIN_LOG_LEVEL=1 \
    -e HF_DATASETS_CACHE=/client/data \
    -e HF_HOME=/tmp/hf_home \
    flhe:client \
    python /pretrain.py "${ARGS[@]:-}"
