#!/bin/bash
# Runs pretrain.py inside the flhe:client container with GPU access.
#
# Usage:
#   ./run_pretrain.sh [--model vit_tiny] [--dataset cifar100] [--epochs 100]
#                     [--batch-size 128] [--lr 3e-4] [--gpu 5]
#
# Env vars:
#   RUNTIME   docker (default) | singularity
#   SIF_DIR   directory containing flhe_client.sif (singularity only, default .)

set -euo pipefail

RUNTIME="${RUNTIME:-docker}"
SIF_DIR="${SIF_DIR:-.}"
GPU_ID="${GPU_ID:-5}"

# Parse --gpu from args; forward everything else to pretrain.py
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

echo "Pre-training on GPU $GPU_ID | runtime=$RUNTIME"
echo "Args: ${ARGS[*]:-<defaults>}"
echo ""

mkdir -p logs/models/pretrained
KERAS_CACHE="${HOME}/.keras"
mkdir -p "$KERAS_CACHE"

# ── Docker ────────────────────────────────────────────────────────────────────
if [[ "$RUNTIME" == "docker" ]]; then

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

# ── Singularity ───────────────────────────────────────────────────────────────
elif [[ "$RUNTIME" == "singularity" ]]; then

    singularity exec --nv \
        --bind "$PWD/pretrain.py:/pretrain.py:ro" \
        --bind "$PWD/client:/client:ro" \
        --bind "$PWD/logs:/logs" \
        --bind "$KERAS_CACHE:/root/.keras" \
        --env CUDA_VISIBLE_DEVICES=$GPU_ID \
        --env TF_CPP_MIN_LOG_LEVEL=1 \
        --env HF_DATASETS_CACHE=/client/data \
        --env HF_HOME=/tmp/hf_home \
        "${SIF_DIR}/flhe_client.sif" \
        python /pretrain.py "${ARGS[@]:-}"

else
    echo "ERROR: unknown RUNTIME='$RUNTIME' (expected: docker | singularity)" >&2
    exit 1
fi
