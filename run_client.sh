#!/bin/bash
# Launches one FL client in a Docker container.
# Called by start_all.sh — do not run directly unless you know what you're doing.
#
# Args: CID SOLUTION TECHNIQUE NCLIENTES GPU_ID POLY COEFFS SCALE KEY_TYPE
#       NIID DATASET SEED PERCENTAGE LOGFOLDER PLAIN_MODULUS N_LENGTH

CID=$1
SOLUTION=$2
TECHNIQUE=$3
NCLIENTES=$4
GPU_ID=$5
POLY=$6
COEFFS=$7
SCALE=$8
KEY_TYPE=$9
NIID=${10}
DATASET=${11}
SEED=${12}
PERCENTAGE=${13}
LOGFOLDER=${14}
PLAIN_MODULUS=${15}
N_LENGTH=${16}

echo "Starting client $CID | solution=$SOLUTION technique=$TECHNIQUE gpu=$GPU_ID seed=$SEED"
echo "  dataset=$DATASET niid=$NIID percentage=$PERCENTAGE logfolder=$LOGFOLDER"
echo "  poly=$POLY coeffs=$COEFFS scale=$SCALE key_type=$KEY_TYPE plain=$PLAIN_MODULUS n_length=$N_LENGTH"

NAME="client-$CID"
docker rm -f "$NAME" 2>/dev/null || true

docker run --rm \
    --name "$NAME" \
    --gpus="device=$GPU_ID" \
    --add-host=host.docker.internal:host-gateway \
    -v "$PWD/context:/context:ro" \
    -v "$PWD/logs:/logs" \
    -v "$PWD/client:/client" \
    -e SERVER_IP=host.docker.internal:9999 \
    -e CID=$CID \
    -e SOLUTION=$SOLUTION \
    -e TECHNIQUE=$TECHNIQUE \
    -e NCLIENTS=$NCLIENTES \
    -e DATASET=$DATASET \
    -e NIID=$NIID \
    -e PERCENTAGE=$PERCENTAGE \
    -e LOGFOLDER=$LOGFOLDER \
    -e POLY=$POLY \
    -e COEFFS=$COEFFS \
    -e SCALE=$SCALE \
    -e KEY_TYPE=$KEY_TYPE \
    -e PLAIN_MODULUS=$PLAIN_MODULUS \
    -e N_LENGTH=$N_LENGTH \
    -e PYTHONHASHSEED=$SEED \
    -e CKKS_BACKEND=${CKKS_BACKEND:-tenseal} \
    -e SIMPLIFY_N=${SIMPLIFY_N:-8192} \
    -e SIMPLIFY_BITS=${SIMPLIFY_BITS:-62} \
    -e SIMPLIFY_SCALE_BITS=${SIMPLIFY_SCALE_BITS:-60} \
    -e PYCRYPT_N=${PYCRYPT_N:-8192} \
    -e PYCRYPT_BITS=${PYCRYPT_BITS:-60} \
    -e OPENFHE_RING_DIM=${OPENFHE_RING_DIM:-8192} \
    -e OPENFHE_MULT_DEPTH=${OPENFHE_MULT_DEPTH:-1} \
    -e OPENFHE_SCALE_BITS=${OPENFHE_SCALE_BITS:-50} \
    -e MODEL_ARCH=${MODEL_ARCH:-} \
    -e EXPERIMENT_NAME=${EXPERIMENT_NAME:-} \
    -e PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-} \
    -e HF_DATASETS_CACHE=/client/data \
    -e HF_HOME=/tmp/hf_home \
    -e HF_HUB_READ_TIMEOUT=300 \
    -e TRANSFORMERS_CACHE=/tmp/hf_cache \
    -e TOKENIZERS_PARALLELISM=false \
    -e TORCH_HOME=/tmp/torch_cache \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e TF_CPP_MIN_LOG_LEVEL=0 \
    flhe:client \
    python /client/client.py
