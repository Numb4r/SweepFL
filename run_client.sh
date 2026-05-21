#!/bin/bash
# Launches one FL client in a Docker or Singularity container.
# Called by start_all.sh — do not run directly unless you know what you're doing.
#
# Args: CID SOLUTION TECHNIQUE NCLIENTES GPU_ID POLY COEFFS SCALE KEY_TYPE
#       NIID DATASET SEED PERCENTAGE LOGFOLDER PLAIN_MODULUS N_LENGTH
#
# Env vars consumed (set by start_all.sh / lib.sh):
#   RUNTIME          docker (default) | singularity
#   SIF_DIR          directory containing flhe_client.sif (singularity only, default .)
#   SERVER_HOST      override server hostname (singularity default: localhost,
#                    docker default: host.docker.internal)

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

RUNTIME="${RUNTIME:-docker}"
SIF_DIR="${SIF_DIR:-.}"

echo "Starting client $CID | runtime=$RUNTIME solution=$SOLUTION technique=$TECHNIQUE gpu=$GPU_ID seed=$SEED"
echo "  dataset=$DATASET niid=$NIID percentage=$PERCENTAGE logfolder=$LOGFOLDER"
echo "  poly=$POLY coeffs=$COEFFS scale=$SCALE key_type=$KEY_TYPE plain=$PLAIN_MODULUS n_length=$N_LENGTH"

# ── Docker ────────────────────────────────────────────────────────────────────
if [[ "$RUNTIME" == "docker" ]]; then

    NAME="client-$CID"
    docker rm -f "$NAME" 2>/dev/null || true

    # GPU flag: skip entirely when USE_GPU=false (CPU fallback for machines without
    # a working nvidia-persistenced socket). Otherwise use the classic --gpus flag.
    if [[ "${USE_GPU:-true}" == "false" ]]; then
        GPU_FLAG=""
    else
        GPU_FLAG="--gpus=device=$GPU_ID"
    fi

    docker run --rm \
        --name "$NAME" \
        $GPU_FLAG \
        --add-host=host.docker.internal:host-gateway \
        -v "$PWD/context:/context:ro" \
        -v "$PWD/logs:/logs" \
        -v "$PWD/client:/client" \
        -e SERVER_IP="${SERVER_HOST:-host.docker.internal}:9999" \
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
        -e BATCH_SIZE=${BATCH_SIZE:-4} \
        -e RR_START_ROUND=${RR_START_ROUND:-200} \
        -e RESUME_LOGFOLDER=${RESUME_LOGFOLDER:-} \
        -e HF_DATASETS_CACHE=/client/data \
        -e HF_HOME=/tmp/hf_home \
        -e HF_HUB_READ_TIMEOUT=300 \
        -e TRANSFORMERS_CACHE=/tmp/hf_cache \
        -e TOKENIZERS_PARALLELISM=false \
        -e TORCH_HOME=/tmp/torch_cache \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e TF_CPP_MIN_LOG_LEVEL=2 \
        -e TF_ENABLE_ONEDNN_OPTS=0 \
        flhe:client \
        python /client/client.py

# ── Singularity ───────────────────────────────────────────────────────────────
elif [[ "$RUNTIME" == "singularity" ]]; then

    singularity exec --nv \
        --bind "${CONTEXT_DIR:-$PWD/context}:/context:ro" \
        --bind "$PWD/logs:/logs" \
        --bind "$PWD/client:/client" \
        --env SERVER_IP="${SERVER_HOST:-localhost}:9999" \
        --env CID=$CID \
        --env SOLUTION=$SOLUTION \
        --env TECHNIQUE=$TECHNIQUE \
        --env NCLIENTS=$NCLIENTES \
        --env DATASET=$DATASET \
        --env NIID=$NIID \
        --env PERCENTAGE=$PERCENTAGE \
        --env LOGFOLDER=$LOGFOLDER \
        --env POLY=$POLY \
        --env COEFFS=$COEFFS \
        --env SCALE=$SCALE \
        --env KEY_TYPE=$KEY_TYPE \
        --env PLAIN_MODULUS=$PLAIN_MODULUS \
        --env N_LENGTH=$N_LENGTH \
        --env PYTHONHASHSEED=$SEED \
        --env CKKS_BACKEND=${CKKS_BACKEND:-tenseal} \
        --env SIMPLIFY_N=${SIMPLIFY_N:-8192} \
        --env SIMPLIFY_BITS=${SIMPLIFY_BITS:-62} \
        --env SIMPLIFY_SCALE_BITS=${SIMPLIFY_SCALE_BITS:-60} \
        --env PYCRYPT_N=${PYCRYPT_N:-8192} \
        --env PYCRYPT_BITS=${PYCRYPT_BITS:-60} \
        --env OPENFHE_RING_DIM=${OPENFHE_RING_DIM:-8192} \
        --env OPENFHE_MULT_DEPTH=${OPENFHE_MULT_DEPTH:-1} \
        --env OPENFHE_SCALE_BITS=${OPENFHE_SCALE_BITS:-50} \
        --env MODEL_ARCH=${MODEL_ARCH:-} \
        --env EXPERIMENT_NAME=${EXPERIMENT_NAME:-} \
        --env PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-} \
        --env BATCH_SIZE=${BATCH_SIZE:-4} \
        --env RR_START_ROUND=${RR_START_ROUND:-200} \
        --env RESUME_LOGFOLDER=${RESUME_LOGFOLDER:-} \
        --env HF_DATASETS_CACHE=/client/data \
        --env HF_HOME=/tmp/hf_home \
        --env HF_HUB_READ_TIMEOUT=300 \
        --env TRANSFORMERS_CACHE=/tmp/hf_cache \
        --env TOKENIZERS_PARALLELISM=false \
        --env TORCH_HOME=/tmp/torch_cache \
        --env CUDA_VISIBLE_DEVICES=$GPU_ID \
        --env TF_CPP_MIN_LOG_LEVEL=2 \
        --env TF_ENABLE_ONEDNN_OPTS=0 \
        "${SIF_DIR}/flhe_client.sif" \
        python /client/client.py

else
    echo "ERROR: unknown RUNTIME='$RUNTIME' (expected: docker | singularity)" >&2
    exit 1
fi
