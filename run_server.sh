#!/bin/bash
# Launches the FL aggregation server in a Docker or Singularity container.
# Called by start_all.sh — do not run directly unless you know what you're doing.
#
# Args: SOLUTION NCLIENTES NUM_ROUNDS POLY COEFFS SCALE KEY_TYPE
#       DATASET NIID SEED PERCENTAGE LOGFOLDER PLAIN_MODULUS N_LENGTH
#
# Env vars consumed (set by start_all.sh / lib.sh):
#   RUNTIME   docker (default) | singularity
#   SIF_DIR   directory containing flhe_server.sif (singularity only, default .)

SOLUTION=$1
NCLIENTES=$2
NUM_ROUNDS=$3
POLY=$4
COEFFS=$5
SCALE=$6
KEY_TYPE=$7
DATASET=$8
NIID=$9
SEED=${10}
PERCENTAGE=${11}
LOGFOLDER=${12}
PLAIN_MODULUS=${13}
N_LENGTH=${14}

RUNTIME="${RUNTIME:-docker}"
SIF_DIR="${SIF_DIR:-.}"

TELEGRAM_TOKEN="7947375786:AAHmPy5ZlI2dhxXBEOC7cOVMb8fLK5ZbaWo"
TELEGRAM_CHAT="111824133"

echo "Starting server | runtime=$RUNTIME solution=$SOLUTION clients=$NCLIENTES rounds=$NUM_ROUNDS dataset=$DATASET"
echo "  niid=$NIID seed=$SEED percentage=$PERCENTAGE logfolder=$LOGFOLDER"
echo "  poly=$POLY coeffs=$COEFFS scale=$SCALE key_type=$KEY_TYPE plain=$PLAIN_MODULUS n_length=$N_LENGTH"

# ── Docker ────────────────────────────────────────────────────────────────────
if [[ "$RUNTIME" == "docker" ]]; then

    NAME="fl_server-gpu"
    docker rm -f "$NAME" 2>/dev/null || true

    docker run --rm \
        --name "$NAME" \
        -p 9999:9999 \
        -v "$PWD/server:/server:ro" \
        -v "$PWD/context:/context:ro" \
        -v "$PWD/logs:/logs" \
        -e SERVER_IP=0.0.0.0:9999 \
        -e NCLIENTS=$NCLIENTES \
        -e NUM_ROUNDS=$NUM_ROUNDS \
        -e SOLUTION=$SOLUTION \
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
        -e EXPERIMENT_NAME=${EXPERIMENT_NAME:-} \
        -e RESUME_LOGFOLDER=${RESUME_LOGFOLDER:-} \
        -e JOB_DEADLINE_EPOCH=${JOB_DEADLINE_EPOCH:-0} \
        -e CHECKPOINT_THRESHOLD_SECS=${CHECKPOINT_THRESHOLD_SECS:-300} \
        -e CKKS_BACKEND=${CKKS_BACKEND:-tenseal} \
        -e SIMPLIFY_N=${SIMPLIFY_N:-8192} \
        -e SIMPLIFY_BITS=${SIMPLIFY_BITS:-62} \
        -e SIMPLIFY_SCALE_BITS=${SIMPLIFY_SCALE_BITS:-60} \
        -e PYCRYPT_N=${PYCRYPT_N:-8192} \
        -e PYCRYPT_BITS=${PYCRYPT_BITS:-60} \
        -e OPENFHE_RING_DIM=${OPENFHE_RING_DIM:-8192} \
        -e OPENFHE_MULT_DEPTH=${OPENFHE_MULT_DEPTH:-1} \
        -e OPENFHE_SCALE_BITS=${OPENFHE_SCALE_BITS:-50} \
        flhe:server \
        python /server/server.py

# ── Singularity ───────────────────────────────────────────────────────────────
elif [[ "$RUNTIME" == "singularity" ]]; then

    singularity exec --nv \
        --bind "$PWD/server:/server:ro" \
        --bind "${CONTEXT_DIR:-$PWD/context}:/context:ro" \
        --bind "$PWD/logs:/logs" \
        --env SERVER_IP=0.0.0.0:9999 \
        --env NCLIENTS=$NCLIENTES \
        --env NUM_ROUNDS=$NUM_ROUNDS \
        --env SOLUTION=$SOLUTION \
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
        --env EXPERIMENT_NAME=${EXPERIMENT_NAME:-} \
        --env RESUME_LOGFOLDER=${RESUME_LOGFOLDER:-} \
        --env JOB_DEADLINE_EPOCH=${JOB_DEADLINE_EPOCH:-0} \
        --env CHECKPOINT_THRESHOLD_SECS=${CHECKPOINT_THRESHOLD_SECS:-300} \
        --env CKKS_BACKEND=${CKKS_BACKEND:-tenseal} \
        --env SIMPLIFY_N=${SIMPLIFY_N:-8192} \
        --env SIMPLIFY_BITS=${SIMPLIFY_BITS:-62} \
        --env SIMPLIFY_SCALE_BITS=${SIMPLIFY_SCALE_BITS:-60} \
        --env PYCRYPT_N=${PYCRYPT_N:-8192} \
        --env PYCRYPT_BITS=${PYCRYPT_BITS:-60} \
        --env OPENFHE_RING_DIM=${OPENFHE_RING_DIM:-8192} \
        --env OPENFHE_MULT_DEPTH=${OPENFHE_MULT_DEPTH:-1} \
        --env OPENFHE_SCALE_BITS=${OPENFHE_SCALE_BITS:-50} \
        "${SIF_DIR}/flhe_server.sif" \
        python /server/server.py

else
    echo "ERROR: unknown RUNTIME='$RUNTIME' (expected: docker | singularity)" >&2
    exit 1
fi

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="$TELEGRAM_CHAT" -d text="Server finished: $SOLUTION $DATASET niid=$NIID" > /dev/null
echo "Server finished."
