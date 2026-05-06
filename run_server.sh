#!/bin/bash
# Launches the FL aggregation server in a Docker container.
# Called by start_all.sh — do not run directly unless you know what you're doing.
#
# Args: SOLUTION NCLIENTES NUM_ROUNDS POLY COEFFS SCALE KEY_TYPE
#       DATASET NIID SEED PERCENTAGE LOGFOLDER PLAIN_MODULUS N_LENGTH

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

TELEGRAM_TOKEN="7947375786:AAHmPy5ZlI2dhxXBEOC7cOVMb8fLK5ZbaWo"
TELEGRAM_CHAT="111824133"

echo "Starting server | solution=$SOLUTION clients=$NCLIENTES rounds=$NUM_ROUNDS dataset=$DATASET"
echo "  niid=$NIID seed=$SEED percentage=$PERCENTAGE logfolder=$LOGFOLDER"
echo "  poly=$POLY coeffs=$COEFFS scale=$SCALE key_type=$KEY_TYPE plain=$PLAIN_MODULUS n_length=$N_LENGTH"

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

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="$TELEGRAM_CHAT" -d text="Server finished: $SOLUTION $DATASET niid=$NIID" > /dev/null
echo "Server finished."
