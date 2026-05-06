#!/bin/bash
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
PLAIN=${15}
LENGHT=${16}
echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16}"
echo "Iniciando cliente $CID com solução $SOLUTION, técnica $TECHNIQUE em GPU $GPU_ID com seed $SEED LOGFOLDER: $LOGFOLDER Dataset: $DATASET NIID: $NIID"
echo "Parâmetros: POLY=$POLY, COEFFS=$COEFFS, SCALE=$SCALE, KEY_TYPE=$KEY_TYPE, PLAIN=$PLAIN,NLENGHT=$LENGHT, PERCENTAGE=$PERCENTAGE"
NAME="client-$CID"

docker rm -f $NAME 2>/dev/null

docker run -it \
   --gpus="device=$GPU_ID" \
   --name $NAME \
   --add-host=host.docker.internal:host-gateway \
   -e SERVER_IP=host.docker.internal:9999 \
   -v $PWD/context:/context:ro \
   -v $PWD/logs:/logs \
   -v $PWD/client:/client \
   -e TRANSFORMERS_CACHE=/tmp/hf_cache \
   -e HF_HOME=/tmp/hf_home \
   -e TOKENIZERS_PARALLELISM=false \
   -e TORCH_HOME=/tmp/torch_cache \
   -e HF_DATASETS_CACHE=/client/data\
   -e HF_HUB_READ_TIMEOUT=300\
   -e CID=$CID \
   -e SOLUTION=$SOLUTION \
   -e TECHNIQUE=$TECHNIQUE \
   -e NCLIENTS=$NCLIENTES \
   -e DATASET_NAME=$DATASET_NAME \
   -e PYTHONHASHSEED=$SEED \
    -e POLY=$POLY \
    -e COEFFS=$COEFFS \
    -e SCALE=$SCALE \
    -e KEY_TYPE=$KEY_TYPE \
    -e NIID=$NIID \
    -e DATASET=$DATASET \
    -e PERCENTAGE=$PERCENTAGE \
    -e LOGFOLDER=$LOGFOLDER \
    -e PLAIN_MODULUS=$PLAIN \
    -e N_LENGTH=$LENGHT \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
   -e TF_CPP_MIN_LOG_LEVEL=0 \
   flhe:client \
   python /client/client.py


# docker logs -f $NAME >> logs/$NAME.log

