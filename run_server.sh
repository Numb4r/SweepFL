#!/bin/bash
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
PLAIN=${13}
LENGHT=${14}
echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14}"
echo "Parâmetros do servidor: SOLUTION=$SOLUTION, NCLIENTES=$NCLIENTES, NUM_ROUNDS=$NUM_ROUNDS, POLY=$POLY, COEFFS=$COEFFS, SCALE=$SCALE, KEY_TYPE=$KEY_TYPE, DATASET=$DATASET, NIID=$NIID, SEED=$SEED, PERCENTAGE=$PERCENTAGE, LOGFOLDER=$LOGFOLDER"
echo "Iniciando Servidor com $NCLIENTES clientes, $NUM_ROUNDS rounds Dataset $DATASET solução $SOLUTION Distribuição Niid $NIID SEED $SEED PERCENTAGE $PERCENTAGE"
echo "Poly $POLY Coeffs $COEFFS Scale $SCALE Key_Type $KEY_TYPE, PLAIN=$PLAIN, LENGHT=$LENGHT"
NAME="fl_server-gpu"



docker rm -f $NAME 2>/dev/null
echo "Iniciando container Docker do Servidor FLHE"
docker run -it --name $NAME \
  -p 9999:9999 \
  -v $PWD/server:/server:ro \
  -v $PWD/context:/context:ro \
  -v $PWD/logs:/logs \
  -e SERVER_IP=0.0.0.0:9999 \
  -e NCLIENTS=$((NCLIENTES)) \
  -e NUM_ROUNDS=$NUM_ROUNDS \
  -e SOLUTION=$SOLUTION \
  -e POLY=$POLY \
  -e COEFFS=$COEFFS \
  -e SCALE=$SCALE \
  -e KEY_TYPE=$KEY_TYPE \
  -e DATASET=$DATASET \
  -e FRAC_FIT=0 \
  -e PYTHONHASHSEED=$SEED \
  -e NIID=$NIID \
  -e PERCENTAGE=$PERCENTAGE \
  -e LOGFOLDER=$LOGFOLDER \
  -e PLAIN_MODULUS=$PLAIN \
  -e N_LENGTH=$LENGHT \
  flhe:server \
  python /server/server.py

# docker logs -f $NAME >> logs/$NAME.log

TOKEN="7947375786:AAHmPy5ZlI2dhxXBEOC7cOVMb8fLK5ZbaWo"
CHAT_ID="111824133"
curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id=$CHAT_ID \
    -d text="Servidor finalizado 🤖"
echo "Servidor finalizado!"