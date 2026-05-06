#!/bin/bash
TOKEN="7947375786:AAHmPy5ZlI2dhxXBEOC7cOVMb8fLK5ZbaWo"
CHAT_ID="111824133"
declare -A dict

dict["resnet20"]="CIFAR10 tanganke/gtsrb" 
dict["lenet5"]="FASHION_MNIST"
dict['DNN']="MNIST"
NIIDS=(0.1 1)
SEEDS=(42)
PERCENTAGES=(0.3 0.5 0.8)
NCLIENTES=9
NUM_ROUNDS=50

# PLACEHOLDERS
SOLUTION="plaintext"
POLY=0
COEFFS="0,0,0"
SCALE=0
plain=-1
nlength=-1
KEY_TYPE="placeholder"
TECHNIQUE="none"

mkdir -p context
GPU_IDS=(5 6 7)
CLIENTS_PER_GPU=3
NGPUS=${#GPU_IDS[@]}
NCLIENTES=$((NGPUS * CLIENTS_PER_GPU))

mkdir -p logs/models
# chmod -R 777 logs/

echo ""
echo "=========================================="
echo "MAPEAMENTO DE CLIENTES → GPUS (BFV)"
echo "=========================================="
for i in $(seq 0 $((NCLIENTES - 1))); do
  GPU_INDEX=$(( i / CLIENTS_PER_GPU ))
  GPU_ID=${GPU_IDS[$GPU_INDEX]}
  echo "CLIENTE $i → GPU física $GPU_ID"
done
echo "=========================================="
echo ""

run_experiment_tmux() {
    local POLY=$1
    local PLAIN=$2
    local COEFFS=$3
    local SCALE=$4
    local KEY_TYPE=$5
    local model=$6
    local DATASET=$7
    local NIID=$8
    local SEED=$9
    local SOLUTION=${10}
    local PERCENTAGE=${11}
    local NUM_ROUNDS=${12}
    local NCLIENTES=${13}
    local CLIENTS_PER_GPU=${14}
    local EXPERIMENT_DATE=${15}
    local LOGFOLDER=${16}

    local ERROR_FLAG="logs/${LOGFOLDER}/EXPERIMENT_FAILED"
    rm -f "$ERROR_FLAG"
    local SERVER_SESSION="SERVER"
    echo -e '{ 
                        "POLY":'$POLY', \n 
                        "PLAIN":'$PLAIN', \n
                        "COEFFS":['$COEFFS'], \n 
                        "SCALE":'$SCALE', \n 
                        "KEY_TYPE":"'$KEY_TYPE'", \n 
                        "model":"'$model'", \n 
                        "DATASET":"'$DATASET'", \n 
                        "NIID":'$NIID' , \n 
                        "SEED":'$SEED', \n
                        "SOLUTION":"'$SOLUTION'", \n
                        "PERCENTAGE":'$PERCENTAGE', \n
                        "NUM_ROUNDS":'$NUM_ROUNDS',\n
                        "NCLIENTES":'$NCLIENTES', \n
                        "CLIENTS_PER_GPU":'$CLIENTS_PER_GPU', \n
                        "EXPERIMENT_DATE":"'$EXPERIMENT_DATE'" \n
                        }' > logs/$LOGFOLDER/experiment_details.json
                        
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null

    echo "[SERVER] Iniciando sessão tmux: $SERVER_SESSION"
    tmux new-session -d -s "$SERVER_SESSION" \
        "./run_server.sh $SOLUTION $NCLIENTES $NUM_ROUNDS $POLY $COEFFS $SCALE $KEY_TYPE $DATASET $NIID $SEED $PERCENTAGE $LOGFOLDER $PLAIN $nlength >> logs/${LOGFOLDER}/tmux_${SERVER_SESSION}.log 2>&1"

    for i in $(seq 0 $((NCLIENTES - 1))); do
        local CLIENT_SESSION="client-$i"
        local GPU_INDEX=$(( i / CLIENTS_PER_GPU ))
        local GPU_ID=${GPU_IDS[$GPU_INDEX]}

        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null
        while ! tmux has-session -t "$CLIENT_SESSION" 2>/dev/null; do
            echo "[CLIENT] Iniciando sessão: $CLIENT_SESSION → GPU $GPU_ID NIID $NIID"
            tmux new-session -d -s "$CLIENT_SESSION" \
                "./run_client.sh $i $SOLUTION $TECHNIQUE $NCLIENTES $GPU_ID $POLY $COEFFS $SCALE $KEY_TYPE $NIID $DATASET $SEED $PERCENTAGE $LOGFOLDER $PLAIN $nlength >> logs/${LOGFOLDER}/tmux_${CLIENT_SESSION}.log 2>&1"
            sleep 4
        done
    done

    echo "Aguardando término da execução BFV..."

    while true; do
        if ! tmux has-session -t "$SERVER_SESSION" 2>/dev/null; then
            break
        fi
        sleep 5
    done
    
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null
    for i in $(seq 0 $((NCLIENTES - 1))); do
        tmux kill-session -t "client-$i" 2>/dev/null
    done

    if [ -f "$ERROR_FLAG" ]; then return 1; else return 0; fi
}

# ================================
# CONFIGURAÇÕES BFV: POLY, COEFFS, PLAIN_MODULUS
# ================================
bfv_configs=(
    "8192 50,50,50,50 1073872897"
)

KEY_TYPEs=('None')

run_combo_keys() {
    local type="$1"
    local json="$2"
    python3 - <<EOF
import json
from generate_keys import combo_keys
args_dict = json.loads('''$json''')
combo_keys("$type", args=args_dict)
EOF
}

# ------------------ LOOP PRINCIPAL BFV -------------------------
for entry in "${bfv_configs[@]}"; do
    POLY=$(echo "$entry" | awk '{print $1}')
    COEFFS=$(echo "$entry" | awk '{print $2}')
    plain=$(echo "$entry" | awk '{print $3}') 
    SCALE=0 

    for KEY_TYPE in "${KEY_TYPEs[@]}"; do
json=$(cat <<EOF
{
"poly_modulus_degree": $POLY,
"coeff_mod_bit_sizes": [$COEFFS],
"plain_modulus": $plain,
"key_type": "$KEY_TYPE"
}
EOF
)   
        echo "Gerando BFV → POLY=$POLY COEFFS=[$COEFFS] PLAIN=$plain key=$KEY_TYPE"       
        # run_combo_keys "bfv" "$json"
        
        for SOLUTION in 'bfv'; do 
            for model in "${!dict[@]}"; do
                for dataset in ${dict[$model]}; do
                    for niid in "${NIIDS[@]}"; do
                        for SEED in "${SEEDS[@]}"; do
                            
                            if [[ "$dataset" == "tanganke/gtsrb" ]]; then
                                dataset_folder="tanganke_gtsrb"
                            else
                                dataset_folder="$dataset"
                            fi
                            
                            PERCENTAGE=-1
           
                            TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
                            LOGFOLDER=${model}_${dataset_folder}_${SOLUTION}_${TECHNIQUE}_${NUM_ROUNDS}rounds_${NCLIENTES}clients_seed${SEED}_niid${niid}_percentage${PERCENTAGE}_${POLY}_${plain}_${nlength}_${COEFFS}_${SCALE}_${KEY_TYPE}_${TIMESTAMP}

                            
                            mkdir -p "logs/${LOGFOLDER}/model"
                            chmod -R 777 logs/${LOGFOLDER}

                            run_experiment_tmux $POLY $plain $COEFFS $SCALE $KEY_TYPE \
                                                $model $dataset $niid $SEED \
                                                $SOLUTION $PERCENTAGE \
                                                $NUM_ROUNDS $NCLIENTES $CLIENTS_PER_GPU $TIMESTAMP $LOGFOLDER

                            tmux kill-session -t "clearlogs" 2>/dev/null
                            tmux new-session -d -s "clearlogs" "./clean_logs.sh" 2>"./logs/tmux_clearlogs.log"
                            sleep 1
                            mkdir -p mv_logs
                            mv logs/${LOGFOLDER} mv_logs/
                        done
                    done
                done
            done 
        done 
    done
done

docker run -it -v $PWD/logs:/logs flhe:client bash -c 'chmod 777 -R  logs/*'
curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id=$CHAT_ID -d text="🎉 Todas as execuções BFV foram concluídas! 🤖"