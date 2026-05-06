#!/bin/bash
TOKEN="7947375786:AAHmPy5ZlI2dhxXBEOC7cOVMb8fLK5ZbaWo"
CHAT_ID="111824133"
declare -A dict

dict["resnet20"]="CIFAR10" #CIFAR10 tanganke/gtsrb
# dict["lenet5"]="FASHION_MNIST"
# dict['DNN']="MNIST"
NIIDS=(1) #0.1 0.5 1
SEEDS=(42) # 42
PERCENTAGES=( 0.8 ) # 0.3 0.5 0.8
NCLIENTES=9
NUM_ROUNDS=50
# DNN niid 0.1 percentage 0.1
# resnet CIFAR10 niid1 percentage 0.8


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
# IDs das GPUs que podem ser usadas (exemplo: 0 4 8)
GPU_IDS=(5 6 7)
# Quantos clientes por GPU
CLIENTS_PER_GPU=3
# Quantidade total de GPUs utilizáveis
NGPUS=${#GPU_IDS[@]}
# Número total de clientes = GPUs * clientes por GPU
NCLIENTES=$((NGPUS * CLIENTS_PER_GPU))
# Número de rodadas



mkdir -p logs
mkdir -p logs/models
chmod -R 777 logs/


# ================================
# FUNÇÃO: mostra o mapeamento cliente → GPU real
# ================================
echo ""
echo "=========================================="
echo "MAPEAMENTO DE CLIENTES → GPUS (selecionadas: ${GPU_IDS[*]})"
echo "=========================================="
for i in $(seq 0 $((NCLIENTES - 1))); do
  # Define qual GPU física usar com base na lista GPU_IDS
  GPU_INDEX=$(( i / CLIENTS_PER_GPU ))
  GPU_ID=${GPU_IDS[$GPU_INDEX]}
  echo "CLIENTE $i → GPU física $GPU_ID"
done
echo "=========================================="
echo ""


run_experiment_tmux() {
    tmux kill-session -t "notify" 2>/dev/null

    local POLY=$POLY
    local PLAIN=$plain
    local COEFFS=$COEFFS
    local SCALE=$SCALE
    local KEY_TYPE=$KEY_TYPE
    local model=$model
    local DATASET=$dataset
    local NIID=$niid
    local SEED=$SEED
    local TECHNIQUE=$TECHNIQUE
    local SOLUTION=$SOLUTION
    local PERCENTAGE=$PERCENTAGE
    local NUM_ROUNDS=$NUM_ROUNDS
    local NCLIENTES=$NCLIENTES
    local CLIENTS_PER_GPU=$CLIENTS_PER_GPU
    local EXPERIMENT_DATE=$TIMESTAMP

    local LOGFOLDER=$LOGFOLDER
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
    # Mata sessão antiga do servidor
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null

    echo "[SERVER] Iniciando sessão tmux: $SERVER_SESSION"

    tmux new-session -d -s "$SERVER_SESSION" \
        "./run_server.sh $SOLUTION $NCLIENTES $NUM_ROUNDS $POLY $COEFFS $SCALE $KEY_TYPE $DATASET $NIID $SEED $PERCENTAGE $LOGFOLDER >> logs/${LOGFOLDER}/tmux_${SERVER_SESSION}.log 2>&1"

    # ============================
    # CLIENTES
    # ============================
    for i in $(seq 0 $((NCLIENTES - 1))); do

        local CLIENT_SESSION="client-$i"

        # Seleciona GPU física conforme posição
        local GPU_INDEX=$(( i / CLIENTS_PER_GPU ))
        local GPU_ID=${GPU_IDS[$GPU_INDEX]}

        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null

        while ! tmux has-session -t "$CLIENT_SESSION" 2>/dev/null; do
            echo "[CLIENT] Iniciando sessão: $CLIENT_SESSION → GPU $GPU_ID NIID $niid"

            tmux new-session -d -s "$CLIENT_SESSION" \
                "./run_client.sh $i $SOLUTION $TECHNIQUE $NCLIENTES $GPU_ID $POLY $COEFFS $SCALE $KEY_TYPE $NIID $DATASET $SEED $PERCENTAGE $LOGFOLDER >> logs/${LOGFOLDER}/tmux_${CLIENT_SESSION}.log 2>&1"

            sleep 4
        done
    done

    echo "Experimentos iniciados para ${SOLUTION} com POLY=$POLY COEFFS=[$COEFFS]"
    echo "Executando com SCALE=2^$POWER=$SCALE KEY_TYPE=$KEY_TYPE"
    echo "Todas as sessões tmux iniciadas."
    echo "Aguardando término da execução..."
    echo "Disparando notificações para erros (Traceback) a cada $INTERVAL segundos..."
    echo "Logs monitorados: logs/${LOGFOLDER}/*.log"
    tmux new-session -d -s "notify" "bash notify.sh $LOGFOLDER"
    # ============================
    # AGUARDA O FIM DO SERVIDOR
    # ============================
    while tmux has-session -t "$SERVER_SESSION" 2>/dev/null; do
        sleep 10
    done
    tmux kill-session -t "notify" 2>/dev/null
    echo "Sessão do servidor finalizada. Encerrando monitoramento de logs."
        if [ -f "$ERROR_FLAG" ]; then
            echo "⚠️  Experimento falhou! Verifique os logs em logs/${LOGFOLDER}/ para detalhes."
            curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
            -d chat_id=$CHAT_ID \
            -d text="⚠️ Experimento falhou! Verifique os logs em logs/${LOGFOLDER}/ para detalhes."
        else
            echo "✅ Experimento concluído com sucesso! Logs disponíveis em logs/${LOGFOLDER}/"
            curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
            -d chat_id=$CHAT_ID \
            -d text="✅ Experimento concluído com sucesso! Logs disponíveis em logs/${LOGFOLDER}/"
        fi
    echo "Execução finalizada."
}



ckks_configs=(
    #  "ckks 8192 60,40,40,60" # FEDPHE
    # 'ckks 8192 40,46,46,46,40'
    # 'ckks 8192 30,17,30'
    'ckks 8192 50,18,50'
    # 'ckks 8192 30,60,30'
    'ckks 8192 60,43,60'
    # 'ckks 8192 30,52,52,30'
    # 'ckks 8192 60,38,38,60'
    # 'ckks 8192 40,39,39,39,40'
    # 'ckks 8192 40,46,46,46,40'
    # 'ckks 8192 60,40,40,60'
)

# KEY_TYPEs=("all" "galois" "relin" "")
KEY_TYPEs=('None')


# ---------- Função que chama combo_keys() no Python ----------
run_combo_keys() {
    local type="$1"
    local json="$2"

    echo "Chamando combo_keys($type) com JSON: $json"

    python3 - <<EOF
import json
from generate_keys import combo_keys

args_dict = json.loads('''$json''')
combo_keys("$type", args=args_dict)
EOF
}





# ------------------ LOOP PRINCIPAL -------------------------
for entry in "${ckks_configs[@]}"; do
    SCHEMA=$(echo "$entry" | awk '{print $1}')
    POLY=$(echo "$entry" | awk '{print $2}')
    COEFFS=$(echo "$entry" | awk '{print $3}')
    POWER=$(echo "$entry" | awk '{print $4}')

    IFS=',' read -r -a coeff_array <<< "$COEFFS"

    for KEY_TYPE in "${KEY_TYPEs[@]}"; do

        for margin in 0  ; do
            
            
            # margin: 4,3,2
        
            min_coeff=${coeff_array[0]}
            for c in "${coeff_array[@]}"; do
                (( c < min_coeff )) && min_coeff=$c
            done

            POWER=$((min_coeff - margin))
            # POWER=40
        
            SCALE=$((2 ** POWER))
            
json=$(cat <<EOF
{
"poly_modulus_degree": $POLY,
"coeff_mod_bit_sizes": [$COEFFS],
"plain_modulus": $plain,
"global_scale": $SCALE,
"key_type": "$KEY_TYPE"
}
EOF
)   
            echo "Gerando CKKS → POLY=$POLY COEFFS=[$COEFFS] SCALE=2^$POWER=$SCALE key=$KEY_TYPE"       
            run_combo_keys "ckks" "$json"
            for SOLUTION in     'nvnl'; do # 'ckks' 'nvnl 'fedphe'
                for model in "${!dict[@]}"; do
                    for dataset in ${dict[$model]}; do
                        for niid in "${NIIDS[@]}"; do
                            for SEED in "${SEEDS[@]}"; do
                                if [[ "$dataset" == "tanganke/gtsrb" ]]; then
                                    dataset_folder="tanganke_gtsrb"
                                else
                                    dataset_folder="$dataset"
                                fi
                                TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
                                if [[ "$SOLUTION" == *"fedphe"* || "$SOLUTION" == *"nvnl"* ]]; then
                                    # Rodar apenas para fedphe e nvnl

                                    for PERCENTAGE in "${PERCENTAGES[@]}"; do
                                        LOGFOLDER=${model}_${dataset_folder}_${SOLUTION}_${TECHNIQUE}_${NUM_ROUNDS}rounds_${NCLIENTES}clients_seed${SEED}_niid${niid}_percentage${PERCENTAGE}_${POLY}_${plain}_${nlength}_${COEFFS}_${SCALE}_${KEY_TYPE}_${TIMESTAMP}
                                        mkdir -p "logs/${LOGFOLDER}/model"
                                        chmod -R 777 logs/${LOGFOLDER}
                                        run_experiment_tmux $POLY  $plain $COEFFS  $SCALE  $KEY_TYPE  $model  $dataset  $niid  $SEED $SOLUTION $PERCENTAGE $NUM_ROUNDS $NCLIENTES $CLIENTS_PER_GPU $TIMESTAMP
                                        tmux new-session -d -s "clearlogs" "./clean_logs.sh"2>"./logs/tmux_clearlogs.log"
                                        sleep 1
                                        mv logs/${LOGFOLDER} mv_logs/
                                    done
                                else
                                    PERCENTAGE=-1
                                    LOGFOLDER=${model}_${dataset_folder}_${SOLUTION}_${TECHNIQUE}_${NUM_ROUNDS}rounds_${NCLIENTES}clients_seed${SEED}_niid${niid}_percentage${PERCENTAGE}_${POLY}_${plain}_${nlength}_${COEFFS}_${SCALE}_${KEY_TYPE}_${TIMESTAMP}
                                    mkdir -p "logs/${LOGFOLDER}/model"
                                    chmod -R 777 logs/${LOGFOLDER}
                                    run_experiment_tmux $POLY  $plain $COEFFS  $SCALE  $KEY_TYPE  $model  $dataset  $niid  $SEED $SOLUTION $PERCENTAGE $NUM_ROUNDS $NCLIENTES $CLIENTS_PER_GPU $TIMESTAMP
                                     tmux new-session -d -s "clearlogs" "./clean_logs.sh"2>"./logs/tmux_clearlogs.log"
                                    sleep 1
                                    mv logs/${LOGFOLDER} mv_logs/

                                fi     
                                
                               
                            done
                        done
                    done
                done 
        done #SCHEMA
    done
done
done
echo "=========================================="
echo "Dando as permissões necessarias!"
echo "=========================================="

docker run -it \
   -v $PWD/logs:/logs \
   flhe:client \
    bash -c 'chmod 777 -R  logs/*'

echo "=========================================="
echo "Todas as execuções foram concluídas!"
echo "=========================================="

curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
  -d chat_id=$CHAT_ID \
  -d text="🎉 Todas as execuções foram concluídas! 🤖"
echo "Fim do script."