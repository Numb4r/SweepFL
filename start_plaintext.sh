#!/bin/bash
echo "Iniciando experimentos com Plaintext - Sem criptografia"
declare -A dict

dict["resnet20"]="CIFAR10 tanganke/gtsrb" #CIFAR10
dict["lenet5"]="FASHION_MNIST"
dict['DNN']="MNIST"
NIIDS=(1) #0.1 0.5 1
SEEDS=(42) # 42
# FALTANDO


# lenet FASHION_MNIST 0.5 
 
# resnet GTSRB 0.5 



GPU_IDS=(5 6 7)
# Quantos clientes por GPU
CLIENTS_PER_GPU=3
# Quantidade total de GPUs utilizáveis
NGPUS=${#GPU_IDS[@]}
# Número total de clientes = GPUs * clientes por GPU
NCLIENTES=$((NGPUS * CLIENTS_PER_GPU))
# Número de rodadas
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
SOLUTION="plaintext"
NCLIENTES=9
NUM_ROUNDS=1
poly=0
coeffs="0,0,0"
scale=0
plain=-1
nlength=-1
key_type="placeholder"
PERCENTAGE=0
TECHNIQUE="none"

run_experiment_tmux() {

    local SERVER_SESSION="server"
    local SEED=$SEED
    local DATASET=$dataset
    local niid=$niid
    local LOGFOLDER=$LOGFOLDER
    # Mata sessão antiga do servidor
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null

    echo "[SERVER] Iniciando sessão tmux: $SERVER_SESSION"

    tmux new-session -d -s "$SERVER_SESSION" \
        "./run_server.sh $SOLUTION $NCLIENTES $NUM_ROUNDS $poly $coeffs $scale $key_type $DATASET $niid $SEED $PERCENTAGE $LOGFOLDER >> logs/${LOGFOLDER}/tmux_${SERVER_SESSION}.log 2>&1"

    # ============================
    # CLIENTES
    # ============================
    for i in $(seq 0 $((NCLIENTES - 1))); do

        local CLIENT_SESSION="client-$i"
        local TECHNIQUE="topk"

        # Seleciona GPU física conforme posição
        local GPU_INDEX=$(( i / CLIENTS_PER_GPU ))
        local GPU_ID=${GPU_IDS[$GPU_INDEX]}

        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null

        while ! tmux has-session -t "$CLIENT_SESSION" 2>/dev/null; do
            echo "[CLIENT] Iniciando sessão: $CLIENT_SESSION → GPU $GPU_ID NIID $niid"

            tmux new-session -d -s "$CLIENT_SESSION" \
                "./run_client.sh $i $SOLUTION $TECHNIQUE $NCLIENTES $GPU_ID $poly $coeffs $scale $key_type $niid $DATASET $SEED $PERCENTAGE $LOGFOLDER >> logs/${LOGFOLDER}/tmux_${CLIENT_SESSION}.log 2>&1"

            sleep 4
        done
    done

    echo "Experimentos iniciados para ${SOLUTION} com poly=$poly coeffs=[$coeffs]"
    echo "Executando com scale=2^$power=$scale key_type=$key_type"
    echo "Todas as sessões tmux iniciadas."
    echo "Aguardando término da execução..."

    # ============================
    # AGUARDA O FIM DO SERVIDOR
    # ============================
    while tmux has-session -t "$SERVER_SESSION" 2>/dev/null; do
        sleep 10
    done

    echo "Execução finalizada."
}
mkdir -p mv_logs
# self.model_name}_{dataset_folder}_{self.solution}_{self.technique}_{self.num_clients}_{self.seed}_{self.dirichlet_alpha}_{self.percentage}_{self.poly_modulus_degree}_{self.plain_modulus}_{self.n_length}_{self.coeff_mod_bit_sizes}_{self.global_scale}_{self.key_type}_{timestamp_experiment}
for model in "${!dict[@]}"; do
    for dataset in ${dict[$model]}; do
        for niid in "${NIIDS[@]}"; do #
            for SEED in "${SEEDS[@]}"; do
                echo "Verificando existência de logs para $model com dataset $dataset niid=$niid"
                if [[ "$dataset" == "tanganke/gtsrb" ]]; then
                    dataset_folder="tanganke_gtsrb"
                else
                    dataset_folder="$dataset"
                fi
                TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
                LOGFOLDER=${model}_${dataset_folder}_${SOLUTION}_${TECHNIQUE}_${NUM_ROUNDS}rounds_${NCLIENTES}clients_seed${SEED}_niid${niid}_percentage${PERCENTAGE}_${poly}_${plain}_${nlength}_${coeffs}_${scale}_${key_type}_${TIMESTAMP}
                mkdir -p "logs/${LOGFOLDER}/model"
                chmod -R 777 logs/${LOGFOLDER}
                echo -e '{ 
                        "POLY":'$poly', \n 
                        "PLAIN":'$plain', \n
                        "COEFFS":['$coeffs'], \n 
                        "SCALE":'$scale', \n 
                        "KEY_TYPE":"'$key_type'", \n 
                        "model":"'$model'", \n 
                        "DATASET":"'$dataset'", \n 
                        "NIID":'$niid' , \n 
                        "SEED":'$SEED', \n
                        "SOLUTION":"'$SOLUTION'", \n
                        "PERCENTAGE":'$PERCENTAGE', \n
                        "NUM_ROUNDS":'$NUM_ROUNDS',\n
                        "NCLIENTES":'$NCLIENTES', \n
                        "CLIENTS_PER_GPU":'$CLIENTS_PER_GPU', \n
                        "EXPERIMENT_DATE":"'$TIMESTAMP'" \n
                         
                        }' > logs/$LOGFOLDER/experiment_details.json
                
                run_experiment_tmux  "$dataset" "$niid" "$SEED" "$LOGFOLDER"
                # chmod -R 777 logs/*.log
                tmux new-session -d -s "clearlogs" "./clean_logs.sh"2>"./logs/tmux_clearlogs.log"
                sleep 1
                mv logs/${LOGFOLDER} mv_logs/
                # echo "Logs movidos para mv_logs/${LOGFOLDER}"
            done
        done
    done
done


#  if [ ! -d "logs/models/init_weights/$model/$dataset_folder" ]; then
#                     echo "Pasta não existe"
#                     run_experiment_tmux  "$dataset" "$niid" "$SEED"

#                 fi