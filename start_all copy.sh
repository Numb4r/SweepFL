#!/bin/bash
TOKEN="7947375786:AAHmPy5ZlI2dhxXBEOC7cOVMb8fLK5ZbaWo"
CHAT_ID="111824133"

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
NUM_ROUNDS=50
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
# Todas as configs (scheme poly coeffs)
# train_topk-0.8-1-MNIST-fedphe-4096-1032193-[40, 20, 40]-65536.0-all-2048.csv

# Definir modelo inicial
#salvar plaintext por round (aggregado) 
# comparar o erro

# reunir os grafico de acordo com o coeef, nao soo poly
ckks_configs=(
    # "ckks 8192 16,16,16"
    # "ckks 4096 16,16,16"
    # "ckks 8192 40,16,40"
    # "ckks 4096 17,17,17"
    # "ckks 8192 45,45"

    # "ckks 4096 40,40"

    # "ckks 4096 40,40"

    # "ckks 8192 30,20,20,30 18" # descer scale pra 2**12 depois

    # "ckks 8192 50,25,25,50"
    # "ckks 8192 60,40,40,60"


    # "ckks 8192 40, 25, 25, 40" #scale 25
#    "ckks 4096 18,18,18" # scale 16 stress test
   # "ccks 4096 25, 20, 25" scale 15 
   # "ckks 8192 25, 20, 20, 25" scale 17
    


    


    # "ckks 8192 40,21,21,21,21,21,21,21,21,40"
    # "ckks 16384 21,21,21,21,21,21,21,21"
    # "ckks 16384 40,21,21,21,21,21,21,40"
    # "ckks 16384 60,40,40,60"

    # Fazer esses
    # "ckks 8192 21,21,21,21,21,21,21,21"
    # "ckks 8192 40,21,21,21,21,21,21,40"
    # "ckks 8192 21,21,21,21,21,21,21"

    # "ckks 8192 30,20,20,30"


    # "ckks 8192 20,20,20"



    # "ckks 8192 60,17,60"


    # "ckks 8192 30,18,30"
    # "ckks 8192 30,18,30"
    "ckks 8192 60,40,40,60"

    
    # "ckks 8192 60,17,60"
    # "ckks 8192 40,17,40"
    # "ckks 8192 60,40,40,60"
    
    
    
    # "ckks 8192 60,17,60 35"

    # "ckks 8192 60,40,40,60"
    # "ckks 2048 18,18 16"


    # "ckks 2048 18,18,18 14"
    # "ckks 8192 40,21,21,21,21,21,40"
    # "ckks 8192 21,21,21,21,21,21"
    
    # "ckks 8192 40,21,21,21,21,40"
    # "ckks 8192 21,21,21,21,21"
    # "ckks 8192 40,21,21,21,40"
    # "ckks 8192 21,21,21,21"
    # "ckks 8192 21,21,21"

    # "ckks 8192 21,21"
    
    # esses
    # "ckks 8192 20,20,20"

    # "ckks 8192 17,17"


    # "ckks 8192 30,20,20,30"
    # "ckks 4096 30,20,20,30"

    # "ckks 4096 40,20,40"
    # "ckks 4096 30,20,30"
    # "ckks 4096 20,20,20"
    # "ckks 4096 19,19,19"
    # "ckks 4096 18,18,18"
   
    # "ckks 4096 18,18"
    # "ckks 4096 17,17"

    # "ckks 2048 20,20"
    # "ckks 2048 18,18"
    # "ckks 2048 16,16"

    # "bfv 8192 40,21,21,21,21,21,21,40"
    # "bfv 8192 40,21,21,21,21,21,40"
    # "bfv 8192 40,21,21,21,21,40"
    # "bfv 8192 40,21,21,21,40"
    # "bfv 8192 40,21,21,40"
    # "bfv 8192 40,20,40"
    # "bfv 4096 40,20,40"
    # "bfv 4096 30,20,30"
    # "bfv 4096 20,20,20"
    # "bfv 4096 19,19,19"
    # "bfv 4096 18,18,18"
    # "bfv 2048 20,20"
)

# key_types=("all" "galois" "relin" "")
key_types=("galois")


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
run_experiment_tmux() {

    local SERVER_SESSION="server"

    # Mata sessão antiga do servidor
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null

    echo "[SERVER] Iniciando sessão tmux: $SERVER_SESSION"

    tmux new-session -d -s "$SERVER_SESSION" \
        "./run_server.sh $SOLUTION $NCLIENTES $NUM_ROUNDS $poly $coeffs $scale $key_type $DATASET $niid >> logs/tmux_${SERVER_SESSION}.log 2>&1"

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
                "./run_client.sh $i $SOLUTION $TECHNIQUE $NCLIENTES $GPU_ID $poly $coeffs $scale $key_type $niid $DATASET >> logs/tmux_${CLIENT_SESSION}.log 2>&1"

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




# ------------------ LOOP PRINCIPAL -------------------------
for entry in "${ckks_configs[@]}"; do
    scheme=$(echo "$entry" | awk '{print $1}')
    poly=$(echo "$entry" | awk '{print $2}')
    coeffs=$(echo "$entry" | awk '{print $3}')
    power=$(echo "$entry" | awk '{print $4}')

    IFS=',' read -r -a coeff_array <<< "$coeffs"

    for key_type in "${key_types[@]}"; do

        if [[ "$scheme" == "ckks" ]]; then
            for margin in 0  ; do
                
                
                # margin: 4,3,2
            
                min_coeff=${coeff_array[0]}
                for c in "${coeff_array[@]}"; do
                    (( c < min_coeff )) && min_coeff=$c
                done

                power=$((min_coeff - margin))
                # power=40
            
                scale=$((2 ** power))
            
json=$(cat <<EOF
{
"poly_modulus_degree": $poly,
"coeff_mod_bit_sizes": [$coeffs],
"global_scale": $scale,
"key_type": "$key_type"
}
EOF
)   
# 'randall-lab/medmnist'
# 'fedphe' 'nvnl' 
 # 'AISNP/PathMNIST'
# 'ckks'  'socas'
# 'MNIST' 'CIFAR10' 'FASHION_MNIST'
# TODO RODAR FEDPHE-SCOUT E NVNL-SCOUT NO 30,18,30
# 
                for DATASET in  'CIFAR10''FASHION_MNIST'  'MNIST'  ; do
                    for SOLUTION in  'ckks' 'nvnl' 'fedphe'   ; do
                        for niid in 0.1 0.5 1 ; do
                            for seed in 42; do
                                if [[ "$SOLUTION" == *"fedphe"* || "$SOLUTION" == *"nvnl"* ]]; then
                                    # Rodar apenas para fedphe e nvnl
                                    for PERCENTAGE in 0.8 0.5 0.3; do
                                        # echo "Rodando $SOLUTION com $DATASET niid=$niid percentage=$PERCENTAGE"
                                        run_experiment_tmux  $SOLUTION $NCLIENTES $NUM_ROUNDS $POLY $COEFFS $SCALE $KEY_TYPE $DATASET $NIID $SEED $PERCENTAGE
                                    done
                                else 
                                    run_experiment_tmux $SOLUTION $NCLIENTES $NUM_ROUNDS $POLY $COEFFS $SCALE $KEY_TYPE $DATASET $NIID $SEED 0.8
                                fi
                            done
                        done #niid
                    done #Solution
                done #dataset

            done #scheme

        elif [[ "$scheme" == "bfv" ]]; then

json=$(cat <<EOF
{
  "poly_modulus_degree": $poly,
  "plain_modulus": 1032193,
  "key_type": "$key_type"
}
EOF
)
echo "Gerando BFV → poly=$poly key=$key_type"
                run_combo_keys "bfv" "$json"
                    SOLUTION="bfv"
                    coeffs="0,0"
                    scale=0
                    SERVER_SESSION="server"
                    tmux kill-session -t $SERVER_SESSION 2>/dev/null
                    echo "[SERVER] Iniciando sessão tmux: $SERVER_SESSION"
                    tmux new-session -d -s ${SERVER_SESSION} "./run_server.sh $SOLUTION $NCLIENTES $NUM_ROUNDS"
                    for i in $(seq 0 $((NCLIENTES - 1))); do
                        CLIENT_SESSION="client-$i"  
                        TECHNIQUE="topk"
                        # Seleciona GPU física conforme posição na lista
                        GPU_INDEX=$(( i / CLIENTS_PER_GPU ))
                        GPU_ID=${GPU_IDS[$GPU_INDEX]}
                        echo $CLIENT_SESSION
                        tmux kill-session -t ${CLIENT_SESSION} 2>/dev/null
                        echo "[CLIENT] Iniciando sessão: $CLIENT_SESSION → GPU $GPU_ID"
                        tmux new-session -d -s ${CLIENT_SESSION} "./run_client.sh $i $SOLUTION $TECHNIQUE $NCLIENTES $GPU_ID $poly $coeffs $scale $key_type "
                        
                    done

                    echo "Todas as sessões tmux iniciadas para $SOLUTION / $TECHNIQUE."
                    echo "Aguardando término da execução..."

                    # ============================
                    # AGUARDA O FIM DAS SESSÕES
                    # ============================
                    while tmux has-session -t $SERVER_SESSION 2>/dev/null; do
                        sleep 10
                    done
                

        elif [[ "$scheme" == "Paillier" ]]; then
            echo "Configuração não reconhecida: $entry"
        else
            echo "Esquema desconhecido: $scheme"
        fi
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