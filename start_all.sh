#!/bin/bash
# Runs the full FL experiment grid using Docker containers in tmux sessions.

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit this section to define your experiments
# ══════════════════════════════════════════════════════════════════

# -- GPU layout -------------------------------------------------------
GPU_IDS=(5 6 7)        # Physical GPU indices to use
CLIENTS_PER_GPU=3      # Clients per GPU

# -- Experiment grid --------------------------------------------------
SOLUTIONS=('nvnl')     # ckks | bfv | batchcrypt | fedphe | nvnl | plaintext | sw | rw
NIIDS=(0.1)            # Dirichlet α  (0.1 = non-IID, 1.0 ≈ IID)
SEEDS=(42)
PERCENTAGES=(0.1 0.3)  # Pack fraction  (packing solutions only: fedphe, nvnl, sw, rw)
NUM_ROUNDS=50
TECHNIQUE="none"       # topk | robin_round | slided_window | none

declare -A MODEL_DATASET
MODEL_DATASET["resnet20"]="CIFAR10"
# MODEL_DATASET["lenet5"]="FASHION_MNIST"
# MODEL_DATASET["DNN"]="MNIST"

# -- HE key parameters (TenSEAL / shared) ------------------------------------
CKKS_CONFIGS=(
    "ckks 8192 30,60,30"
    # "ckks 8192 60,40,40,60"
)
KEY_TYPES=('None')
SCALE_POWER=60         # SCALE = 2^SCALE_POWER

PLAIN_MODULUS=-1       # BFV plain modulus   (-1 if unused)
N_LENGTH=-1            # Paillier key length (-1 if unused)

# -- CKKS backend selection ---------------------------------------------------
# Valid values: tenseal | ckks_python_simplify | ckks_python | openfhe
CKKS_BACKEND="tenseal"

# ckks_python_simplify params (used when CKKS_BACKEND=ckks_python_simplify)
SIMPLIFY_N=8192
SIMPLIFY_BITS=62
SIMPLIFY_SCALE_BITS=60

# ckks_python params (used when CKKS_BACKEND=ckks_python)
PYCRYPT_N=8192
PYCRYPT_BITS=60

# OpenFHE params (used when CKKS_BACKEND=openfhe)
OPENFHE_RING_DIM=8192
OPENFHE_MULT_DEPTH=1
OPENFHE_SCALE_BITS=50

# -- Telegram notification --------------------------------------------
TELEGRAM_TOKEN="7947375786:AAHmPy5ZlI2dhxXBEOC7cOVMb8fLK5ZbaWo"
TELEGRAM_CHAT="111824133"

# ══════════════════════════════════════════════════════════════════
#  DERIVED VALUES
# ══════════════════════════════════════════════════════════════════
NCLIENTES=$(( ${#GPU_IDS[@]} * CLIENTS_PER_GPU ))

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

notify() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
         -d chat_id="$TELEGRAM_CHAT" -d text="$1" > /dev/null
}

# Returns 0 if the solution uses packing + selective encryption
is_packing_solution() {
    case "$1" in fedphe|nvnl|fedphe-scout|nvnl-scout|sw|rw) return 0 ;; esac
    return 1
}

# Generates HE keys by calling generate_keys.py
generate_keys() {
    local schema="$1" poly="$2" coeffs="$3" scale="$4" key_type="$5"
    echo "Generating $schema keys: backend=$CKKS_BACKEND poly=$poly coeffs=[$coeffs] scale=$scale"
    python3 - <<EOF
from generate_keys import combo_keys
combo_keys("$schema", args={
    # TenSEAL / shared
    "poly_modulus_degree": $poly,
    "coeff_mod_bit_sizes": [$coeffs],
    "plain_modulus":       $PLAIN_MODULUS,
    "global_scale":        $scale,
    "key_type":            "$key_type",
    "n_length":            $N_LENGTH,
    # Backend selector
    "ckks_backend":        "$CKKS_BACKEND",
    # ckks_python_simplify
    "simplify_N":          $SIMPLIFY_N,
    "simplify_bits":       $SIMPLIFY_BITS,
    "simplify_scale_bits": $SIMPLIFY_SCALE_BITS,
    # ckks_python
    "pycrypt_N":           $PYCRYPT_N,
    "pycrypt_bits":        $PYCRYPT_BITS,
    # OpenFHE
    "openfhe_ring_dim":    $OPENFHE_RING_DIM,
    "openfhe_mult_depth":  $OPENFHE_MULT_DEPTH,
    "openfhe_scale_bits":  $OPENFHE_SCALE_BITS,
})
EOF
}

# Starts server + all clients in tmux, waits for the server to finish, then cleans up.
# All experiment parameters are read from global variables.
run_experiment() {
    local LOGFOLDER="$1"
    local SERVER_SESSION="SERVER"

    mkdir -p "logs/${LOGFOLDER}/model"
    chmod -R 777 "logs/${LOGFOLDER}"

    cat > "logs/${LOGFOLDER}/experiment_details.json" <<JSON
{
  "SOLUTION":        "$SOLUTION",
  "MODEL":           "$MODEL",
  "DATASET":         "$DATASET",
  "NIID":            $NIID,
  "SEED":            $SEED,
  "PERCENTAGE":      $PERCENTAGE,
  "TECHNIQUE":       "$TECHNIQUE",
  "NUM_ROUNDS":      $NUM_ROUNDS,
  "NCLIENTES":       $NCLIENTES,
  "CLIENTS_PER_GPU": $CLIENTS_PER_GPU,
  "POLY":            $POLY,
  "COEFFS":          [$COEFFS],
  "SCALE":           $SCALE,
  "KEY_TYPE":        "$KEY_TYPE",
  "PLAIN_MODULUS":   $PLAIN_MODULUS,
  "N_LENGTH":        $N_LENGTH
}
JSON

    # Launch server
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SERVER_SESSION" \
        "./run_server.sh $SOLUTION $NCLIENTES $NUM_ROUNDS $POLY $COEFFS $SCALE $KEY_TYPE \
         $DATASET $NIID $SEED $PERCENTAGE $LOGFOLDER $PLAIN_MODULUS $N_LENGTH \
         >> logs/${LOGFOLDER}/tmux_SERVER.log 2>&1"

    # Launch clients (staggered by 4 s to let the server come up)
    for i in $(seq 0 $(( NCLIENTES - 1 ))); do
        local GPU_ID=${GPU_IDS[$(( i / CLIENTS_PER_GPU ))]}
        local CLIENT_SESSION="client-$i"

        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null || true
        tmux new-session -d -s "$CLIENT_SESSION" \
            "./run_client.sh $i $SOLUTION $TECHNIQUE $NCLIENTES $GPU_ID \
             $POLY $COEFFS $SCALE $KEY_TYPE $NIID $DATASET $SEED \
             $PERCENTAGE $LOGFOLDER $PLAIN_MODULUS $N_LENGTH \
             >> logs/${LOGFOLDER}/tmux_${CLIENT_SESSION}.log 2>&1"
        sleep 4
    done

    echo "Waiting for experiment: $LOGFOLDER"
    while tmux has-session -t "$SERVER_SESSION" 2>/dev/null; do
        sleep 5
    done

    for i in $(seq 0 $(( NCLIENTES - 1 ))); do
        tmux kill-session -t "client-$i" 2>/dev/null || true
    done

    echo "[OK] $LOGFOLDER"
}

# ══════════════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════════════
mkdir -p context logs logs/models

echo "GPU mapping (${NCLIENTES} clients across GPUs: ${GPU_IDS[*]}):"
for i in $(seq 0 $(( NCLIENTES - 1 ))); do
    echo "  client-$i → GPU ${GPU_IDS[$(( i / CLIENTS_PER_GPU ))]}"
done

# ══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════
for config in "${CKKS_CONFIGS[@]}"; do
    SCHEMA=$(awk '{print $1}' <<< "$config")
    POLY=$(awk '{print $2}' <<< "$config")
    COEFFS=$(awk '{print $3}' <<< "$config")
    SCALE=$(( 2 ** SCALE_POWER ))

    for KEY_TYPE in "${KEY_TYPES[@]}"; do

        generate_keys "$SCHEMA" "$POLY" "$COEFFS" "$SCALE" "$KEY_TYPE"

        for SOLUTION in "${SOLUTIONS[@]}"; do
        for MODEL in "${!MODEL_DATASET[@]}"; do
        for DATASET in ${MODEL_DATASET[$MODEL]}; do
        for NIID in "${NIIDS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            DATASET_FOLDER="${DATASET//\//_}"

            if is_packing_solution "$SOLUTION"; then
                ITER_PERCENTAGES=("${PERCENTAGES[@]}")
            else
                ITER_PERCENTAGES=(-1)
            fi

            for PERCENTAGE in "${ITER_PERCENTAGES[@]}"; do
                TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
                LOGFOLDER="${MODEL}_${DATASET_FOLDER}_${SOLUTION}_${TECHNIQUE}_${NUM_ROUNDS}rounds_${NCLIENTES}clients_seed${SEED}_niid${NIID}_percentage${PERCENTAGE}_${POLY}_${PLAIN_MODULUS}_${N_LENGTH}_${COEFFS}_${SCALE}_${KEY_TYPE}_${TIMESTAMP}"

                run_experiment "$LOGFOLDER"

                tmux kill-session -t "clearlogs" 2>/dev/null || true
                tmux new-session -d -s "clearlogs" "./clean_logs.sh" 2>/dev/null
                sleep 1
                mkdir -p mv_logs
                mv "logs/${LOGFOLDER}" mv_logs/
            done

        done; done; done; done; done

    done
done

notify "All experiments finished 🎉"
echo "Done."
