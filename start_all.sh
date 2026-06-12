#!/bin/bash
# Runs the full FL experiment grid using Docker containers in tmux sessions.

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit this section to define your experiments
# ══════════════════════════════════════════════════════════════════

# -- Container runtime ------------------------------------------------
RUNTIME="docker"       # docker | singularity
SIF_DIR="."            # path to .sif files (singularity only)
SERVER_HOST=""         # override server hostname (leave empty for auto-detect:
                       #   docker → host.docker.internal, singularity → localhost)

# -- GPU layout -------------------------------------------------------
GPU_IDS=(5 6 7)        # Physical GPU indices to use
CLIENTS_PER_GPU=3      # Clients per GPU

# -- Experiment grid --------------------------------------------------
SOLUTIONS=('nvnl')     # ckks | bfv | batchcrypt | fedphe | nvnl | plaintext | sw | rw | topkrr | srr
NIIDS=(0.1)            # Dirichlet α  (0.1 = non-IID, 1.0 ≈ IID)
SEEDS=(42)
PERCENTAGES=(0.1 0.3)  # Pack fraction  (packing solutions only: fedphe, nvnl, sw, rw)
NUM_ROUNDS=50
TECHNIQUE="none"       # topk | robin_round | slided_window | none

MODELS=("resnet20")
# MODELS=("resnet20" "lenet5" "dnn" "dnn_base" "dnn_tiny" "mobilenetv2" "vit_tiny" "simple_cnn")

DATASETS=("CIFAR10")
# DATASETS=("CIFAR10" "FASHION_MNIST" "MNIST" "AISNP/PathMNIST" "tanganke/gtsrb")

# Human-readable experiment label added to every log row.
# Leave empty to auto-generate as "{model}_{dataset}_{solution}".
EXPERIMENT_NAME=""

# Optional subfolder inside logs/ to group related experiments.
# e.g. FOLDER="sliding_window_ablation"  →  logs/sliding_window_ablation/{exp}/
# Leave empty to place experiments directly inside logs/.
FOLDER=""

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
# CLIENT_FRACTIONS sweep — default to (1.0) for backward-compat.
[[ ${#CLIENT_FRACTIONS[@]} -eq 0 ]] && CLIENT_FRACTIONS=(1.0)

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

notify() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
         -d chat_id="$TELEGRAM_CHAT" -d text="$1" > /dev/null
}

# Returns 0 if the solution uses packing + selective encryption
is_packing_solution() {
    case "$1" in fedphe|nvnl|fedphe-scout|nvnl-scout|sw|rw|topkrr|srr) return 0 ;; esac
    return 1
}

# Finds the best incomplete checkpoint for the current experiment config.
# Sets RESUME_LOGFOLDER (path relative to logs/) and RESUME_ROUND.
_find_resume_checkpoint() {
    RESUME_LOGFOLDER=""
    RESUME_ROUND=0

    local search_base="logs/${FOLDER:+${FOLDER}/}"
    local name_prefix="${MODEL}_${DATASET_FOLDER}_${SOLUTION}_${TECHNIQUE}_"
    # Inclui o token cf<frac> no sufixo para que runs com CLIENT_FRACTION != 1.0
    # não retomem checkpoints de runs a 100% (e vice-versa).
    local _cf_token=""
    case "${CLIENT_FRACTION:-1.0}" in 1.0|1|1.00) ;; *) _cf_token="cf${CLIENT_FRACTION}_" ;; esac
    local name_suffix="_${NCLIENTES}clients_seed${SEED}_niid${NIID}_percentage${PERCENTAGE}_${POLY}_${PLAIN_MODULUS}_${N_LENGTH}_${COEFFS}_${SCALE}_${KEY_TYPE}_${_cf_token}"

    local best_dir="" best_round=0
    for dir in "${search_base}${name_prefix}"*rounds"${name_suffix}"????-??-??_??-??-??; do
        [[ -d "$dir" ]] || continue
        local meta="${dir}/checkpoint/meta.json"
        [[ -f "$meta" ]] || continue
        local ckpt_round
        ckpt_round=$(python3 -c "import json,sys; d=json.load(open('$meta')); sys.stdout.write(str(d.get('round',0)))" 2>/dev/null) || continue
        [[ "$ckpt_round" =~ ^[0-9]+$ ]] || continue
        (( ckpt_round < NUM_ROUNDS )) || continue
        if (( ckpt_round > best_round )); then
            best_round=$ckpt_round
            best_dir="$dir"
        fi
    done

    if [[ -n "$best_dir" ]]; then
        RESUME_LOGFOLDER="${best_dir#logs/}"
        RESUME_ROUND=$best_round
        echo "[RESUME] Found checkpoint: $best_dir (round $best_round / $NUM_ROUNDS)"
    fi
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

_write_incomplete_log() {
    local logfolder="$1"
    local meta="logs/${logfolder}/checkpoint/meta.json"
    [[ -f "$meta" ]] || return 0
    local completed
    completed=$(python3 -c "import json; print(json.load(open('$meta'))['round'])" 2>/dev/null) || return 0
    [[ "$completed" =~ ^[0-9]+$ ]] || return 0
    (( completed < NUM_ROUNDS )) || return 0
    local remaining=$(( NUM_ROUNDS - completed ))
    local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
    cat > "logs/${logfolder}/INCOMPLETE.log" <<EOF
INTERRUPTED : $ts
Round       : $completed / $NUM_ROUNDS
Remaining   : $remaining
LOGFOLDER   : $logfolder
Solution    : $SOLUTION
Model       : $MODEL
Dataset     : $DATASET
Seed        : $SEED
NIID        : $NIID
Percentage  : $PERCENTAGE
EOF
    echo "[INCOMPLETE] $logfolder (round $completed / $NUM_ROUNDS)"
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
  "EXPERIMENT_NAME": "$EXP_NAME",
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
  "N_LENGTH":        $N_LENGTH,

  "CLIENT_FRACTION": ${CLIENT_FRACTION:-1.0},

  "RESUMED_FROM":    "${RESUME_LOGFOLDER:-}",
  "RESUME_ROUND":    ${RESUME_ROUND:-0}
}
JSON

    # Launch server
    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SERVER_SESSION" \
        "RUNTIME='$RUNTIME' SIF_DIR='$SIF_DIR' EXPERIMENT_NAME='$EXP_NAME' \
         RESUME_LOGFOLDER='${RESUME_LOGFOLDER:-}' \
         CLIENT_FRACTION='${CLIENT_FRACTION:-1.0}' \
         CKKS_BACKEND='${CKKS_BACKEND:-tenseal}' \
         SIMPLIFY_N='${SIMPLIFY_N:-8192}' SIMPLIFY_BITS='${SIMPLIFY_BITS:-62}' SIMPLIFY_SCALE_BITS='${SIMPLIFY_SCALE_BITS:-60}' \
         PYCRYPT_N='${PYCRYPT_N:-8192}' PYCRYPT_BITS='${PYCRYPT_BITS:-60}' \
         OPENFHE_RING_DIM='${OPENFHE_RING_DIM:-8192}' OPENFHE_MULT_DEPTH='${OPENFHE_MULT_DEPTH:-1}' OPENFHE_SCALE_BITS='${OPENFHE_SCALE_BITS:-50}' \
         ./run_server.sh $SOLUTION $NCLIENTES $NUM_ROUNDS $POLY $COEFFS $SCALE $KEY_TYPE \
         $DATASET $NIID $SEED $PERCENTAGE $LOGFOLDER $PLAIN_MODULUS $N_LENGTH \
         >> logs/${LOGFOLDER}/tmux_SERVER.log 2>&1"

    # Launch clients (staggered by 4 s to let the server come up)
    for i in $(seq 0 $(( NCLIENTES - 1 ))); do
        local GPU_ID=${GPU_IDS[$(( i / CLIENTS_PER_GPU ))]}
        local CLIENT_SESSION="client-$i"

        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null || true
        tmux new-session -d -s "$CLIENT_SESSION" \
            "RUNTIME='$RUNTIME' SIF_DIR='$SIF_DIR' SERVER_HOST='${SERVER_HOST:-}' \
             MODEL_ARCH='$MODEL' EXPERIMENT_NAME='$EXP_NAME' \
             PRETRAINED_WEIGHTS='${PRETRAINED_WEIGHTS:-}' USE_GPU='${USE_GPU:-true}' \
             RESUME_LOGFOLDER='${RESUME_LOGFOLDER:-}' \
             CKKS_BACKEND='${CKKS_BACKEND:-tenseal}' \
             SIMPLIFY_N='${SIMPLIFY_N:-8192}' SIMPLIFY_BITS='${SIMPLIFY_BITS:-62}' SIMPLIFY_SCALE_BITS='${SIMPLIFY_SCALE_BITS:-60}' \
             PYCRYPT_N='${PYCRYPT_N:-8192}' PYCRYPT_BITS='${PYCRYPT_BITS:-60}' \
             OPENFHE_RING_DIM='${OPENFHE_RING_DIM:-8192}' OPENFHE_MULT_DEPTH='${OPENFHE_MULT_DEPTH:-1}' OPENFHE_SCALE_BITS='${OPENFHE_SCALE_BITS:-50}' \
             ./run_client.sh $i $SOLUTION $TECHNIQUE $NCLIENTES $GPU_ID \
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

    _write_incomplete_log "$LOGFOLDER"
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
        for MODEL in "${MODELS[@]}"; do

            # Auto-detect pre-trained weights for this model.
            PRETRAINED_WEIGHTS=$(
                find "logs/models/pretrained/${MODEL}" -name "pretrained.weights.h5" \
                     2>/dev/null -printf "%T@ %p\n" | sort -rn | head -1 | awk '{print $2}'
            )
            [[ -n "$PRETRAINED_WEIGHTS" ]] && echo "  [pretrain] $MODEL → $PRETRAINED_WEIGHTS"

        for DATASET in "${DATASETS[@]}"; do
        for NIID in "${NIIDS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            DATASET_FOLDER="${DATASET//\//_}"
            EXP_NAME="${EXPERIMENT_NAME:-${MODEL}_${DATASET_FOLDER}_${SOLUTION}}"

            if is_packing_solution "$SOLUTION"; then
                ITER_PERCENTAGES=("${PERCENTAGES[@]}")
            else
                ITER_PERCENTAGES=(-1)
            fi

            for CLIENT_FRACTION in "${CLIENT_FRACTIONS[@]}"; do
            for PERCENTAGE in "${ITER_PERCENTAGES[@]}"; do
                # Token condicional: só inserido quando CLIENT_FRACTION != 1.0
                # para preservar retrocompatibilidade com pastas existentes (100%).
                CF_TOKEN=""
                case "${CLIENT_FRACTION:-1.0}" in 1.0|1|1.00) ;; *) CF_TOKEN="cf${CLIENT_FRACTION}_" ;; esac
                _find_resume_checkpoint
                TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
                LOGFOLDER="${FOLDER:+${FOLDER}/}${MODEL}_${DATASET_FOLDER}_${SOLUTION}_${TECHNIQUE}_${NUM_ROUNDS}rounds_${NCLIENTES}clients_seed${SEED}_niid${NIID}_percentage${PERCENTAGE}_${POLY}_${PLAIN_MODULUS}_${N_LENGTH}_${COEFFS}_${SCALE}_${KEY_TYPE}_${CF_TOKEN}${TIMESTAMP}"

                run_experiment "$LOGFOLDER"
            done
            done

        done; done; done; done; done

    done
done

notify "All experiments finished 🎉"
echo "Done."
