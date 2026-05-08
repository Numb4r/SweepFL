#!/bin/bash
# experiments/lib.sh — Shared infrastructure for themed experiment batches.
#
# HOW TO USE:
#   1. Copy experiments/template.sh → experiments/my_theme.sh
#   2. Fill in your config variables
#   3. Run from the repo root:  bash experiments/my_theme.sh

# ── Guard: must be run from repo root ─────────────────────────────
_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(dirname "$_lib_dir")"
if [[ "$PWD" != "$_repo_root" ]]; then
    echo "[lib.sh] Changing to repo root: $_repo_root"
    cd "$_repo_root" || { echo "[lib.sh] ERROR: cannot cd to $_repo_root"; exit 1; }
fi

# ── Defaults for optional backend params ──────────────────────────
SIMPLIFY_N="${SIMPLIFY_N:-8192}"
SIMPLIFY_BITS="${SIMPLIFY_BITS:-62}"
SIMPLIFY_SCALE_BITS="${SIMPLIFY_SCALE_BITS:-60}"
PYCRYPT_N="${PYCRYPT_N:-8192}"
PYCRYPT_BITS="${PYCRYPT_BITS:-60}"
OPENFHE_RING_DIM="${OPENFHE_RING_DIM:-8192}"
OPENFHE_MULT_DEPTH="${OPENFHE_MULT_DEPTH:-1}"
OPENFHE_SCALE_BITS="${OPENFHE_SCALE_BITS:-50}"

# ── Telegram ──────────────────────────────────────────────────────
TELEGRAM_TOKEN="${TELEGRAM_TOKEN:-7947375786:AAHmPy5ZlI2dhxXBEOC7cOVMb8fLK5ZbaWo}"
TELEGRAM_CHAT="${TELEGRAM_CHAT:-111824133}"

notify() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
         -d chat_id="$TELEGRAM_CHAT" -d text="$1" > /dev/null
}

# ── Helpers ───────────────────────────────────────────────────────
is_packing_solution() {
    case "$1" in fedphe|nvnl|fedphe-scout|nvnl-scout|sw|rw) return 0 ;; esac
    return 1
}

generate_keys() {
    local schema="$1" poly="$2" coeffs="$3" scale="$4" key_type="$5"
    echo "Generating $schema keys: backend=$CKKS_BACKEND poly=$poly coeffs=[$coeffs] scale=$scale"
    python3 - <<EOF
from generate_keys import combo_keys
combo_keys("$schema", args={
    "poly_modulus_degree": $poly,
    "coeff_mod_bit_sizes": [$coeffs],
    "plain_modulus":       $PLAIN_MODULUS,
    "global_scale":        $scale,
    "key_type":            "$key_type",
    "n_length":            $N_LENGTH,
    "ckks_backend":        "$CKKS_BACKEND",
    "simplify_N":          $SIMPLIFY_N,
    "simplify_bits":       $SIMPLIFY_BITS,
    "simplify_scale_bits": $SIMPLIFY_SCALE_BITS,
    "pycrypt_N":           $PYCRYPT_N,
    "pycrypt_bits":        $PYCRYPT_BITS,
    "openfhe_ring_dim":    $OPENFHE_RING_DIM,
    "openfhe_mult_depth":  $OPENFHE_MULT_DEPTH,
    "openfhe_scale_bits":  $OPENFHE_SCALE_BITS,
})
EOF
}

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
  "N_LENGTH":        $N_LENGTH
}
JSON

    tmux kill-session -t "$SERVER_SESSION" 2>/dev/null || true
    EXPERIMENT_NAME="$EXP_NAME" tmux new-session -d -s "$SERVER_SESSION" \
        "./run_server.sh $SOLUTION $NCLIENTES $NUM_ROUNDS $POLY $COEFFS $SCALE $KEY_TYPE \
         $DATASET $NIID $SEED $PERCENTAGE $LOGFOLDER $PLAIN_MODULUS $N_LENGTH \
         >> logs/${LOGFOLDER}/tmux_SERVER.log 2>&1"

    for i in $(seq 0 $(( NCLIENTES - 1 ))); do
        local GPU_ID=${GPU_IDS[$(( i / CLIENTS_PER_GPU ))]}
        local CLIENT_SESSION="client-$i"

        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null || true
        MODEL_ARCH="$MODEL" EXPERIMENT_NAME="$EXP_NAME" tmux new-session -d -s "$CLIENT_SESSION" \
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

# ── Main loop (called at the end of every theme script) ───────────
# Reads all config variables from the calling script's environment.
run_theme() {
    NCLIENTES=$(( ${#GPU_IDS[@]} * CLIENTS_PER_GPU ))

    mkdir -p context logs logs/models

    echo "══════════════════════════════════════════════"
    echo " Theme : ${FOLDER:-<no folder>}"
    echo " GPU mapping (${NCLIENTES} clients across GPUs: ${GPU_IDS[*]}):"
    for i in $(seq 0 $(( NCLIENTES - 1 ))); do
        echo "   client-$i → GPU ${GPU_IDS[$(( i / CLIENTS_PER_GPU ))]}"
    done
    echo "══════════════════════════════════════════════"

    # Count total experiments upfront
    local total=0
    for config in "${CKKS_CONFIGS[@]}"; do
      for KEY_TYPE in "${KEY_TYPES[@]}"; do
        for SOLUTION in "${SOLUTIONS[@]}"; do
          for pair in "${MODEL_DATASET_PAIRS[@]}"; do
            for NIID in "${NIIDS[@]}"; do
              for SEED in "${SEEDS[@]}"; do
                if is_packing_solution "$SOLUTION"; then
                    (( total += ${#PERCENTAGES[@]} ))
                else
                    (( total += 1 ))
                fi
              done
            done
          done
        done
      done
    done
    local done_count=0
    echo "Total experiments to run: $total"
    echo ""

    for config in "${CKKS_CONFIGS[@]}"; do
        SCHEMA=$(awk '{print $1}' <<< "$config")
        POLY=$(awk '{print $2}' <<< "$config")
        COEFFS=$(awk '{print $3}' <<< "$config")
        SCALE=$(( 2 ** SCALE_POWER ))

        for KEY_TYPE in "${KEY_TYPES[@]}"; do
            generate_keys "$SCHEMA" "$POLY" "$COEFFS" "$SCALE" "$KEY_TYPE"

            for SOLUTION in "${SOLUTIONS[@]}"; do
            for pair in "${MODEL_DATASET_PAIRS[@]}"; do
                MODEL="${pair%%:*}"
                DATASET="${pair##*:}"

                for NIID in "${NIIDS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    DATASET_FOLDER="${DATASET//\//_}"
                    EXP_NAME="${EXPERIMENT_NAME:-${MODEL}_${DATASET_FOLDER}_${SOLUTION}}"

                    if is_packing_solution "$SOLUTION"; then
                        ITER_PERCENTAGES=("${PERCENTAGES[@]}")
                    else
                        ITER_PERCENTAGES=(-1)
                    fi

                    for PERCENTAGE in "${ITER_PERCENTAGES[@]}"; do
                        (( done_count++ ))
                        TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
                        LOGFOLDER="${FOLDER:+${FOLDER}/}${MODEL}_${DATASET_FOLDER}_${SOLUTION}_${TECHNIQUE}_${NUM_ROUNDS}rounds_${NCLIENTES}clients_seed${SEED}_niid${NIID}_percentage${PERCENTAGE}_${POLY}_${PLAIN_MODULUS}_${N_LENGTH}_${COEFFS}_${SCALE}_${KEY_TYPE}_${TIMESTAMP}"

                        echo "[$done_count/$total] $MODEL @ $DATASET | $SOLUTION | pct=$PERCENTAGE | seed=$SEED | niid=$NIID"
                        run_experiment "$LOGFOLDER"
                    done

                done; done
            done; done
        done
    done

    notify "Theme '${FOLDER:-default}' finished 🎉 ($total experiments)"
    echo "Done. ($total experiments)"
}
