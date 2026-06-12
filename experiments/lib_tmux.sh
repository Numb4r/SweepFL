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

# ── Runtime ───────────────────────────────────────────────────────
RUNTIME="${RUNTIME:-docker}"   # docker | singularity
SIF_DIR="${SIF_DIR:-.}"        # directory containing .sif files (singularity only)
SERVER_HOST="${SERVER_HOST:-}" # override server hostname; empty = auto per runtime

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
    case "$1" in fedphe|nvnl|fedphe-scout|nvnl-scout|sw|rw|topkrr|srr) return 0 ;; esac
    return 1
}

# Finds the best incomplete checkpoint for the current experiment config.
# Sets RESUME_LOGFOLDER (path relative to logs/) and RESUME_ROUND.
# Requires: FOLDER, MODEL, DATASET_FOLDER, SOLUTION, TECHNIQUE, NCLIENTES,
#           SEED, NIID, PERCENTAGE, POLY, PLAIN_MODULUS, N_LENGTH, COEFFS,
#           SCALE, KEY_TYPE, NUM_ROUNDS
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

run_experiment() {
    local LOGFOLDER="$1"
    local SERVER_SESSION="SERVER"

    mkdir -p "logs/${LOGFOLDER}/model"
    chmod -R 777 "logs/${LOGFOLDER}"

    cat > "logs/${LOGFOLDER}/experiment_details.json" <<JSON
{
  "EXPERIMENT_NAME": "$EXP_NAME",
  "FOLDER":          "${FOLDER:-}",
  "TIMESTAMP":       "$TIMESTAMP",

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
  "GPU_IDS":         [$(IFS=,; echo "${GPU_IDS[*]}")],
  "RUNTIME":         "${RUNTIME:-docker}",
  "USE_GPU":         "${USE_GPU:-true}",
  "BATCH_SIZE":      ${BATCH_SIZE:-4},
  "RR_START_ROUND":    ${RR_START_ROUND:-200},
  "MAGG_RENORM":       ${MAGG_RENORM:-1},
  "CLIENT_FRACTION":   ${CLIENT_FRACTION:-1.0},
  "PRETRAINED_WEIGHTS": "${PRETRAINED_WEIGHTS:-}",

  "POLY":            $POLY,
  "COEFFS":          [$COEFFS],
  "SCALE":           $SCALE,
  "KEY_TYPE":        "$KEY_TYPE",
  "PLAIN_MODULUS":   $PLAIN_MODULUS,
  "N_LENGTH":        $N_LENGTH,

  "CKKS_BACKEND":          "${CKKS_BACKEND:-tenseal}",
  "SIMPLIFY_N":            ${SIMPLIFY_N:-8192},
  "SIMPLIFY_BITS":         ${SIMPLIFY_BITS:-62},
  "SIMPLIFY_SCALE_BITS":   ${SIMPLIFY_SCALE_BITS:-60},
  "PYCRYPT_N":             ${PYCRYPT_N:-8192},
  "PYCRYPT_BITS":          ${PYCRYPT_BITS:-60},
  "OPENFHE_RING_DIM":      ${OPENFHE_RING_DIM:-8192},
  "OPENFHE_MULT_DEPTH":    ${OPENFHE_MULT_DEPTH:-1},
  "OPENFHE_SCALE_BITS":    ${OPENFHE_SCALE_BITS:-50},

  "RESUMED_FROM":          "${RESUME_LOGFOLDER:-}",
  "RESUME_ROUND":          ${RESUME_ROUND:-0}
}
JSON

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

    for i in $(seq 0 $(( NCLIENTES - 1 ))); do
        local GPU_ID=${GPU_IDS[$(( i / CLIENTS_PER_GPU ))]}
        local CLIENT_SESSION="client-$i"

        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null || true
        tmux new-session -d -s "$CLIENT_SESSION" \
            "RUNTIME='$RUNTIME' SIF_DIR='$SIF_DIR' SERVER_HOST='${SERVER_HOST:-}' \
             MODEL_ARCH='$MODEL' EXPERIMENT_NAME='$EXP_NAME' \
             PRETRAINED_WEIGHTS='${PRETRAINED_WEIGHTS:-}' USE_GPU='${USE_GPU:-true}' \
             BATCH_SIZE='${BATCH_SIZE:-4}' \
             RR_START_ROUND='${RR_START_ROUND:-200}' \
             MAGG_RENORM='${MAGG_RENORM:-1}' \
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

    # Force-remove Docker containers: tmux SIGHUP kills the 'docker run' client
    # process but leaves the container running — the next experiment would then
    # see "docker rm -f" kill the previous server and "Address already in use"
    # for Singularity runs.  Clean up unconditionally so port 9999 is free.
    if [[ "${RUNTIME:-docker}" == "docker" ]]; then
        docker rm -f fl_server-gpu 2>/dev/null || true
        for i in $(seq 0 $(( NCLIENTES - 1 ))); do
            docker rm -f "client-$i" 2>/dev/null || true
        done
    fi

    # Wait for port 9999 to be fully released before returning
    local _port_wait=0
    while ss -tlnp 2>/dev/null | grep -q ':9999[^0-9]' && (( _port_wait < 15 )); do
        sleep 1; (( _port_wait++ ))
    done

    _write_incomplete_log "$LOGFOLDER"
    echo "[OK] $LOGFOLDER"
}

# ── Main loop (called at the end of every theme script) ───────────
# Reads all config variables from the calling script's environment.
run_theme() {
    # Single-pair override: when MODEL_DATASET_PAIR is set, restrict this run to
    # just that one pair. Used by sbatch/submit_per_pair.sh to fan a theme out into
    # one SLURM job per pair. All other sweeps (solution/niid/percentage/seed) are
    # unaffected. Also handy locally:
    #   MODEL_DATASET_PAIR=lenet5:FASHION_MNIST bash experiments/foo.sh
    if [[ -n "${MODEL_DATASET_PAIR:-}" ]]; then
        MODEL_DATASET_PAIRS=("$MODEL_DATASET_PAIR")
    fi
    # List mode: print the (possibly overridden) pairs and exit so the launcher
    # can enumerate them without triggering any tmux/docker/key-gen work.
    if [[ -n "${PRINT_PAIRS_ONLY:-}" ]]; then
        printf '%s\n' "${MODEL_DATASET_PAIRS[@]}"
        return 0
    fi

    NCLIENTES=$(( ${#GPU_IDS[@]} * CLIENTS_PER_GPU ))
    # CLIENT_FRACTIONS array: default to (1.0) when the script doesn't define it
    # so that existing scripts run exactly once at 100% (backward-compatible).
    [[ ${#CLIENT_FRACTIONS[@]} -eq 0 ]] && CLIENT_FRACTIONS=(1.0)

    mkdir -p "${CONTEXT_DIR:-context}" logs logs/models
    # Pre-create the FOLDER subtree with host-user ownership so Docker containers
    # (which run as root) never get a chance to create it first as root.
    [[ -n "${FOLDER:-}" ]] && mkdir -p "logs/${FOLDER}"

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
                    (( total += ${#PERCENTAGES[@]} * ${#CLIENT_FRACTIONS[@]} ))
                else
                    (( total += ${#CLIENT_FRACTIONS[@]} ))
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

                # Per-model overrides: reset defaults, then call model_config if defined.
                BATCH_SIZE=4
                if declare -f model_config &>/dev/null; then
                    model_config "$MODEL" "$DATASET"
                fi

                # Auto-detect pre-trained weights for this model.
                # Uses the most recently modified .h5 found under logs/models/pretrained/<model>/.
                PRETRAINED_WEIGHTS=$(
                    find "logs/models/pretrained/${MODEL}" -name "pretrained.weights.h5" \
                         2>/dev/null -printf "%T@ %p\n" | sort -rn | head -1 | awk '{print $2}'
                ) || PRETRAINED_WEIGHTS=""
                if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
                    echo "  [pretrain] $MODEL → $PRETRAINED_WEIGHTS"
                fi

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
                        (( done_count++ ))
                        # Token condicional: só inserido quando CLIENT_FRACTION != 1.0
                        # para preservar retrocompatibilidade com pastas existentes (100%).
                        CF_TOKEN=""
                        case "${CLIENT_FRACTION:-1.0}" in 1.0|1|1.00) ;; *) CF_TOKEN="cf${CLIENT_FRACTION}_" ;; esac
                        _find_resume_checkpoint
                        TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
                        LOGFOLDER="${FOLDER:+${FOLDER}/}${MODEL}_${DATASET_FOLDER}_${SOLUTION}_${TECHNIQUE}_${NUM_ROUNDS}rounds_${NCLIENTES}clients_seed${SEED}_niid${NIID}_percentage${PERCENTAGE}_${POLY}_${PLAIN_MODULUS}_${N_LENGTH}_${COEFFS}_${SCALE}_${KEY_TYPE}_${CF_TOKEN}${TIMESTAMP}"

                        echo "[$done_count/$total] $MODEL @ $DATASET | $SOLUTION | pct=$PERCENTAGE | cf=$CLIENT_FRACTION | seed=$SEED | niid=$NIID"
                        run_experiment "$LOGFOLDER"
                    done
                    done

                done; done
            done; done
        done
    done

    notify "Theme '${FOLDER:-default}' finished 🎉 ($total experiments)"
    echo "Done. ($total experiments)"
}
