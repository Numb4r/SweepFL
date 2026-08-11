#!/bin/bash
# experiments/lib_slurm.sh — SLURM/sbatch backend for themed experiment batches.
#
# Same interface as lib_tmux.sh but without tmux: server and clients run as
# background processes within the SLURM allocation.
#
# GPU_IDS is always derived from SLURM_GPUS_ON_NODE (ignores hardcoded IDs
# from experiment scripts — those are physical IDs from the local machine).
#
# Do NOT source this file directly; use lib.sh, which auto-dispatches.

# ── Guard: must be run from repo root ─────────────────────────────
_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(dirname "$_lib_dir")"
if [[ "$PWD" != "$_repo_root" ]]; then
    echo "[lib_slurm.sh] Changing to repo root: $_repo_root"
    cd "$_repo_root" || { echo "[lib_slurm.sh] ERROR: cannot cd to $_repo_root"; exit 1; }
fi

# ── Runtime (always singularity for SLURM) ────────────────────────
RUNTIME="${RUNTIME:-singularity}"
SIF_DIR="${SIF_DIR:-.}"
SERVER_HOST="${SERVER_HOST:-localhost}"   # on SLURM, server and clients share the node
# Per-job context dir — isolates keys when two jobs run simultaneously on shared NFS
CONTEXT_DIR="${CONTEXT_DIR:-$PWD/context_${SLURM_JOB_ID}}"

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

# ── Email ──────────────────────────────────────────────────────────
NOTIFY_EMAIL="${NOTIFY_EMAIL:-y218172@dac.unicamp.br}"

send_email() {
    local subject="$1" body="$2"
    [[ -z "${NOTIFY_EMAIL:-}" ]] && return
    if command -v mail &>/dev/null; then
        echo "$body" | mail -s "$subject" "$NOTIFY_EMAIL"
    elif command -v sendmail &>/dev/null; then
        printf "To: %s\nSubject: %s\n\n%s\n" "$NOTIFY_EMAIL" "$subject" "$body" \
            | sendmail "$NOTIFY_EMAIL"
    else
        echo "[WARN] send_email: mail/sendmail not found — skipping email to $NOTIFY_EMAIL"
    fi
}

# Sends both Telegram and email. $1 = short message, $2 = subject suffix, $3 = body (optional).
_notify_all() {
    local msg="$1" subject="$2" body="${3:-$1}"
    notify "$msg"
    send_email "[Slidhe] $subject" "$body"
}

# ── Helpers ───────────────────────────────────────────────────────
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

generate_keys() {
    local schema="$1" poly="$2" coeffs="$3" scale="$4" key_type="$5"
    echo "Generating $schema keys: backend=$CKKS_BACKEND poly=$poly coeffs=[$coeffs] scale=$scale"
    # Run inside the Singularity container: all Python deps are there, not on the host.
    # --bind $PWD:$PWD makes the repo available at its real path so relative paths
    # (context/, encryption/) resolve correctly inside the container.
    singularity exec \
        --bind "$PWD:$PWD" \
        --env CKKS_BACKEND="${CKKS_BACKEND:-tenseal}" \
        --env CONTEXT_DIR="$CONTEXT_DIR" \
        --env SIMPLIFY_N="${SIMPLIFY_N:-8192}" \
        --env SIMPLIFY_BITS="${SIMPLIFY_BITS:-62}" \
        --env SIMPLIFY_SCALE_BITS="${SIMPLIFY_SCALE_BITS:-60}" \
        --env PYCRYPT_N="${PYCRYPT_N:-8192}" \
        --env PYCRYPT_BITS="${PYCRYPT_BITS:-60}" \
        --env OPENFHE_RING_DIM="${OPENFHE_RING_DIM:-8192}" \
        --env OPENFHE_MULT_DEPTH="${OPENFHE_MULT_DEPTH:-1}" \
        --env OPENFHE_SCALE_BITS="${OPENFHE_SCALE_BITS:-50}" \
        "${SIF_DIR}/flhe_client.sif" \
        bash -c "cd '$PWD' && python3 -" <<EOF
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

# ── run_experiment ────────────────────────────────────────────────
# Runs server + clients as background processes. Waits for the server
# to finish (it exits after all FL rounds), then cleans up clients.
run_experiment() {
    local LOGFOLDER="$1"

    mkdir -p "logs/${LOGFOLDER}/model"
    chmod -R 777 "logs/${LOGFOLDER}"

    # ── RAM monitor (host-side, catches OOM kills) ───────────────────
    local MEM_LOG="logs/${LOGFOLDER}/memory.log"
    (
        echo "time        used_MB  avail_MB total_MB  node"
        while true; do
            free -m | awk -v ts="$(date +%H:%M:%S)" -v node="${SLURMD_NODENAME:-$(hostname -s)}" \
                '/Mem:/{printf "%s %8d %8d %8d  %s\n", ts, $3, $7, $2, node}'
            sleep 30
        done
    ) >> "$MEM_LOG" &
    local MEM_MONITOR_PID=$!

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
  "RUNTIME":         "singularity",
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

    # ── Server ──────────────────────────────────────────────────────
    echo "[SLURM] Starting server (background)..."
    RUNTIME="$RUNTIME" SIF_DIR="$SIF_DIR" CONTEXT_DIR="$CONTEXT_DIR" \
    EXPERIMENT_NAME="$EXP_NAME" \
    RESUME_LOGFOLDER="${RESUME_LOGFOLDER:-}" \
    JOB_DEADLINE_EPOCH="${JOB_DEADLINE_EPOCH:-0}" \
    CHECKPOINT_THRESHOLD_SECS="${CHECKPOINT_THRESHOLD_SECS:-300}" \
    CLIENT_FRACTION="${CLIENT_FRACTION:-1.0}" \
    CKKS_BACKEND="${CKKS_BACKEND:-tenseal}" \
    SIMPLIFY_N="${SIMPLIFY_N:-8192}" SIMPLIFY_BITS="${SIMPLIFY_BITS:-62}" SIMPLIFY_SCALE_BITS="${SIMPLIFY_SCALE_BITS:-60}" \
    PYCRYPT_N="${PYCRYPT_N:-8192}" PYCRYPT_BITS="${PYCRYPT_BITS:-60}" \
    OPENFHE_RING_DIM="${OPENFHE_RING_DIM:-8192}" OPENFHE_MULT_DEPTH="${OPENFHE_MULT_DEPTH:-1}" OPENFHE_SCALE_BITS="${OPENFHE_SCALE_BITS:-50}" \
        ./run_server.sh "$SOLUTION" "$NCLIENTES" "$NUM_ROUNDS" "$POLY" "$COEFFS" "$SCALE" "$KEY_TYPE" \
        "$DATASET" "$NIID" "$SEED" "$PERCENTAGE" "$LOGFOLDER" "$PLAIN_MODULUS" "$N_LENGTH" \
        > "logs/${LOGFOLDER}/server.log" 2>&1 &
    local SERVER_PID=$!
    echo "[SLURM] Server PID=$SERVER_PID — waiting 8s for startup..."
    sleep 8

    # ── Clients ─────────────────────────────────────────────────────
    local CLIENT_PIDS=()
    for i in $(seq 0 $(( NCLIENTES - 1 ))); do
        local GPU_ID=${GPU_IDS[$(( i / CLIENTS_PER_GPU ))]}
        echo "[SLURM] Starting client-$i on GPU $GPU_ID..."
        RUNTIME="$RUNTIME" SIF_DIR="$SIF_DIR" SERVER_HOST="$SERVER_HOST" CONTEXT_DIR="$CONTEXT_DIR" \
        MODEL_ARCH="$MODEL" EXPERIMENT_NAME="$EXP_NAME" \
        PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}" USE_GPU="${USE_GPU:-true}" \
        BATCH_SIZE="${BATCH_SIZE:-4}" \
        RR_START_ROUND="${RR_START_ROUND:-200}" \
        MAGG_RENORM="${MAGG_RENORM:-1}" \
        RESUME_LOGFOLDER="${RESUME_LOGFOLDER:-}" \
        CKKS_BACKEND="${CKKS_BACKEND:-tenseal}" \
        SIMPLIFY_N="${SIMPLIFY_N:-8192}" SIMPLIFY_BITS="${SIMPLIFY_BITS:-62}" SIMPLIFY_SCALE_BITS="${SIMPLIFY_SCALE_BITS:-60}" \
        PYCRYPT_N="${PYCRYPT_N:-8192}" PYCRYPT_BITS="${PYCRYPT_BITS:-60}" \
        OPENFHE_RING_DIM="${OPENFHE_RING_DIM:-8192}" OPENFHE_MULT_DEPTH="${OPENFHE_MULT_DEPTH:-1}" OPENFHE_SCALE_BITS="${OPENFHE_SCALE_BITS:-50}" \
            ./run_client.sh "$i" "$SOLUTION" "$TECHNIQUE" "$NCLIENTES" "$GPU_ID" \
            "$POLY" "$COEFFS" "$SCALE" "$KEY_TYPE" "$NIID" "$DATASET" "$SEED" \
            "$PERCENTAGE" "$LOGFOLDER" "$PLAIN_MODULUS" "$N_LENGTH" \
            > "logs/${LOGFOLDER}/client-${i}.log" 2>&1 &
        CLIENT_PIDS+=($!)
        sleep 2
    done

    # ── Wait ────────────────────────────────────────────────────────
    echo "[SLURM] Waiting for server (PID=$SERVER_PID)..."
    wait "$SERVER_PID"
    echo "[SLURM] Server finished. Waiting up to 60s for clients..."

    # Give clients time to write final logs and exit cleanly
    local _waited=0
    while kill -0 "${CLIENT_PIDS[@]}" 2>/dev/null && (( _waited < 60 )); do
        sleep 2; (( _waited += 2 ))
    done

    # Kill any stragglers
    for pid in "${CLIENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${CLIENT_PIDS[@]}" 2>/dev/null

    kill "$MEM_MONITOR_PID" 2>/dev/null || true

    # Wait for port 9999 to be fully released
    local _port_wait=0
    while ss -tlnp 2>/dev/null | grep -q ':9999[^0-9]' && (( _port_wait < 15 )); do
        sleep 1; (( _port_wait++ ))
    done

    _write_incomplete_log "$LOGFOLDER"
    echo "[OK] $LOGFOLDER"
}

# ── run_theme ─────────────────────────────────────────────────────
run_theme() {
    # Single-pair override: when MODEL_DATASET_PAIR is set, restrict this run to
    # just that one pair. Used by sbatch/submit_per_pair.sh to fan a theme out into
    # one SLURM job per (pair, seed). The remaining sweeps (solution/niid/percentage)
    # are unaffected. Also handy locally:
    #   MODEL_DATASET_PAIR=lenet5:FASHION_MNIST bash experiments/foo.sh
    if [[ -n "${MODEL_DATASET_PAIR:-}" ]]; then
        MODEL_DATASET_PAIRS=("$MODEL_DATASET_PAIR")
    fi
    # Single-seed override: same mechanism as MODEL_DATASET_PAIR above, second axis
    # of the fan-out. Note it replaces the whole SEEDS array, so exporting it in a
    # shell pins every later run in that shell to one seed.
    #   SEED_OVERRIDE=7 bash experiments/foo.sh
    if [[ -n "${SEED_OVERRIDE:-}" ]]; then
        SEEDS=("$SEED_OVERRIDE")
    fi
    # Default before any list mode can read it, mirroring the CLIENT_FRACTIONS
    # fallback further down: a script that omits SEEDS still enumerates as (42).
    [[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(42)

    # List mode: print the (possibly overridden) pairs and exit so the launcher
    # can enumerate them without triggering any tmux/docker/key-gen work.
    if [[ -n "${PRINT_PAIRS_ONLY:-}" ]]; then
        printf '%s\n' "${MODEL_DATASET_PAIRS[@]}"
        return 0
    fi
    # Same contract for seeds, so the launcher can build the pair × seed product.
    if [[ -n "${PRINT_SEEDS_ONLY:-}" ]]; then
        printf '%s\n' "${SEEDS[@]}"
        return 0
    fi

    # Derive GPU_IDS from SLURM allocation — ignores hardcoded IDs in experiment
    # scripts (those are physical IDs from the local machine, meaningless on SLURM).
    # SLURM remaps allocated GPUs to virtual 0-based IDs via CUDA_VISIBLE_DEVICES.
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        GPU_IDS=()
        for ((g=0; g<SLURM_GPUS_ON_NODE; g++)); do GPU_IDS+=($g); done
        echo "[SLURM] GPU_IDS overridden to (${GPU_IDS[*]}) from SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
    elif [[ ${#GPU_IDS[@]} -eq 0 ]]; then
        GPU_IDS=(0)
        echo "[SLURM] SLURM_GPUS_ON_NODE not set and GPU_IDS empty — defaulting to (0)"
    fi

    NCLIENTES=$(( ${#GPU_IDS[@]} * CLIENTS_PER_GPU ))
    # CLIENT_FRACTIONS array: default to (1.0) when the script doesn't define it
    # so that existing scripts run exactly once at 100% (backward-compatible).
    [[ ${#CLIENT_FRACTIONS[@]} -eq 0 ]] && CLIENT_FRACTIONS=(1.0)

    mkdir -p "$CONTEXT_DIR" logs logs/models
    [[ -n "${FOLDER:-}" ]] && mkdir -p "logs/${FOLDER}"

    # Compute job deadline epoch for time-aware shutdown
    JOB_DEADLINE_EPOCH=0
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        _job_end=$(squeue -j "$SLURM_JOB_ID" -h -o "%e" 2>/dev/null)
        if [[ -n "$_job_end" ]]; then
            JOB_DEADLINE_EPOCH=$(date -d "$_job_end" +%s 2>/dev/null || echo 0)
        fi
        echo "[SLURM] JOB_DEADLINE_EPOCH=$JOB_DEADLINE_EPOCH ($(date -d @$JOB_DEADLINE_EPOCH 2>/dev/null || echo unknown))"
    fi
    export JOB_DEADLINE_EPOCH

    echo "══════════════════════════════════════════════"
    echo " Theme  : ${FOLDER:-<no folder>}"
    echo " Runtime: SLURM/singularity (job ${SLURM_JOB_ID:-?})"
    echo " Node   : ${SLURMD_NODENAME:-$(hostname)}"
    echo " GPU mapping (${NCLIENTES} clients across ${#GPU_IDS[@]} GPU(s)):"
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

                PRETRAINED_WEIGHTS=$(
                    find "logs/models/pretrained/${MODEL}" -name "pretrained.weights.h5" \
                         2>/dev/null -printf "%T@ %p\n" | sort -rn | head -1 | awk '{print $2}'
                ) || PRETRAINED_WEIGHTS=""
                [[ -n "$PRETRAINED_WEIGHTS" ]] && echo "  [pretrain] $MODEL → $PRETRAINED_WEIGHTS"

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

    local _node="${SLURMD_NODENAME:-$(hostname -s)}"
    local _job="${SLURM_JOB_ID:-local}"
    local _body
    _body="$(printf 'Theme   : %s\nJob     : %s\nNode    : %s\nDone    : %d experiments\nFinished: %s' \
        "${FOLDER:-default}" "$_job" "$_node" "$total" "$(date '+%Y-%m-%d %H:%M:%S')")"

    _notify_all \
        "✅ Theme '${FOLDER:-default}' finished ($total experiments) [job $_job @ $_node]" \
        "✅ ${FOLDER:-default} done — job $_job" \
        "$_body"
    echo "Done. ($total experiments)"
}

# ── Failure trap (registered by sbatch entry-point) ───────────────
# Call this from sbatch scripts: trap '_slurm_fail_notify $?' EXIT
# It only fires a notification when the exit code is non-zero.
_slurm_fail_notify() {
    local exit_code="${1:-?}"
    [[ "$exit_code" == "0" ]] && return
    local _node="${SLURMD_NODENAME:-$(hostname -s)}"
    local _job="${SLURM_JOB_ID:-local}"
    local _body
    _body="$(printf 'Theme   : %s\nJob     : %s\nNode    : %s\nExit    : %s\nFailed  : %s' \
        "${FOLDER:-default}" "$_job" "$_node" "$exit_code" "$(date '+%Y-%m-%d %H:%M:%S')")"

    _notify_all \
        "❌ Theme '${FOLDER:-default}' FAILED (exit=$exit_code) [job $_job @ $_node]" \
        "❌ ${FOLDER:-default} FAILED — job $_job" \
        "$_body"
}
