#!/bin/bash
# sbatch/submit_per_pair.sh — Submit one independent SLURM job per (model:dataset, seed).
#
# Instead of running all MODEL_DATASET_PAIRS × SEEDS sequentially inside a single job,
# this launcher submits one sbatch job per combination, letting the cluster schedule
# them in parallel. The remaining sweeps — solution, niid, percentage, client_fraction —
# run inside each job unchanged.
#
# Usage (from repo root):
#   bash sbatch/submit_per_pair.sh <sbatch_script> [--dry-run]
#
# Examples:
#   bash sbatch/submit_per_pair.sh sbatch/sparsification_percentage/sbatch_sparsification_percentage_srr.sh --dry-run
#   bash sbatch/submit_per_pair.sh sbatch/client_frac/sbatch_client_frac_sweepfl.sh
#
# How it works:
#   1. Reads EXPERIMENT_SCRIPT from the sbatch script.
#   2. Runs that experiment script twice in list mode (no tmux/docker/key-gen):
#      PRINT_PAIRS_ONLY=1 for MODEL_DATASET_PAIRS, PRINT_SEEDS_ONLY=1 for SEEDS.
#   3. Calls sbatch once per (pair, seed), passing MODEL_DATASET_PAIR=<pair> and
#      SEED_OVERRIDE=<seed> via --export. run_theme() in lib_slurm.sh picks both up
#      and restricts the run to that single combination.
#
# Job naming:  <original-job-name>_<model>_<dataset>[_seed<N>]
#              The seed suffix appears only when the theme sweeps more than one seed,
#              so single-seed themes keep byte-identical job and log names.
# Log files:   logs/slurm_<job-name>_<jobid>.out  (one file per job)
#
# Backward compat: direct  sbatch sbatch/sbatch_*.sh  still runs the full sweep in one job.

set -euo pipefail

SBATCH_SCRIPT="${1:-}"
DRY=0
[[ "${2:-}" == "--dry-run" ]] && DRY=1

# ── Argument validation ───────────────────────────────────────────────────────
if [[ -z "$SBATCH_SCRIPT" ]]; then
    echo "usage: bash sbatch/submit_per_pair.sh <sbatch_script> [--dry-run]" >&2
    echo ""
    echo "Available sbatch scripts:"
    find sbatch -name 'sbatch_*.sh' 2>/dev/null | sort | sed 's|^|  |' || true
    exit 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: $SBATCH_SCRIPT not found" >&2
    exit 1
fi

# ── Resolve the experiment script ────────────────────────────────────────────
EXP=$(grep -oP 'EXPERIMENT_SCRIPT="\K[^"]+' "$SBATCH_SCRIPT" | head -1)
if [[ -z "$EXP" ]]; then
    echo "ERROR: could not find EXPERIMENT_SCRIPT=\"...\" in $SBATCH_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$EXP" ]]; then
    echo "ERROR: EXPERIMENT_SCRIPT=$EXP does not exist" >&2
    exit 1
fi

# ── GPU guardrail: a GPU request must never be silently lost ─────────────────
# Extract the gres value from the sbatch script (stops at first whitespace,
# so it works correctly even if the line still has an inline comment).
GRES=$(grep -oP '^\s*#SBATCH\s+--gres=\K\S+' "$SBATCH_SCRIPT" | head -1) || true
if [[ -z "$GRES" ]]; then
    echo "ERROR: no '#SBATCH --gres=' directive found in $SBATCH_SCRIPT" >&2
    echo "       refusing to submit jobs without a GPU request" >&2
    exit 1
fi
# Accept gpu:N or gpu:<type>:N; require N >= 1
if [[ "$GRES" =~ ^gpu:([A-Za-z0-9_]+:)?([0-9]+)$ ]]; then
    if (( BASH_REMATCH[2] < 1 )); then
        echo "ERROR: --gres=$GRES requests 0 GPUs in $SBATCH_SCRIPT" >&2
        exit 1
    fi
else
    echo "ERROR: malformed --gres=$GRES in $SBATCH_SCRIPT" >&2
    echo "       expected gpu:N or gpu:<type>:N  (e.g. gpu:3 or gpu:l40s:3)" >&2
    exit 1
fi
echo "GPU guardrail: re-asserting --gres=$GRES on every submission"
echo ""

# ── Enumerate MODEL_DATASET_PAIRS (fast, no side effects) ────────────────────
# PRINT_PAIRS_ONLY=1 makes run_theme() print the pairs and return before any
# tmux/docker/key-gen work.  lib.sh dispatches to lib_tmux.sh here (no SLURM_JOB_ID).
mapfile -t PAIRS < <(PRINT_PAIRS_ONLY=1 bash "$EXP" 2>/dev/null)

if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "ERROR: no MODEL_DATASET_PAIRS found in $EXP" >&2
    echo "  (check that the array has at least one uncommented entry)" >&2
    exit 1
fi

# ── Enumerate SEEDS the same way (second fan-out axis) ───────────────────────
mapfile -t SEED_LIST < <(PRINT_SEEDS_ONLY=1 bash "$EXP" 2>/dev/null)

if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
    echo "ERROR: no SEEDS found in $EXP" >&2
    echo "  (run_theme defaults to (42) when SEEDS is unset — an empty list means" >&2
    echo "   the script failed to run; try: PRINT_SEEDS_ONLY=1 bash $EXP)" >&2
    exit 1
fi

# ── Base job name from the sbatch script ─────────────────────────────────────
BASE=$(grep -oP '#SBATCH\s+--job-name=\K\S+' "$SBATCH_SCRIPT" | head -1)
BASE="${BASE:-$(basename "$SBATCH_SCRIPT" .sh)}"

# ── Require sbatch before entering the submission loop ───────────────────────
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: 'sbatch' não encontrado no PATH — rode isto no login node do cluster" >&2
    echo "       (ou use --dry-run para apenas listar as submissões)" >&2
    exit 1
fi

# ── Submit one job per (pair, seed) ───────────────────────────────────────────
TOTAL=$(( ${#PAIRS[@]} * ${#SEED_LIST[@]} ))
echo "Submitting ${TOTAL} job(s) from $SBATCH_SCRIPT"
echo "  ${#PAIRS[@]} pair(s) × ${#SEED_LIST[@]} seed(s): ${SEED_LIST[*]}"
[[ $DRY -eq 1 ]] && echo "(dry-run — no jobs will be submitted)"
echo ""

for pair in "${PAIRS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    # Sanitise pair into a valid job-name suffix:
    #   lenet5:FASHION_MNIST          → lenet5_FASHION_MNIST
    #   vit_tiny:tanganke/gtsrb       → vit_tiny_tanganke_gtsrb
    tag=$(echo "$pair" | tr ':/' '__')
    name="${BASE}_${tag}"
    # Seed token only when actually sweeping seeds, mirroring the CF_TOKEN convention
    # in lib_slurm.sh — keeps names byte-identical for single-seed themes.
    (( ${#SEED_LIST[@]} > 1 )) && name="${name}_seed${seed}"

    if [[ $DRY -eq 1 ]]; then
        echo "[dry-run] sbatch --job-name=${name} \\"
        echo "               --output=logs/slurm_%x_%j.out \\"
        echo "               --error=logs/slurm_%x_%j.err \\"
        echo "               --export=ALL,MODEL_DATASET_PAIR=${pair},SEED_OVERRIDE=${seed} \\"
        echo "               --gres=${GRES} \\"
        echo "               ${SBATCH_SCRIPT}"
    else
        jid=$(sbatch --parsable \
                     --job-name="${name}" \
                     --output="logs/slurm_%x_%j.out" \
                     --error="logs/slurm_%x_%j.err" \
                     --export=ALL,MODEL_DATASET_PAIR="${pair}",SEED_OVERRIDE="${seed}" \
                     --gres="${GRES}" \
                     "$SBATCH_SCRIPT")
        echo "submitted job ${jid}  →  ${pair}  seed=${seed}"
        echo "  name   : ${name}"
        echo "  log    : logs/slurm_${name}_${jid}.out"
        echo "  monitor: tail -f logs/slurm_${name}_${jid}.out"
    fi
    echo ""
  done
done

if [[ $DRY -eq 0 ]]; then
    echo "All ${TOTAL} job(s) submitted.  Check status: squeue -u \$USER"
fi
