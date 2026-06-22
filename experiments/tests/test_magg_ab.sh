#!/bin/bash
# experiments/test_magg_ab.sh — A/B experiment for the MAGG_RENORM fix.
#
# Compares 5 conditions (all Docker, ckks_python_simplify backend):
#   Phase 0 — plaintext           → baseline; also seeds init_weights (required)
#   Phase 1 — fedphe MAGG_RENORM=0  (legacy, biased for sparse/data-dependent masks)
#   Phase 2 — fedphe MAGG_RENORM=1  (corrected, proper per-pack FedAvg)
#   Phase 3 — nvnl   MAGG_RENORM=0  (control: RR masks = identical → sparsity no-op)
#   Phase 4 — nvnl   MAGG_RENORM=1  (control corrected: also fixes data-weighting)
#
# Setup: 3 clients · 25 rounds · FashionMNIST · LeNet-5 · niid=0.1 · ζ=0.1
#   ζ=0.1 (10% of packs) maximises sparse-pack coverage → largest bias gap.
#   niid=0.1 (strong heterogeneity) → unequal Ds → largest data-weighting error.
#
# Expected outcome:
#   fedphe MAGG_RENORM=1 ≥ MAGG_RENORM=0 in final accuracy (bug fix visible).
#   nvnl   MAGG_RENORM=1 ≥ MAGG_RENORM=0 (data-weighting fix; gap smaller than fedphe).
#   plaintext sets the upper bound (~1.45% envelope per paper T6).
#
# Usage (from repo root):
#   bash experiments/test_magg_ab.sh
#
# Results in logs/test_magg_ab/  — compare server_eval.csv mean_acc by round.

# ── GPU layout (3 clients on one GPU) ────────────────────────────────────────
GPU_IDS=(0)
CLIENTS_PER_GPU=3
if [[ ! -S /run/nvidia-persistenced/socket ]]; then
    USE_GPU=false
    echo "[WARN] nvidia-persistenced socket not found — running clients in CPU mode"
fi

# ── HE parameters ─────────────────────────────────────────────────────────────
CKKS_CONFIGS=("ckks 8192 60,40,40,60")
KEY_TYPES=('None')
SCALE_POWER=40
PLAIN_MODULUS=-1
N_LENGTH=-1
CKKS_BACKEND="ckks_python_simplify"
SIMPLIFY_N=8192
SIMPLIFY_BITS=62
SIMPLIFY_SCALE_BITS=40

# ── Shared experiment parameters ──────────────────────────────────────────────
NIIDS=(0.1)      # strong non-IID → unequal Ds → both biases active
SEEDS=(42)
TECHNIQUE="none"
NUM_ROUNDS=25
PERCENTAGES=(0.1) # ζ=0.1: only 10% of packs sent → maximum sparsity gap
MODEL_DATASET_PAIRS=("lenet5:FASHION_MNIST")
EXPERIMENT_NAME=""

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

_cleanup() {
    echo "── Cleanup: stopping FL containers, releasing port 9999..."
    docker rm -f fl_server-gpu 2>/dev/null || true
    local _n=${NCLIENTES:-3}
    for _c in $(seq 0 $(( _n - 1 ))); do
        docker rm -f "client-$_c" 2>/dev/null || true
    done
    local _t=0
    while ss -tlnp 2>/dev/null | grep -q ':9999[^0-9]' && (( _t < 20 )); do
        echo "  waiting for port 9999... ($((++_t))/20)"
        sleep 1
    done
    echo "── Cleanup done."
}

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  MAGG_RENORM A/B Experiment — fedphe & nvnl                     ║"
echo "║  3 clients · 25 rounds · FashionMNIST · LeNet-5 · ζ=0.1        ║"
echo "║  niid=0.1 · ckks_python_simplify backend                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ── Phase 0: plaintext baseline (also seeds init_weights) ────────────────────
echo ""
echo "▶ Phase 0: plaintext baseline"
FOLDER="test_magg_ab/phase0_plaintext"
SOLUTIONS=("plaintext")
RUNTIME="docker"
SIF_DIR="."
run_theme
_cleanup

# ── Phase 1: fedphe MAGG_RENORM=0 (legacy/biased) ────────────────────────────
echo ""
echo "▶ Phase 1: fedphe MAGG_RENORM=0 (legacy — biased renorm)"
FOLDER="test_magg_ab/phase1_fedphe_legacy"
SOLUTIONS=("fedphe")
RUNTIME="docker"
SIF_DIR="."
MAGG_RENORM=0 run_theme
_cleanup

# ── Phase 2: fedphe MAGG_RENORM=1 (corrected) ────────────────────────────────
echo ""
echo "▶ Phase 2: fedphe MAGG_RENORM=1 (corrected — proper per-pack FedAvg)"
FOLDER="test_magg_ab/phase2_fedphe_corrected"
SOLUTIONS=("fedphe")
RUNTIME="docker"
SIF_DIR="."
MAGG_RENORM=1 run_theme
_cleanup

# ── Phase 3: nvnl MAGG_RENORM=0 (control — RR masks, legacy) ─────────────────
echo ""
echo "▶ Phase 3: nvnl MAGG_RENORM=0 (control — RR masks, legacy)"
FOLDER="test_magg_ab/phase3_nvnl_legacy"
SOLUTIONS=("nvnl")
RUNTIME="docker"
SIF_DIR="."
MAGG_RENORM=0 run_theme
_cleanup

# ── Phase 4: nvnl MAGG_RENORM=1 (control corrected) ─────────────────────────
echo ""
echo "▶ Phase 4: nvnl MAGG_RENORM=1 (control — RR masks, corrected)"
FOLDER="test_magg_ab/phase4_nvnl_corrected"
SOLUTIONS=("nvnl")
RUNTIME="docker"
SIF_DIR="."
MAGG_RENORM=1 run_theme
_cleanup

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════ SUMMARY ════════════════════════════════"
echo "Condition                      | Final acc (server_eval.csv last row)"
echo "───────────────────────────────|──────────────────────────────────────"
declare -A LABELS=(
    [phase0_plaintext]="plaintext  (baseline)"
    [phase1_fedphe_legacy]="fedphe     MAGG_RENORM=0 (legacy)"
    [phase2_fedphe_corrected]="fedphe     MAGG_RENORM=1 (corrected)"
    [phase3_nvnl_legacy]="nvnl       MAGG_RENORM=0 (control legacy)"
    [phase4_nvnl_corrected]="nvnl       MAGG_RENORM=1 (control corrected)"
)
all_ok=true
for phase in phase0_plaintext phase1_fedphe_legacy phase2_fedphe_corrected \
             phase3_nvnl_legacy phase4_nvnl_corrected; do
    label="${LABELS[$phase]}"
    found=false
    for dir in logs/test_magg_ab/${phase}/*/; do
        [[ -d "$dir" ]] || continue
        eval_csv="${dir}server_eval.csv"
        if [[ -f "$eval_csv" ]]; then
            last_acc=$(tail -1 "$eval_csv" | awk -F',' '{print $NF}')
            rounds=$(( $(wc -l < "$eval_csv") - 1 ))
            printf "  %-32s | acc=%-8s  rounds=%d\n" "$label" "$last_acc" "$rounds"
            found=true
        fi
    done
    if ! $found; then
        echo "  [FAIL] $phase — no server_eval.csv found"
        all_ok=false
    fi
done
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Expected: fedphe corrected acc ≥ fedphe legacy (fix matters for sparse+non-IID)."
echo "          nvnl   corrected acc ≥ nvnl   legacy (data-weighting fix, smaller gap)."
echo "          plaintext sets upper bound."
echo ""
echo "Compare full convergence curves:"
echo "  python3 studies/sparsification_percentage/load_data.py  (manual inspect)"
echo "  or: for f in logs/test_magg_ab/*/*/server_eval.csv; do echo \"\$f\"; tail -5 \"\$f\"; done"
if $all_ok; then
    echo ""
    echo "All phases completed."
else
    echo ""
    echo "Some experiments may have failed — check logs above."
    exit 1
fi
