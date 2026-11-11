#!/bin/bash
# resume_all.sh — Gera scripts de resume para todos os experimentos interrompidos
# e submete via sbatch.
#
# Uso:
#   ./resume_all.sh                     # gera + submete todos
#   ./resume_all.sh --dry-run           # lista o que seria submetido (sem criar arquivos)
#   ./resume_all.sh --generate-only     # gera scripts sem submeter
#   ./resume_all.sh --merge-only        # junta pares (original+resumido) já concluídos
#   ./resume_all.sh --partition p100    # override de partição SLURM
#   ./resume_all.sh --time 1-00:00:00   # override do limite de tempo
#   ./resume_all.sh --logs-dir /path    # diretório de logs alternativo
#   ./resume_all.sh --target seed_experiment_v2              # só esse subset (substring)
#   ./resume_all.sh --target 'seed_exp*/mobilenetv2*sweepfl*' --target '*percentage0.7*'  # glob, repetível (OR)
#   ./resume_all.sh --debug            # set -x + progresso cronometrado do gerador (stderr)
#   ./resume_all.sh --min-idle 30       # ignora pastas com checkpoint atualizado há <30 min (default 60)
#
# Detecção: checkpoint/meta.json round < NUM_ROUNDS (INCOMPLETE.log não é exigido —
# jobs mortos pelo SLURM não chegam a escrevê-lo). Pastas em archive/ são ignoradas.
#
# Saída: paths dos resume_sbatch.sh gerados (e submetidos, se não --dry-run/--generate-only).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DRY_RUN=false
GENERATE_ONLY=false
MERGE_ONLY=false
LOGS_DIR="logs"
_PREV_ARG=""
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]]       && DRY_RUN=true
    [[ "$arg" == "--generate-only" ]] && GENERATE_ONLY=true
    [[ "$arg" == "--merge-only" ]]    && MERGE_ONLY=true
    [[ "$arg" == "--debug" ]]         && set -x
    [[ "$_PREV_ARG" == "--logs-dir" ]] && LOGS_DIR="$arg"
    _PREV_ARG="$arg"
done

# ── Merge-only mode: join completed resume pairs, no submission ───────────────
if $MERGE_ONLY; then
    echo "[resume_all] Merge-only mode: joining completed (original + resumed) experiment pairs..."
    python3 experiments/merge_resume.py --logs-dir "$LOGS_DIR"
    exit 0
fi

# Run generator — info to stderr, sbatch paths to stdout
SBATCH_FILES=$(python3 experiments/generate_resume_jobs.py "$@")

if [[ -z "$SBATCH_FILES" ]]; then
    echo "Nothing to resume."
    exit 0
fi

if $DRY_RUN; then
    echo "Dry run — would submit:"
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        echo "  $f"
    done <<< "$SBATCH_FILES"
    exit 0
fi

if $GENERATE_ONLY; then
    echo "Generated (not submitted):"
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        echo "  $f"
    done <<< "$SBATCH_FILES"
    exit 0
fi

COUNT=0
while IFS= read -r sbatch_file; do
    [[ -z "$sbatch_file" ]] && continue
    echo "Submitting: $sbatch_file"
    sbatch "$sbatch_file"
    COUNT=$(( COUNT + 1 ))
done <<< "$SBATCH_FILES"

echo ""
echo "Submitted $COUNT job(s)."
