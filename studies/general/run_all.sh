#!/bin/bash
# studies/general/run_all.sh — Run all general analysis scripts for a given experiment type.
#
# Usage (from repo root or from studies/general/):
#   bash studies/general/run_all.sh <nome_do_experimento>
#   bash studies/general/run_all.sh client_frac
#   bash studies/general/run_all.sh sparsification_percentage
#   bash studies/general/run_all.sh client_frac --logs-dir /custom/path
#
# Each script in studies/general/ reads logs via LOGS_DIR env var (already supported
# in load_data.py) and writes its output to output/figures/, output/tables/,
# output/notebooks/ (relative to studies/general/).
#
# After all scripts finish, the output is moved to studies/<nome_do_experimento>/
# so each experiment type has its own output directory.

set -e

# ── Locate the script directory and resolve repo root ─────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ── Argument: nome do experimento ─────────────────────────────────────────────
if [[ -z "${1:-}" ]]; then
    echo "Uso: bash $0 <nome_do_experimento> [flags extras]" >&2
    echo "  Exemplos:" >&2
    echo "    bash $0 client_frac" >&2
    echo "    bash $0 sparsification_percentage" >&2
    exit 1
fi

NOME="$1"
shift
EXTRA_ARGS=("$@")

# ── Resolve LOGS_DIR (let --logs-dir override if provided) ───────────────────
if [[ -z "${LOGS_DIR:-}" ]]; then
    # Check if --logs-dir is in EXTRA_ARGS
    _custom_logs=""
    for _i in "${!EXTRA_ARGS[@]}"; do
        if [[ "${EXTRA_ARGS[$_i]}" == "--logs-dir" ]]; then
            _custom_logs="${EXTRA_ARGS[$((_i+1))]:-}"
        fi
    done
    if [[ -n "$_custom_logs" ]]; then
        export LOGS_DIR="$_custom_logs"
    else
        export LOGS_DIR="$REPO/logs/$NOME"
    fi
fi

# ── Output destination ────────────────────────────────────────────────────────
OUT_DIR="$REPO/studies/$NOME"
mkdir -p "$OUT_DIR/figures" "$OUT_DIR/tables" "$OUT_DIR/notebooks"

# Temp output inside general/ (where scripts hardcode their output/)
TEMP_OUT="$HERE/output"
mkdir -p "$TEMP_OUT/figures" "$TEMP_OUT/tables" "$TEMP_OUT/notebooks"

cd "$HERE"

echo "============================================"
echo " Analysis Suite: $NOME"
echo " LOGS_DIR : $LOGS_DIR"
echo " Output   : $OUT_DIR"
echo "============================================"

for script in 01_accuracy_curves.py \
              02_solution_comparison.py \
              03_overhead_breakdown.py \
              04_communication.py \
              05_niid_impact.py \
              06_memory_profile.py \
              07_convergence_speed.py \
              08_efficiency_frontier.py \
              09_summary_table.py \
              10_total_data_convergence.py \
              11_data_vs_coverage.py \
              12_bytes_vs_accuracy.py \
              12_bytes_vs_accuracy_individual.py \
              13_overhead_vs_accuracy.py \
              13_overhead_vs_accuracy_individual.py \
              14_neurons_vs_accuracy.py \
              14_neurons_vs_accuracy_individual.py \
              15_bytes_vs_maxacc.py \
              15_bytes_vs_maxacc_individual.py \
              16_overhead_vs_maxacc.py \
              16_overhead_vs_maxacc_individual.py \
              17_coverage_unique.py \
              17_coverage_unique_individual.py \
              18_coverage_volume.py \
              18_coverage_volume_individual.py \
              19_coverage_accuracy.py \
              19_coverage_accuracy_individual.py \
              20_pack_frequency.py \
              20_pack_frequency_individual.py \
              21_layer_frequency.py \
              21_layer_frequency_individual.py \
              22_layer_heatmap.py \
              22_layer_heatmap_individual.py \
              23_pack_heatmap.py \
              23_pack_heatmap_individual.py \
              24_full_coverage_convergence.py; do
    [[ -f "$script" ]] || continue
    echo ""
    echo "==> $script"
    python3 "$script" "${EXTRA_ARGS[@]}"
done

echo ""
echo "==> notebook.ipynb (executing...)"
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=300 \
    notebook.ipynb \
    --output "$TEMP_OUT/notebooks/notebook_executed.ipynb" \
    2>/dev/null && echo "    Saved $TEMP_OUT/notebooks/notebook_executed.ipynb" \
              || echo "    [WARNING] Notebook execution failed (jupyter/ipywidgets not available). Skipping."

# ── Move output to studies/<nome>/ ────────────────────────────────────────────
echo ""
echo "Moving output → $OUT_DIR/"
for subdir in figures tables notebooks; do
    if [[ -d "$TEMP_OUT/$subdir" ]] && [[ "$(ls -A "$TEMP_OUT/$subdir" 2>/dev/null)" ]]; then
        cp -r "$TEMP_OUT/$subdir/." "$OUT_DIR/$subdir/"
        rm -rf "$TEMP_OUT/$subdir"
        mkdir -p "$TEMP_OUT/$subdir"
    fi
done

echo ""
echo "============================================"
echo " Concluído. Saída em studies/$NOME/:"
echo "   figures/  : $(ls "$OUT_DIR/figures/"  2>/dev/null | wc -l) arquivos"
echo "   tables/   : $(ls "$OUT_DIR/tables/"   2>/dev/null | wc -l) arquivos"
echo "   notebooks/: $(ls "$OUT_DIR/notebooks/" 2>/dev/null | wc -l) arquivos"
echo "============================================"
