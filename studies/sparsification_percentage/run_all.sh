#!/bin/bash
# Run all sparsification_percentage study scripts in sequence.
# Usage:
#   bash run_all.sh
#   bash run_all.sh --logs-dir /path/to/logs/sparsification_percentage

set -e
cd "$(dirname "$0")"

mkdir -p output/figures output/tables output/notebooks

EXTRA_ARGS="$@"

echo "============================================"
echo " Sparsification Percentage — Analysis Suite"
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
              13_overhead_vs_accuracy.py \
              14_neurons_vs_accuracy.py \
              15_bytes_vs_maxacc.py \
              16_overhead_vs_maxacc.py \
              17_coverage_unique.py \
              18_coverage_volume.py \
              19_coverage_accuracy.py \
              20_pack_frequency.py \
              21_layer_frequency.py \
              22_layer_heatmap.py \
              23_pack_heatmap.py; do
    echo ""
    echo "==> $script"
    python3 "$script" $EXTRA_ARGS
done

echo ""
echo "==> notebook.ipynb (executing...)"
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=300 \
    notebook.ipynb \
    --output output/notebooks/notebook_executed.ipynb \
    2>/dev/null && echo "    Saved output/notebooks/notebook_executed.ipynb" \
              || echo "    [WARNING] Notebook execution failed (jupyter/ipywidgets not available). Skipping."

echo ""
echo "============================================"
echo " Done. Outputs:"
echo "   figures/  : $(ls output/figures/ 2>/dev/null | wc -l) files"
echo "   tables/   : $(ls output/tables/  2>/dev/null | wc -l) files"
echo "   notebooks/: $(ls output/notebooks/ 2>/dev/null | wc -l) files"
echo "============================================"
