#!/usr/bin/env bash
# Regenerates every journal figure/table into studies/journal/output/. Run from anywhere.
# SOURCE_DATE_EPOCH makes the PDFs byte-reproducible (no embedded creation time).
set -euo pipefail
cd "$(dirname "$0")"
export SOURCE_DATE_EPOCH=1700000000 MPLBACKEND=Agg PYTHONHASHSEED=0
export LOGS_DIR="$PWD/../../logs/sparsification_percentage"
for s in accuracy_per_round_en accuracy_vs_budget_en starvation_vs_packets_en downlink_en seed_curves_en \
         pack_heatmap_en clientfrac_en 27_three_panel 20_pack_frequency_individual \
         15_bytes_vs_maxacc_pt 26_union_coverage_downlink_pt 28_download_vs_maxacc_pt \
         paper_numbers check_pendencies; do
  echo "== $s"; python3 "$s.py" > /dev/null
done
python3 mask_agreement.py --plot > /dev/null
python3 29_seed_variance.py --out output/seed_variance > /dev/null
