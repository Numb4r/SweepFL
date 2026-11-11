# studies/journal — journal figure scripts

Copied from `journal-nvnl/figures/scripts/`. Paths are relative to this repo. Figures and numbers go to
`studies/journal/output/` (override with `FIG_OUT=/path`). Run from this folder.

| Figure in paper | Script | Logs read |
|---|---|---|
| `three-panel-resnet20-p10` | `LOGS_DIR=../../logs/sparsification_percentage python3 27_three_panel.py` | sparsification_percentage |
| `accuracy-per-round` | `accuracy_per_round_en.py` | sparsification_percentage |
| `starvation-vs-packets` | `starvation_vs_packets_en.py` | sparsification_percentage, range_fix |
| `accuracy-vs-budget` | `accuracy_vs_budget_en.py` | sparsification_percentage, range_fix |
| `seed-curves` | `seed_curves_en.py` | sparsification_percentage, range_fix, seed_experiment_v2 (mobilenetv2 only) |
| `downlink-resnet20` | `downlink_en.py` | sparsification_percentage |
| `mask-agreement` | `python3 mask_agreement.py --plot` | sparsification_percentage |
| `clientfrac` | `clientfrac_en.py` (reconstructed from bytecode 2026-11-11; validated against Table B.1, 180/180 cells) | client_frac |

Extras: `paper_numbers.py` (all numbers cited, -> `output/paper_numbers.md`), `check_pendencies.py`,
`pack_heatmap_en.py`, `20_pack_frequency_individual.py` (packfreq). `15/26/28_*_pt.py` are Portuguese
(dissertation) and are not used by the manuscript.
`29_seed_variance.py` (seed tables, -> output/seed_variance) also copied. `used_logs.txt` lists every run folder these scripts opened (audit-hook trace). `./run_all.sh` regenerates everything (byte-reproducible PDFs). `deleted_logs_2026-11-11.txt` lists the log entries removed from `logs/`.
