#!/bin/bash
# experiments/lib.sh — Auto-dispatcher: tmux (local) vs SLURM (cluster).
#
# Inside a SLURM job ($SLURM_JOB_ID is set) → sources lib_slurm.sh
#   • No tmux; server + clients run as background processes.
#   • GPU_IDS derived automatically from SLURM_GPUS_ON_NODE.
#
# Outside SLURM → sources lib_tmux.sh
#   • Original tmux-based workflow (Docker or Singularity locally).
#
# Experiment scripts (e.g. sparsification_percentage.sh) don't need to
# change — they always end with:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
#   run_theme

_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    source "${_lib_dir}/lib_slurm.sh"
else
    source "${_lib_dir}/lib_tmux.sh"
fi
