#!/usr/bin/env python3
"""
experiments/generate_resume_jobs.py
Scan logs/ for interrupted experiments and generate resume.sh + resume_sbatch.sh
for each one that hasn't since been completed.

An experiment is interrupted when checkpoint/meta.json round < NUM_ROUNDS.
INCOMPLETE.log is NOT required: bash only writes it when the server exits
cleanly, so SLURM kills (wall-time, OOM, scancel, node failure) leave none.
Folders whose checkpoint changed in the last --min-idle minutes are treated as
still running and skipped.

Usage:
    python3 experiments/generate_resume_jobs.py
    python3 experiments/generate_resume_jobs.py --dry-run
    python3 experiments/generate_resume_jobs.py --partition p100 --time 1-00:00:00
    python3 experiments/generate_resume_jobs.py --logs-dir /custom/logs
    python3 experiments/generate_resume_jobs.py --target seed_experiment_v2
    python3 experiments/generate_resume_jobs.py --target 'seed_experiment*/mobilenetv2*sweepfl*' --target '*percentage0.7*'

Outputs paths of generated (or would-be generated) resume_sbatch.sh to stdout,
one per line.  Informational messages go to stderr.
"""

import argparse
import fnmatch
import glob
import json
import math
import os
import stat
import sys
import time
from collections import defaultdict
from datetime import datetime

PACKING_SOLUTIONS = frozenset(
    {"fedphe", "sweepfl", "fedphe-scout", "sweepfl-scout", "sw", "rw", "topkrr", "srr"}
)


DEBUG = False
_T0 = time.time()


def info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def debug(msg: str) -> None:
    if DEBUG:
        info(f"[DEBUG {time.time() - _T0:7.2f}s] {msg}")


# Never descend into these: huge (model snapshots) or not experiment roots.
_PRUNE_DIRS = frozenset({"model", "checkpoint", "archive", "models", "__pycache__"})


def find_meta_files(logs_dir: str) -> list:
    """Return every <exp>/checkpoint/meta.json under logs_dir.

    os.walk does not follow symlinks (recursive glob does, and loops on a
    symlink cycle) and we stop descending at experiment folders, so snapshot
    dirs with thousands of files are never listed.
    """
    found = []
    n_dirs = 0
    for root, dirs, files in os.walk(logs_dir):
        n_dirs += 1
        if n_dirs % 500 == 0:
            debug(f"walked {n_dirs} dirs, at {os.path.relpath(root, logs_dir)}")
        if "experiment_details.json" in files:
            dirs[:] = []  # experiment root: nothing interesting deeper
            meta = os.path.join(root, "checkpoint", "meta.json")
            if os.path.exists(meta):
                found.append(meta)
            continue
        dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
    debug(f"walk done: {n_dirs} dirs, {len(found)} checkpoints")
    return sorted(found)


def load_details(folder: str):
    path = os.path.join(folder, "experiment_details.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        info(f"  [WARN] Cannot read experiment_details.json in {folder}: {exc}")
        return None


def get_checkpoint_round(folder: str):
    meta_path = os.path.join(folder, "checkpoint", "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return int(meta.get("round", 0))
    except Exception:
        return None


def cf_token(details: dict) -> str:
    """LOGFOLDER token for partial participation (mirrors CF_TOKEN in lib_*.sh)."""
    cf = str(details.get("CLIENT_FRACTION", "1.0"))
    return "" if cf in ("1.0", "1", "1.00") else f"cf{cf}_"


def matches_target(logfolder_rel: str, targets: list) -> bool:
    """Glob match against the path relative to logs/; plain words match as substrings."""
    if not targets:
        return True
    return any(
        fnmatch.fnmatch(logfolder_rel, t if any(c in t for c in "*?[") else f"*{t}*")
        for t in targets
    )


def config_fingerprint(details: dict) -> tuple:
    """Stable key for an experiment config (excludes timestamp, includes NUM_ROUNDS)."""
    coeffs_str = ",".join(str(c) for c in details.get("COEFFS", []))
    return (
        details.get("FOLDER", ""),
        details.get("MODEL", ""),
        details.get("DATASET", ""),
        details.get("SOLUTION", ""),
        details.get("TECHNIQUE", ""),
        int(details.get("NUM_ROUNDS", 0)),
        int(details.get("NCLIENTES", 0)),
        str(details.get("SEED", 0)),
        str(details.get("NIID", 0)),
        str(details.get("PERCENTAGE", -1)),
        int(details.get("POLY", 0)),
        int(details.get("PLAIN_MODULUS", -1)),
        int(details.get("N_LENGTH", -1)),
        coeffs_str,
        int(details.get("SCALE", 0)),
        str(details.get("KEY_TYPE", "")),
        str(details.get("CLIENT_FRACTION", "1.0")),
    )


def is_completed(details: dict, logs_dir: str) -> bool:
    """Return True if any run with the same config has checkpoint round >= NUM_ROUNDS."""
    folder = details.get("FOLDER", "")
    model = details["MODEL"]
    dataset_folder = details["DATASET"].replace("/", "_")
    solution = details["SOLUTION"]
    technique = details["TECHNIQUE"]
    nclientes = details["NCLIENTES"]
    seed = details["SEED"]
    niid = details["NIID"]
    percentage = details["PERCENTAGE"]
    poly = details["POLY"]
    plain_modulus = details["PLAIN_MODULUS"]
    n_length = details["N_LENGTH"]
    coeffs_csv = ",".join(str(c) for c in details.get("COEFFS", []))
    scale = details["SCALE"]
    key_type = details["KEY_TYPE"]
    num_rounds = int(details["NUM_ROUNDS"])

    search_base = os.path.join(logs_dir, folder) if folder else logs_dir
    prefix = f"{model}_{dataset_folder}_{solution}_{technique}_"
    suffix = (
        f"_{nclientes}clients_seed{seed}_niid{niid}_percentage{percentage}"
        f"_{poly}_{plain_modulus}_{n_length}_{coeffs_csv}_{scale}_{key_type}_{cf_token(details)}"
    )
    pattern = os.path.join(search_base, f"{prefix}*rounds{suffix}????-??-??_??-??-??")

    debug(f"is_completed glob: {pattern}")
    for d in glob.glob(pattern):
        if not os.path.isdir(d):
            continue
        ckpt_round = get_checkpoint_round(d)
        if ckpt_round is not None and ckpt_round >= num_rounds:
            return True
    return False


def _find_lib_sh_block() -> str:
    """Generate the bash snippet that locates and sources experiments/lib.sh."""
    return (
        "# Locate repo root by walking up from this script's directory\n"
        "_SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "_REPO_ROOT=\"$_SCRIPT_DIR\"\n"
        "while [[ \"$_REPO_ROOT\" != \"/\" ]] && [[ ! -f \"${_REPO_ROOT}/experiments/lib.sh\" ]]; do\n"
        "    _REPO_ROOT=\"$(dirname \"$_REPO_ROOT\")\"\n"
        "done\n"
        "if [[ ! -f \"${_REPO_ROOT}/experiments/lib.sh\" ]]; then\n"
        "    echo \"ERROR: cannot find experiments/lib.sh from $_SCRIPT_DIR\"; exit 1\n"
        "fi\n"
        "source \"${_REPO_ROOT}/experiments/lib.sh\""
    )


def generate_resume_script(details: dict, logfolder_rel: str, completed_round: int) -> str:
    gpu_ids = details.get("GPU_IDS", [0])
    clients_per_gpu = details.get("CLIENTS_PER_GPU", 1)
    folder = details.get("FOLDER", "")
    model = details["MODEL"]
    dataset = details["DATASET"]
    solution = details["SOLUTION"]
    technique = details["TECHNIQUE"]
    niid = details["NIID"]
    seed = details["SEED"]
    percentage = details["PERCENTAGE"]
    num_rounds = int(details["NUM_ROUNDS"])
    experiment_name = details.get("EXPERIMENT_NAME", "")

    poly = details["POLY"]
    coeffs_csv = ",".join(str(c) for c in details.get("COEFFS", []))
    scale = int(details["SCALE"])
    scale_power = int(math.log2(scale)) if scale > 0 else 0
    key_type = details["KEY_TYPE"]
    plain_modulus = details["PLAIN_MODULUS"]
    n_length = details["N_LENGTH"]

    ckks_backend = details.get("CKKS_BACKEND", "tenseal")
    simplify_n = details.get("SIMPLIFY_N", 8192)
    simplify_bits = details.get("SIMPLIFY_BITS", 62)
    simplify_scale_bits = details.get("SIMPLIFY_SCALE_BITS", 60)
    pycrypt_n = details.get("PYCRYPT_N", 8192)
    pycrypt_bits = details.get("PYCRYPT_BITS", 60)
    openfhe_ring_dim = details.get("OPENFHE_RING_DIM", 8192)
    openfhe_mult_depth = details.get("OPENFHE_MULT_DEPTH", 1)
    openfhe_scale_bits = details.get("OPENFHE_SCALE_BITS", 50)
    batch_size = details.get("BATCH_SIZE", 4)
    rr_start_round = details.get("RR_START_ROUND", 200)
    magg_renorm = details.get("MAGG_RENORM", 1)
    client_fraction = details.get("CLIENT_FRACTION", 1.0)

    gpu_ids_str = " ".join(str(g) for g in gpu_ids)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lib_block = _find_lib_sh_block()

    return (
        "#!/bin/bash\n"
        f"# Auto-generated resume script\n"
        f"# Source folder : {logfolder_rel}\n"
        f"# Generated     : {ts}\n"
        f"# Interrupted at: round {completed_round} / {num_rounds}\n"
        "#\n"
        f"# Run locally : bash logs/{logfolder_rel}/resume.sh\n"
        f"# Run on SLURM: sbatch logs/{logfolder_rel}/resume_sbatch.sh\n"
        "\n"
        f"GPU_IDS=({gpu_ids_str})\n"
        f"CLIENTS_PER_GPU={clients_per_gpu}\n"
        "\n"
        f'FOLDER="{folder}"\n'
        f"SOLUTIONS=('{solution}')\n"
        f"NIIDS=({niid})\n"
        f"SEEDS=({seed})\n"
        f"PERCENTAGES=({percentage})\n"
        f"CLIENT_FRACTIONS=({client_fraction})\n"
        f"NUM_ROUNDS={num_rounds}\n"
        f'TECHNIQUE="{technique}"\n'
        f'EXPERIMENT_NAME="{experiment_name}"\n'
        "\n"
        f'MODEL_DATASET_PAIRS=("{model}:{dataset}")\n'
        "\n"
        f'CKKS_CONFIGS=("ckks {poly} {coeffs_csv}")\n'
        f"KEY_TYPES=('{key_type}')\n"
        f"SCALE_POWER={scale_power}\n"
        f"PLAIN_MODULUS={plain_modulus}\n"
        f"N_LENGTH={n_length}\n"
        "\n"
        f'CKKS_BACKEND="{ckks_backend}"\n'
        f"SIMPLIFY_N={simplify_n}\n"
        f"SIMPLIFY_BITS={simplify_bits}\n"
        f"SIMPLIFY_SCALE_BITS={simplify_scale_bits}\n"
        f"PYCRYPT_N={pycrypt_n}\n"
        f"PYCRYPT_BITS={pycrypt_bits}\n"
        f"OPENFHE_RING_DIM={openfhe_ring_dim}\n"
        f"OPENFHE_MULT_DEPTH={openfhe_mult_depth}\n"
        f"OPENFHE_SCALE_BITS={openfhe_scale_bits}\n"
        "\n"
        f"BATCH_SIZE={batch_size}\n"
        f"RR_START_ROUND={rr_start_round}\n"
        f"MAGG_RENORM={magg_renorm}\n"
        "\n"
        f"{lib_block}\n"
        "run_theme\n"
        "\n"
        "# ── Auto-merge: join this resumed experiment with the original folder ──────\n"
        '_MERGE_SCRIPT="${_REPO_ROOT}/experiments/merge_resume.py"\n'
        'if [[ -f "$_MERGE_SCRIPT" ]]; then\n'
        '    echo ""\n'
        '    echo "[merge] Merging resumed experiment into original folder..."\n'
        '    python3 "$_MERGE_SCRIPT" --logs-dir "${_REPO_ROOT}/logs" --yes\n'
        "fi\n"
    )


def generate_sbatch_script(
    details: dict,
    logfolder_rel: str,
    completed_round: int,
    partition: str,
    time_limit: str,
) -> str:
    gpu_ids = details.get("GPU_IDS", [0])
    n_gpus = len(gpu_ids)
    model = details["MODEL"]
    solution = details["SOLUTION"]
    folder = details.get("FOLDER", "")
    num_rounds = int(details["NUM_ROUNDS"])

    job_parts = ["resume"]
    if folder:
        job_parts.append(folder.replace("/", "_"))
    job_parts.extend([solution, model])
    job_name = "_".join(job_parts)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Bash variable expansions use {{ }} to escape Python f-string braces.
    return (
        "#!/bin/bash\n"
        f"# Auto-generated sbatch — resume of interrupted experiment\n"
        f"# Source folder : {logfolder_rel}\n"
        f"# Generated     : {ts}\n"
        f"# Interrupted at: round {completed_round} / {num_rounds}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --partition={partition}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --gres=gpu:{n_gpus}\n"
        "#SBATCH --cpus-per-task=12\n"
        f"#SBATCH --time={time_limit}\n"
        "#SBATCH --output=logs/slurm_%j.out\n"
        "#SBATCH --error=logs/slurm_%j.err\n"
        "#SBATCH --mail-type=FAIL,TIME_LIMIT\n"
        "#SBATCH --mail-user=y218172@dac.unicamp.br\n"
        "#SBATCH --mem=64G\n"
        "\n"
        "export RUNTIME=singularity\n"
        'export SIF_DIR="${SIF_DIR:-$HOME/keys}"\n'
        "export USE_GPU=true\n"
        "\n"
        'REPO="$HOME/keys"\n'
        'cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }\n'
        "\n"
        "_SBATCH_TRAP_REGISTERED=0\n"
        "_sbatch_exit_hook() {\n"
        "    local code=$?\n"
        "    if declare -f _slurm_fail_notify &>/dev/null; then\n"
        '        _slurm_fail_notify "$code"\n'
        "    fi\n"
        "}\n"
        "trap '_sbatch_exit_hook' EXIT\n"
        "\n"
        'echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"\n'
        'echo "Partition: $SLURM_JOB_PARTITION  |  GPUs on node: $SLURM_GPUS_ON_NODE"\n'
        'echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"\n'
        'echo ""\n'
        "\n"
        "mkdir -p logs\n"
        "\n"
        f'EXPERIMENT_SCRIPT="logs/{logfolder_rel}/resume.sh"\n'
        'bash "$EXPERIMENT_SCRIPT"\n'
        "\n"
        'echo ""\n'
        'echo "Job $SLURM_JOB_ID finished at $(date)"\n'
    )


def _make_executable(path: str) -> None:
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate resume scripts for interrupted FL experiments."
    )
    parser.add_argument(
        "--logs-dir", default="logs", help="Root logs directory (default: logs/)"
    )
    parser.add_argument(
        "--partition", default="l40s", help="SLURM partition (default: l40s)"
    )
    parser.add_argument(
        "--time", default="2-00:00:00", help="SLURM wall-time limit (default: 2-00:00:00)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating files",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Only resume folders whose path relative to logs/ matches this glob "
        "(plain words match as substrings). Repeatable; OR semantics.",
    )
    parser.add_argument(
        "--min-idle",
        type=float,
        default=60,
        help="Skip folders whose checkpoint was updated less than N minutes ago "
        "(probably still running). Default: 60",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print timed progress to stderr"
    )
    # Accepted and ignored — used by resume_all.sh to suppress submission
    parser.add_argument("--generate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug
    logs_dir = os.path.abspath(args.logs_dir)
    debug(f"logs_dir={logs_dir} targets={args.target} min_idle={args.min_idle}")
    if not os.path.isdir(logs_dir):
        info(f"[WARN] Logs directory not found: {logs_dir}")
        return

    meta_files = find_meta_files(logs_dir)

    # Collect valid candidates: checkpoint round < NUM_ROUNDS
    candidates = []
    now = time.time()
    for meta in meta_files:
        folder_abs = os.path.dirname(os.path.dirname(meta))
        logfolder_rel = os.path.relpath(folder_abs, logs_dir)
        # archive/ holds merge backups and already-merged resumed folders
        if "archive" in logfolder_rel.split(os.sep):
            continue
        if not matches_target(logfolder_rel, args.target):
            continue
        debug(f"candidate: {logfolder_rel}")
        details = load_details(folder_abs)
        if details is None:
            info(f"  [SKIP] No experiment_details.json: {logfolder_rel}")
            continue
        ckpt_round = get_checkpoint_round(folder_abs)
        if ckpt_round is None or ckpt_round >= int(details.get("NUM_ROUNDS", 0)):
            continue
        idle_min = (now - os.path.getmtime(meta)) / 60
        if idle_min < args.min_idle:
            info(f"  [SKIP] Updated {idle_min:.0f} min ago (running?): {logfolder_rel}")
            continue
        candidates.append((folder_abs, logfolder_rel, details, ckpt_round))

    info(f"[INFO] Found {len(candidates)} interrupted experiment folder(s)"
         + (f" matching {args.target}" if args.target else "") + ".")

    if not candidates:
        info("[INFO] No valid incomplete experiments found.")
        return

    # Group by config fingerprint, deduplicate
    fingerprint_groups: dict = defaultdict(list)
    for entry in candidates:
        fp = config_fingerprint(entry[2])
        fingerprint_groups[fp].append(entry)

    generated = []

    for fp, group in sorted(fingerprint_groups.items()):
        rep_details = group[0][2]

        if is_completed(rep_details, logs_dir):
            for _, logfolder_rel, _, _ in group:
                info(f"  [SKIP] Already completed: {logfolder_rel}")
            continue

        # Best candidate = highest checkpoint round
        best = max(group, key=lambda e: e[3])
        folder_abs, logfolder_rel, details, ckpt_round = best

        if len(group) > 1:
            skipped = [e[1] for e in group if e is not best]
            info(
                f"  [DEDUP] {len(group)} candidates; using best (round={ckpt_round}): {logfolder_rel}"
            )
            for s in skipped:
                info(f"    └─ skipped: {s}")

        num_rounds = int(details["NUM_ROUNDS"])
        info(f"  [GEN] round {ckpt_round}/{num_rounds}: {logfolder_rel}")

        resume_sh = os.path.join(folder_abs, "resume.sh")
        resume_sbatch = os.path.join(folder_abs, "resume_sbatch.sh")

        if not args.dry_run:
            with open(resume_sh, "w") as f:
                f.write(generate_resume_script(details, logfolder_rel, ckpt_round))
            _make_executable(resume_sh)

            with open(resume_sbatch, "w") as f:
                f.write(
                    generate_sbatch_script(
                        details, logfolder_rel, ckpt_round, args.partition, args.time
                    )
                )
            _make_executable(resume_sbatch)

        generated.append(resume_sbatch)

    if generated:
        qualifier = " (dry run)" if args.dry_run else ""
        info(f"\n[INFO] {len(generated)} resume job(s) generated{qualifier}.")
        for path in generated:
            print(path)
    else:
        info("[INFO] Nothing to generate.")


if __name__ == "__main__":
    main()
