#!/usr/bin/env python3
"""
check_status.py — Audit which experiments have been done, are partial, or are missing.

Usage:
    python3 experiments/check_status.py
    python3 experiments/check_status.py --all          # show done + partial + missing
    python3 experiments/check_status.py --scripts sparsification_percentage_sw.sh
    python3 experiments/check_status.py --no-color
    python3 experiments/check_status.py --summary-only
    python3 experiments/check_status.py --include-test
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from datetime import date
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

PACKING_SOLUTIONS = frozenset(
    {"fedphe", "nvnl", "fedphe-scout", "nvnl-scout", "sw", "rw", "topkrr", "srr"}
)

# Scripts to exclude even when they contain run_theme
EXCLUDE_SCRIPTS = frozenset(
    {"lib.sh", "lib_tmux.sh", "lib_slurm.sh", "template.sh"}
)

# Test scripts excluded by default
TEST_SCRIPTS_RE = re.compile(r"^test_")

# ── ANSI colors ───────────────────────────────────────────────────────────────

class Colors:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"

USE_COLOR = True

def _c(text, *codes):
    if not USE_COLOR:
        return text
    return "".join(codes) + text + Colors.RESET

def green(t):  return _c(t, Colors.GREEN)
def yellow(t): return _c(t, Colors.YELLOW)
def red(t):    return _c(t, Colors.RED)
def cyan(t):   return _c(t, Colors.CYAN)
def bold(t):   return _c(t, Colors.BOLD)
def dim(t):    return _c(t, Colors.DIM)

# ── Bash parser ───────────────────────────────────────────────────────────────

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or \
       (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def parse_bash_script(path: str) -> dict:
    """
    Parse a bash experiment script and return a dict of config variables.
    Handles:
    - Multi-line array definitions
    - Commented-out entries inside arrays (lines containing only # text)
    - Both single and double quoted values
    """
    with open(path) as f:
        content = f.read()

    lines = content.splitlines()

    cfg = {
        # Defaults matching the template / lib defaults
        "GPU_IDS": [],
        "CLIENTS_PER_GPU": 3,
        "FOLDER": "",
        "EXPERIMENT_NAME": "",
        "NUM_ROUNDS": 50,
        "NIIDS": ["0.1"],
        "SEEDS": ["42"],
        "TECHNIQUE": "none",
        "SOLUTIONS": [],
        "PERCENTAGES": [],
        "MODEL_DATASET_PAIRS": [],
        "CKKS_CONFIGS": ["ckks 8192 60,40,40,60"],
        "KEY_TYPES": ["None"],
        "SCALE_POWER": 40,
        "PLAIN_MODULUS": -1,
        "N_LENGTH": -1,
        "CKKS_BACKEND": "tenseal",
        "CLIENT_FRACTIONS": ["1.0"],
    }

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Skip full-line comments and empty lines at top level
        if line.startswith("#") or not line:
            i += 1
            continue

        # ── Detect array assignment ──────────────────────────────────────────
        # Matches: VAR=( ... ) possibly on one line, or opening of multi-line
        arr_inline = re.match(r'^(\w+)=\(([^)]*)\)\s*(?:#.*)?$', line)
        arr_open   = re.match(r'^(\w+)=\(\s*(?:#.*)?$', line)

        if arr_inline:
            varname, raw_val = arr_inline.group(1), arr_inline.group(2)
            # Use tokenizer to handle quoted strings with spaces inside them
            items = _tokenize_array_line(raw_val)
            _set_cfg(cfg, varname, items)
            i += 1
            continue

        if arr_open:
            varname = arr_open.group(1)
            items = []
            i += 1
            # Collect lines until closing )
            while i < len(lines):
                inner = lines[i].strip()
                i += 1
                if inner == ")" or inner.startswith(")"):
                    break
                # Skip commented-out entries
                if inner.startswith("#") or not inner:
                    continue
                # Strip inline comment from the value
                inner = re.sub(r'\s+#.*$', '', inner)
                for token in _tokenize_array_line(inner):
                    items.append(token)
            _set_cfg(cfg, varname, items)
            continue

        # ── Detect scalar assignment ─────────────────────────────────────────
        # VAR=value  or  VAR="value"  (but not VAR+=...)
        scalar = re.match(r'^(\w+)=([^(].*?)?\s*(?:#.*)?$', line)
        if scalar:
            varname, val = scalar.group(1), (scalar.group(2) or "").strip()
            val = _strip_quotes(val)
            _set_cfg_scalar(cfg, varname, val)
            i += 1
            continue

        # ── Multi-assignment on one line: A=x; B=y ──────────────────────────
        # e.g. SIMPLIFY_N=8192; SIMPLIFY_BITS=62; SIMPLIFY_SCALE_BITS=40
        if ";" in line:
            for part in line.split(";"):
                part = part.strip()
                m = re.match(r'^(\w+)=([^(].*?)?\s*(?:#.*)?$', part)
                if m:
                    vn, vv = m.group(1), (m.group(2) or "").strip()
                    _set_cfg_scalar(cfg, vn, _strip_quotes(vv))

        i += 1

    return cfg


def _tokenize_array_line(line: str) -> list:
    """
    Tokenize a line from inside a bash array, handling quoted strings with spaces.
    e.g. '"ckks 8192 60"' → ['ckks 8192 60']
         '"dnn_tiny:FASHION_MNIST" "lenet5:FASHION_MNIST"' → [...]
    """
    tokens = []
    # Match quoted strings or unquoted words
    for m in re.finditer(r'"([^"]*)"' + r"|'([^']*)'" + r'|(\S+)', line):
        tokens.append(m.group(1) or m.group(2) or m.group(3))
    return tokens


def _parse_array_items(tokens: list) -> list:
    """Strip quotes from each token in an inline array."""
    result = []
    for t in tokens:
        t = _strip_quotes(t)
        if t and not t.startswith("#"):
            result.append(t)
    return result


def _set_cfg(cfg: dict, varname: str, items: list):
    """Map a bash array name to a cfg key and store the parsed items."""
    if varname in cfg:
        cfg[varname] = items
    # Also handle a few common aliases
    elif varname == "GPU_IDS":
        cfg["GPU_IDS"] = items


def _set_cfg_scalar(cfg: dict, varname: str, val: str):
    """Map a bash scalar to a cfg key, with type coercion."""
    if varname == "GPU_IDS":
        # Sometimes written as scalar-ish (rare) — skip; handled by array parser
        return
    # Backward-compat: old scripts use scalar CLIENT_FRACTION=<val>;
    # new scripts use array CLIENT_FRACTIONS=(...).  Map the old form.
    if varname == "CLIENT_FRACTION":
        cfg["CLIENT_FRACTIONS"] = [val]
        return
    int_vars = {"CLIENTS_PER_GPU", "NUM_ROUNDS", "SCALE_POWER", "PLAIN_MODULUS", "N_LENGTH"}
    if varname in int_vars:
        try:
            cfg[varname] = int(val)
        except (ValueError, TypeError):
            pass
    elif varname in cfg:
        cfg[varname] = val


# ── Experiment enumeration ────────────────────────────────────────────────────

def enumerate_expected(cfg: dict) -> list:
    """
    Replicate run_theme()'s nested loop to produce all expected experiment keys.
    Returns a list of dicts, each describing one expected experiment.
    """
    gpu_ids = cfg.get("GPU_IDS", [])
    clients_per_gpu = int(cfg.get("CLIENTS_PER_GPU", 3))
    nclientes = len(gpu_ids) * clients_per_gpu

    folder         = cfg.get("FOLDER", "")
    num_rounds     = int(cfg.get("NUM_ROUNDS", 50))
    technique      = cfg.get("TECHNIQUE", "none")
    plain_modulus  = cfg.get("PLAIN_MODULUS", -1)
    n_length       = cfg.get("N_LENGTH", -1)
    scale_power    = int(cfg.get("SCALE_POWER", 40))
    scale          = 2 ** scale_power

    solutions        = cfg.get("SOLUTIONS", [])
    percentages      = cfg.get("PERCENTAGES", [])
    niids            = cfg.get("NIIDS", [])
    seeds            = cfg.get("SEEDS", [])
    model_pairs      = cfg.get("MODEL_DATASET_PAIRS", [])
    ckks_configs     = cfg.get("CKKS_CONFIGS", ["ckks 8192 60,40,40,60"])
    key_types        = cfg.get("KEY_TYPES", ["None"])
    client_fractions = cfg.get("CLIENT_FRACTIONS", ["1.0"])

    experiments = []

    for config in ckks_configs:
        parts = config.split()
        if len(parts) < 3:
            continue
        schema, poly, coeffs = parts[0], parts[1], parts[2]

        for key_type in key_types:
            for solution in solutions:
                for pair in model_pairs:
                    if ":" not in pair:
                        continue
                    model, dataset = pair.split(":", 1)
                    dataset_folder = dataset.replace("/", "_")

                    for niid in niids:
                        for seed in seeds:
                            if solution in PACKING_SOLUTIONS:
                                iter_pct = percentages if percentages else ["-1"]
                            else:
                                iter_pct = ["-1"]

                            for cf in client_fractions:
                                for pct in iter_pct:
                                    experiments.append({
                                        "folder":          folder,
                                        "model":           model,
                                        "dataset":         dataset,
                                        "dataset_folder":  dataset_folder,
                                        "solution":        solution,
                                        "technique":       technique,
                                        "num_rounds":      num_rounds,
                                        "nclientes":       nclientes,
                                        "seed":            str(seed),
                                        "niid":            str(niid),
                                        "pct":             str(pct),
                                        "poly":            str(poly),
                                        "plain_modulus":   str(plain_modulus),
                                        "n_length":        str(n_length),
                                        "coeffs":          str(coeffs),
                                        "scale":           str(scale),
                                        "key_type":        str(key_type),
                                        "ckks_backend":    cfg.get("CKKS_BACKEND", "tenseal"),
                                        "client_fraction": str(cf),
                                    })
    return experiments


# ── Log scanning ──────────────────────────────────────────────────────────────

def get_max_round(logfolder: str) -> int:
    """Return the maximum completed round from server_eval.csv, or 0 if absent."""
    csv_path = os.path.join(logfolder, "server_eval.csv")
    if not os.path.isfile(csv_path):
        return 0
    last_line = ""
    try:
        with open(csv_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
    except OSError:
        return 0
    if not last_line or last_line.startswith("experiment_name"):
        return 0
    parts = last_line.split(",")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return 0


def read_experiment_details(logfolder: str) -> dict:
    """Read experiment_details.json from a log folder. Returns {} if absent/invalid."""
    path = os.path.join(logfolder, "experiment_details.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def find_log_matches(exp: dict, logs_dir: str) -> list:
    """
    Glob for log folders matching the experiment, ignoring client count and timestamp.
    Returns list of (max_round, logfolder_path) sorted best-first.
    """
    folder        = exp["folder"]
    model         = exp["model"]
    dataset_folder= exp["dataset_folder"]
    solution      = exp["solution"]
    technique     = exp["technique"]
    num_rounds    = exp["num_rounds"]
    seed          = exp["seed"]
    niid          = exp["niid"]
    pct           = exp["pct"]
    poly          = exp["poly"]
    plain_modulus = exp["plain_modulus"]
    n_length      = exp["n_length"]
    coeffs        = exp["coeffs"]
    scale         = exp["scale"]
    key_type      = exp["key_type"]

    prefix = os.path.join(logs_dir, folder) if folder else logs_dir

    # Token condicional cf<frac> — espelha a regra de start_all.sh / lib_*.sh:
    # fração 1.0 → sem token (casa pastas existentes); fração != 1.0 → "cf<frac>_".
    client_fraction = str(exp.get("client_fraction", "1.0"))
    cf_token = "" if client_fraction in ("1.0", "1", "1.00") else f"cf{client_fraction}_"
    pattern = (
        f"{model}_{dataset_folder}_{solution}_{technique}_"
        f"{num_rounds}rounds_*clients_"
        f"seed{seed}_niid{niid}_percentage{pct}_"
        f"{poly}_{plain_modulus}_{n_length}_{coeffs}_{scale}_{key_type}_"
        f"{cf_token}????-??-??_??-??-??"
    )

    full_pattern = os.path.join(prefix, pattern)
    matches = glob.glob(full_pattern)

    results = []
    for m in matches:
        r = get_max_round(m)
        results.append((r, m))

    # Best first: highest round, then most recent folder name (lexicographic)
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return results


def classify(exp: dict, logs_dir: str) -> tuple:
    """
    Returns (status, max_round, best_logfolder, found_backend) where status is
    'done', 'partial', 'missing', or 'wrong_backend'.

    found_backend is non-empty only when status == 'wrong_backend', indicating
    the CKKS_BACKEND actually recorded in experiment_details.json.
    """
    matches = find_log_matches(exp, logs_dir)
    if not matches:
        return ("missing", 0, "", "")

    expected_backend = exp.get("ckks_backend", "")
    valid   = []   # (max_round, folder)
    wrong   = []   # (max_round, folder, found_backend)

    for max_round, folder in matches:
        details      = read_experiment_details(folder)
        found_backend = details.get("CKKS_BACKEND", "")
        if not found_backend or found_backend == expected_backend:
            valid.append((max_round, folder))
        else:
            wrong.append((max_round, folder, found_backend))

    if valid:
        best_round, best_folder = valid[0]
        num_rounds = exp["num_rounds"]
        if best_round >= num_rounds:
            return ("done",    best_round, best_folder, "")
        elif best_round > 0:
            return ("partial", best_round, best_folder, "")
        else:
            return ("partial", 0,          best_folder, "")

    if wrong:
        best_round, best_folder, found = wrong[0]
        return ("wrong_backend", best_round, best_folder, found)

    return ("missing", 0, "", "")


# ── Output rendering ──────────────────────────────────────────────────────────

STATUS_ICON = {
    "done":          "✅",
    "partial":       "⚠️ ",
    "missing":       "❌",
    "wrong_backend": "🔀",
}
STATUS_LABEL = {
    "done":          "done",
    "partial":       "partial",
    "missing":       "missing",
    "wrong_backend": "wrong backend",
}

def _status_str(status: str, extra: str = "") -> str:
    """Return a colored status string. extra appends info (e.g. found backend)."""
    icon  = STATUS_ICON.get(status, "?")
    label = STATUS_LABEL.get(status, status)
    suffix = f" ({extra})" if extra else ""
    if USE_COLOR:
        if status == "done":
            return f"{icon} {green(label)}{suffix}"
        elif status == "partial":
            return f"{icon} {yellow(label)}{suffix}"
        elif status == "wrong_backend":
            return f"{icon} {yellow(label)}{suffix}"
        else:
            return f"{icon} {red(label)}{suffix}"
    return f"{icon} {label}{suffix}"


def _col_widths(rows: list, headers: list) -> list:
    """Compute column widths for a table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            # Strip ANSI codes for width calculation
            plain = re.sub(r'\033\[[0-9;]*m', '', str(cell))
            widths[j] = max(widths[j], len(plain))
    return widths


def _pad(cell: str, width: int) -> str:
    """Pad a cell to the given display width (ignoring ANSI codes)."""
    plain_len = len(re.sub(r'\033\[[0-9;]*m', '', str(cell)))
    return str(cell) + " " * (width - plain_len)


def render_table(headers: list, rows: list, title: str = ""):
    """Print a Unicode box-drawing table."""
    widths = _col_widths(rows, headers)

    top    = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    mid    = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bottom = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    def row_str(cells):
        return "│" + "│".join(
            " " + _pad(c, widths[i]) + " " for i, c in enumerate(cells)
        ) + "│"

    if title:
        print(title)
    print(top)
    print(row_str([bold(h) for h in headers]))
    print(mid)
    for row in rows:
        print(row_str(row))
    print(bottom)


def render_script_section(script_name: str, cfg: dict, results: list, show_all: bool):
    """
    Print the section for one script.
    results: list of (exp, status, max_round, logfolder, found_backend)
    """
    folder     = cfg.get("FOLDER", "(no folder)")
    num_rounds = cfg.get("NUM_ROUNDS", "?")
    total         = len(results)
    n_done        = sum(1 for r in results if r[1] == "done")
    n_partial     = sum(1 for r in results if r[1] == "partial")
    n_missing     = sum(1 for r in results if r[1] == "missing")
    n_wrong       = sum(1 for r in results if r[1] == "wrong_backend")

    # Filter rows based on mode
    if show_all:
        display = results
    else:
        display = [r for r in results if r[1] in ("partial", "missing", "wrong_backend")]

    script_title = bold(f"\nScript: {script_name}") + dim(f"    [FOLDER: {folder}]")
    print(script_title)

    if not display:
        # Nothing to show (all done)
        print(green(f"  ✅ All {total} experiments done!"))
    else:
        headers = [
            "Solution", "Model", "Dataset", "niid", "%pct", "cf%",
            "seed", "clients", "poly", "coeffs", "scale",
            "backend", "Status", "Rounds",
        ]
        rows = []
        for exp, status, max_round, logfolder, found_backend in display:
            pct_str    = exp["pct"] if exp["pct"] != "-1" else "—"
            cf_str     = exp.get("client_fraction", "1.0")
            rounds_str = f"{max_round}/{num_rounds}" if max_round else f"—/{num_rounds}"
            try:
                scale_exp = int(math.log2(int(exp["scale"])))
                scale_str = f"2^{scale_exp}"
            except (ValueError, ZeroDivisionError):
                scale_str = exp["scale"]
            # For wrong_backend: show the found backend in the status cell
            status_cell = _status_str(status, found_backend if status == "wrong_backend" else "")
            rows.append([
                exp["solution"],
                exp["model"],
                exp["dataset_folder"],
                exp["niid"],
                pct_str,
                cf_str,
                exp["seed"],
                str(exp["nclientes"]),
                exp["poly"],
                exp["coeffs"],
                scale_str,
                exp.get("ckks_backend", "tenseal"),
                status_cell,
                rounds_str,
            ])
        render_table(headers, rows)

    # Per-script summary line
    parts = []
    if n_done:    parts.append(green(f"{n_done} ✅"))
    if n_partial: parts.append(yellow(f"{n_partial} ⚠️ "))
    if n_wrong:   parts.append(yellow(f"{n_wrong} 🔀"))
    if n_missing: parts.append(red(f"{n_missing} ❌"))
    summary = "  " + "  ".join(parts) + dim(f"  (total {total})")
    print(summary)


# ── Script discovery ──────────────────────────────────────────────────────────

def discover_scripts(experiments_dir: str, include_test: bool,
                     filter_scripts: list) -> list:
    """
    Find all .sh files that call run_theme and aren't lib/template/test files.
    Returns list of script paths sorted alphabetically.
    """
    candidates = sorted(Path(experiments_dir).rglob("*.sh"))
    result = []
    for p in candidates:
        name = p.name
        if name in EXCLUDE_SCRIPTS:
            continue
        if not include_test and TEST_SCRIPTS_RE.match(name):
            continue
        if filter_scripts and name not in filter_scripts:
            # Also allow without .sh extension
            if name.removesuffix(".sh") not in filter_scripts:
                continue
        # Only include scripts that call run_theme
        try:
            text = p.read_text()
        except OSError:
            continue
        if "run_theme" not in text:
            continue
        # Exclude scripts that only define run_theme (lib files)
        if re.search(r'^run_theme\s*\(\)', text, re.MULTILINE):
            continue
        result.append(str(p))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="Audit experiment status: done / partial / missing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--logs-dir", default="logs",
                        help="Path to the logs directory (default: logs/)")
    parser.add_argument("--experiments-dir", default="experiments",
                        help="Path to the experiments directory (default: experiments/)")
    parser.add_argument("--scripts", nargs="+", default=[],
                        help="Only check these script names (e.g. sparsification_percentage_sw.sh)")
    parser.add_argument("--include-test", action="store_true",
                        help="Include test_*.sh scripts")
    parser.add_argument("--no-color", action="store_true",
                        help="Plain text output without ANSI colors")
    parser.add_argument("--all", dest="show_all", action="store_true",
                        help="Show done + partial + missing (default: only partial + missing)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only the global summary table")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        USE_COLOR = False

    logs_dir = args.logs_dir
    exp_dir  = args.experiments_dir

    # ── Discover scripts ─────────────────────────────────────────────────────
    scripts = discover_scripts(exp_dir, args.include_test, args.scripts)
    if not scripts:
        print("No experiment scripts found.", file=sys.stderr)
        sys.exit(1)

    # ── Header ───────────────────────────────────────────────────────────────
    today = date.today().isoformat()
    sep = "═" * 65
    print(bold(sep))
    print(bold(f" 📋  Experiment Status — {today}"))
    print(dim(f" Logs dir      : {logs_dir}/"))
    print(dim(f" Experiments   : {exp_dir}/"))
    print(bold(sep))

    # ── Per-script processing ─────────────────────────────────────────────────
    total_expected = 0
    total_done     = 0
    total_partial  = 0
    total_wrong    = 0
    total_missing  = 0
    summary_rows   = []   # for summary table

    for script_path in scripts:
        script_name = os.path.basename(script_path)
        try:
            cfg = parse_bash_script(script_path)
        except Exception as e:
            print(f"\n⚠️  Could not parse {script_name}: {e}", file=sys.stderr)
            continue

        expected = enumerate_expected(cfg)
        if not expected:
            # Script parsed but no combinations produced (e.g. empty MODEL_DATASET_PAIRS)
            print(f"\n{dim(script_name)}: {yellow('no experiments enumerated (check MODEL_DATASET_PAIRS)')}")
            continue

        results = []
        for exp in expected:
            status, max_round, logfolder, found_backend = classify(exp, logs_dir)
            results.append((exp, status, max_round, logfolder, found_backend))

        n_done    = sum(1 for r in results if r[1] == "done")
        n_partial = sum(1 for r in results if r[1] == "partial")
        n_wrong   = sum(1 for r in results if r[1] == "wrong_backend")
        n_missing = sum(1 for r in results if r[1] == "missing")
        total_expected += len(results)
        total_done     += n_done
        total_partial  += n_partial
        total_wrong    += n_wrong
        total_missing  += n_missing

        summary_rows.append((script_name, len(results), n_done, n_partial, n_wrong, n_missing))

        if not args.summary_only:
            render_script_section(script_name, cfg, results, args.show_all)

    # ── Global summary ────────────────────────────────────────────────────────
    print()
    print(bold(sep))
    print(bold(" SUMMARY"))
    print(bold(sep))

    sum_headers = ["Script", "Total", "✅ Done", "⚠️  Partial", "🔀 Wrong cfg", "❌ Missing"]
    sum_rows = []
    for name, tot, nd, np_, nw, nm in summary_rows:
        sum_rows.append([
            name,
            str(tot),
            green(str(nd))    if nd else dim("0"),
            yellow(str(np_))  if np_ else dim("0"),
            yellow(str(nw))   if nw else dim("0"),
            red(str(nm))      if nm else dim("0"),
        ])
    render_table(sum_headers, sum_rows)

    pct_done    = total_done    * 100 // total_expected if total_expected else 0
    pct_partial = total_partial * 100 // total_expected if total_expected else 0
    pct_wrong   = total_wrong   * 100 // total_expected if total_expected else 0
    pct_missing = total_missing * 100 // total_expected if total_expected else 0

    print()
    print(f"  Scripts scanned : {len(scripts)}")
    print(f"  Total expected  : {total_expected}")
    print(f"  {green('✅ Done')         :30s}: {total_done:4d}  ({pct_done}%)")
    print(f"  {yellow('⚠️  Partial')    :30s}: {total_partial:4d}  ({pct_partial}%)")
    print(f"  {yellow('🔀 Wrong backend'):30s}: {total_wrong:4d}  ({pct_wrong}%)")
    print(f"  {red('❌ Missing')        :30s}: {total_missing:4d}  ({pct_missing}%)")

    if total_missing == 0 and total_partial == 0 and total_wrong == 0:
        print()
        print(green(bold("  🎉 All experiments are done!")))
    print()


if __name__ == "__main__":
    main()
