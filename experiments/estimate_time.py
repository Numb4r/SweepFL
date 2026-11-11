#!/usr/bin/env python3
"""
estimate_time.py — Estimador de tempo de execução para experimentos FL Slidhe.

Uso:
    python3 experiments/estimate_time.py experiments/sparsification_percentage.sh
    python3 experiments/estimate_time.py experiments/sparsification_percentage.sh --machine h100
    python3 experiments/estimate_time.py experiments/sparsification_percentage.sh --all
    python3 experiments/estimate_time.py experiments/sparsification_percentage.sh --margin 2.0

Veja experiments/ESTIMATION_METHODOLOGY.md para a descrição completa do modelo.
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

# ─── Constantes de referência ─────────────────────────────────────────────────
# Medidos empiricamente nos logs disponíveis:
#   resnet20 / CIFAR10 / 2 clients / batch=4 / CPU (esta máquina)
REF_TRAIN_TIME_S   = 163.0   # segundos por round (1 época) por cliente (GPU-bound)
REF_N_PARAMS       = 274_042
REF_DATASET        = "CIFAR10"
REF_DATASET_SIZE   = 50_000  # amostras de treino totais
REF_N_CLIENTS      = 2
REF_BATCH_SIZE     = 4

# Referência HE (CPU-bound): encrypt + pack + serialize + decrypt por cliente por round
REF_HE_TIME_S      = 0.50    # segundos HE por cliente por round na máquina local
REF_POLY_DEGREE    = 8_192   # poly_degree da referência → chunk_size = 4096
REF_N_CHUNKS       = math.ceil(REF_N_PARAMS / (REF_POLY_DEGREE // 2))  # ≈ 67

# Parâmetros conhecidos dos modelos (n_params real, corrigido por benchmark_resources)
MODEL_PARAMS = {
    "resnet20":    274_042,
    "lenet5":       61_706,   # medido: 61706 (era 44000)
    "dnn":         309_000,
    "dnn_base":    309_000,
    "dnn_tiny":    111_146,   # medido: 111146 (era 111000)
    "mobilenetv2": 719_034,   # medido: 719034 (era 460000)
    "vit_tiny":    206_000,
    "simple_cnn":   80_000,
}

# Batch size padrão por modelo (usado quando não especificado no script)
MODEL_BATCH_SIZE = {
    "mobilenetv2": 32,
    "vit_tiny":    16,
    # todos os outros: REF_BATCH_SIZE=4
}

# Referências empíricas medidas em benchmark_resources (2026-05-14)
# Chave: (model_lower, dataset, batch_size) → s/round/cliente (CPU, plaintext, 2 clientes)
EMPIRICAL_TRAIN_TIME_S = {
    ("dnn_tiny",    "FASHION_MNIST", 4):  11.1,
    ("lenet5",      "FASHION_MNIST", 4):  19.4,
    ("mobilenetv2", "CIFAR10",       32): 242.7,
}

# RAM medida em benchmark_resources (peak_mb por cliente, plaintext CPU, 2 clientes)
RAM_BASE_MB          = 6_500   # overhead fixo TF/Keras+dataset (dominante em modelos pequenos)
RAM_PER_REF_PARAMS_MB =  700   # MB adicionais por "unidade REF_N_PARAMS" de parâmetros

MODEL_RAM_PEAK_MB = {   # medido empiricamente; fonte: benchmark_resources/*/train.csv
    "dnn_tiny":    6_720,
    "lenet5":      6_720,
    "mobilenetv2": 8_520,
}

# Payload serializado por cliente por round (bytes, plaintext, fonte: benchmark_resources)
MODEL_PAYLOAD_BYTES = {
    "dnn_tiny":    903_272,
    "lenet5":      547_048,
    "mobilenetv2": 6_310_040,
}

# Tamanho dos datasets (amostras de treino)
DATASET_SIZE = {
    "FASHION_MNIST":          60_000,
    "CIFAR10":                50_000,
    "CIFAR100":               50_000,
    "tanganke/gtsrb":         39_209,
    "AISNP/PathMNIST":        89_996,
    "randall-lab/medmnist":   10_000,
}

# Soluções que iteram sobre PERCENTAGES
PACKING_SOLUTIONS = {"fedphe", "sweepfl", "sw", "rw", "fedphe-scout", "sweepfl-scout"}

# Overhead de setup por experimento (geração de chaves, startup de containers)
SETUP_OVERHEAD_S = 90.0


# ─── Leitura do SLURM.md ──────────────────────────────────────────────────────

def parse_slurm_md(slurm_path: Path) -> dict:
    """
    Lê o SLURM.md e retorna dict keyed por partição.
    """
    machines = {}
    if not slurm_path.exists():
        print(f"[AVISO] SLURM.md não encontrado em {slurm_path}. Usando apenas 'local'.",
              file=sys.stderr)
        return {"local": {"gpu": "CPU", "max_days": None, "speed": 1.0,
                          "gpu_speed": 1.0, "cpu_speed": 1.0, "note": "Referência"}}

    content = slurm_path.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines()]

    header_idx = None
    col_map = {}
    for i, line in enumerate(lines):
        if "|" not in line:
            continue
        cells = [c.strip().lower() for c in line.split("|") if c.strip()]
        if len(cells) < 4:
            continue
        if "partição" in cells or "particao" in cells or "partição" in "".join(cells):
            for j, c in enumerate(cells):
                if "parti" in c:
                    col_map["partition"] = j
                elif "nó" in c or "no" == c:
                    col_map["nodes"] = j
                elif "cpu speed" in c or "cpu_speed" in c:
                    col_map["cpu_speed"] = j
                elif "cpu" in c and "speed" not in c:
                    col_map["cpu"] = j
                elif "gpu speed" in c or "gpu_speed" in c:
                    col_map["gpu_speed"] = j
                elif "gpu" in c and "speed" not in c:
                    col_map["gpu"] = j
                elif "tempo" in c or "máx" in c or "max" in c:
                    col_map["max_time"] = j
                elif "speed" in c or "veloc" in c:
                    col_map["gpu_speed"] = j
                elif "nota" in c or "note" in c:
                    col_map["note"] = j
            header_idx = i
            break

    if header_idx is None:
        print("[AVISO] Tabela de partições não encontrada no SLURM.md.", file=sys.stderr)
        return {}

    has_speed     = "speed" in col_map or "gpu_speed" in col_map
    has_cpu_speed = "cpu_speed" in col_map
    if not has_speed:
        print("[AVISO] Coluna 'GPU Speed' / 'Speed' não encontrada no SLURM.md.",
              file=sys.stderr)

    for line in lines[header_idx + 1:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if all(c.startswith("-") for c in cells):
            continue
        if not cells:
            continue

        def get(key, default=""):
            idx = col_map.get(key)
            return cells[idx] if idx is not None and idx < len(cells) else default

        raw_partition = get("partition")
        partition = re.sub(r"\*+", "", raw_partition).strip()
        partition = re.sub(r"\(.*?\)", "", partition).strip()
        if not partition or partition.startswith("-"):
            continue

        gpu  = get("gpu")
        note = get("note")

        max_time_raw = get("max_time", "")
        max_days = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*dia", max_time_raw, re.I)
        if not m:
            m = re.search(r"(\d+(?:\.\d+)?)\s*day", max_time_raw, re.I)
        if m:
            max_days = float(m.group(1))

        gpu_speed = 1.0
        if has_speed:
            speed_raw = get("gpu_speed", "1.0")
            try:
                gpu_speed = float(re.sub(r"[^\d.]", "", speed_raw) or "1.0")
            except ValueError:
                gpu_speed = 1.0

        cpu_speed = 1.0
        if has_cpu_speed:
            cpu_speed_raw = get("cpu_speed", "1.0")
            try:
                cpu_speed = float(re.sub(r"[^\d.]", "", cpu_speed_raw) or "1.0")
            except ValueError:
                cpu_speed = 1.0

        cpu_model = get("cpu", "")

        machines[partition] = {
            "gpu":       gpu,
            "cpu":       cpu_model,
            "max_days":  max_days,
            "speed":     gpu_speed,
            "gpu_speed": gpu_speed,
            "cpu_speed": cpu_speed,
            "note":      note,
        }

    return machines


# ─── Parsing do script de experimento ─────────────────────────────────────────

def _parse_scalar(s: str) -> str:
    return s.strip().strip("'\"")


def _parse_array(s: str) -> list:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    tokens = re.findall(r"'[^']*'|\"[^\"]*\"|[^\s]+", s)
    return [t.strip("'\"") for t in tokens if t.strip("'\"")]


def _parse_model_config_batch(content: str) -> dict:
    """
    Extrai mapeamentos modelo→BATCH_SIZE do corpo de model_config() no script bash.

    Procura padrões como:
        mobilenetv2) BATCH_SIZE=16 ;;
        vit_tiny)    BATCH_SIZE=16 ;;
    Retorna {model_lower: batch_size}.
    """
    result = {}
    m = re.search(r'model_config\s*\(\s*\)(.*?)^}', content, re.M | re.S)
    if not m:
        return result
    body = m.group(1)
    for match in re.finditer(r'(\w+)\s*\)\s*BATCH_SIZE=(\d+)', body):
        model = match.group(1).lower()
        bs    = int(match.group(2))
        result[model] = bs
    return result


def parse_experiment_script(script_path: Path) -> dict:
    """
    Extrai variáveis relevantes de um script bash de experimento via regex.
    """
    content = script_path.read_text(encoding="utf-8")

    # Remove comentários de linha
    content_nc = re.sub(r"#.*", "", content)

    def get_scalar(name: str, default: str = "") -> str:
        m = re.search(rf'^\s*{name}="([^"]*)"', content_nc, re.M)
        if not m:
            m = re.search(rf"^\s*{name}='([^']*)'", content_nc, re.M)
        if not m:
            m = re.search(rf"^\s*{name}=([^\s(][^\s]*)", content_nc, re.M)
        return _parse_scalar(m.group(1)) if m else default

    def get_array(name: str, default: list = None) -> list:
        m = re.search(rf'^\s*{name}=\((.*?)\)', content_nc, re.M | re.S)
        if m:
            return _parse_array(m.group(1))
        m = re.search(rf'^\s*{name}=([^\s(][^\s]*)', content_nc, re.M)
        if m:
            return [_parse_scalar(m.group(1))]
        return default if default is not None else []

    gpu_ids = get_array("GPU_IDS", default=["0"])
    try:
        clients_per_gpu = int(get_scalar("CLIENTS_PER_GPU", "1"))
    except ValueError:
        clients_per_gpu = 1

    # BATCH_SIZE global (fallback quando model_config não define por modelo)
    try:
        global_batch_size = int(get_scalar("BATCH_SIZE", "0") or "0")
    except ValueError:
        global_batch_size = 0

    # Por-modelo: parse model_config()
    model_batch_sizes = _parse_model_config_batch(content_nc)

    params = {
        "GPU_IDS":           gpu_ids,
        "CLIENTS_PER_GPU":   clients_per_gpu,
        "NCLIENTES":         len(gpu_ids) * clients_per_gpu,

        "NUM_ROUNDS":        int(get_scalar("NUM_ROUNDS", "10")),
        "TECHNIQUE":         get_scalar("TECHNIQUE", "none"),
        "CKKS_BACKEND":      get_scalar("CKKS_BACKEND", "tenseal"),

        "SOLUTIONS":         get_array("SOLUTIONS", ["plaintext"]),
        "PERCENTAGES":       get_array("PERCENTAGES", ["1.0"]),
        "NIIDS":             get_array("NIIDS", ["0.1"]),
        "SEEDS":             get_array("SEEDS", ["42"]),
        "CKKS_CONFIGS":      get_array("CKKS_CONFIGS", ["ckks 8192 60"]),
        "KEY_TYPES":         get_array("KEY_TYPES", ["None"]),
        "MODEL_DATASET_PAIRS": get_array("MODEL_DATASET_PAIRS", []),

        "BATCH_SIZE":        global_batch_size,
        "MODEL_BATCH_SIZES": model_batch_sizes,
    }

    # Backward compat: MODELS + DATASETS separados
    if not params["MODEL_DATASET_PAIRS"]:
        models   = get_array("MODELS", [])
        datasets = get_array("DATASETS", [])
        if models and datasets:
            params["MODEL_DATASET_PAIRS"] = [
                f"{m}:{d}" for m in models for d in datasets
            ]

    return params


# ─── Contagem de experimentos ──────────────────────────────────────────────────

def count_experiments(params: dict) -> list:
    experiments = []
    for ckks_config in params["CKKS_CONFIGS"]:
        for key_type in params["KEY_TYPES"]:
            for solution in params["SOLUTIONS"]:
                sol_lower = solution.lower()
                for pair in params["MODEL_DATASET_PAIRS"]:
                    if ":" in pair:
                        model, dataset = pair.split(":", 1)
                    else:
                        model, dataset = pair, "FASHION_MNIST"
                    for niid in params["NIIDS"]:
                        for seed in params["SEEDS"]:
                            if sol_lower in PACKING_SOLUTIONS:
                                pcts = params["PERCENTAGES"]
                            else:
                                pcts = ["-1"]
                            for pct in pcts:
                                experiments.append({
                                    "solution":    solution,
                                    "model":       model,
                                    "dataset":     dataset,
                                    "percentage":  pct,
                                    "niid":        niid,
                                    "seed":        seed,
                                    "ckks_config": ckks_config,
                                    "key_type":    key_type,
                                })
    return experiments


# ─── Estimativa de tempo e RAM ────────────────────────────────────────────────

def _resolve_batch_size(model: str, params: dict) -> int:
    """Resolve batch_size para um modelo: model_config > BATCH_SIZE global > dict interno > 4."""
    m = model.lower()
    bs = params.get("MODEL_BATCH_SIZES", {}).get(m)
    if bs:
        return bs
    if params.get("BATCH_SIZE"):
        return params["BATCH_SIZE"]
    return MODEL_BATCH_SIZE.get(m, REF_BATCH_SIZE)


def estimate_round_time_local(model: str, dataset: str, n_clients: int,
                               clients_per_gpu: int, batch_size: int = None) -> dict:
    """
    Estima o tempo de wall-clock de UM round nesta máquina de referência.

    Prioridade para treino:
      1. EMPIRICAL_TRAIN_TIME_S[(model, dataset, batch_size)] — medição direta
      2. Modelo linear: T_ref × (n_params/P_ref) × (steps/steps_ref)

    Retorna: {t_train_s, t_he_s, ram_peak_mb, batch_size_used, empirical}
    """
    m_lower   = model.lower()
    n_params  = MODEL_PARAMS.get(m_lower, MODEL_PARAMS.get(model, REF_N_PARAMS))
    ds_size   = DATASET_SIZE.get(dataset, DATASET_SIZE.get(dataset.split("/")[-1], REF_DATASET_SIZE))

    if batch_size is None:
        batch_size = MODEL_BATCH_SIZE.get(m_lower, REF_BATCH_SIZE)

    samples_per_client = ds_size / n_clients

    # 1. Empirical lookup
    empirical_key = (m_lower, dataset, batch_size)
    empirical = empirical_key in EMPIRICAL_TRAIN_TIME_S
    if empirical:
        t_single_train = EMPIRICAL_TRAIN_TIME_S[empirical_key]
    else:
        steps     = math.ceil(samples_per_client / batch_size)
        ref_steps = math.ceil((REF_DATASET_SIZE / REF_N_CLIENTS) / REF_BATCH_SIZE)
        t_single_train = REF_TRAIN_TIME_S * (n_params / REF_N_PARAMS) * (steps / ref_steps)

    t_train = clients_per_gpu * t_single_train

    # HE (CPU-bound) — escala com número de chunks
    n_chunks    = math.ceil(n_params / (REF_POLY_DEGREE // 2))
    t_single_he = REF_HE_TIME_S * (n_chunks / REF_N_CHUNKS)
    t_he        = clients_per_gpu * t_single_he

    # RAM: empirical dict ou modelo linear
    ram_peak = MODEL_RAM_PEAK_MB.get(
        m_lower,
        RAM_BASE_MB + int(RAM_PER_REF_PARAMS_MB * (n_params / REF_N_PARAMS))
    )

    return {
        "t_train_s":    t_train,
        "t_he_s":       t_he,
        "ram_peak_mb":  ram_peak,
        "batch_size":   batch_size,
        "empirical":    empirical,
    }


def estimate_total_time(experiments: list, params: dict, gpu_speed: float,
                        cpu_speed: float, margin: float) -> dict:
    """
    Estima o tempo total do conjunto de experimentos.
    """
    n_clients       = params["NCLIENTES"]
    clients_per_gpu = params["CLIENTS_PER_GPU"]
    num_rounds      = params["NUM_ROUNDS"]
    total_s         = 0.0
    breakdown       = []

    for exp in experiments:
        batch_size = _resolve_batch_size(exp["model"], params)
        local = estimate_round_time_local(
            exp["model"], exp["dataset"], n_clients, clients_per_gpu,
            batch_size=batch_size
        )
        t_round = local["t_train_s"] / gpu_speed + local["t_he_s"] / cpu_speed
        t_exp   = num_rounds * t_round + SETUP_OVERHEAD_S
        total_s += t_exp
        breakdown.append({
            **exp,
            "t_train_local_s": local["t_train_s"],
            "t_he_local_s":    local["t_he_s"],
            "t_round_s":       t_round,
            "t_exp_s":         t_exp,
            "ram_peak_mb":     local["ram_peak_mb"],
            "batch_size":      local["batch_size"],
            "empirical":       local["empirical"],
        })

    return {
        "total_s":             total_s,
        "total_s_with_margin": total_s * margin,
        "n_experiments":       len(experiments),
        "n_rounds_total":      len(experiments) * num_rounds,
        "breakdown":           breakdown,
    }


# ─── Formatação ───────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if s or not parts: parts.append(f"{s}s")
    return " ".join(parts)


def fmt_days(seconds: float) -> str:
    return f"{seconds / 86400:.2f} dias"


def fmt_bytes(b: int) -> str:
    if b >= 1_073_741_824:
        return f"{b/1_073_741_824:.1f} GB"
    if b >= 1_048_576:
        return f"{b/1_048_576:.1f} MB"
    return f"{b/1_024:.1f} KB"


def fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb/1024:.1f} GB"
    return f"{mb:.0f} MB"


# ─── Seção de RAM ─────────────────────────────────────────────────────────────

def print_ram_section(experiments: list, params: dict, margin: float) -> None:
    clients_per_gpu = params["CLIENTS_PER_GPU"]

    # Coleta modelos únicos e seu ram_peak
    seen = {}
    for exp in experiments:
        m = exp["model"].lower()
        if m not in seen:
            n_params = MODEL_PARAMS.get(m, REF_N_PARAMS)
            ram = MODEL_RAM_PEAK_MB.get(
                m, RAM_BASE_MB + int(RAM_PER_REF_PARAMS_MB * (n_params / REF_N_PARAMS))
            )
            seen[m] = ram

    he_note = "HE (CKKS) adiciona ~100–300 MB/cliente (context + ciphertexts): incluso na margem."

    print()
    print("─" * 72)
    print("  RAM estimada por cliente (plaintext; fonte: benchmark_resources)")
    print("─" * 72)
    print(f"  {'Modelo':14} {'Peak/cliente':>13}  "
          f"{f'{clients_per_gpu} cl/GPU':>12}  "
          f"{f'{clients_per_gpu} cl/GPU + {margin:.1f}× margem':>22}")
    print("  " + "-" * 68)
    for model, ram_mb in sorted(seen.items()):
        total_mb    = ram_mb * clients_per_gpu
        with_margin = total_mb * margin
        print(f"  {model:14} {fmt_mb(ram_mb):>13}  "
              f"{fmt_mb(total_mb):>12}  "
              f"{fmt_mb(with_margin):>22}")
    print()
    print(f"  Clientes na mesma GPU rodam CONCORRENTEMENTE → RAM é aditiva.")
    print(f"  {he_note}")
    print(f"  Servidor (agregação): ~200 MB (desprezível).")

    # Recomendação geral
    max_with_margin = max(r * clients_per_gpu * margin for r in seen.values())
    print()
    print(f"  ► Recomendação para {clients_per_gpu} clientes/GPU com margem {margin:.1f}×:")
    print(f"    RAM mínima por nó: {fmt_mb(max_with_margin)}")
    print("─" * 72)


# ─── Seção de disco (verbose) ─────────────────────────────────────────────────

def print_disk_section(experiments: list, params: dict) -> None:
    n_clients  = params["NCLIENTES"]
    num_rounds = params["NUM_ROUNDS"]

    print()
    print("─" * 72)
    print("  Estimativa de espaço em disco (logs/)")
    print("─" * 72)
    print(f"  {'Modelo':14} {'Payload/round/cl':>17}  {'CSVs+logs/exp':>14}  {'model/ npz/exp':>15}")
    print("  " + "-" * 66)

    total_bytes = 0
    for model in sorted({e["model"].lower() for e in experiments}):
        n_exps   = sum(1 for e in experiments if e["model"].lower() == model)
        n_params = MODEL_PARAMS.get(model, REF_N_PARAMS)
        payload  = MODEL_PAYLOAD_BYTES.get(model, n_params * 4)

        # Logs CSV: train.csv (~300B/row × n_clients × n_rounds) + evaluate (~200B) + server
        csv_per_exp = (300 * n_clients * num_rounds + 200 * n_clients * num_rounds
                       + 100 * num_rounds + 100 * num_rounds)
        # model/ npz: float16, só cid==0, todos os rounds
        npz_per_exp = n_params * 2 * num_rounds

        total_per_exp = payload * n_clients * num_rounds + csv_per_exp + npz_per_exp
        total_bytes  += total_per_exp * n_exps

        print(f"  {model:14} {fmt_bytes(payload):>17}  "
              f"{fmt_bytes(csv_per_exp + npz_per_exp):>14}  "
              f"{fmt_bytes(npz_per_exp):>15}")

    print()
    print(f"  Total estimado para {len(experiments)} experimentos: {fmt_bytes(total_bytes)}")
    print("─" * 72)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estima o tempo de execução de um script de experimento FL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("script", help="Caminho para o script .sh do experimento")
    parser.add_argument(
        "--machine", "-m", default=None,
        help="Partição SLURM alvo (ex: l40s, h100). Default: todas do SLURM.md",
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="Mostra estimativa para todas as máquinas listadas no SLURM.md",
    )
    parser.add_argument(
        "--margin", type=float, default=1.5,
        help="Margem de segurança multiplicativa (default: 1.5 = +50%%)",
    )
    parser.add_argument(
        "--slurm", default=None,
        help="Caminho para SLURM.md (default: detecta automaticamente)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Exibe breakdown detalhado por experimento e estimativa de disco",
    )
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"[ERRO] Script não encontrado: {script_path}", file=sys.stderr)
        sys.exit(1)

    if args.slurm:
        slurm_path = Path(args.slurm).resolve()
    else:
        slurm_path = script_path.parent.parent / "SLURM.md"
        if not slurm_path.exists():
            slurm_path = script_path.parent / "SLURM.md"

    machines = parse_slurm_md(slurm_path)
    params   = parse_experiment_script(script_path)
    exps     = count_experiments(params)

    if not exps:
        print("[ERRO] Nenhum experimento identificado no script. "
              "Verifique MODEL_DATASET_PAIRS / MODELS / DATASETS.", file=sys.stderr)
        sys.exit(1)

    if args.machine:
        if args.machine not in machines:
            print(f"[ERRO] Máquina '{args.machine}' não encontrada no SLURM.md.",
                  file=sys.stderr)
            print(f"       Disponíveis: {', '.join(machines.keys())}", file=sys.stderr)
            sys.exit(1)
        selected = {args.machine: machines[args.machine]}
    elif args.all or len(machines) > 1:
        selected = machines
    else:
        selected = machines

    # ── Cabeçalho ────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  Estimativa de Tempo — {script_path.name}")
    print("=" * 72)
    print(f"  Script: {script_path}")
    print(f"  Experimentos totais : {len(exps)}")
    print(f"  Rounds por experim. : {params['NUM_ROUNDS']}")
    print(f"  Rounds totais       : {len(exps) * params['NUM_ROUNDS']}")
    print(f"  Clientes            : {params['NCLIENTES']} "
          f"({params['CLIENTS_PER_GPU']} por GPU × {len(params['GPU_IDS'])} GPU(s))")
    print(f"  Soluções            : {', '.join(params['SOLUTIONS'])}")
    packing_pcts = [e['percentage'] for e in exps if e['solution'].lower() in PACKING_SOLUTIONS]
    if packing_pcts:
        unique_pcts = sorted(set(packing_pcts), key=float)
        print(f"  Percentagens (pack) : {', '.join(unique_pcts)}")
    print(f"  IID/non-IID (NIIDs) : {', '.join(params['NIIDS'])}  "
          f"[não afeta tempo/round; apenas contagem de experimentos]")
    print(f"  Margem de segurança : {args.margin:.1f}×  (+{(args.margin-1)*100:.0f}%)")
    print()

    # Batch sizes resolvidos por modelo
    model_bs_info = {}
    for exp in exps:
        m = exp["model"].lower()
        if m not in model_bs_info:
            bs = _resolve_batch_size(exp["model"], params)
            src = ("model_config()" if m in params.get("MODEL_BATCH_SIZES", {})
                   else "BATCH_SIZE global" if params.get("BATCH_SIZE")
                   else "padrão interno")
            model_bs_info[m] = (bs, src)
    if model_bs_info:
        model_dataset = {e["model"].lower(): e["dataset"] for e in exps}
        print("  Batch sizes (por modelo):")
        for m, (bs, src) in sorted(model_bs_info.items()):
            ds       = model_dataset.get(m, "")
            emp_flag = "★ empírico" if (m, ds, bs) in EMPIRICAL_TRAIN_TIME_S else ""
            print(f"    {m:14} batch={bs:>3}  ({src})  {emp_flag}")
        print()

    # Resumo das combinações
    print("  Combinações modelo×solução:")
    from collections import Counter
    combos = Counter(f"{e['model']:12s} × {e['solution']:10s} @ {e['dataset']}" for e in exps)
    for combo, count in sorted(combos.items()):
        print(f"    {combo}  → {count} experimento(s)")
    print()

    # ── Tabela de tempo ──────────────────────────────────────────────────────
    print("-" * 84)
    print(f"  {'Máquina':<12} {'GPU':<20} {'GPU Spd':>7} {'CPU Spd':>7}  "
          f"{'Sem margem':>14}  {'COM MARGEM':>14}  {'vs limite'}")
    print("-" * 84)

    for partition, info in selected.items():
        result = estimate_total_time(
            exps, params,
            gpu_speed=info["gpu_speed"],
            cpu_speed=info["cpu_speed"],
            margin=args.margin,
        )
        t_total  = result["total_s"]
        t_margin = result["total_s_with_margin"]

        limit_str = ""
        if info.get("max_days"):
            limit_s = info["max_days"] * 86400
            ratio   = t_margin / limit_s
            if ratio > 1.0:
                limit_str = f"⚠  {ratio:.1f}× o limite ({info['max_days']}d)!"
            else:
                limit_str = f"OK ({ratio*100:.0f}% do limite)"

        gpu_str = info["gpu"][:18]
        print(f"  {partition:<12} {gpu_str:<20} {info['gpu_speed']:>7.1f} {info['cpu_speed']:>7.1f}  "
              f"{fmt_time(t_total):>14}  {fmt_time(t_margin):>14}  {limit_str}")

    print("-" * 84)

    # ── Seção RAM ────────────────────────────────────────────────────────────
    print_ram_section(exps, params, args.margin)

    # ── Verbose: breakdown + disco ───────────────────────────────────────────
    if args.verbose:
        first_machine = next(iter(selected))
        info   = selected[first_machine]
        result = estimate_total_time(
            exps, params,
            gpu_speed=info["gpu_speed"],
            cpu_speed=info["cpu_speed"],
            margin=args.margin,
        )
        print()
        print(f"  Breakdown detalhado para '{first_machine}' "
              f"(GPU_speed={info['gpu_speed']}, CPU_speed={info['cpu_speed']}):")
        print(f"  {'#':>3}  {'Solução':10} {'Modelo':12} {'Dataset':15} {'Pct':5} {'Batch':>5} "
              f"{'T/round':>9} {'T/exp':>11} {'T/exp+mg':>11} {'RAM/cl':>8} {'Emp?':>5}")
        print("  " + "-" * 95)
        for i, b in enumerate(result["breakdown"], 1):
            t_exp_mg = b["t_exp_s"] * args.margin
            emp_flag = "★" if b["empirical"] else ""
            print(f"  {i:>3}  {b['solution']:10} {b['model']:12} {b['dataset']:15} "
                  f"{b['percentage']:>5}  {b['batch_size']:>5}  {fmt_time(b['t_round_s']):>9}  "
                  f"{fmt_time(b['t_exp_s']):>11}  {fmt_time(t_exp_mg):>11} "
                  f"{fmt_mb(b['ram_peak_mb']):>8} {emp_flag:>5}")
        print()
        print("  ★ = tempo medido empiricamente (benchmark_resources); demais são estimativas lineares.")

        print_disk_section(exps, params)

    # ── Notas metodológicas ──────────────────────────────────────────────────
    print()
    print("  NOTAS:")
    print(f"  • Treino GPU (ref): resnet20/CIFAR10/2-clients = {REF_TRAIN_TIME_S:.0f}s/round na máquina local")
    print(f"  • HE CPU    (ref): resnet20/tenseal/poly=8192   = {REF_HE_TIME_S:.2f}s/round na máquina local")
    print(f"  • Dados empíricos (★): dnn_tiny=11.1s, lenet5=19.4s, mobilenetv2=242.7s/round")
    print("  • IID/non-IID: não afeta tempo por round — apenas duplica contagem de experimentos")
    print("  • Escalonamento sem dado empírico: treino por n_params×steps; HE por n_chunks")
    print("  • Clientes por GPU tratados como concorrentes (hipótese de soma de RAM)")
    print(f"  • Overhead de setup por experimento: {SETUP_OVERHEAD_S:.0f}s")
    print("  • GPU/CPU Speed são estimativas; calibre com runs reais atualizando SLURM.md")
    print("  • Veja experiments/ESTIMATION_METHODOLOGY.md para detalhes do modelo")
    print()


if __name__ == "__main__":
    main()
