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
# Medido nos logs (train.csv, tenseal, poly_degree=8192, resnet20):
#   encrypt_time_s  ≈ 0.05–0.30 s
#   serialize_time_s ≈ 0.05–0.10 s
#   decrypt_time_s  ≈ 0.05–0.10 s
# Usando 0.50 s como referência conservadora para o conjunto dessas operações.
REF_HE_TIME_S      = 0.50    # segundos HE por cliente por round na máquina local
REF_POLY_DEGREE    = 8_192   # poly_degree da referência → chunk_size = 4096
REF_N_CHUNKS       = math.ceil(REF_N_PARAMS / (REF_POLY_DEGREE // 2))  # ≈ 67

# Parâmetros conhecidos dos modelos (n_params aproximado)
MODEL_PARAMS = {
    "resnet20":    274_042,
    "lenet5":       44_000,
    "dnn":         309_000,
    "dnn_base":    309_000,
    "dnn_tiny":    111_000,
    "mobilenetv2": 460_000,
    "vit_tiny":    206_000,
    "simple_cnn":   80_000,
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
PACKING_SOLUTIONS = {"fedphe", "nvnl", "sw", "rw", "fedphe-scout", "nvnl-scout"}

# Overhead de setup por experimento (geração de chaves, startup de containers)
SETUP_OVERHEAD_S = 90.0


# ─── Leitura do SLURM.md ──────────────────────────────────────────────────────

def parse_slurm_md(slurm_path: Path) -> dict:
    """
    Lê o SLURM.md e retorna dict keyed por partição:
        { 'l40s': {'gpu': 'L40S 46GB', 'max_days': 4, 'speed': 12.0, 'note': '...'}, ... }

    Formato esperado do header:
        | Partição | Nós | GPUs | Tempo máx | Speed | Nota |

    A coluna 'Speed' é obrigatória; se ausente, o script avisa e usa 1.0.
    """
    machines = {}
    if not slurm_path.exists():
        print(f"[AVISO] SLURM.md não encontrado em {slurm_path}. Usando apenas 'local'.",
              file=sys.stderr)
        return {"local": {"gpu": "CPU", "max_days": None, "speed": 1.0, "note": "Referência"}}

    content = slurm_path.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines()]

    # Detectar linha de header da tabela
    header_idx = None
    col_map = {}
    for i, line in enumerate(lines):
        if "|" not in line:
            continue
        cells = [c.strip().lower() for c in line.split("|") if c.strip()]
        if len(cells) < 4:
            continue  # ignora linhas curtas (comentários, texto solto com "|")
        if "partição" in cells or "particao" in cells or "partição" in "".join(cells):
            # mapear índices de coluna
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
                    # Fallback: coluna genérica "Speed" (formato antigo)
                    col_map["gpu_speed"] = j
                elif "nota" in c or "note" in c:
                    col_map["note"] = j
            header_idx = i
            break

    if header_idx is None:
        print("[AVISO] Tabela de partições não encontrada no SLURM.md.", file=sys.stderr)
        return {}

    has_speed = "speed" in col_map or "gpu_speed" in col_map or "gpu speed" in col_map
    has_cpu_speed = "cpu_speed" in col_map or "cpu speed" in col_map
    if not has_speed:
        print("[AVISO] Coluna 'GPU Speed' / 'Speed' não encontrada no SLURM.md. "
              "Adicione a coluna para estimativas por máquina.", file=sys.stderr)

    for line in lines[header_idx + 1:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if all(c.startswith("-") for c in cells):
            continue  # linha separadora
        if not cells:
            continue

        def get(key, default=""):
            idx = col_map.get(key)
            return cells[idx] if idx is not None and idx < len(cells) else default

        raw_partition = get("partition")
        # limpar markdown bold (**text**)
        partition = re.sub(r"\*+", "", raw_partition).strip()
        # remover sufixo "(default)" etc.
        partition = re.sub(r"\(.*?\)", "", partition).strip()
        if not partition or partition.startswith("-"):
            continue

        gpu = get("gpu")
        note = get("note")

        # Tempo máximo: "4 dias", "2 days", etc.
        max_time_raw = get("max_time", "")
        max_days = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*dia", max_time_raw, re.I)
        if not m:
            m = re.search(r"(\d+(?:\.\d+)?)\s*day", max_time_raw, re.I)
        if m:
            max_days = float(m.group(1))

        # GPU speed factor
        gpu_speed = 1.0
        if has_speed:
            speed_raw = get("gpu_speed", "1.0")
            try:
                gpu_speed = float(re.sub(r"[^\d.]", "", speed_raw) or "1.0")
            except ValueError:
                gpu_speed = 1.0

        # CPU speed factor (for HE operations)
        cpu_speed = 1.0
        if has_cpu_speed:
            cpu_speed_raw = get("cpu_speed", "1.0")
            try:
                cpu_speed = float(re.sub(r"[^\d.]", "", cpu_speed_raw) or "1.0")
            except ValueError:
                cpu_speed = 1.0

        cpu_model = get("cpu", "")

        machines[partition] = {
            "gpu": gpu,
            "cpu": cpu_model,
            "max_days": max_days,
            "speed": gpu_speed,      # kept as "speed" for backward compat
            "gpu_speed": gpu_speed,
            "cpu_speed": cpu_speed,
            "note": note,
        }

    return machines


# ─── Parsing do script de experimento ─────────────────────────────────────────

def _parse_scalar(s: str) -> str:
    """Remove aspas e espaços de um valor escalar bash."""
    return s.strip().strip("'\"")


def _parse_array(s: str) -> list:
    """
    Converte string bash '(val1 val2 val3)' em lista de strings.
    Trata aspas simples e duplas dentro do array.
    """
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    tokens = re.findall(r"'[^']*'|\"[^\"]*\"|[^\s]+", s)
    return [t.strip("'\"") for t in tokens if t.strip("'\"")]


def parse_experiment_script(script_path: Path) -> dict:
    """
    Extrai variáveis relevantes de um script bash de experimento via regex.
    Retorna dict com os valores encontrados (listas para arrays, string para escalares).
    Não executa o bash — apenas analisa o texto.
    """
    content = script_path.read_text(encoding="utf-8")

    # Remove comentários
    content_nc = re.sub(r"#.*", "", content)

    def get_scalar(name: str, default: str = "") -> str:
        m = re.search(rf'^\s*{name}="([^"]*)"', content_nc, re.M)
        if not m:
            m = re.search(rf"^\s*{name}='([^']*)'", content_nc, re.M)
        if not m:
            m = re.search(rf"^\s*{name}=([^\s(][^\s]*)", content_nc, re.M)
        return _parse_scalar(m.group(1)) if m else default

    def get_array(name: str, default: list = None) -> list:
        # Multiline array: VAR=(\n  val1\n  val2\n)
        m = re.search(
            rf'^\s*{name}=\((.*?)\)',
            content_nc,
            re.M | re.S,
        )
        if m:
            return _parse_array(m.group(1))
        # Inline scalar fallback
        m = re.search(rf'^\s*{name}=([^\s(][^\s]*)', content_nc, re.M)
        if m:
            return [_parse_scalar(m.group(1))]
        return default if default is not None else []

    # GPU layout
    gpu_ids = get_array("GPU_IDS", default=["0"])
    try:
        clients_per_gpu = int(get_scalar("CLIENTS_PER_GPU", "1"))
    except ValueError:
        clients_per_gpu = 1

    params = {
        "GPU_IDS":          gpu_ids,
        "CLIENTS_PER_GPU":  clients_per_gpu,
        "NCLIENTES":        len(gpu_ids) * clients_per_gpu,

        "NUM_ROUNDS":       int(get_scalar("NUM_ROUNDS", "10")),
        "TECHNIQUE":        get_scalar("TECHNIQUE", "none"),
        "CKKS_BACKEND":     get_scalar("CKKS_BACKEND", "tenseal"),

        "SOLUTIONS":        get_array("SOLUTIONS", ["plaintext"]),
        "PERCENTAGES":      get_array("PERCENTAGES", ["1.0"]),
        "NIIDS":            get_array("NIIDS", ["0.1"]),
        "SEEDS":            get_array("SEEDS", ["42"]),
        "CKKS_CONFIGS":     get_array("CKKS_CONFIGS", ["ckks 8192 60"]),
        "KEY_TYPES":        get_array("KEY_TYPES", ["None"]),
        "MODEL_DATASET_PAIRS": get_array("MODEL_DATASET_PAIRS", []),
    }

    # Backward compat: MODELS + DATASETS separados (template antigo)
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
    """
    Retorna lista de dicts, cada um descrevendo um experimento:
      { solution, model, dataset, percentage, niid, seed, ckks_config, key_type }
    """
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


# ─── Estimativa de tempo ──────────────────────────────────────────────────────

def estimate_round_time_local(model: str, dataset: str, n_clients: int,
                              clients_per_gpu: int) -> dict:
    """
    Estima o tempo de wall-clock de UM round (1 época por cliente) nesta máquina de referência.
    Retorna dict com t_train_s (GPU-bound) e t_he_s (CPU-bound) separados.

    Treino (GPU-bound):
        T_train = clients_per_gpu × T_ref × (n_params / P_ref) × (steps / steps_ref)

    HE (CPU-bound): encrypt + pack + serialize + decrypt escala com nº de chunks.
        n_chunks = ceil(n_params / chunk_size)   chunk_size = poly_degree/2
        T_he = clients_per_gpu × REF_HE_TIME_S × (n_chunks / REF_N_CHUNKS)

    Clientes na mesma GPU rodam sequencialmente (hipótese conservadora).
    Clientes em GPUs diferentes são paralelos — não acumulam.
    """
    n_params = MODEL_PARAMS.get(model.lower(), MODEL_PARAMS.get(model, REF_N_PARAMS))

    ds_size = DATASET_SIZE.get(dataset, DATASET_SIZE.get(dataset.split("/")[-1], REF_DATASET_SIZE))
    samples_per_client = ds_size / n_clients

    batch_size = REF_BATCH_SIZE
    steps      = math.ceil(samples_per_client / batch_size)
    ref_steps  = math.ceil((REF_DATASET_SIZE / REF_N_CLIENTS) / batch_size)

    # Treino GPU
    t_single_train = REF_TRAIN_TIME_S * (n_params / REF_N_PARAMS) * (steps / ref_steps)
    t_train = clients_per_gpu * t_single_train

    # HE CPU — escala com número de chunks (packs)
    n_chunks = math.ceil(n_params / (REF_POLY_DEGREE // 2))
    t_single_he = REF_HE_TIME_S * (n_chunks / REF_N_CHUNKS)
    t_he = clients_per_gpu * t_single_he

    return {"t_train_s": t_train, "t_he_s": t_he}


def estimate_total_time(experiments: list, params: dict, gpu_speed: float,
                        cpu_speed: float, margin: float) -> dict:
    """
    Estima o tempo total do conjunto de experimentos.

    T_round_machine = T_train_local/gpu_speed + T_he_local/cpu_speed

    Retorna dict com total_s, total_s_with_margin, n_experiments, n_rounds_total,
    breakdown (lista por experimento).
    """
    n_clients       = params["NCLIENTES"]
    clients_per_gpu = params["CLIENTS_PER_GPU"]
    num_rounds      = params["NUM_ROUNDS"]
    total_s         = 0.0
    breakdown       = []

    for exp in experiments:
        local = estimate_round_time_local(
            exp["model"], exp["dataset"], n_clients, clients_per_gpu
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
    """Converte segundos em string legível: Xd Yh Zm Ws."""
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estima o tempo de execução de um script de experimento FL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("script", help="Caminho para o script .sh do experimento")
    parser.add_argument(
        "--machine", "-m",
        default=None,
        help="Partição SLURM alvo (ex: l40s, h100). Default: todas do SLURM.md",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Mostra estimativa para todas as máquinas listadas no SLURM.md",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.5,
        help="Margem de segurança multiplicativa (default: 1.5 = +50%%)",
    )
    parser.add_argument(
        "--slurm",
        default=None,
        help="Caminho para SLURM.md (default: detecta automaticamente a partir do script)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Exibe breakdown detalhado por experimento",
    )
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"[ERRO] Script não encontrado: {script_path}", file=sys.stderr)
        sys.exit(1)

    # Localizar SLURM.md: argumento > diretório do script > raiz do repo
    if args.slurm:
        slurm_path = Path(args.slurm).resolve()
    else:
        # Procura subindo de experiments/ até a raiz do repo
        slurm_path = script_path.parent.parent / "SLURM.md"
        if not slurm_path.exists():
            slurm_path = script_path.parent / "SLURM.md"

    # Parse
    machines = parse_slurm_md(slurm_path)
    params   = parse_experiment_script(script_path)
    exps     = count_experiments(params)

    if not exps:
        print("[ERRO] Nenhum experimento identificado no script. "
              "Verifique MODEL_DATASET_PAIRS / MODELS / DATASETS.", file=sys.stderr)
        sys.exit(1)

    # Quais máquinas mostrar
    if args.machine:
        if args.machine not in machines:
            print(f"[ERRO] Máquina '{args.machine}' não encontrada no SLURM.md.", file=sys.stderr)
            print(f"       Disponíveis: {', '.join(machines.keys())}", file=sys.stderr)
            sys.exit(1)
        selected = {args.machine: machines[args.machine]}
    elif args.all or len(machines) > 1:
        selected = machines
    else:
        selected = machines  # mostra todas se --all não especificado e só tem uma

    # Cabeçalho
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
    print(f"  Margem de segurança : {args.margin:.1f}×  (+{(args.margin-1)*100:.0f}%)")
    print()

    # Resumo dos experimentos por (modelo, dataset, solução)
    print("  Combinações modelo×solução:")
    from collections import Counter
    combos = Counter(f"{e['model']:12s} × {e['solution']:10s} @ {e['dataset']}" for e in exps)
    for combo, count in sorted(combos.items()):
        print(f"    {combo}  → {count} experimento(s)")
    print()

    # Estimativa por máquina
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
    print()

    # Verbose: breakdown por experimento para a primeira máquina selecionada
    if args.verbose:
        first_machine = next(iter(selected))
        info = selected[first_machine]
        result = estimate_total_time(
            exps, params,
            gpu_speed=info["gpu_speed"],
            cpu_speed=info["cpu_speed"],
            margin=args.margin,
        )
        print(f"  Breakdown detalhado para '{first_machine}' "
              f"(GPU_speed={info['gpu_speed']}, CPU_speed={info['cpu_speed']}):")
        print(f"  {'#':>3}  {'Solução':10} {'Modelo':12} {'Dataset':15} {'Pct':5} "
              f"{'T/round':>9} {'T/exp':>11} {'T/exp+mg':>11}")
        print("  " + "-" * 82)
        for i, b in enumerate(result["breakdown"], 1):
            t_exp_mg = b["t_exp_s"] * args.margin
            print(f"  {i:>3}  {b['solution']:10} {b['model']:12} {b['dataset']:15} "
                  f"{b['percentage']:>5}  {fmt_time(b['t_round_s']):>9}  "
                  f"{fmt_time(b['t_exp_s']):>11}  {fmt_time(t_exp_mg):>11}")
        print()

    # Notas metodológicas
    print("  NOTAS:")
    print(f"  • Treino GPU (ref): resnet20/CIFAR10/2-clients = {REF_TRAIN_TIME_S:.0f}s/round na máquina local")
    print(f"  • HE CPU    (ref): resnet20/tenseal/poly=8192   = {REF_HE_TIME_S:.2f}s/round na máquina local")
    print("  • Escalonamento: treino por n_params×steps; HE por n_chunks=ceil(n_params/4096)")
    print("  • Clientes por GPU tratados como sequenciais (hipótese conservadora)")
    print(f"  • Overhead de setup por experimento: {SETUP_OVERHEAD_S:.0f}s")
    print("  • GPU/CPU Speed são estimativas; calibre com runs reais atualizando SLURM.md")
    print("  • Veja experiments/ESTIMATION_METHODOLOGY.md para detalhes do modelo")
    print()


if __name__ == "__main__":
    main()
