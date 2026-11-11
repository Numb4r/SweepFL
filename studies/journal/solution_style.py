"""
Shared solution styling for studies/general/*_pt.py scripts (PT-BR figures).

Centralizes solution colors, markers, PT-BR display labels, and ordering so
every translated script renders the same technique the same way. Import via
the same importlib pattern used for load_data.py:

    spec = importlib.util.spec_from_file_location("solution_style", HERE / "solution_style.py")
    solution_style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution_style)
"""

SOL_COLORS = {
    "plaintext":  "#333333",
    "sweepfl":       "#2196F3",
    "sw":         "#FF5722",
    "rw":         "#4CAF50",
    "fedphe":     "#9C27B0",
    "topkrr":     "#E91E63",
    "srr":        "#009688",
    "ckks":       "#FF9800",
    "bfv":        "#00BCD4",
    "batchcrypt": "#795548",
}

SOL_MARKERS = {
    "plaintext": "D", "sweepfl": "o", "sw": "s", "rw": "^",
    "fedphe": "P", "topkrr": "p", "srr": "h",
    "ckks": "v", "bfv": "X", "batchcrypt": "H",
}

# PT-BR display names with the correct literature naming (FedAvg for plaintext).
SOL_LABELS = {
    "plaintext":  "FedAvg",
    "sweepfl":       "Round Robin",
    "fedphe":     "Top-$k$ (FedPHE)",
    "sw":         "Sliding Window",
    "rw":         "Random",
    "topkrr":     "TopK-RR",
    "srr":        "SRR",
    "ckks":       "CKKS",
    "bfv":        "BFV",
    "batchcrypt": "BatchCrypt",
}

# Full-model (non-packing) baselines: rows with percentage=None in load_data.
FULL_MODEL_SOLS = {"plaintext", "ckks", "bfv", "batchcrypt"}

# Packing/sparsification techniques: rows with a real percentage.
PACKING_SOLS = {"sweepfl", "fedphe", "sw", "rw", "topkrr", "srr",
                "sweepfl-scout", "fedphe-scout"}

# Display order: full-model baselines first, then packing techniques.
SOL_ORDER = ["plaintext", "ckks", "bfv", "batchcrypt",
             "sweepfl", "fedphe", "sw", "rw", "topkrr", "srr"]


def label(sol: str) -> str:
    """PT-BR display name for a solution; falls back to the raw string."""
    return SOL_LABELS.get(sol, sol)


def color(sol: str) -> str:
    return SOL_COLORS.get(sol, "#888888")


def marker(sol: str) -> str:
    return SOL_MARKERS.get(sol, "o")


def sorted_solutions(sols) -> list:
    """Order a solution list: known solutions in SOL_ORDER, unknown appended sorted."""
    sols = set(sols)
    known = [s for s in SOL_ORDER if s in sols]
    unknown = sorted(s for s in sols if s not in SOL_ORDER)
    return known + unknown


def cf_suffix(cf: float) -> str:
    """Filename token for client_fraction; empty when cf==1.0 (sparsification theme)."""
    if cf is None or float(cf) == 1.0:
        return ""
    return f"_cf{cf:g}"


def cf_title(cf: float) -> str:
    """Title fragment for client_fraction; empty when cf==1.0 (sparsification theme)."""
    if cf is None or float(cf) == 1.0:
        return ""
    return f" | client fraction={cf:.0%}"
