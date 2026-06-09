"""
Shared data loading utility for sparsification_percentage studies.

Usage:
    from load_data import load_all, load_masks, last_round, rounds_to_threshold, auc_accuracy, discover_dims

    dfs = load_all()          # uses default logs dir
    dfs = load_all("/path/to/logs/sparsification_percentage")
    masks = load_masks()      # loads mask.csv from every experiment folder
    # or: LOGS_DIR env var override
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_LOGS_RELATIVE = "../../logs/sparsification_percentage"

_FOLDER_RE = re.compile(
    r"^(?P<model>.+?)_(?P<dataset>(?:tanganke/)?[^_]+(?:_[^_]+)?)_"
    r"(?P<solution>[^_]+)_(?:none|[^_]+)_"
    r"(?P<rounds>\d+)rounds_(?P<clients>\d+)clients_"
    r"seed(?P<seed>\d+)_niid(?P<niid>[0-9.]+)_"
    r"percentage(?P<percentage>-?[0-9.]+)_"
    r"(?P<poly>\d+)_"
)

_META_COLS = ["model", "dataset", "solution", "niid", "percentage", "seed", "rounds", "clients"]
_GROUP_COLS = ["model", "dataset", "solution", "niid", "percentage", "seed"]


def _resolve_logs_dir(logs_dir=None):
    if logs_dir is not None:
        return Path(logs_dir)
    env = os.environ.get("LOGS_DIR")
    if env:
        return Path(env)
    return Path(__file__).parent / _DEFAULT_LOGS_RELATIVE


def _parse_folder(name: str) -> dict | None:
    m = _FOLDER_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    pct = float(d["percentage"])
    sol = d["solution"].lower()  # normalize: NVNL → nvnl

    # Skip erroneous case: nvnl with no packing (uppercase NVNL + percentage=-1)
    if sol == "nvnl" and pct == -1:
        return None

    return {
        "model": d["model"],
        "dataset": d["dataset"],
        "solution": sol,
        "niid": float(d["niid"]),
        "percentage": None if pct == -1 else pct,
        "seed": int(d["seed"]),
        "rounds": int(d["rounds"]),
        "clients": int(d["clients"]),
    }


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_all(logs_dir=None) -> dict[str, pd.DataFrame]:
    """Load all experiment CSVs from logs_dir into four DataFrames.

    Returns dict with keys: 'train', 'evaluate', 'server_eval', 'server_agg'.
    Each DataFrame has extra columns: model, dataset, solution, niid, percentage, seed.
    percentage=None means plaintext/no-packing baseline (was -1 in folder name).
    """
    root = _resolve_logs_dir(logs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Logs directory not found: {root.resolve()}")

    train_parts, eval_parts, seval_parts, sagg_parts = [], [], [], []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        meta = _parse_folder(folder.name)
        if meta is None:
            continue

        for csv_name, parts in [
            ("train.csv", train_parts),
            ("evaluate.csv", eval_parts),
            ("server_eval.csv", seval_parts),
            ("server_agg.csv", sagg_parts),
        ]:
            df = _read_csv_safe(folder / csv_name)
            if df is not None and len(df) > 0:
                for k, v in meta.items():
                    df[k] = v
                parts.append(df)

    def _concat(parts):
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        # ensure numeric types
        for col in ["round", "niid", "seed", "rounds", "clients"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    return {
        "train": _concat(train_parts),
        "evaluate": _concat(eval_parts),
        "server_eval": _concat(seval_parts),
        "server_agg": _concat(sagg_parts),
    }


def last_round(df: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    """Return only the rows from the last (max) round per group."""
    if df.empty:
        return df
    gc = group_cols or [c for c in _GROUP_COLS if c in df.columns]
    max_rounds = df.groupby(gc, dropna=False)["round"].transform("max")
    return df[df["round"] == max_rounds].copy()


def rounds_to_threshold(
    df: pd.DataFrame, metric: str = "mean_acc", threshold: float = 0.80
) -> pd.DataFrame:
    """Return the first round where metric >= threshold for each group.

    Returns DataFrame with group columns + 'convergence_round' (NaN if never reached).
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    gc = [c for c in _GROUP_COLS if c in df.columns]

    def _first(g):
        reached = g[g[metric] >= threshold]
        if reached.empty:
            return pd.Series({"convergence_round": np.nan})
        return pd.Series({"convergence_round": reached["round"].min()})

    return df.sort_values("round").groupby(gc, dropna=False).apply(_first).reset_index()


def auc_accuracy(df: pd.DataFrame, metric: str = "mean_acc") -> pd.DataFrame:
    """Compute trapezoidal AUC of metric vs round for each group.

    Returns DataFrame with group columns + 'auc'.
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    gc = [c for c in _GROUP_COLS if c in df.columns]

    def _auc(g):
        g = g.sort_values("round")
        area = np.trapz(g[metric].values, g["round"].values)
        n_rounds = g["round"].max() - g["round"].min()
        norm = area / n_rounds if n_rounds > 0 else 0.0
        return pd.Series({"auc": area, "auc_norm": norm})

    return df.groupby(gc, dropna=False).apply(_auc).reset_index()


def _parse_sel(val) -> list:
    """Parse a ';'-separated pack-index string into a list of ints.

    Handles empty strings and NaN (returns []).
    """
    if not isinstance(val, str) or val.strip() == "":
        return []
    try:
        return [int(x) for x in val.split(";") if x.strip() != ""]
    except ValueError:
        return []


def load_masks(logs_dir=None) -> pd.DataFrame:
    """Load mask.csv from every experiment folder into a single DataFrame.

    Columns from the CSV: experiment_name, round, cid, solution, model, dataset,
    distribution, n_packs, packs_sent, selected_packs.
    Meta injected from folder name (overrides csv columns): model, dataset, solution,
    niid, percentage, seed, rounds, clients.
    Extra column added: 'selected' = list[int] parsed from selected_packs.
    percentage=None means plaintext/no-packing (was -1 in folder name).
    """
    root = _resolve_logs_dir(logs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Logs directory not found: {root.resolve()}")

    parts = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        meta = _parse_folder(folder.name)
        if meta is None:
            continue
        df = _read_csv_safe(folder / "mask.csv")
        if df is None or len(df) == 0:
            continue
        for k, v in meta.items():
            df[k] = v
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    for col in ["round", "n_packs", "packs_sent", "niid", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["selected"] = df["selected_packs"].map(_parse_sel)
    return df


def discover_dims(df: pd.DataFrame) -> dict[str, list]:
    """Return unique values for each experiment dimension found in df."""
    dims = {}
    for col in ["model", "dataset", "solution", "niid", "percentage", "seed"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist())
            dims[col] = vals
    return dims


# ── CLI: quick sanity check ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and inspect sparsification logs")
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()

    print("Loading data...")
    dfs = load_all(args.logs_dir)
    for name, df in dfs.items():
        print(f"  {name:12s}: {len(df):7,} rows  columns={list(df.columns[:6])}...")
    print("\nDimensions discovered (server_eval):")
    if not dfs["server_eval"].empty:
        for k, v in discover_dims(dfs["server_eval"]).items():
            print(f"  {k:12s}: {v}")
