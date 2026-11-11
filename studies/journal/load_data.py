"""
Shared data loading utility for experiment studies (generic — works for any experiment type).

Usage:
    from load_data import load_all, load_masks, last_round, rounds_to_threshold, auc_accuracy, discover_dims

    # LOGS_DIR must be set (env var or explicit path) — no implicit fallback.
    dfs = load_all("/path/to/logs/client_frac")
    dfs = load_all()          # uses LOGS_DIR env var (must be set by run_all.sh or caller)
    masks = load_masks()      # loads mask.csv from every experiment folder
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

# Trailing "_YYYY-MM-DD_HH-MM-SS" of every experiment folder name. Stripping it
# yields the config identity shared by reruns and by resume fragments.
_TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def _resolve_logs_dir(logs_dir=None):
    if logs_dir is not None:
        return Path(logs_dir)
    env = os.environ.get("LOGS_DIR")
    if env:
        return Path(env)
    raise ValueError(
        "LOGS_DIR não definido. Rode via 'bash studies/general/run_all.sh <nome>' ou "
        "passe --logs-dir <caminho> (ex.: --logs-dir logs/sparsification_percentage)."
    )


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

    # Optional client_fraction token: _cf<frac>_ anywhere in folder name.
    # Default 1.0 for folders without the token (e.g. sparsification_percentage).
    cf_m = re.search(r"_cf([0-9.]+)_", name)
    client_fraction = float(cf_m.group(1)) if cf_m else 1.0

    return {
        "model": d["model"],
        "dataset": d["dataset"],
        "solution": sol,
        "niid": float(d["niid"]),
        "percentage": None if pct == -1 else pct,
        "seed": int(d["seed"]),
        "rounds": int(d["rounds"]),
        "clients": int(d["clients"]),
        "client_fraction": client_fraction,
    }


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _folder_rounds(folder: Path) -> set[int] | None:
    """Rounds present in a folder's train.csv, or None when it has no usable rows."""
    df = _read_csv_safe(folder / "train.csv")
    if df is None or len(df) == 0 or "round" not in df.columns:
        return None
    rounds = pd.to_numeric(df["round"], errors="coerce").dropna()
    if rounds.empty:
        return None
    return set(rounds.astype(int).tolist())


_MAX_BRUTE_FORCE_RUNS = 10


def _best_disjoint_subset(runs):
    """Pick the pairwise-disjoint subset of (folder, rounds) with widest coverage.

    `runs` must be sorted by folder name, so timestamps are ascending and the
    newest-wins tie-break falls out of preferring later indices.
    """
    if len(runs) > _MAX_BRUTE_FORCE_RUNS:
        # Degenerate case, not seen in practice: fall back to longest-first greedy.
        taken: set[int] = set()
        chosen = []
        for folder, rounds in sorted(runs, key=lambda i: (-len(i[1]), i[0].name)):
            if not (rounds & taken):
                chosen.append((folder, rounds))
                taken |= rounds
        return chosen

    best, best_key = [], (-1, -1, ())
    for bits in range(1, 1 << len(runs)):
        subset = [runs[i] for i in range(len(runs)) if bits >> i & 1]
        union: set[int] = set()
        for _, rounds in subset:
            if rounds & union:
                union = set()
                break
            union |= rounds
        if not union:
            continue
        key = (len(union), max(union), tuple(i for i in range(len(runs)) if bits >> i & 1))
        if key > best_key:
            best, best_key = subset, key
    return best


def select_folders(folders: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split experiment folders into the ones to load and the duplicates to skip.

    Folders sharing a config (same name minus the trailing timestamp) come in two
    flavours, and only one of them may be concatenated:

    * disjoint round ranges (e.g. 1-124 and 125-200) are resume fragments that
      `merge_resume.py` never merged. Their union is the real run, so all of them
      are kept.
    * overlapping round ranges are independent reruns of the same config. Only one
      is kept: studies that sum a per-client metric over a round (bytes in study
      15, comm overhead in study 16) would otherwise count one experiment twice
      and report double the true cost.

    The two cases can be mixed inside one config (an abandoned attempt alongside a
    complete resume chain), so the kept set is the pairwise-disjoint subset with
    the largest round coverage rather than a greedy longest-first pick. Ties go to
    the subset reaching the highest round, then to the newest timestamps.

    Returns (kept, skipped), both sorted by folder name.
    """
    groups: dict[str, list[Path]] = {}
    for folder in folders:
        m = _TS_RE.search(folder.name)
        config = folder.name[: m.start()] if m else folder.name
        groups.setdefault(config, []).append(folder)

    keep: list[Path] = []
    skip: list[Path] = []
    for config in sorted(groups):
        members = groups[config]
        if len(members) == 1:
            keep.append(members[0])
            continue

        scored = [(f, _folder_rounds(f)) for f in members]
        # Folders with no readable train.csv have no rounds to clash over; keeping
        # them preserves whatever other CSVs they carry.
        keep.extend(f for f, rounds in scored if rounds is None)

        runs = sorted(((f, r) for f, r in scored if r is not None),
                      key=lambda item: item[0].name)
        if not runs:
            continue

        best = _best_disjoint_subset(runs)
        keep.extend(f for f, _ in best)
        chosen = {f.name for f, _ in best}
        skip.extend(f for f, _ in runs if f.name not in chosen)

    return sorted(keep, key=lambda f: f.name), sorted(skip, key=lambda f: f.name)


def config_runs(root, prefix: str, tokens=()) -> list[list[Path]]:
    """Runs of one config, each a list of folders ordered by round.

    The config is every folder under root whose name starts with `prefix` and
    contains every string in `tokens`. Each complete folder (all rounds 1..R of its
    name) is an independent run and is returned on its own. When none is complete,
    the resume fragments chosen by select_folders form a single stitched run.
    """
    folders = [f for f in sorted(Path(root).iterdir())
               if f.is_dir() and f.name.startswith(prefix) and all(t in f.name for t in tokens)]
    if not folders:
        return []
    complete = []
    for f in folders:
        m = re.search(r"_(\d+)rounds_", f.name)
        rounds = _folder_rounds(f)
        if m and rounds and set(range(1, int(m.group(1)) + 1)) <= rounds:
            complete.append([f])
    # if complete:
    #     return complete
    if complete:
    # One run per configuration: when a configuration has several complete
    # folders, the older ones are superseded executions and are dropped.
        newest = {}
        for run in complete:
            name = run[0].name
            m = _TS_RE.search(name)
            key = name[: m.start()] if m else name
            if key not in newest or name > newest[key][0].name:
                newest[key] = run
        return [newest[k] for k in sorted(newest)]
    keep, _ = select_folders(folders)
    keep = [f for f in keep if _folder_rounds(f)]
    return [sorted(keep, key=lambda f: min(_folder_rounds(f)))] if keep else []


def read_run(run: list[Path], name: str) -> pd.DataFrame | None:
    """Concatenate one CSV across the folders of a run, first folder wins per round."""
    parts = [df for df in (_read_csv_safe(f / name) for f in run) if df is not None and len(df)]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    if name in ("server_eval.csv", "server_agg.csv"):
        df = df.drop_duplicates("round", keep="first")
    return df.sort_values("round", kind="stable").reset_index(drop=True)


def final_acc(run: list[Path], rounds: int = 200):
    """Accuracy at round `rounds`; 'div' when the loss went NaN; None when the run is short."""
    ev = read_run(run, "server_eval.csv")
    if ev is None or ev["round"].max() < rounds:
        return None
    if ev["mean_loss"].isna().any():
        return "div"
    return float(ev.loc[ev["round"] == rounds, "mean_acc"].iloc[0])


def last_clean_acc(run: list[Path], rounds: int = 200):
    """(accuracy, round, collapsed) of one run.

    A handful of MobileNetV2 runs return a corrupted aggregate partway through: every
    client drops to chance with a non-finite evaluation loss, while local training keeps
    improving. For those the accuracy is read at the last round before the first
    non-finite loss, since later finite rounds already carry the damaged state.
    """
    ev = read_run(run, "server_eval.csv")
    if ev is None or ev["round"].max() < rounds:
        return None, None, False
    bad = ev["mean_loss"].isna()
    if not bad.any():
        return float(ev.loc[ev["round"] == rounds, "mean_acc"].iloc[0]), rounds, False
    clean = ev[ev["round"] < int(ev.loc[bad, "round"].min())]
    if clean.empty:
        return None, None, True
    return float(clean["mean_acc"].iloc[-1]), int(clean["round"].iloc[-1]), True


def run_masks(run: list[Path], first_round: int = 1):
    """(n_packs, {round: [set of packets per client]}) of one run, from mask.csv."""
    df = read_run(run, "mask.csv")
    if df is None:
        return None, {}
    masks: dict[int, list[set]] = {}
    for rnd, raw in zip(df["round"].astype(int), df["selected_packs"].fillna("").astype(str)):
        if rnd >= first_round:
            masks.setdefault(rnd, []).append({int(float(x)) for x in raw.split(";") if x.strip()})
    return int(df["n_packs"].iloc[0]), masks


def _experiment_folders(root: Path, dedup: bool = True) -> list[Path]:
    """Experiment folders under root, deduplicated by default (see select_folders).

    Pass dedup=False for metrics that average rather than sum. Independent reruns
    of one config are separate observations of accuracy and must both count, while
    a metric summed over rounds would double-count them.
    """
    candidates = [f for f in sorted(root.iterdir())
                  if f.is_dir() and _parse_folder(f.name) is not None]
    if not dedup:
        return candidates
    keep, skip = select_folders(candidates)
    if skip:
        print(
            f"[load_data] {len(skip)} pasta(s) ignorada(s) por serem reruns "
            f"duplicados da mesma config; {len(keep)} carregada(s).",
            file=sys.stderr,
        )
    return keep


def count_experiments(logs_dir=None) -> int:
    """Return the number of valid experiment folders in logs_dir.

    Useful for pre-flight validation before creating output directories.
    Raises FileNotFoundError if the directory doesn't exist.
    """
    root = _resolve_logs_dir(logs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Logs directory not found: {root.resolve()}")
    return len(_experiment_folders(root))


def load_all(logs_dir=None, dedup: bool = True) -> dict[str, pd.DataFrame]:
    """Load all experiment CSVs from logs_dir into four DataFrames.

    Returns dict with keys: 'train', 'evaluate', 'server_eval', 'server_agg'.
    Each DataFrame has extra columns: model, dataset, solution, niid, percentage, seed.
    percentage=None means plaintext/no-packing baseline (was -1 in folder name).
    """
    root = _resolve_logs_dir(logs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Logs directory not found: {root.resolve()}")

    train_parts, eval_parts, seval_parts, sagg_parts = [], [], [], []
    valid_folders = 0

    for folder in _experiment_folders(root, dedup):
        meta = _parse_folder(folder.name)

        valid_folders += 1
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
                df["run"] = folder.name
                parts.append(df)

    if valid_folders == 0:
        raise FileNotFoundError(
            f"Nenhum experimento válido encontrado em: {root.resolve()}\n"
            "  Verifique se o nome do experimento está correto e se os logs foram gerados."
        )

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
    for folder in _experiment_folders(root):
        meta = _parse_folder(folder.name)
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
