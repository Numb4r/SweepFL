"""
24 — Full-model coverage convergence table (sparsification_percentage).

Answers: for each technique × model × dataset, in which federated round does the
server first receive ALL model packs at least once (100% unique coverage)?  And
what is the accuracy, cumulative wall-clock time, and cumulative data uploaded at
that exact round?

Definition of 100% coverage
  Server-side union of pack indices across all clients per round is accumulated in a
  running set `covered`.  The first round where |covered| == n_packs is the
  convergence round.  This is the first 100%-hit of the sawtooth in study 17 — we
  stop at that point (no reset).  Techniques that transmit the same fixed packs
  every round (fedphe / top-k) never cover the whole model → shown as "—".

Output layout
  One table per sparsification percentage.  Each table has rows keyed by
  (solution, model, dataset).  Two main column groups, one per IID setting:
    • IID  (niid = 1.0) : round_100, accuracy (%), time (s), MB
    • NIID (niid = 0.1) : round_100, accuracy (%), time (s), MB
  Values are averaged across seeds.  NaN → "—" in all outputs.

Time convention
  Wall-clock time per round ≈ average over clients of the sum of all per-client
  timing columns (clients run in parallel), plus the server-side selection_time_s
  and aggregation_time_s for that round.  Cumulative = sum over rounds 1…conv_round.

Bytes convention (raw float32 model data, pre-encryption)
  packing solutions (nvnl, sw, rw, fedphe, …): packs_sent × pack_size × 4 bytes / client
  other (plaintext, ckks, bfv):                 n_params × 4 bytes / client
  Total per round = sum over all clients.  `payload_bytes` is NOT used — it records
  only mask metadata for packing solutions, not the actual ciphertext size.

Source data: mask.csv  (via load_data.load_masks)
             train.csv + server_eval.csv + server_agg.csv (via load_data.load_all)

Output
  output/tables/24_full_coverage_convergence_p<NNN>.{csv,tex}  — one per percentage
  output/tables/24_full_coverage_convergence_all.csv            — consolidated
"""

import argparse
import importlib.util
import sys
from math import isnan
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("load_data", HERE / "load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)

OUT_TAB = HERE / "output" / "tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ── Dimensions ────────────────────────────────────────────────────────────────

SEED_DIMS   = ["model", "dataset", "solution", "niid", "percentage", "seed"]
GROUP_DIMS  = ["model", "dataset", "solution", "niid", "percentage"]

# niid → column-group label
NIID_LABEL = {1.0: "IID", 0.1: "NIID"}

# Packing solutions (bytes estimated from packs_sent * pack_size * 4)
PACKING_SOLS = {"nvnl", "sw", "rw", "fedphe", "fedphe-scout", "nvnl-scout",
                "srr", "topkrr"}

# Client-side timing columns to sum for per-round wall-clock estimate
TIME_COLS = [
    "train_time_s",
    "decrypt_time_s",
    "flatten_time_s",
    "quantize_time_s",
    "batch_padding_time_s",
    "pack_time_s",
    "mask_select_time_s",
    "encrypt_time_s",
    "serialize_time_s",
]


# ── Step 1: Round of first 100% coverage (per config+seed) ───────────────────

def _first_full_coverage_round(group_df: pd.DataFrame) -> float:
    """Return the first round where all packs have been sent at least once.

    Accumulates a set of seen pack indices across rounds (server-side union of
    all clients per round).  Returns the round number when the set first reaches
    n_packs, or NaN if it never does.
    """
    n_packs = int(group_df["n_packs"].iloc[0])
    if n_packs <= 0:
        return float("nan")

    covered: set = set()
    for rnd, rnd_df in group_df.sort_values("round").groupby("round", sort=False):
        server_union: set = set()
        for sel in rnd_df["selected"]:
            server_union.update(sel)
        covered.update(server_union)
        if len(covered) >= n_packs:
            return float(int(rnd))

    return float("nan")


def compute_conv_rounds(masks: pd.DataFrame) -> pd.DataFrame:
    """Compute the first-100%-coverage round for every (config+seed) group.

    Returns a DataFrame with columns SEED_DIMS + ['conv_round'].
    """
    gcols = [c for c in SEED_DIMS if c in masks.columns]
    rows = []
    for key, grp in masks.groupby(gcols, dropna=False):
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        rows.append({**meta, "conv_round": _first_full_coverage_round(grp)})
    return pd.DataFrame(rows)


# ── Step 2: Cumulative bytes up to conv_round ─────────────────────────────────

def _est_bytes_per_client(row) -> float:
    """Raw float32 model data per client per round (pre-encryption)."""
    sol = str(row.get("solution", ""))
    if sol in PACKING_SOLS:
        packs     = float(row.get("packs_sent") or 0)
        pack_size = float(row.get("pack_size") or 0)
        return packs * pack_size * 4  # float32
    else:
        n = float(row.get("n_params") or 0)
        return n * 4  # float32


def compute_cumulative_bytes(train: pd.DataFrame) -> pd.DataFrame:
    """Sum bytes over clients per round, cumsum over rounds per experiment.

    Returns SEED_DIMS + ['round', 'cumulative_mb'].
    """
    df = train.copy()
    df["bytes_client"] = df.apply(_est_bytes_per_client, axis=1)

    round_group = [c for c in SEED_DIMS + ["round"] if c in df.columns]
    per_round = (
        df.groupby(round_group, dropna=False)["bytes_client"]
        .sum()
        .reset_index()
        .rename(columns={"bytes_client": "bytes_round"})
    )

    seed_group = [c for c in SEED_DIMS if c in per_round.columns]
    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_mb"] = (
        per_round.groupby(seed_group, dropna=False)["bytes_round"]
        .cumsum() / 1e6
    )
    return per_round[seed_group + ["round", "cumulative_mb"]]


# ── Step 3: Cumulative wall-clock time up to conv_round ──────────────────────

def compute_cumulative_time(train: pd.DataFrame,
                             server_agg: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative wall-clock time (s) per round per experiment.

    Per round:
      client_time = mean over clients of sum(TIME_COLS)  (parallel execution)
      server_time = selection_time_s + aggregation_time_s  (from server_agg)
      total_round = client_time + server_time

    Returns SEED_DIMS + ['round', 'cumulative_time_s'].
    """
    df = train.copy()
    present_time = [c for c in TIME_COLS if c in df.columns]
    if not present_time:
        # No timing columns available; return zeros
        seed_group = [c for c in SEED_DIMS if c in df.columns]
        round_group = seed_group + ["round"]
        per_round = df.groupby(round_group, dropna=False).size().reset_index(name="_n")
        per_round["cumulative_time_s"] = 0.0
        return per_round[seed_group + ["round", "cumulative_time_s"]]

    df["client_round_time"] = df[present_time].fillna(0).sum(axis=1)

    seed_group = [c for c in SEED_DIMS if c in df.columns]
    round_group = seed_group + ["round"]

    # Mean over clients within each round (parallel execution model)
    per_round = (
        df.groupby(round_group, dropna=False)["client_round_time"]
        .mean()
        .reset_index()
        .rename(columns={"client_round_time": "client_time_s"})
    )

    # Add server-side time (selection + aggregation)
    if not server_agg.empty:
        sagg_keys = [c for c in SEED_DIMS + ["round"] if c in server_agg.columns]
        sagg_time_cols = [c for c in ["selection_time_s", "aggregation_time_s"]
                         if c in server_agg.columns]
        if sagg_time_cols:
            sagg = server_agg[sagg_keys + sagg_time_cols].copy()
            sagg["server_time_s"] = sagg[sagg_time_cols].fillna(0).sum(axis=1)
            sagg = sagg[[c for c in sagg_keys if c in per_round.columns] + ["server_time_s"]]
            per_round = per_round.merge(sagg, on=round_group, how="left")
            per_round["server_time_s"] = per_round["server_time_s"].fillna(0)
        else:
            per_round["server_time_s"] = 0.0
    else:
        per_round["server_time_s"] = 0.0

    per_round["round_time_s"] = per_round["client_time_s"] + per_round["server_time_s"]

    per_round = per_round.sort_values(seed_group + ["round"])
    per_round["cumulative_time_s"] = (
        per_round.groupby(seed_group, dropna=False)["round_time_s"]
        .cumsum()
    )
    return per_round[seed_group + ["round", "cumulative_time_s"]]


# ── Step 4: Accuracy at conv_round ───────────────────────────────────────────

def _lookup_at_round(df: pd.DataFrame, round_col: str,
                     value_col: str, query_round: float) -> float:
    """Return value_col where round_col == query_round (first match), or NaN."""
    if pd.isna(query_round):
        return float("nan")
    r = int(query_round)
    hit = df[df[round_col] == r]
    if hit.empty:
        return float("nan")
    return float(hit[value_col].iloc[0])


# ── Step 5: Build per-seed summary ───────────────────────────────────────────

def build_per_seed_summary(conv_rounds: pd.DataFrame,
                            server_eval: pd.DataFrame,
                            cum_bytes: pd.DataFrame,
                            cum_time: pd.DataFrame) -> pd.DataFrame:
    """For each (config+seed), collect: conv_round, acc, time_s, mb."""
    gcols = [c for c in SEED_DIMS if c in conv_rounds.columns]
    rows = []
    for key, grp in conv_rounds.groupby(gcols, dropna=False):
        meta = dict(zip(gcols, key if isinstance(key, tuple) else (key,)))
        conv_round = float(grp["conv_round"].iloc[0])

        # Filter the joined DFs to this (config+seed)
        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            mask = pd.Series(True, index=df.index)
            for k, v in meta.items():
                if k in df.columns:
                    if pd.isna(v):
                        mask &= df[k].isna()
                    else:
                        mask &= (df[k] == v)
            return df[mask]

        se_sub  = _filter(server_eval)
        cb_sub  = _filter(cum_bytes)
        ct_sub  = _filter(cum_time)

        acc    = _lookup_at_round(se_sub,  "round", "mean_acc",       conv_round)
        mb     = _lookup_at_round(cb_sub,  "round", "cumulative_mb",  conv_round)
        time_s = _lookup_at_round(ct_sub,  "round", "cumulative_time_s", conv_round)

        rows.append({
            **meta,
            "conv_round": conv_round,
            "acc":        acc,
            "mb":         mb,
            "time_s":     time_s,
        })
    return pd.DataFrame(rows)


# ── Step 6: Average over seeds ────────────────────────────────────────────────

def average_over_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean of conv_round, acc, mb, time_s over seeds per GROUP_DIMS."""
    gcols = [c for c in GROUP_DIMS if c in per_seed.columns]
    return (
        per_seed.groupby(gcols, dropna=False)[["conv_round", "acc", "mb", "time_s"]]
        .mean()
        .reset_index()
    )


# ── Step 7: Pivot IID / NIID ──────────────────────────────────────────────────

def pivot_iid_niid(averaged: pd.DataFrame,
                   niid_label_map: dict) -> pd.DataFrame:
    """Pivot so IID and NIID become column groups on the same row.

    Index: (solution, model, dataset, percentage)
    Columns per group: round_100, acc, time_s, mb
    """
    idx_cols = ["solution", "model", "dataset", "percentage"]
    idx_cols = [c for c in idx_cols if c in averaged.columns]
    value_cols = ["conv_round", "acc", "time_s", "mb"]

    parts = []
    for niid_val, label in niid_label_map.items():
        sub = averaged[averaged["niid"] == niid_val].copy() if "niid" in averaged.columns else averaged.copy()
        sub = sub[[c for c in idx_cols + value_cols if c in sub.columns]]
        rename = {
            "conv_round": f"{label.lower()}_round",
            "acc":        f"{label.lower()}_acc",
            "time_s":     f"{label.lower()}_time_s",
            "mb":         f"{label.lower()}_mb",
        }
        sub = sub.rename(columns=rename)
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    result = parts[0]
    for p in parts[1:]:
        result = result.merge(p, on=idx_cols, how="outer")

    result = result.sort_values(
        [c for c in ["solution", "model", "dataset"] if c in result.columns],
        na_position="last",
    )
    return result


# ── Output helpers ────────────────────────────────────────────────────────────

def _pct_tag(pct: float) -> str:
    """e.g. 0.1 → 'p010', 0.5 → 'p050'"""
    return f"p{int(round(pct * 100)):03d}"


def save_csv(table: pd.DataFrame, path: Path):
    for c in table.select_dtypes(include=float).columns:
        table[c] = table[c].round(3)
    table.to_csv(path, index=False)
    print(f"Saved {path}")


def save_latex(table: pd.DataFrame, pct: float, path: Path,
               niid_labels: list[str]):
    """Write a longtable with a two-row header: technique cols + IID/NIID groups."""

    # Fixed left cols
    left_cols  = [c for c in ["solution", "model", "dataset"] if c in table.columns]
    # Dynamic value cols, ordered by IID then NIID
    val_suffixes = ["round", "acc", "time_s", "mb"]
    grp_cols = {}
    for lbl in niid_labels:
        gcols = [f"{lbl.lower()}_{s}" for s in val_suffixes if f"{lbl.lower()}_{s}" in table.columns]
        if gcols:
            grp_cols[lbl] = gcols

    all_cols = left_cols + [c for gc in grp_cols.values() for c in gc]

    # Best (min) round per (model, dataset) × IID/NIID group — key: (model, dataset, round_col)
    best: dict[tuple, float | None] = {}
    grp_key_cols = [c for c in ["model", "dataset"] if c in table.columns]
    for lbl in niid_labels:
        round_col = f"{lbl.lower()}_round"
        if round_col not in table.columns:
            continue
        if grp_key_cols:
            for keys, sub in table.groupby(grp_key_cols, dropna=False):
                keys_t = keys if isinstance(keys, tuple) else (keys,)
                vals = sub[round_col].dropna()
                best[(*keys_t, round_col)] = float(vals.min()) if not vals.empty else None
        else:
            vals = table[round_col].dropna()
            best[(round_col,)] = float(vals.min()) if not vals.empty else None

    def _best_for_row(row, round_col: str) -> float | None:
        keys_t = tuple(row.get(c) for c in grp_key_cols)
        return best.get((*keys_t, round_col), best.get((round_col,)))

    col_display = {
        "solution": "Solution", "model": "Model", "dataset": "Dataset",
    }
    for lbl in niid_labels:
        col_display[f"{lbl.lower()}_round"]  = "Round"
        col_display[f"{lbl.lower()}_acc"]    = "Acc. (\\%)"
        col_display[f"{lbl.lower()}_time_s"] = "Time (s)"
        col_display[f"{lbl.lower()}_mb"]     = "MB"

    def _fmt(row, col):
        val = row.get(col)
        if val is None or (isinstance(val, float) and isnan(val)):
            return "—"
        if col.endswith("_round"):
            s = str(int(val))
            b = _best_for_row(row, col)
            if b is not None and val == b:
                s = r"\textbf{" + s + "}"
            return s
        if col.endswith("_acc"):
            return f"{val * 100:.2f}"
        if col.endswith("_time_s"):
            return f"{val:.1f}"
        if col.endswith("_mb"):
            return f"{val:.2f}"
        return str(val)

    n_left  = len(left_cols)
    n_total = len(all_cols)
    col_spec = "l" * n_left + ("|" + "r" * 4) * len(grp_cols)

    pct_lbl = f"{pct * 100:.0f}\\%"

    # Two-row header
    # Row 1: left spacers + \multicolumn per niid group
    hdr1_parts = [""] * n_left
    for lbl, gcols in grp_cols.items():
        hdr1_parts.append(r"\multicolumn{" + str(len(gcols)) + r"}{c}{\textbf{" + lbl + r"}}")
    hdr1 = " & ".join(hdr1_parts) + r" \\"
    # Row 2: per-column labels
    hdr2_parts = [col_display.get(c, c) for c in left_cols]
    for gcols in grp_cols.values():
        hdr2_parts += [col_display.get(c, c) for c in gcols]
    hdr2 = " & ".join(hdr2_parts) + r" \\"

    # Build cmidrule spans: one per niid group
    cmidrule_parts = []
    col_offset = n_left + 1
    for gcols in grp_cols.values():
        cmidrule_parts.append(
            r"\cmidrule(lr){" + f"{col_offset}-{col_offset + len(gcols) - 1}" + r"}"
        )
        col_offset += len(gcols)
    cmidrule_line = " ".join(cmidrule_parts)

    lines = [
        r"\begin{longtable}{" + col_spec + "}",
        r"\caption{100\% model coverage convergence --- sparsification " + pct_lbl + r"}",
        r"\label{tab:full_coverage_" + _pct_tag(pct) + r"} \\",
        r"\toprule",
        hdr1, cmidrule_line,
        hdr2,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        hdr1, cmidrule_line,
        hdr2,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(n_total) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    prev_sol = None
    for _, row in table.iterrows():
        sol = row.get("solution", "")
        if prev_sol is not None and sol != prev_sol:
            lines.append(r"\midrule")
        prev_sol = sol
        lines.append(" & ".join(_fmt(row, c) for c in all_cols) + r" \\")

    lines.append(r"\end{longtable}")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(logs_dir=None):
    print("Loading data...")
    try:
        masks = load_data.load_masks(logs_dir)
    except FileNotFoundError as e:
        print(f"Error loading masks: {e}")
        return

    if masks.empty:
        print("No mask data found.")
        return

    # Keep only packing techniques (percentage not null = has mask)
    masks = masks[masks["percentage"].notna()].copy()
    if masks.empty:
        print("No packing-technique data found in mask.csv.")
        return

    dfs        = load_data.load_all(logs_dir)
    train      = dfs.get("train",      pd.DataFrame())
    server_eval = dfs.get("server_eval", pd.DataFrame())
    server_agg  = dfs.get("server_agg",  pd.DataFrame())

    dims = load_data.discover_dims(masks)
    print(f"Models      : {dims.get('model', [])}")
    print(f"Solutions   : {dims.get('solution', [])}")
    print(f"Percentages : {sorted(masks['percentage'].dropna().unique())}")
    print(f"Mask rows   : {len(masks):,}")
    print()

    # ── Compute per-seed results ──────────────────────────────────────────────
    print("Computing first-100%-coverage rounds...")
    conv_rounds = compute_conv_rounds(masks)

    print("Computing cumulative bytes...")
    if not train.empty:
        cum_bytes = compute_cumulative_bytes(train)
    else:
        print("  [WARNING] train.csv not available — MB column will be NaN")
        gcols = [c for c in SEED_DIMS + ["round"] if c in conv_rounds.columns]
        cum_bytes = pd.DataFrame(columns=gcols + ["cumulative_mb"])

    print("Computing cumulative time...")
    if not train.empty:
        cum_time = compute_cumulative_time(train, server_agg)
    else:
        print("  [WARNING] train.csv not available — time column will be NaN")
        gcols = [c for c in SEED_DIMS + ["round"] if c in conv_rounds.columns]
        cum_time = pd.DataFrame(columns=gcols + ["cumulative_time_s"])

    print("Joining accuracy at convergence round...")
    per_seed = build_per_seed_summary(conv_rounds, server_eval, cum_bytes, cum_time)

    print("Averaging over seeds...")
    averaged = average_over_seeds(per_seed)

    # ── Determine IID/NIID labels present in data ─────────────────────────────
    if "niid" in averaged.columns:
        present_niids = sorted(averaged["niid"].dropna().unique())
    else:
        present_niids = []
    niid_label_map = {n: lbl for n, lbl in NIID_LABEL.items() if n in present_niids}
    if not niid_label_map:
        # Fallback: use whatever niid values exist
        niid_label_map = {n: f"niid={n}" for n in present_niids}
    niid_labels = list(niid_label_map.values())

    print(f"NIID values found: {list(niid_label_map.keys())} → {niid_labels}")
    print()

    # ── Consolidated CSV (all percentages) ────────────────────────────────────
    consolidated = per_seed.copy()
    for c in consolidated.select_dtypes(include=float).columns:
        consolidated[c] = consolidated[c].round(3)
    consolidated.to_csv(OUT_TAB / "24_full_coverage_convergence_all.csv", index=False)
    print(f"Saved {OUT_TAB / '24_full_coverage_convergence_all.csv'}")

    # ── One table per percentage ───────────────────────────────────────────────
    percentages = sorted(averaged["percentage"].dropna().unique()) if "percentage" in averaged.columns else []
    if not percentages:
        print("No percentage dimension found — writing a single combined table.")
        percentages = [None]

    for pct in percentages:
        if pct is None:
            sub_avg = averaged.copy()
            tag = "all"
        else:
            sub_avg = averaged[averaged["percentage"] == pct].copy()
            tag = _pct_tag(pct)

        if sub_avg.empty:
            continue

        pivoted = pivot_iid_niid(sub_avg, niid_label_map)
        if pivoted.empty:
            continue

        # Drop percentage column from display (it's in the filename/caption)
        if "percentage" in pivoted.columns:
            pivoted = pivoted.drop(columns=["percentage"])

        csv_path = OUT_TAB / f"24_full_coverage_convergence_{tag}.csv"
        tex_path = OUT_TAB / f"24_full_coverage_convergence_{tag}.tex"

        save_csv(pivoted.copy(), csv_path)
        save_latex(pivoted, pct if pct is not None else 0.0, tex_path, niid_labels)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study 24: full-model coverage convergence table")
    parser.add_argument(
        "--logs-dir", default=None,
        help="Path to logs/sparsification_percentage directory (overrides LOGS_DIR env var)")
    args = parser.parse_args()
    main(args.logs_dir)
