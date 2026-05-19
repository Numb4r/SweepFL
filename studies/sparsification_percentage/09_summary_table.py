"""
09 — Summary tables (CSV + LaTeX).

CSV: one row per (model, dataset, solution, niid, percentage) with:
  final_acc, final_loss, convergence_round_80pct, mean_payload_mb,
  mean_encrypt_time_s, mean_he_overhead_s, auc_acc

LaTeX: longtable grouped by solution, best values bold,
  caption + label included.

Output: output/tables/summary.{csv,tex}
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("load_data", HERE / "load_data.py")
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)

OUT_TABLES = HERE / "output" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

HE_OPS = ["encrypt_time_s", "serialize_time_s", "pack_time_s", "mask_select_time_s"]


def build_summary(dfs) -> pd.DataFrame:
    se = dfs["server_eval"]
    train = dfs["train"]

    group = ["model", "dataset", "solution", "niid", "percentage"]
    group = [c for c in group if c in se.columns]

    # final accuracy & loss
    lr = load_data.last_round(se, group_cols=group + ["seed"])
    acc = lr.groupby(group, dropna=False)[["mean_acc", "mean_loss"]].mean().reset_index()
    acc.rename(columns={"mean_acc": "final_acc", "mean_loss": "final_loss"}, inplace=True)

    # convergence round
    conv = load_data.rounds_to_threshold(se, metric="mean_acc", threshold=0.80)
    conv_g = [c for c in group if c in conv.columns]
    if not conv.empty:
        conv_agg = conv.groupby(conv_g, dropna=False)["convergence_round"].mean().reset_index()
        conv_agg.rename(columns={"convergence_round": "convergence_round_80pct"}, inplace=True)
        acc = acc.merge(conv_agg, on=conv_g, how="left")

    # AUC
    auc = load_data.auc_accuracy(se)
    auc_g = [c for c in group if c in auc.columns]
    if not auc.empty:
        auc_agg = auc.groupby(auc_g, dropna=False)["auc_norm"].mean().reset_index()
        auc_agg.rename(columns={"auc_norm": "auc_acc"}, inplace=True)
        acc = acc.merge(auc_agg, on=auc_g, how="left")

    # payload + timing
    if not train.empty:
        train_g = [c for c in group if c in train.columns]
        cols_to_agg = {}
        if "payload_bytes" in train.columns:
            cols_to_agg["payload_bytes"] = "mean"
        if "encrypt_time_s" in train.columns:
            cols_to_agg["encrypt_time_s"] = "mean"
        present_he = [c for c in HE_OPS if c in train.columns]

        if cols_to_agg or present_he:
            t_agg = train.groupby(train_g, dropna=False).agg(
                **{k: (k, v) for k, v in cols_to_agg.items()}
            ).reset_index()
            if present_he:
                he_sum = train.copy()
                he_sum["he_overhead"] = he_sum[present_he].sum(axis=1)
                he_agg = he_sum.groupby(train_g, dropna=False)["he_overhead"].mean().reset_index()
                t_agg = t_agg.merge(he_agg, on=train_g, how="left")
                t_agg.rename(columns={"he_overhead": "mean_he_overhead_s"}, inplace=True)
            if "payload_bytes" in t_agg.columns:
                t_agg["mean_payload_mb"] = t_agg["payload_bytes"] / 1e6
                t_agg.drop(columns=["payload_bytes"], inplace=True)
            if "encrypt_time_s" in t_agg.columns:
                t_agg.rename(columns={"encrypt_time_s": "mean_encrypt_time_s"}, inplace=True)
            acc = acc.merge(t_agg, on=train_g, how="left")

    # round to 4 decimal places
    for col in acc.select_dtypes(include=[float]).columns:
        acc[col] = acc[col].round(4)

    return acc.sort_values(["solution", "model", "niid", "percentage"], na_position="last")


def to_latex(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    # higher is better for acc/auc; lower is better for times/payload/convergence
    higher_better = ["final_acc", "auc_acc"]
    lower_better = ["final_loss", "convergence_round_80pct", "mean_payload_mb",
                    "mean_encrypt_time_s", "mean_he_overhead_s"]

    best_vals = {}
    for col in numeric_cols:
        vals = df[col].dropna()
        if vals.empty:
            continue
        best_vals[col] = vals.max() if col in higher_better else vals.min()

    def _fmt(row, col):
        val = row[col]
        if pd.isna(val):
            return "—"
        s = f"{val:.4f}" if isinstance(val, float) else str(val)
        if col in best_vals and not pd.isna(best_vals[col]):
            if col in higher_better and val == best_vals[col]:
                s = r"\textbf{" + s + "}"
            elif col in lower_better and val == best_vals[col]:
                s = r"\textbf{" + s + "}"
        return s

    col_labels = {
        "model": "Model", "dataset": "Dataset", "solution": "Solution",
        "niid": "NIID $\\alpha$", "percentage": "Pct.",
        "final_acc": "Acc.", "final_loss": "Loss",
        "convergence_round_80pct": "Conv. Rd.", "mean_payload_mb": "Payload (MB)",
        "mean_encrypt_time_s": "Enc. (s)", "mean_he_overhead_s": "HE ovhd (s)",
        "auc_acc": "AUC",
    }

    display_cols = [c for c in df.columns if c in col_labels]
    header = " & ".join(col_labels.get(c, c) for c in display_cols) + r" \\"

    lines = [
        r"\begin{longtable}{" + "l" * len(display_cols) + "}",
        r"\caption{Sparsification percentage experiment summary}",
        r"\label{tab:sparsification_summary} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(len(display_cols)) + r"}{r}{\textit{continued\ldots}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    prev_solution = None
    for _, row in df.iterrows():
        sol = row.get("solution", "")
        if prev_solution is not None and sol != prev_solution:
            lines.append(r"\midrule")
        prev_solution = sol
        cells = " & ".join(_fmt(row, c) for c in display_cols)
        lines.append(cells + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def main(logs_dir=None):
    dfs = load_data.load_all(logs_dir)
    if dfs["server_eval"].empty:
        print("No server_eval data found.")
        return

    summary = build_summary(dfs)
    print(f"Summary: {len(summary)} rows, {list(summary.columns)}")

    csv_path = OUT_TABLES / "summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    tex = to_latex(summary)
    tex_path = OUT_TABLES / "summary.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=None)
    args = parser.parse_args()
    main(args.logs_dir)
