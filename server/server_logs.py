"""
Server-side CSV logging for FL+HE experiments.

server_agg.csv  — one row per FL round (aggregate_fit):
  experiment_name, round, n_clients, total_examples,
  selection_time_s, aggregation_time_s

server_eval.csv — one row per FL round (aggregate_evaluate):
  experiment_name, round, n_clients, mean_loss, mean_acc

coverage.csv — one row per (round, pack_idx) for packing solutions:
  experiment_name, round, solution, pack_idx,
  n_contributors, n_clients_round
  n_contributors = number of clients that sent this pack in this round (|S_k|)
  n_clients_round = total clients that reported results this round
"""

from __future__ import annotations

import os

_AGG_HEADER      = "experiment_name,round,n_clients,total_examples,selection_time_s,aggregation_time_s"
_EVAL_HEADER     = "experiment_name,round,n_clients,mean_loss,mean_acc"
_COVERAGE_HEADER = "experiment_name,round,solution,pack_idx,n_contributors,n_clients_round"


def _append(filename: str, header: str, row: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        if os.path.getsize(filename) == 0:
            f.write(header + "\n")
        f.write(row + "\n")


def write_agg_logs(
    log_folder: str,
    experiment_name: str,
    server_round: int,
    n_clients: int,
    total_examples: int,
    selection_time_s: float,
    aggregation_time_s: float,
) -> None:
    row = (
        f"{experiment_name},{server_round},{n_clients},{total_examples},"
        f"{selection_time_s:.4f},{aggregation_time_s:.4f}"
    )
    _append(f"logs/{log_folder}/server_agg.csv", _AGG_HEADER, row)


def write_eval_logs(
    log_folder: str,
    experiment_name: str,
    server_round: int,
    n_clients: int,
    mean_loss: float,
    mean_acc: float,
) -> None:
    row = (
        f"{experiment_name},{server_round},{n_clients},"
        f"{mean_loss:.6f},{mean_acc:.6f}"
    )
    _append(f"logs/{log_folder}/server_eval.csv", _EVAL_HEADER, row)


def write_coverage_logs(
    log_folder: str,
    experiment_name: str,
    server_round: int,
    solution: str,
    contributor_counts: list,
    n_clients_round: int,
) -> None:
    """Write per-pack contributor counts for one round to logs/{log_folder}/coverage.csv.

    One row per pack_idx. n_contributors = |S_k| = number of clients that sent
    pack k in this round. n_clients_round = total clients that responded.
    """
    path = f"logs/{log_folder}/coverage.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as f:
        if is_new:
            f.write(_COVERAGE_HEADER + "\n")
        for pack_idx, count in enumerate(contributor_counts):
            f.write(
                f"{experiment_name},{server_round},{solution},"
                f"{pack_idx},{count},{n_clients_round}\n"
            )
