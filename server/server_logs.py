"""
Server-side CSV logging for FL+HE experiments.

server_agg.csv  — one row per FL round (aggregate_fit):
  experiment_name, round, n_clients, total_examples,
  selection_time_s, aggregation_time_s

server_eval.csv — one row per FL round (aggregate_evaluate):
  experiment_name, round, n_clients, mean_loss, mean_acc
"""

from __future__ import annotations

import os

_AGG_HEADER  = "experiment_name,round,n_clients,total_examples,selection_time_s,aggregation_time_s"
_EVAL_HEADER = "experiment_name,round,n_clients,mean_loss,mean_acc"


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
