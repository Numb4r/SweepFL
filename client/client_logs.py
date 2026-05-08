"""
Client-side CSV logging for FL+HE experiments.

train.csv — one row per (client, round):
  experiment_name, round, cid, solution, schema, model, dataset,
  distribution, dirichlet_alpha,
  train_loss, train_acc,
  train_time_s, encrypt_time_s, decrypt_time_s,
  payload_bytes, n_params,
  n_packs, packs_sent, pack_size, sigma_ckks,
  poly_modulus_degree, coeff_mod_bits, scale_bits, plain_modulus

evaluate.csv — one row per (client, round):
  experiment_name, round, cid, solution, schema, model, dataset,
  distribution, dirichlet_alpha,
  eval_loss, eval_acc, decrypt_time_s,
  poly_modulus_degree, coeff_mod_bits, scale_bits
"""

from __future__ import annotations

import math
import os

# ── Column headers ────────────────────────────────────────────────────────────

_TRAIN_HEADER = (
    "experiment_name,round,cid,"
    "solution,schema,model,dataset,distribution,dirichlet_alpha,"
    "train_loss,train_acc,"
    "train_time_s,encrypt_time_s,decrypt_time_s,"
    "payload_bytes,n_params,"
    "n_packs,packs_sent,pack_size,sigma_ckks,"
    "poly_modulus_degree,coeff_mod_bits,scale_bits,plain_modulus"
)

_EVAL_HEADER = (
    "experiment_name,round,cid,"
    "solution,schema,model,dataset,distribution,dirichlet_alpha,"
    "eval_loss,eval_acc,decrypt_time_s,"
    "poly_modulus_degree,coeff_mod_bits,scale_bits"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _coeff_str(coeff_mod_bit_sizes) -> str:
    if not coeff_mod_bit_sizes:
        return ""
    return ";".join(map(str, coeff_mod_bit_sizes))


def _scale_bits(global_scale: int) -> str:
    if not global_scale or global_scale <= 0:
        return ""
    try:
        return str(int(round(math.log2(global_scale))))
    except (ValueError, OverflowError):
        return ""


def _append(filename: str, header: str, row: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        if os.path.getsize(filename) == 0:
            f.write(header + "\n")
        f.write(row + "\n")


# ── Public API ────────────────────────────────────────────────────────────────

def write_train_logs(
    self,
    rnd: int,
    loss: float,
    acc: float,
    model_size: int,
    times: dict,
    qnt_packing: float = 0,
    size_pack: int = 0,
    sigma: float | None = None,
) -> None:
    """Write one training-round row to logs/{log_folder}/train.csv.

    Parameters
    ----------
    times : dict with keys train_time_clock, encrypt_time_clock (or
            cypher_time_clock), decrypt_time_clock (or decypher_time_clock).
    qnt_packing : number of packs actually sent (len(packs) * percentage).
    size_pack   : pack size in elements (poly_modulus_degree // 2).
    sigma       : estimated CKKS additive-noise std-dev (None if not CKKS).
    """
    distribution = "niid" if self.niid else "iid"
    dataset      = self.dataset.replace("/", "_")
    coeff_bits   = _coeff_str(self.coeff_mod_bit_sizes)
    scale_b      = _scale_bits(self.global_scale)
    exp_name     = getattr(self, "experiment_name", "")

    # Support both key conventions used across fit_*.py files
    train_t   = times.get("train_time_clock",   times.get("train_time", 0))
    encrypt_t = times.get("cypher_time_clock",  times.get("encrypt_time_clock", 0))
    decrypt_t = times.get("decypher_time_clock", times.get("decrypt_time_clock", 0))

    packs_sent = int(qnt_packing)
    n_packs    = (self.len_shared_data // size_pack + 1) if size_pack > 0 else 0

    row = (
        f"{exp_name},{rnd},{self.cid},"
        f"{self.solution},{self.homomorphic_type},{self.model_name},{dataset},"
        f"{distribution},{self.dirichlet_alpha},"
        f"{loss},{acc},"
        f"{train_t},{encrypt_t},{decrypt_t},"
        f"{model_size},{self.len_shared_data},"
        f"{n_packs},{packs_sent},{size_pack},"
        f"{sigma if sigma is not None else ''},"
        f"{self.poly_modulus_degree},{coeff_bits},{scale_b},{self.plain_modulus}"
    )
    _append(f"logs/{self.log_folder}/train.csv", _TRAIN_HEADER, row)


def write_evaluate_logs(
    self,
    rnd: int,
    loss: float,
    acc: float,
    decrypt_time_clock: float,
    decrypt_time_process: float,
) -> None:
    """Write one evaluation-round row to logs/{log_folder}/evaluate.csv."""
    distribution = "niid" if self.niid else "iid"
    dataset      = self.dataset.replace("/", "_")
    coeff_bits   = _coeff_str(self.coeff_mod_bit_sizes)
    scale_b      = _scale_bits(self.global_scale)
    exp_name     = getattr(self, "experiment_name", "")

    row = (
        f"{exp_name},{rnd},{self.cid},"
        f"{self.solution},{self.homomorphic_type},{self.model_name},{dataset},"
        f"{distribution},{self.dirichlet_alpha},"
        f"{loss},{acc},{decrypt_time_clock},"
        f"{self.poly_modulus_degree},{coeff_bits},{scale_b}"
    )
    _append(f"logs/{self.log_folder}/evaluate.csv", _EVAL_HEADER, row)
