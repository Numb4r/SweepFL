"""
Client-side CSV logging for FL+HE experiments.

train.csv — one row per (client, round):
  experiment_name, round, cid, solution, schema, model, dataset,
  distribution, dirichlet_alpha,
  train_loss, train_acc,
  decrypt_time_s, decrypt_time_cpu,
  train_time_s, train_time_cpu,
  flatten_time_s, flatten_time_cpu,
  quantize_time_s, quantize_time_cpu,
  batch_padding_time_s, batch_padding_time_cpu,
  pack_time_s, pack_time_cpu,
  mask_select_time_s, mask_select_time_cpu,
  encrypt_time_s, encrypt_time_cpu,
  serialize_time_s, serialize_time_cpu,
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


def _read_mem_mb() -> tuple[float, float]:
    """Return (rss_mb, peak_mb) from /proc/self/status. Returns (0, 0) on failure."""
    rss = peak = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmPeak:"):
                    peak = int(line.split()[1])
    except Exception:
        pass
    return rss / 1024, peak / 1024

# ── Column headers ────────────────────────────────────────────────────────────

_TRAIN_HEADER = (
    "experiment_name,round,cid,"
    "solution,schema,model,dataset,distribution,dirichlet_alpha,"
    "train_loss,train_acc,"
    "decrypt_time_s,decrypt_time_cpu,"
    "train_time_s,train_time_cpu,"
    "flatten_time_s,flatten_time_cpu,"
    "quantize_time_s,quantize_time_cpu,"
    "batch_padding_time_s,batch_padding_time_cpu,"
    "pack_time_s,pack_time_cpu,"
    "mask_select_time_s,mask_select_time_cpu,"
    "encrypt_time_s,encrypt_time_cpu,"
    "serialize_time_s,serialize_time_cpu,"
    "payload_bytes,n_params,"
    "n_packs,packs_sent,pack_size,sigma_ckks,"
    "poly_modulus_degree,coeff_mod_bits,scale_bits,plain_modulus,"
    "ram_rss_mb,ram_peak_mb,"
    "magg_renorm,ckks_backend,client_fraction"
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


def _g(times: dict, *keys, default: float = 0.0) -> float:
    """Get first matching key from times dict, fallback to default."""
    for k in keys:
        if k in times:
            return times[k]
    return default


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

    times keys (all optional, 0 if absent):
      decrypt_time_clock, decrypt_time_cpu      — set_parameters phase
      train_time_clock, train_time_process       — model.fit phase
      flatten_time_clock, flatten_time_cpu       — flat_parameters call
      quantize_time_clock, quantize_time_cpu     — BFV / BatchCrypt quantization
      batch_padding_time_clock, batch_padding_time_cpu — BatchCrypt batch-padding
      pack_time_clock, pack_time_cpu             — packing() call
      mask_select_time_clock, mask_select_time_cpu — mask selection
      encrypt_time_clock, encrypt_time_cpu       — HE encryption (cypher_packs / ts.ckks_vector / etc.)
      serialize_time_clock, serialize_time_cpu   — serialization (pickle.dumps / .serialize())

    Legacy aliases still accepted:
      cypher_time_clock → encrypt_time_clock
      decypher_time_clock → decrypt_time_clock
      train_time_process → train_time_cpu
    """
    distribution = "niid" if self.niid else "iid"
    dataset      = self.dataset.replace("/", "_")
    coeff_bits   = _coeff_str(self.coeff_mod_bit_sizes)
    scale_b      = _scale_bits(self.global_scale)
    exp_name     = getattr(self, "experiment_name", "")

    decrypt_s    = _g(times, "decrypt_time_clock",  "decypher_time_clock")
    decrypt_cpu  = _g(times, "decrypt_time_cpu",    "decypher_time_process")
    train_s      = _g(times, "train_time_clock")
    train_cpu    = _g(times, "train_time_cpu",      "train_time_process")
    flatten_s    = _g(times, "flatten_time_clock")
    flatten_cpu  = _g(times, "flatten_time_cpu")
    quantize_s   = _g(times, "quantize_time_clock")
    quantize_cpu = _g(times, "quantize_time_cpu")
    bpad_s       = _g(times, "batch_padding_time_clock")
    bpad_cpu     = _g(times, "batch_padding_time_cpu")
    pack_s       = _g(times, "pack_time_clock")
    pack_cpu     = _g(times, "pack_time_cpu")
    mask_s       = _g(times, "mask_select_time_clock")
    mask_cpu     = _g(times, "mask_select_time_cpu")
    encrypt_s    = _g(times, "encrypt_time_clock",  "cypher_time_clock")
    encrypt_cpu  = _g(times, "encrypt_time_cpu",    "cypher_time_process")
    serial_s     = _g(times, "serialize_time_clock")
    serial_cpu   = _g(times, "serialize_time_cpu")

    packs_sent = int(qnt_packing)
    n_packs    = (self.len_shared_data // size_pack + 1) if size_pack > 0 else 0
    rss_mb, peak_mb = _read_mem_mb()
    magg_renorm    = os.environ.get("MAGG_RENORM", "1")
    ckks_backend   = os.environ.get("CKKS_BACKEND", "tenseal")
    client_fraction = os.environ.get("CLIENT_FRACTION", "1.0")

    row = (
        f"{exp_name},{rnd},{self.cid},"
        f"{self.solution},{self.homomorphic_type},{self.model_name},{dataset},"
        f"{distribution},{self.dirichlet_alpha},"
        f"{loss},{acc},"
        f"{decrypt_s},{decrypt_cpu},"
        f"{train_s},{train_cpu},"
        f"{flatten_s},{flatten_cpu},"
        f"{quantize_s},{quantize_cpu},"
        f"{bpad_s},{bpad_cpu},"
        f"{pack_s},{pack_cpu},"
        f"{mask_s},{mask_cpu},"
        f"{encrypt_s},{encrypt_cpu},"
        f"{serial_s},{serial_cpu},"
        f"{model_size},{self.len_shared_data},"
        f"{n_packs},{packs_sent},{size_pack},"
        f"{sigma if sigma is not None else ''},"
        f"{self.poly_modulus_degree},{coeff_bits},{scale_b},{self.plain_modulus},"
        f"{rss_mb:.1f},{peak_mb:.1f},"
        f"{magg_renorm},{ckks_backend},{client_fraction}"
    )
    _append(f"logs/{self.log_folder}/train.csv", _TRAIN_HEADER, row)


_MASK_HEADER = (
    "experiment_name,round,cid,"
    "solution,model,dataset,distribution,"
    "n_packs,packs_sent,selected_packs"
)


def write_mask_logs(self, rnd: int, mask) -> None:
    """Write one selected-pack row to logs/{log_folder}/mask.csv (one per client/round).

    Normalizes both return formats from mask helpers:
      - ndarray float64 (sw, NVNL, rw)     → flatnonzero(arr > 0)
      - list of int8   (fedphe, srr, topkrr) → same after np.asarray
    selected_packs uses ';' as separator (same convention as coeff_mod_bits).
    """
    import numpy as np
    arr        = np.asarray(mask).ravel()
    selected   = np.flatnonzero(arr > 0)
    n_packs    = int(arr.size)
    packs_sent = int(selected.size)
    selected_str = ";".join(map(str, selected.tolist()))
    distribution = "niid" if self.niid else "iid"
    dataset      = self.dataset.replace("/", "_")
    exp_name     = getattr(self, "experiment_name", "")
    row = (
        f"{exp_name},{rnd},{self.cid},"
        f"{self.solution},{self.model_name},{dataset},{distribution},"
        f"{n_packs},{packs_sent},{selected_str}"
    )
    _append(f"logs/{self.log_folder}/mask.csv", _MASK_HEADER, row)


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
