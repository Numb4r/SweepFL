import os
import pickle
import time

import numpy as np

from .fedphe import get_robin_round_mask, get_topk_mask
from .set_parameters import he_packs_to_model
from client_logs import write_mask_logs, write_train_logs
from client_utils import cypher_packs, flat_parameters, get_size, packing


def fit_topkrr(self, parameters, config):
    """FedAvg fit with top-k selection for the first rr_start_round rounds,
    then round-robin for all subsequent rounds."""
    rr_start = self.rr_start_round
    current_round = config['round']

    # ── Decrypt ───────────────────────────────────────────────────────────────
    self.last_decrypt_clock = 0.0
    self.last_decrypt_cpu   = 0.0
    if len(config['he']) > 0 and self.homomorphic:
        if self.packing:
            he_packs_to_model(self, config)
    # he_packs_to_model sets self.last_decrypt_* timing only the model
    # manipulation (deserialize + decrypt + reshape), excluding disk writes.
    decrypt_time_clock = self.last_decrypt_clock
    decrypt_time_cpu   = self.last_decrypt_cpu

    # ── Train ─────────────────────────────────────────────────────────────────
    train_time_clock = time.time()
    train_time_cpu   = time.process_time()
    history = self.model.fit(
        self.x_train, self.y_train,
        epochs=1,
        batch_size=int(os.environ.get("BATCH_SIZE", "4")),
        verbose=0,
    )
    for v in self.model.trainable_variables:
        _ = v.numpy()
    train_time_cpu   = time.process_time() - train_time_cpu
    train_time_clock = time.time() - train_time_clock

    acc  = np.mean(history.history['accuracy'])
    loss = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters = []
    model_size    = 0
    chunk_size    = 0
    qnt_packing   = 0

    flatten_time_clock = pack_time_clock = mask_select_time_clock = 0.0
    flatten_time_cpu   = pack_time_cpu   = mask_select_time_cpu   = 0.0
    encrypt_time_clock = serialize_time_clock = 0.0
    encrypt_time_cpu   = serialize_time_cpu   = 0.0

    if self.packing:
        chunk_size = self.key_config["poly_modulus_degree"] // 2

        # ── Flatten ───────────────────────────────────────────────────────────
        flatten_time_clock = time.time()
        flatten_time_cpu   = time.process_time()
        flatted = flat_parameters(trained_parameters)
        _ckks_backend = self.key_config.get("ckks_backend") or os.environ.get("CKKS_BACKEND", "tenseal")
        _magg_renorm = int(os.environ.get("MAGG_RENORM", "1"))
        if self.only_sum or (_ckks_backend == "ckks_python_simplify" and _magg_renorm):
            # Pre-weight by local dataset size D_i so the server accumulates Σ D_i·w_i.
            # Per-pack renorm by M_agg[k]=Σ_{S_k} D_i (in set_parameters) then gives the
            # correct FedAvg weighted average per pack, even for sparse packs.
            flatted = np.array(flatted) * len(self.x_train)
        elif _ckks_backend == "ckks_python_simplify":
            # MAGG_RENORM=0 legacy: pre-division by K; phase-1 top-k is data-dependent (biased).
            flatted = np.array(flatted) / self.num_clients
        flatten_time_cpu   = time.process_time() - flatten_time_cpu
        flatten_time_clock = time.time() - flatten_time_clock

        # ── Pack ──────────────────────────────────────────────────────────────
        pack_time_clock = time.time()
        pack_time_cpu   = time.process_time()
        packed = packing(flatted, chunk_size=chunk_size)
        pack_time_cpu   = time.process_time() - pack_time_cpu
        pack_time_clock = time.time() - pack_time_clock

        # ── Mask: top-k phase → round-robin phase ─────────────────────────────
        mask_select_time_clock = time.time()
        mask_select_time_cpu   = time.process_time()
        if current_round < rr_start:
            mask = get_topk_mask(packed, self.percentage)
        else:
            # Offset so RR window starts at position 0 on the first RR round.
            rr_round = current_round - rr_start
            mask = get_robin_round_mask(
                round=rr_round, packs=packed,
                size_window=self.percentage, percentage=True,
            )
        mask_select_time_cpu   = time.process_time() - mask_select_time_cpu
        mask_select_time_clock = time.time() - mask_select_time_clock
        write_mask_logs(self, current_round, mask)

        # ── Encrypt ───────────────────────────────────────────────────────────
        encrypt_time_clock = time.time()
        encrypt_time_cpu   = time.process_time()
        cyphered = cypher_packs(self, packed, mask)
        encrypt_time_cpu   = time.process_time() - encrypt_time_cpu
        encrypt_time_clock = time.time() - encrypt_time_clock

        # ── Serialize ─────────────────────────────────────────────────────────
        serialize_time_clock = time.time()
        serialize_time_cpu   = time.process_time()
        he_parameters = pickle.dumps(cyphered)
        serialize_time_cpu   = time.process_time() - serialize_time_cpu
        serialize_time_clock = time.time() - serialize_time_clock

        model_size  = get_size(cyphered)
        qnt_packing = len(packed) * self.percentage

    times = {
        'decrypt_time_clock':      decrypt_time_clock,
        'decrypt_time_cpu':        decrypt_time_cpu,
        'train_time_clock':        train_time_clock,
        'train_time_cpu':          train_time_cpu,
        'flatten_time_clock':      flatten_time_clock,
        'flatten_time_cpu':        flatten_time_cpu,
        'pack_time_clock':         pack_time_clock,
        'pack_time_cpu':           pack_time_cpu,
        'mask_select_time_clock':  mask_select_time_clock,
        'mask_select_time_cpu':    mask_select_time_cpu,
        'encrypt_time_clock':      encrypt_time_clock,
        'encrypt_time_cpu':        encrypt_time_cpu,
        'serialize_time_clock':    serialize_time_clock,
        'serialize_time_cpu':      serialize_time_cpu,
    }
    write_train_logs(self, current_round, loss, acc, model_size, times, qnt_packing, chunk_size)

    fit_msg = self.create_fit_msg(train_time_cpu, acc, loss, model_size, he_parameters, mask)
    return fit_msg
