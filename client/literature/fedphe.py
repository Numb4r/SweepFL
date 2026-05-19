import os
import time
import numpy as np
import pickle
import math
from .set_parameters import he_packs_to_model
from client_utils import cypher_packs, flat_parameters, packing, get_size
from client_logs import write_train_logs


def fit_fedphe(self, parameters, config):

    decrypt_time_clock = time.time()
    decrypt_time_cpu   = time.process_time()
    if len(config['he']) > 0 and self.homomorphic:
        if self.packing:
            he_packs_to_model(self, config)
    decrypt_time_cpu   = time.process_time() - decrypt_time_cpu
    decrypt_time_clock = time.time() - decrypt_time_clock

    train_time_clock = time.time()
    train_time_cpu   = time.process_time()
    history = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=int(os.environ.get("BATCH_SIZE", "4")), verbose=0)
    for v in self.model.trainable_variables:
        _ = v.numpy()
    train_time_cpu   = time.process_time() - train_time_cpu
    train_time_clock = time.time() - train_time_clock

    acc  = np.mean(history.history['accuracy'])
    loss = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    model_size         = 0
    chunk_size         = 0

    flatten_time_clock    = 0.0
    flatten_time_cpu      = 0.0
    pack_time_clock       = 0.0
    pack_time_cpu         = 0.0
    mask_select_time_clock = 0.0
    mask_select_time_cpu  = 0.0
    encrypt_time_clock    = 0.0
    encrypt_time_cpu      = 0.0
    serialize_time_clock  = 0.0
    serialize_time_cpu    = 0.0
    qnt_packing           = 0

    if self.packing:
        chunk_size = self.key_config["poly_modulus_degree"] // 2

        flatten_time_clock = time.time()
        flatten_time_cpu   = time.process_time()
        flatted_parameters = flat_parameters(trained_parameters)
        _ckks_backend = self.key_config.get("ckks_backend") or os.environ.get("CKKS_BACKEND", "tenseal")
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
        elif _ckks_backend == "ckks_python_simplify":
            # Pre-division protocol: server sums ciphertexts only; sum(w_i/K) = average(w_i)
            flatted_parameters = np.array(flatted_parameters) / self.num_clients
        flatten_time_cpu   = time.process_time() - flatten_time_cpu
        flatten_time_clock = time.time() - flatten_time_clock

        pack_time_clock = time.time()
        pack_time_cpu   = time.process_time()
        packed_parameters = packing(flatted_parameters, chunk_size=chunk_size)
        pack_time_cpu   = time.process_time() - pack_time_cpu
        pack_time_clock = time.time() - pack_time_clock

        mask_select_time_clock = time.time()
        mask_select_time_cpu   = time.process_time()
        mask = get_topk_mask(packed_parameters, self.percentage)
        mask_select_time_cpu   = time.process_time() - mask_select_time_cpu
        mask_select_time_clock = time.time() - mask_select_time_clock

        encrypt_time_clock = time.time()
        encrypt_time_cpu   = time.process_time()
        cyphered_packs = cypher_packs(self, packed_parameters, mask)
        encrypt_time_cpu   = time.process_time() - encrypt_time_cpu
        encrypt_time_clock = time.time() - encrypt_time_clock

        serialize_time_clock = time.time()
        serialize_time_cpu   = time.process_time()
        he_parameters = pickle.dumps(cyphered_packs)
        serialize_time_cpu   = time.process_time() - serialize_time_cpu
        serialize_time_clock = time.time() - serialize_time_clock

        model_size  = get_size(cyphered_packs)
        qnt_packing = len(packed_parameters) * self.percentage

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
    write_train_logs(self, config['round'], loss, acc, model_size, times, qnt_packing, chunk_size)

    fit_msg = self.create_fit_msg(train_time_cpu, acc, loss, model_size, he_parameters, mask)
    return fit_msg


def get_topk_mask(packs, percentage):
    n = len(packs)
    if n == 0:
        return []

    percentage = float(percentage)
    percentage = max(0.0, min(1.0, percentage))

    topk = int(math.ceil(n * percentage))
    if topk <= 0:
        return [0] * n

    avg = np.array([np.mean(np.abs(pack)) for pack in packs], dtype=np.float64)
    avg = np.nan_to_num(avg, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    idx = np.argsort(avg)[::-1][:topk]
    idx.sort()

    mask = np.zeros(n, dtype=np.int8)
    mask[idx] = 1
    return mask.tolist()


def get_slide_window_mask(round, packs, size_window=0.1, stride=3, percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
        stride = math.ceil(size_window * stride)

    array_z = np.zeros(len(packs))
    start = round * stride
    end   = round * stride + size_window
    for i in range(start, end):
        array_z[i % len(array_z)] = 1
    return array_z


def get_robin_round_mask(round, packs, size_window, percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
    return get_slide_window_mask(round, packs, size_window, size_window)


def get_spread_rr_mask(round, packs, size_window, percentage=False):
    n = len(packs)
    if percentage:
        k = math.ceil(n * size_window)
    else:
        k = int(size_window)
    if n == 0 or k == 0:
        return [0] * n
    if k >= n:
        return [1] * n
    stride = max(1, n // k)
    offset = round % stride
    mask = np.zeros(n, dtype=np.int8)
    for i in range(k):
        idx = (offset + i * stride) % n
        mask[idx] = 1
    return mask.tolist()


def get_pondering_random_mask(packs, k, weights_packs, penalty=0.5, reward=0.1, percentage=False):
    n = len(packs)
    if percentage:
        k = math.ceil(k * n)
    probabilities = weights_packs / weights_packs.sum()
    mask = np.zeros(n)

    chosen = np.random.choice(range(n), size=k, replace=False, p=probabilities)

    for i in range(n):
        if i in chosen:
            weights_packs[i] *= penalty
            mask[i] = 1
        else:
            weights_packs[i] += reward

    weights_packs = np.maximum(weights_packs, 0.01)
    return mask
