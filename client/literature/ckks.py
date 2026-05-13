import time
import sys
import tenseal as ts
import numpy as np
import math
import os
from .set_parameters import he_parameters_to_model
from client_utils import flat_parameters
from client_logs import write_train_logs


def estimate_ckks_sigma(context, v, n_clients, dim=1000, trials=50):
    v = np.array(v)
    errs = []
    for _ in range(trials):
        enc = ts.ckks_vector(context, v)
        dec = np.array(enc.decrypt())
        errs.append(dec - v)
    sigma_ckks = np.std(np.concatenate(errs))
    return math.sqrt(n_clients) * sigma_ckks


def fit_ckks(self, parameters, config):

    decrypt_time_clock = time.time()
    decrypt_time_cpu   = time.process_time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decrypt_time_cpu   = time.process_time() - decrypt_time_cpu
    decrypt_time_clock = time.time() - decrypt_time_clock

    if config["round"] % 5 == 0:
        v = flat_parameters(self.model.get_weights())
        self.sigma = estimate_ckks_sigma(self.context, v, self.num_clients)

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
    topk_mask          = ''

    flatten_time_clock  = 0.0
    flatten_time_cpu    = 0.0
    encrypt_time_clock  = 0.0
    encrypt_time_cpu    = 0.0
    serialize_time_clock = 0.0
    serialize_time_cpu  = 0.0

    if self.homomorphic:
        flatten_time_clock = time.time()
        flatten_time_cpu   = time.process_time()
        flatted_parameters = flat_parameters(trained_parameters)
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
        flatten_time_cpu   = time.process_time() - flatten_time_cpu
        flatten_time_clock = time.time() - flatten_time_clock

        encrypt_time_clock = time.time()
        encrypt_time_cpu   = time.process_time()
        he_parameters = ts.ckks_vector(self.context, flatted_parameters)
        encrypt_time_cpu   = time.process_time() - encrypt_time_cpu
        encrypt_time_clock = time.time() - encrypt_time_clock

        serialize_time_clock = time.time()
        serialize_time_cpu   = time.process_time()
        he_parameters = he_parameters.serialize()
        serialize_time_cpu   = time.process_time() - serialize_time_cpu
        serialize_time_clock = time.time() - serialize_time_clock

        model_size = sys.getsizeof(he_parameters)

    times = {
        'decrypt_time_clock':   decrypt_time_clock,
        'decrypt_time_cpu':     decrypt_time_cpu,
        'train_time_clock':     train_time_clock,
        'train_time_cpu':       train_time_cpu,
        'flatten_time_clock':   flatten_time_clock,
        'flatten_time_cpu':     flatten_time_cpu,
        'encrypt_time_clock':   encrypt_time_clock,
        'encrypt_time_cpu':     encrypt_time_cpu,
        'serialize_time_clock': serialize_time_clock,
        'serialize_time_cpu':   serialize_time_cpu,
    }
    write_train_logs(self, config['round'], loss, acc, model_size, times, sigma=self.sigma)

    fit_msg = self.create_fit_msg(train_time_cpu, acc, loss, model_size, he_parameters, topk_mask)
    return fit_msg
