import time
import pickle
import sys
import numpy as np
from .set_parameters import he_parameters_to_model
from encryption.quantize_batchcrypt import quantize_batchcrypt, batch_padding_batchcrypt
from client_utils import flat_parameters
from client_logs import write_train_logs


def fit_batchcrypt(self, parameters, config):

    decrypt_time_clock = time.time()
    decrypt_time_cpu   = time.process_time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decrypt_time_cpu   = time.process_time() - decrypt_time_cpu
    decrypt_time_clock = time.time() - decrypt_time_clock

    train_time_clock = time.time()
    train_time_cpu   = time.process_time()
    history = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time_cpu   = time.process_time() - train_time_cpu
    train_time_clock = time.time() - train_time_clock

    acc  = np.mean(history.history['accuracy'])
    loss = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    model_size         = 0
    topk_mask          = ''

    flatten_time_clock      = 0.0
    flatten_time_cpu        = 0.0
    quantize_time_clock     = 0.0
    quantize_time_cpu       = 0.0
    batch_padding_time_clock = 0.0
    batch_padding_time_cpu  = 0.0
    encrypt_time_clock      = 0.0
    encrypt_time_cpu        = 0.0
    serialize_time_clock    = 0.0
    serialize_time_cpu      = 0.0

    if self.homomorphic and not self.packing:
        flatten_time_clock = time.time()
        flatten_time_cpu   = time.process_time()
        flatted_parameters = flat_parameters(trained_parameters)
        flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
        flatten_time_cpu   = time.process_time() - flatten_time_cpu
        flatten_time_clock = time.time() - flatten_time_clock

        quan_bits    = 32
        batch_size   = 50
        padding_bits = int(np.ceil(np.log2(self.num_clients + 1)))
        elem_bits    = quan_bits + padding_bits

        quantize_time_clock = time.time()
        quantize_time_cpu   = time.process_time()
        quan_param = quantize_batchcrypt(flatted_parameters, quan_bits, self.num_clients)
        quantize_time_cpu   = time.process_time() - quantize_time_cpu
        quantize_time_clock = time.time() - quantize_time_clock

        batch_padding_time_clock = time.time()
        batch_padding_time_cpu   = time.process_time()
        quan_param = batch_padding_batchcrypt(quan_param, self.context.key_length, elem_bits, batch_size=batch_size)
        batch_padding_time_cpu   = time.process_time() - batch_padding_time_cpu
        batch_padding_time_clock = time.time() - batch_padding_time_clock

        encrypt_time_clock = time.time()
        encrypt_time_cpu   = time.process_time()
        he_parameters = self.context.encrypt(quan_param)
        encrypt_time_cpu   = time.process_time() - encrypt_time_cpu
        encrypt_time_clock = time.time() - encrypt_time_clock

        serialize_time_clock = time.time()
        serialize_time_cpu   = time.process_time()
        he_parameters = pickle.dumps(he_parameters)
        serialize_time_cpu   = time.process_time() - serialize_time_cpu
        serialize_time_clock = time.time() - serialize_time_clock

        model_size = sys.getsizeof(he_parameters)
        self.len_shared_data = len(flatted_parameters)

    times = {
        'decrypt_time_clock':        decrypt_time_clock,
        'decrypt_time_cpu':          decrypt_time_cpu,
        'train_time_clock':          train_time_clock,
        'train_time_cpu':            train_time_cpu,
        'flatten_time_clock':        flatten_time_clock,
        'flatten_time_cpu':          flatten_time_cpu,
        'quantize_time_clock':       quantize_time_clock,
        'quantize_time_cpu':         quantize_time_cpu,
        'batch_padding_time_clock':  batch_padding_time_clock,
        'batch_padding_time_cpu':    batch_padding_time_cpu,
        'encrypt_time_clock':        encrypt_time_clock,
        'encrypt_time_cpu':          encrypt_time_cpu,
        'serialize_time_clock':      serialize_time_clock,
        'serialize_time_cpu':        serialize_time_cpu,
    }
    write_train_logs(self, config['round'], loss, acc, model_size, times, sigma=self.sigma)

    fit_msg = self.create_fit_msg(train_time_cpu, acc, loss, model_size, he_parameters, topk_mask)
    return fit_msg
