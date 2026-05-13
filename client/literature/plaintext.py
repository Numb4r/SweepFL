import numpy as np
import time
import sys
import os
from client_utils import flat_parameters
from client_logs import write_train_logs

def fit_plaintext(self, parameters, config):

    self.model.set_weights(parameters)
    if self.cid == 0:
        folder = f'logs/{self.log_folder}/model'
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(
            f"{folder}/round_{config['round']}",
            *[w.astype(np.float16) for w in parameters],
        )

    train_time_clock = time.time()
    train_time_process = time.process_time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=int(os.environ.get("BATCH_SIZE", "4")), verbose=0)
    for v in self.model.trainable_variables:
        _ = v.numpy()
    train_time_process = time.process_time() - train_time_process
    train_time_clock = time.time() - train_time_clock

    acc     = np.mean(history.history['accuracy'])
    loss    = np.mean(history.history['loss'])

    flatten_time_clock = time.time()
    flatten_time_cpu   = time.process_time()
    trained_parameters = self.model.get_weights()
    flatted_parameters = flat_parameters(trained_parameters)
    flatten_time_cpu   = time.process_time() - flatten_time_cpu
    flatten_time_clock = time.time() - flatten_time_clock

    model_size = sys.getsizeof(flatted_parameters)
    he_parameters = []
    topk_mask = ''

    times = {
        'train_time_clock':   train_time_clock,
        'train_time_cpu':     train_time_process,
        'flatten_time_clock': flatten_time_clock,
        'flatten_time_cpu':   flatten_time_cpu,
    }
    write_train_logs(self, config['round'], loss, acc, model_size, times)

    fit_msg = self.create_fit_msg(train_time_clock, acc, loss, model_size, he_parameters, topk_mask, config['round'])
    return fit_msg
