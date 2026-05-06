    
import time 
import pickle
import sys 
import numpy as np

from .set_parameters import he_parameters_to_model
from encryption.quantize_batchcrypt import quantize_batchcrypt, batch_padding_batchcrypt
from client_utils import flat_parameters
from client_logs import write_train_logs

def fit_batchcrypt(self, parameters, config):
    
    decypher_time_clock = time.time()
    decypher_time_process = time.process_time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time_process = time.process_time() - decypher_time_process
    decypher_time_clock = time.time() - decypher_time_clock
    
    
    
    train_time_clock = time.time()
    train_time_process = time.process_time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time_process = time.process_time() - train_time_process
    train_time_clock = time.time() - train_time_clock
    
    
    acc     = np.mean(history.history['accuracy'])
    loss    = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    cypher_time_clock        = time.time()
    cypher_time_process       = time.process_time()
    
    
    if self.homomorphic and not self.packing:
        flatted_parameters = flat_parameters(trained_parameters) 
        flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
    
        quan_bits     = 32
        batch_size    = 50
        padding_bits  = int(np.ceil(np.log2(self.num_clients + 1)))
        elem_bits     = quan_bits + padding_bits
            
        quan_param    = quantize_batchcrypt(flatted_parameters, quan_bits, self.num_clients)
        quan_param    = batch_padding_batchcrypt(quan_param, self.context.key_length, elem_bits, batch_size=batch_size)

        he_parameters = self.context.encrypt(quan_param)
        he_parameters = pickle.dumps(he_parameters)
        model_size    = sys.getsizeof(he_parameters)
        topk_mask     =  '' 
        self.len_shared_data = len(flatted_parameters)
        
    cypher_time_clock = time.time() - cypher_time_clock
    cypher_time_process = time.process_time() - cypher_time_process
    
    times={
        'train_time_clock': train_time_clock,
        'train_time_process': train_time_process,
        
        'cypher_time_clock': cypher_time_clock,
        'cypher_time_process': cypher_time_process,
        
        'decypher_time_clock': decypher_time_clock,
        'decypher_time_process': decypher_time_process
    }
    write_train_logs(self, config['round'], loss, acc, model_size, times,sigma=self.sigma)
    
    fit_msg = self.create_fit_msg(train_time_process, acc, loss, model_size, he_parameters, topk_mask)
    
    return fit_msg
