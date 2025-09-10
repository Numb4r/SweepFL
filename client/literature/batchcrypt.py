    
import time 
import pickle
import sys 
import numpy as np
from .set_parameters import he_parameters_to_model
from encryption.quantize import quantize, batch_padding
from client_utils import flat_parameters
from client_logs import write_train_logs

def fit_batchcrypt(self, parameters, config):
    
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time = time.time() - decypher_time

    train_time = time.time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time = time.time() - train_time

    acc     = np.mean(history.history['accuracy'])
    loss    = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    cypher_time         = time.time()
    
    
    if self.homomorphic and not self.packing:
        flatted_parameters = flat_parameters(trained_parameters) 
        flatted_parameters = np.array(flatted_parameters) #* len(self.x_train)
    
        quan_bits     = 32
        batch_size    = 50
        padding_bits  = int(np.ceil(np.log2(self.num_clients + 1)))
        elem_bits     = quan_bits + padding_bits
            
        quan_param    = quantize(flatted_parameters, quan_bits, self.num_clients)
        quan_param    = batch_padding(quan_param, self.context.key_length, elem_bits, batch_size=batch_size)

        he_parameters = self.context.encrypt(quan_param)
        he_parameters = pickle.dumps(he_parameters)
        model_size    = sys.getsizeof(he_parameters)
        topk_mask     =  '' 
        self.len_shared_data = len(flatted_parameters)
        
    cypher_time = time.time() - cypher_time
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
    
    return fit_msg
