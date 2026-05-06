
import time 

import sys
import numpy as np
import tenseal as ts
from .set_parameters import he_parameters_to_model
from encryption.quantize import quantize
from client_utils import flat_parameters
from client_logs  import write_train_logs
import encryption.quantize as q
def fit_bfv(self, parameters, config):
    print(q.__file__)

    print(f'tenseal: {self.context}')
    decypher_time_clock = time.time()
    decypher_time_process = time.process_time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time_process = time.process_time() - decypher_time_process
    decypher_time_clock = time.time() - decypher_time_clock

    train_time_clock = time.time()
    train_time_process = time.process_time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    for v in self.model.trainable_variables:
        _ = v.numpy()
    train_time_process = time.process_time() - train_time_process
    train_time_clock = time.time() - train_time_clock

    acc        = np.mean(history.history['accuracy'])
    loss       = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    
    
    cypher_time_clock        = time.time()
    cypher_time_process       = time.process_time()
    print("BEGIN ENCRIPTION")
    if self.homomorphic:
        flatted_parameters = flat_parameters(trained_parameters) 
        
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
            
        print(f"[DEBUG] pre-quantize min={np.min(flatted_parameters):.2f} max={np.max(flatted_parameters):.2f}")
        print(f"[DEBUG] num_examples={len(self.x_train)}")
        print(f"[DEBUG] quantized sample (before true2two): {list(flatted_parameters[:5])}")
        quantized_parameters = quantize(flatted_parameters, 22, self.num_clients)
        print(f"[DEBUG] quantized sample (after true2two): {list(quantized_parameters[:5])}")
        print(f"[DEBUG] pos-quantize min={np.min(quantized_parameters):.2f} max={np.max(quantized_parameters):.2f}")
        


        he_parameters        = ts.bfv_vector(self.context, quantized_parameters)
        he_parameters        = he_parameters.serialize()
        model_size           = sys.getsizeof(he_parameters)
        topk_mask            =  '' 
    print("END ENCRIPTION")
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
    write_train_logs(self, config['round'], loss, acc, model_size, times)
    
    fit_msg = self.create_fit_msg(train_time_process, acc, loss, model_size, he_parameters)
    print("FIT MESSAGE")
    return fit_msg
