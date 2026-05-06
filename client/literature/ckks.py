
import time 

import sys 
import tenseal as ts 
import numpy as np
from .set_parameters import he_parameters_to_model
from  client_utils import flat_parameters
from  client_logs import write_train_logs
import math
import os
import pickle as pk
def estimate_ckks_sigma(context,v,n_clients, dim=1000, trials=50):
    # v = np.random.randn(dim)
    v = np.array(v)
    errs = []
    for _ in range(trials):
        enc = ts.ckks_vector(context, v)
        dec = enc.decrypt()
        dec = np.array(dec)
        errs.append(dec - v)
    sigma_ckks = np.std(np.concatenate(errs))

    # σ total após soma de n clientes
    sigma_total = math.sqrt(n_clients) * sigma_ckks
    return sigma_total
def fit_ckks(self, parameters, config):
    total_time = time.time()
    
        
    decypher_time_clock = time.time()
    decypher_time_process = time.process_time()
    
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time_process = time.process_time() - decypher_time_process
    decypher_time_clock = time.time() - decypher_time_clock
    if config["round"]%5==0:
        v = flat_parameters(self.model.get_weights())
        self.sigma = estimate_ckks_sigma(self.context,v,self.num_clients)    
        
    

    train_time_clock = time.time()
    train_time_process = time.process_time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1,batch_size=4)
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
    

    if self.homomorphic:
        flatted_parameters = flat_parameters(trained_parameters)
       
        # print(f'flatted_parameters max: {max(flatted_parameters)}')
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
            
        he_parameters      = ts.ckks_vector(self.context, flatted_parameters)
        
        he_parameters      = he_parameters.serialize()
        model_size         = sys.getsizeof(he_parameters)
        topk_mask          =  '' 
    
    
    
    cypher_time_clock = time.time() - cypher_time_clock
    cypher_time_process = time.process_time() - cypher_time_process
    
    total_time = time.time() - total_time
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



