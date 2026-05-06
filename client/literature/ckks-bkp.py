
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
    
    
     
    
        
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time = time.time() - decypher_time

    train_time = time.time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time = time.time() - train_time

    acc        = np.mean(history.history['accuracy'])
    loss       = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    cypher_time        = time.time()
    

    if self.homomorphic:
        flatted_parameters = flat_parameters(trained_parameters)
       
        # print(f'flatted_parameters max: {max(flatted_parameters)}')
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
            
        he_parameters      = ts.ckks_vector(self.context, flatted_parameters)
        
        he_parameters      = he_parameters.serialize()
        model_size         = sys.getsizeof(he_parameters)
        topk_mask          =  '' 
    
    
    
    cypher_time = time.time() - cypher_time
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time,sigma=self.sigma)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
    if self.cid == 0:
        folder = f'logs/models/model_client0_{self.solution}_{self.technique}_{self.poly_modulus_degree}_{self.coeff_mod_bit_sizes}_{self.global_scale}_{self.key_type}/'
        os.makedirs(f'{folder}', exist_ok=True)  

        with open(f"{folder}/round_{config['round']}.pkl", 'wb') as f:
            pk.dump(trained_parameters, f)
    
    return fit_msg



