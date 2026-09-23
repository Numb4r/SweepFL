
import time 

import sys 
import tenseal as ts 
import numpy as np
from .set_parameters import he_parameters_to_model
from  client_utils import flat_parameters, packing, cypher_packs, MAGG_D_REF
from  client_logs import write_train_logs
# TODO: packing for remove warning
# TODO: Search for sketch methods for CKKS error correction

def fit_ckks_error(self, parameters, config):
    print(f'tenseal: {self.context}')
    errors={
        'MAE': 0,
        'MSE': 0,
        'RMSE': 0,
        'Max Error': 0,
        'Relative Error Mean': 0
    }
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        error_mean = he_packs_to_model(self, config)
    decypher_time = time.time() - decypher_time

    train_time = time.time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time = time.time() - train_time

    acc        = np.mean(history.history['accuracy'])
    loss       = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    cypher_time        = time.time()
    

    if self.homomorphic and self.packing:
        chunk_size = self.key_config["poly_modulus_degree"]//2

        flatted_parameters = flat_parameters(trained_parameters)
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * (len(self.x_train) / MAGG_D_REF)

        packed_parameters = packing(flatted_parameters,chunk_size = chunk_size)
        mask = [1]*len(packed_parameters)  # Dummy mask 
        cypher_packs = cypher_packs(self, packed_parameters, mask)
        he_parameters      = pickle.dumps(cypher_packs)
        model_size         = sys.getsizeof(cypher_packs)
    
    cypher_time = time.time() - cypher_time
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time,errors=errors)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, mask)
    
    return fit_msg



