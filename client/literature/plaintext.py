import pickle
import numpy as np 

import time
import tenseal as ts 
import sys
import os
from client_utils import flat_parameters
from client_logs import write_train_logs
import pickle as pk

def fit_plaintext(self, parameters, config):
    
    decypher_time = 0
    self.model.set_weights(parameters)
    folder = f'logs/{self.log_folder}/model'
    with open(f"{folder}/round_{config['round']}.pkl", 'wb') as f:
        pk.dump(parameters, f)
        
        
    train_time_clock = time.time()
    train_time_process = time.process_time()
    
    history    = self.model.fit(self.x_train, self.y_train, epochs=1,batch_size=4)
    # Sincronizando relogio
    for v in self.model.trainable_variables:
        _ = v.numpy()
    
    
    
    
    train_time_process = time.process_time() - train_time_process
    train_time_clock = time.time() - train_time_clock

    acc     = np.mean(history.history['accuracy'])
    loss    = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    
    
    flatted_parameters  = flat_parameters(trained_parameters)
    # temp_buf            = pickle.dumps(flatted_parameters)
    model_size          = sys.getsizeof(flatted_parameters)
    topk_mask           =  '' 
    cypher_time         = 0
    times = {
        'train_time_clock': train_time_clock,
        'train_time_process': train_time_process,
        
        'cypher_time_clock': cypher_time,
        'cypher_time_process': cypher_time,
        
        'decypher_time_clock': decypher_time,
        'decypher_time_process': decypher_time
      
    }
    write_train_logs(self, config['round'], loss, acc, model_size, times)
    
    fit_msg = self.create_fit_msg(train_time_clock, acc, loss, model_size, he_parameters, topk_mask,config['round'])
    # if self.cid == 0:
    #     folder = f'logs/models/model_client0_{self.solution}_{self.technique}_{self.poly_modulus_degree}_{self.coeff_mod_bit_sizes}_{self.global_scale}_{self.key_type}/'
    #     os.makedirs(f'{folder}', exist_ok=True)  

    #     with open(f"{folder}/round_{config['round']}.pkl", 'wb') as f:
    #         pk.dump(trained_parameters, f)

    return fit_msg

