import pickle
import numpy as np 

import time
import tenseal as ts 
import sys
import os
from client_utils import flat_parameters
from client_logs import write_train_logs

def fit_plaintext(self, parameters, config):
    
    decypher_time = 0
    self.model.set_weights(parameters)
    

    train_time = time.time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time = time.time() - train_time

    acc     = np.mean(history.history['accuracy'])
    loss    = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    
    
    flatted_parameters  = flat_parameters(trained_parameters)
    # temp_buf            = pickle.dumps(flatted_parameters)
    model_size          = sys.getsizeof(flatted_parameters)
    topk_mask           =  '' 
    cypher_time         = 0
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
       
    return fit_msg

