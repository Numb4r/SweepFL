import time 
import numpy as np
import pickle 
import math
from .set_parameters import he_packs_to_model
from  client_utils import  cypher_packs, flat_parameters, packing, get_size
from  client_logs import write_train_logs

def fit_fedphe(self, parameters, config):
   
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        if self.packing:
            he_packs_to_model(self, config)    
    decypher_time = time.time() - decypher_time

    train_time = time.time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time = time.time() - train_time

    acc     = np.mean(history.history['accuracy'])
    loss    = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    cypher_time        = time.time()
    
    if self.packing:
        flatted_parameters = flat_parameters(trained_parameters)     
        packed_parameters  = packing(flatted_parameters)
        
        if self.only_sum:
            packed_parameters = [np.array(pack) * len(self.x_train) for pack in packed_parameters] 

        if self.technique == "topk": 
            mask          = get_topk_mask(packed_parameters, self.percentage)
        elif self.technique == "robin_round" :    
            mask          = get_robin_round_mask(round=config['round'],packs=packed_parameters,size_window=self.percentage,percentage=True)

        elif self.technique == "slided_window":
            mask          = get_slide_window_mask(round=config['round'],packs=packed_parameters,size_window=self.percentage,stride=0.5,percentage=True)
        elif self.technique == "weight_random":
            if len(self.weights_packs ) == 0 :
                self.weights_packs = np.ones(len(packed_parameters))
            mask          = get_pondering_random_mask(packs=packed_parameters,k=self.percentage,weights_packs=self.weights_packs,percentage=True)
        else:
            return
        cyphered_packs     = cypher_packs(self, packed_parameters, mask)
        he_parameters      = pickle.dumps(cyphered_packs)
        model_size         = get_size(cyphered_packs)
    
    cypher_time = time.time() - cypher_time
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, mask)
    
    return fit_msg


def get_topk_mask(packs, percentage):
    topk = int(np.ceil(len(packs) * percentage))
    
    avg_list     = [np.average(np.abs(pack)) for pack in packs]    
    max_avg_list = np.sort(avg_list)[::-1][:topk]
    mask_list    = []
    
    for i in range(len(max_avg_list)):
        mask_list.append(avg_list.index(max_avg_list[i]))
    mask_list.sort()
    
    res_mask = [0  for i in range(len(packs)) ]
    for i in range(len(packs)):
        if i in mask_list:
            res_mask[i] = 1
            
    return res_mask
def get_slide_window_mask(round,packs,size_window=0.1,stride=3,percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
        stride = math.ceil(size_window*stride)
    
    array_z = np.zeros(len(packs))
    start = round*stride
    end = round*stride+size_window
    for i in range(start, end):
        array_z[i % len(array_z)] = 1
    return array_z

def get_robin_round_mask(round,packs,size_window,percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
    return get_slide_window_mask(round,packs,size_window,size_window)
def get_pondering_random_mask(packs,k,weights_packs,penalty=0.5,reward=0.1,percentage=False):

    n = len(packs)
    if percentage:
        k = math.ceil(k*n)
    probabilities = weights_packs / weights_packs.sum()
    mask = np.zeros(n)

    
    chosen = np.random.choice(range(n), size=k, replace=False, p=probabilities)
    
    for i in range(n):
        if i in chosen:
            weights_packs[i] *= penalty  
            mask[i] = 1
        else:
            weights_packs[i] += reward  # Aumenta o peso dos não escolhidos

    
    weights_packs = np.maximum(weights_packs, 0.01)
    return mask

