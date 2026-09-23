import math
import sys
import numpy as np
from models import flat_parameters
from encryption.ckks_backend import get_backend

# Public pre-scale for the D_i-weighted pack protocol: clients encrypt w·D_i/MAGG_D_REF and
# set_parameters divides by M_agg/MAGG_D_REF, so FedAvg is unchanged. Without it the server
# sum Σ_i D_i·w reaches ~5e4·w and wraps the single-prime CKKS range q/(2Δ)=2^21 on
# MobileNetV2's BN moving_variance (docs/ckks_range_overflow.md). 2^10 keeps it exact.
MAGG_D_REF = 1
def get_size(data):
    total_size = 0
    
    for item in data:
        total_size += sys.getsizeof(item)
        
    return total_size
    
    
def packing(parameters, chunk_size=4096):
    parameters = list(parameters)
    total_parameters = len(parameters)
    number_packs     = total_parameters // chunk_size

    last_chunk      = (number_packs) * chunk_size
    last_chunk_size = parameters[last_chunk:]
    need_to_fill    = chunk_size - last_chunk
    parameters.extend([0] * need_to_fill)

    packs            = []
    
    for pack in range(number_packs + 1):
        
        start  = pack * chunk_size
        end    = start + chunk_size
        packet = np.array(parameters[start : end])
        
        
        packs.append(packet)

    return packs

def cypher_packs(self, packs, mask):
    backend = get_backend(self.key_config.get("ckks_backend"))
    cyphered_packs = []
    for idx, pack in enumerate(packs):
        if mask[idx] == 1:
            ct = backend.encrypt(self.context, pack)
            cyphered_packs.append(backend.serialize(ct))
    return cyphered_packs


def remove_padding(self, packed_parameters):
    total_parameters = len(flat_parameters(self.model.get_weights()))
    return packed_parameters[:total_parameters]

def flat_packs(packed_parameters):
    flat_params = []
    
    for pack in packed_parameters:
        flat_params.extend(pack)
        
    return flat_params

def decypher_packs(packed_parameters, context, backend=None):
    if backend is None:
        backend = get_backend()
    return [backend.decrypt(context, backend.deserialize(context, pack))
            for pack in packed_parameters]
def reshape_model(flatted_packs,a,sentinel=0):
    model =[]
    # print(a)
    
    if type(a) != int and type(a)!=float:
        # print(f'len:{len(a)}')
        for item in a:
            # print(f'item {item},  {type(item)},sentinel {sentinel}')
            
            m,sentinel=reshape_model(flatted_packs,item,sentinel)
            # print(sentinel)
            model.append(m)
    else:
    
        sentinel+=1 
        # print(f'a: {a}')
        return flatted_packs[sentinel-1],sentinel
    return model,sentinel

def packing_yphe(model_layers,chunk_size=-1):

    packs = []

    for idx,layer in enumerate(model_layers):
        weight = layer
        
        
        for w in weight:
            flat = flat_parameters(w)
            total_parameters = len(flat)
            current_chunk_size = chunk_size if chunk_size != -1 else total_parameters            
            number_packs = math.ceil(total_parameters / current_chunk_size )
            for pack in range(number_packs):
                start = pack * current_chunk_size
                end = start + current_chunk_size
                
            
                packs.append(flat[start:end])
            
            if len(packs[-1]) < chunk_size:
                packs[-1].extend([0] * (chunk_size - len(packs[-1])))
            
    return packs
def remove_padding_yphe(layer,target):
    
    total_parameters = len(target)
    return layer[:total_parameters]