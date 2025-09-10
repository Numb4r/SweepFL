import pickle 
import numpy as np
import tenseal as ts

from models import flat_parameters, reshape_parameters

from client_utils import  packing, decypher_packs, flat_packs, remove_padding
from encryption.quantize import  unquantize, unbatching_padding

def he_parameters_to_model(self, config):
        
        if self.homomorphic_type == 'Paillier':
            quan_bits     = 32
            batch_size    = 50
            padding_bits  = int(np.ceil(np.log2(self.num_clients + 1)))
            elem_bits     = quan_bits + padding_bits
            
            he_parameters = pickle.loads(config['he'])
            he_parameters = self.context.decrypt(he_parameters)
            he_parameters = unbatching_padding(he_parameters, elem_bits, batch_size)[:(int(np.prod((self.len_shared_data))))]
            he_parameters = unquantize(he_parameters, quan_bits, self.num_clients)
            if self.only_sum:
                he_parameters = np.array(he_parameters) / float(config['total_examples']) 
            
            
            
            reshaped_parameters = reshape_parameters(self, he_parameters)
            self.model.set_weights(reshaped_parameters)
        
        else:
            if 'CKKS' == self.homomorphic_type:
                he_parameters        = ts.ckks_vector_from(self.context, config['he'])
                local_parameters     = self.model.get_weights()
                decrypted_parameters = he_parameters.decrypt()
                if self.only_sum:
                    decrypted_parameters  = np.array(decrypted_parameters) / float(config['total_examples'])
                    
                reshaped_parameters  = reshape_parameters(self, decrypted_parameters)
                self.model.set_weights(reshaped_parameters)
                
            else:
                he_parameters        = ts.bfv_vector_from(self.context, config['he'])
                local_parameters     = self.model.get_weights()
                decrypted_parameters = he_parameters.decrypt()
                decrypted_parameters = unquantize(decrypted_parameters, 16, self.num_clients)
                if self.only_sum:
                    decrypted_parameters  = np.array(decrypted_parameters) / float(config['total_examples'])
                
                reshaped_parameters  = reshape_parameters(self, decrypted_parameters)
                self.model.set_weights(reshaped_parameters)
            
def he_packs_to_model(self, config):
    packed_parameters  = packing(flat_parameters(self.model.get_weights()))
    cyphered_packs     = pickle.loads(config['he'])
    decyphered_pack    = decypher_packs(cyphered_packs, self.context)
    aggredated_mask    = pickle.loads(config['mask'])
    
    for idx_mask, m in enumerate(aggredated_mask):
        if m > 0:
            if self.only_sum:
                packed_parameters[idx_mask] = np.array(decyphered_pack.pop(0)) / m
            else:
                packed_parameters[idx_mask] = decyphered_pack.pop(0)
    
    flatted_packs       = flat_packs(packed_parameters)
    flatted_packs       = remove_padding(self, flatted_packs)
    reshaped_parameters = reshape_parameters(self, flatted_packs)
    self.model.set_weights(reshaped_parameters)