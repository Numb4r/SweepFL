import pickle 
import numpy as np
import tenseal as ts
import os
import pickle as pk

from models import flat_parameters, reshape_parameters

from client_utils import  packing, decypher_packs, flat_packs, remove_padding
from encryption.quantize import  unquantize, unbatching_padding
from encryption.quantize_batchcrypt import unquantize_batchcrypt,unbatching_padding_batchcrypt
import encryption.quantize as q
def he_parameters_to_model(self, config):
        
        if self.homomorphic_type == 'Paillier':
            quan_bits     = 32
            batch_size    = 50
            padding_bits  = int(np.ceil(np.log2(self.num_clients + 1)))
            elem_bits     = quan_bits + padding_bits
            
            he_parameters = pickle.loads(config['he'])
            he_parameters = self.context.decrypt(he_parameters)
            print(f"[DEBUG] Batchcrypt Pós-Desempacotar: {list(he_parameters[:5])}")
            he_parameters = unbatching_padding_batchcrypt(he_parameters, elem_bits, batch_size)[:(int(np.prod((self.len_shared_data))))]
            print(f"[DEBUG] Batchcrypt Pós-Desquantizar min={np.min(he_parameters):.6f} max={np.max(he_parameters):.6f}")
            he_parameters = unquantize_batchcrypt(he_parameters, quan_bits, self.num_clients)
            if self.only_sum:
                he_parameters = np.array(he_parameters) / float(config['total_examples']) 
            
            
            print(f"[DEBUG] Batchcrypt Pós-Divisão min={np.min(he_parameters):.6f} max={np.max(he_parameters):.6f}")
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
                print("TRYING DECRIPTION")
                he_parameters        = ts.bfv_vector_from(self.context, config['he'])
         
                decrypted_parameters = he_parameters.decrypt()
                print(f"[DEBUG] decrypt min={min(decrypted_parameters):.2f} max={max(decrypted_parameters):.2f}")
                # logo após decrypt, antes do unquantize
                sample = list(decrypted_parameters[:5])
              
                print(q.__file__)
                print(f"[DEBUG] raw sample: {sample}")
                print(f"[DEBUG] raw sample: {list(decrypted_parameters[:10])}")
                print(f"[DEBUG] type: {type(decrypted_parameters[0])}")
                decrypted_parameters = unquantize(decrypted_parameters, 22, self.num_clients)
                print(f"[DEBUG] unquantize min={np.min(decrypted_parameters):.6f} max={np.max(decrypted_parameters):.6f}")
                print("UNQUANTIZE PARAMETERS")
                if self.only_sum:
                    decrypted_parameters  = np.array(decrypted_parameters) / float(config['total_examples'])
                print(f"[DEBUG] after_div min={np.min(decrypted_parameters):.6f} max={np.max(decrypted_parameters):.6f}")
                print(f"[DEBUG] total_examples={config['total_examples']}")
                reshaped_parameters  = reshape_parameters(self, decrypted_parameters)
                print("RESHAPED")
                self.model.set_weights(reshaped_parameters)
                print("ATT COMPLETA")
        # # debug   
        if self.cid == 0:
            # iid = "niid" if self.niid else 'iid'
            folder = f'{self.log_folder}/model'
            os.makedirs(f'{folder}', exist_ok=True)  
            with open(f"{folder}/round_{config['round']}.pkl", 'wb') as f:
                pk.dump(reshaped_parameters, f)
                
                
                
   
        
def he_packs_to_model(self, config):
    chunk_size = self.key_config["poly_modulus_degree"]//2    
    packed_parameters  = packing(flat_parameters(self.model.get_weights()),chunk_size = chunk_size)
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

    if self.cid == 0:
        folder = f'{self.log_folder}/model'
        os.makedirs(f'{folder}', exist_ok=True)  
        with open(f"{folder}/round_{config['round']}.pkl", 'wb') as f:
            pk.dump(reshaped_parameters, f)
