import pickle
import numpy as np
import os
import time
import pickle as pk
from encryption.ckks_backend import get_backend

from models import flat_parameters, reshape_parameters

from client_utils import  packing, decypher_packs, flat_packs, remove_padding
from encryption.quantize import  unquantize, unbatching_padding
from encryption.quantize_batchcrypt import unquantize_batchcrypt,unbatching_padding_batchcrypt
import encryption.quantize as q
def he_parameters_to_model(self, config):
        t0 = time.time()
        c0 = time.process_time()

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
                backend              = get_backend(self.key_config.get("ckks_backend"))
                he_parameters        = backend.deserialize(self.context, config['he'])
                decrypted_parameters = backend.decrypt(self.context, he_parameters)
                if self.only_sum:
                    decrypted_parameters = np.array(decrypted_parameters) / float(config['total_examples'])
                reshaped_parameters  = reshape_parameters(self, decrypted_parameters)
                self.model.set_weights(reshaped_parameters)

            else:
                import tenseal as ts
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

        # decrypt timing covers model manipulation only (deserialize + decrypt +
        # reshape/set_weights). The disk writes below are logging, not part of
        # the decryption cost, so they are intentionally left out of the window.
        self.last_decrypt_clock = time.time() - t0
        self.last_decrypt_cpu   = time.process_time() - c0

        # # debug
        if self.cid == 0:
            folder = f'logs/{self.log_folder}/model'
            os.makedirs(folder, exist_ok=True)
            np.savez_compressed(
                f"{folder}/round_{config['round']}",
                *[w.astype(np.float16) for w in reshaped_parameters],
            )
        ckpt_folder = f'logs/{self.log_folder}/checkpoint'
        os.makedirs(ckpt_folder, exist_ok=True)
        np.savez_compressed(
            f'{ckpt_folder}/client_{self.cid}',
            *[w.astype(np.float16) for w in reshaped_parameters],
        )
                
                
                
   
        
def he_packs_to_model(self, config):
    t0 = time.time()
    c0 = time.process_time()
    chunk_size = self.key_config["poly_modulus_degree"]//2
    packed_parameters  = packing(flat_parameters(self.model.get_weights()), chunk_size=chunk_size)
    cyphered_packs     = pickle.loads(config['he'])
    backend            = get_backend(self.key_config.get("ckks_backend"))
    decyphered_pack    = decypher_packs(cyphered_packs, self.context, backend=backend)
    aggredated_mask    = pickle.loads(config['mask'])


    # MAGG_RENORM=1 (default, corrected): divide each active pack by its per-pack
    # contributor mass M_agg[k] = Σ_{S_k} D_i, giving the correct FedAvg weighted
    # average regardless of how many clients contributed to pack k.
    # MAGG_RENORM=0 (legacy): skip renorm — biased for fedphe/rw (sparse, data-dependent
    # masks), no-op for nvnl/sw/srr (deterministic masks → all clients contribute equally).
    _magg_renorm = int(os.environ.get("MAGG_RENORM", "1"))
    for idx_mask, m in enumerate(aggredated_mask):
        if m > 0:
            if self.only_sum or _magg_renorm:
                packed_parameters[idx_mask] = np.array(decyphered_pack.pop(0)) / m
            else:
                packed_parameters[idx_mask] = decyphered_pack.pop(0)
    
    flatted_packs       = flat_packs(packed_parameters)
    flatted_packs       = remove_padding(self, flatted_packs)
    reshaped_parameters = reshape_parameters(self, flatted_packs)
    self.model.set_weights(reshaped_parameters)

    # decrypt timing covers model manipulation only (deserialize + decrypt +
    # reshape/set_weights). The disk writes below are logging, not part of the
    # decryption cost, so they are intentionally left out of the window.
    self.last_decrypt_clock = time.time() - t0
    self.last_decrypt_cpu   = time.process_time() - c0

    if self.cid == 0:
        folder = f'logs/{self.log_folder}/model'
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(
            f"{folder}/round_{config['round']}",
            *[w.astype(np.float16) for w in reshaped_parameters],
        )
    ckpt_folder = f'logs/{self.log_folder}/checkpoint'
    os.makedirs(ckpt_folder, exist_ok=True)
    np.savez_compressed(
        f'{ckpt_folder}/client_{self.cid}',
        *[w.astype(np.float16) for w in reshaped_parameters],
    )
