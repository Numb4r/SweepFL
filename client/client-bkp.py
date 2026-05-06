
import os
import pickle
import time
import flwr as fl
import tensorflow as tf
import tenseal as ts

from dataset_utils import ManageDatasets, load_data_flowerdataset
from pathlib import Path
import random
import numpy as np
from models import create_dnn, create_lenet5,  flat_parameters,create_bitnet
from client_logs import write_evaluate_logs
from literature.set_parameters import he_parameters_to_model, he_packs_to_model
from literature import fit_ckks, fit_bfv, fit_batchcrypt, fit_fedphe, fit_plaintext, fit_bitnetflip
from  client_utils import flat_parameters
import math


from typing import Optional,Dict,Any
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

class HEClient(fl.client.NumPyClient):
    def __init__(self, cid:int, niid:bool, dataset:str, num_clients:int, 
                 dirichlet_alpha:float, solution:str,percentage:float,technique:str,
                 key_config: Dict[str,Any]):
        
        self.cid              = int(cid)
        self.dataset          = dataset
        self.niid             = niid
        self.num_clients      = num_clients
        self.dirichlet_alpha  = dirichlet_alpha
        self.last_parameters  = None
        self.dataset_manager  = ManageDatasets(self.cid)
        self.solution         = str(solution).lower()
        self.weights_packs    = []
        self.technique        = technique
        self.percentage       = percentage
        self.dirichlet_alpha = dirichlet_alpha
        self.key_config = key_config
        self.sigma = None
        # os.remove(f'logs/client_{self.cid}_train.csv') if os.path.exists(f'logs/client_{self.cid}_train.csv') else None
        os.remove(f'logs/debug/ckks_client_*.txt') if os.path.exists(f'logs/debug/ckks_client_*.txt') else None
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed) # type: ignore
        tf.config.experimental.enable_op_determinism() # type: ignore
        tf.config.threading.set_intra_op_parallelism_threads(1) # type: ignore
        tf.config.threading.set_inter_op_parallelism_threads(1) # type: ignore
        self.poly_modulus_degree = self.key_config.get("poly_modulus_degree")
        # self.plain_modulus = self.key_config.get("plain_modulus")
        self.plain_modulus = None
        self.coeff_mod_bit_sizes = self.key_config.get("coeff_mod_bit_sizes")
        self.global_scale = self.key_config.get("global_scale")
        self.key_type = self.key_config.get("key_type")

        os.environ["TF_DETERMINISTIC_OPS"] = "1"

        if dataset == 'MNIST' or dataset == 'CIFAR10' or dataset == "FASHION_MNIST":
            self.x_train, self.y_train, self.x_test, self.y_test = load_data_flowerdataset(self)
            print(f"""self.x_train: {self.x_train.shape}\n\\
                self.y_train: {self.y_train.shape}\n\\
                self.x_test: {self.x_test.shape}\n\\
                self.y_test: {self.y_test.shape}\n""")
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = self.load_har(dataset) #self.load_data()
        self.model = create_bitnet(self.x_train.shape[1:],10)
        self.model_name="socas"
        # if dataset == 'FASHION_MNIST':
        #     self.model = create_lenet5(self.x_train.shape,10)
        #     self.model_name = "lenet5"
        # elif dataset == "MNIST":
        #     self.model = create_dnn(self.x_train.shape,10,seed)
        #     self.model_name = "DNN"
        # else:
        #     return
        
        self.config_solution()
        if self.homomorphic:
        
            self.context = self.get_client_context()
            v = flat_parameters(self.model.get_weights())
            self.sigma = estimate_ckks_sigma(self.context,v,self.num_clients)
            print(f"SIGMA: {self.sigma}")
        init_weights = f"logs/models/init_weights/init_weights_{cid}.weights.h5"
        os.makedirs(f"logs/models/init_weights/",exist_ok=True)
        if os.path.exists(init_weights):
            self.model.load_weights(init_weights)
            print("Load weights")
        elif self.solution == "plaintext":
            print("Init weights")
            self.model.save_weights(init_weights)
        else:
            raise Exception("Cannot start, generate plaintext first")
        
        
    def get_client_context(self):
        if self.homomorphic_type == 'Paillier':
            with open(f'context/paillier.pkl', 'rb') as file:
                context = pickle.load(file)    
        else:
            if 'CKKS' == self.homomorphic_type:
                with open(f'../context/ckks_{self.poly_modulus_degree}_{self.plain_modulus}_{self.coeff_mod_bit_sizes}_{self.global_scale}_{self.key_type}_secret.pkl', 'rb') as file:
                    secret = pickle.load(file) 
                    context = ts.context_from(secret["context"])
            else:
                with open(f'../context/bfv_{self.poly_modulus_degree}_{self.plain_modulus}_{self.coeff_mod_bit_sizes}_{self.global_scale}_{self.key_type}_secret.pkl', 'rb') as file:
                    secret = pickle.load(file) 
                    context = ts.context_from(secret["context"])
                
        return context

    def get_parameters(self, config):
        parameters           = self.model.get_weights()
        return parameters
    
    def load_har(self, dataset):
        if dataset == 'UCIHAR':
            return self.dataset_manager.load_UCIHAR()
        if dataset == 'ExtraSensory':
            return self.dataset_manager.load_ExtraSensory()
        if dataset == 'MotionSense':
            return self.dataset_manager.load_MotionSense()
 
    def config_solution(self):
        
        if str(self.solution).lower() == 'ckks':
            self.homomorphic      = True
            self.packing          = False
            self.only_sum         = False
            self.homomorphic_type = 'CKKS'
            
        elif str(self.solution).lower() == 'bfv':
            self.homomorphic      = True
            self.packing          = False
            self.only_sum         = True
            self.homomorphic_type = 'BFV'
        
        elif str(self.solution).lower() == 'batchcrypt':
            self.homomorphic      = True
            self.homomorphic_type = 'Paillier'
            self.only_sum         = True
            self.packing          = False
            
        elif str(self.solution).lower() == 'fedphe':
            self.homomorphic      = True
            self.homomorphic_type = 'CKKS'
            self.only_sum         = False
            self.packing          = True
        elif str(self.solution).lower() == 'plaintext':
            self.homomorphic      = False
            self.only_sum         = False
            self.packing          = False
            self.homomorphic_type = 'None'
        elif str(self.solution).lower() == "socas":
            self.homomorphic      = True
            self.packing          = False
            self.only_sum         = False
            self.homomorphic_type = 'CKKS'
            
            

    def fit(self, parameters, config):
        
        if str(self.solution).lower() == 'ckks':
            fit_msg = fit_ckks(self, parameters, config)
        elif str(self.solution).lower() == 'bfv':
            fit_msg = fit_bfv(self, parameters, config)
        elif str(self.solution).lower() == 'batchcrypt':
            fit_msg = fit_batchcrypt(self, parameters, config)
        elif str(self.solution).lower() == 'fedphe':
            fit_msg = fit_fedphe(self, parameters, config)        
        elif str(self.solution).lower() == 'plaintext':
            fit_msg = fit_plaintext(self, parameters, config)
        elif str(self.solution).lower() == "socas":
            fit_msg = fit_bitnetflip(self,parameters,config)
        return self.model.get_weights(), len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        decypher_time = time.time()
        
        if len(config['he']) > 0 and self.homomorphic:
            if self.packing:
                he_packs_to_model(self, config)
            else:
                he_parameters_to_model(self, config)
            
        else:
            self.model.set_weights(parameters)
        decypher_time = time.time() - decypher_time
        loss, acc = self.model.evaluate(self.x_test, self.y_test)

        write_evaluate_logs(self, config['round'], loss, acc, decypher_time)

        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        # with open(f'{self.log_folder}/client_{self.cid}_eval.csv', 'a') as f:
        #     f.write(f"{acc},{loss}\n")


        return loss, len(self.x_test), eval_msg
    
    def create_fit_msg(self, train_time: float, acc:float, loss:float, model_size:int, he_parameters, mask='',round_n=0) -> Dict[str,Any]:
        
        fit_msg: Dict[str,Any] = {
            'cid'         : self.cid,
            'train_time'  : train_time,
            'accuracy'    : acc,
            'loss'        : loss,
            'delay_start' : time.time(),
            'data_size'   : model_size,
            'he'          : he_parameters if self.homomorphic else '',
            'mask'        : pickle.dumps(mask),
            'round'       : round_n,
            
        }
        
        return fit_msg
        
def main():
    tf.keras.backend.clear_session()
    keys_config:Dict[str,Any] = {}
    
    keys_config['poly_modulus_degree'] = int(os.environ.get('POLY', 0))
    keys_config['plain_modulus']       = int(os.environ.get('PLAIN_MODULUS', 0))
    keys_config['coeff_mod_bit_sizes'] = list(map(int, os.environ.get('COEFFS', '').split(',')))
    keys_config['global_scale']        = int(os.environ.get('SCALE', 0))
    keys_config['key_type']            = os.environ.get('KEY_TYPE', None)
    keys_config['n_length']            = int(os.environ.get('N_LENGTH', 0))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"GPUS: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    client =  HEClient(
                        cid             = int(os.environ['CID']), 
                        niid            = True, 
                        dataset         = 'MNIST', 
                        num_clients     = int(os.environ['NCLIENTS']), 
                        dirichlet_alpha = 1,
                     
                        solution        = str(os.environ['SOLUTION']),        
                        percentage      = 0.8,
                        technique       = str(os.environ["TECHNIQUE"]),
                        
                        key_config      = keys_config ,
                        

                                        #   "robin_round"
                                        #   "slided_window"
                                        #   "weight_random"
                        # homomorphic     = os.environ['HOMOMORPHIC'] == 'True',
                        # packing         = os.environ['PACKING'] == 'True',
                        # onlysum         = os.environ['ONLYSUM'] == 'True',
                        # homomorphic_type= str(os.environ['HOMOMORPHIC_TYPE'])
                        )
        
    fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], 
                                client=client)


if __name__ == '__main__':
	main()