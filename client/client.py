
import os
import pickle
import time
import flwr as fl
import tensorflow as tf


from dataset_utils import ManageDatasets, load_data_flowerdataset
from pathlib import Path
import random
import numpy as np
from models import (
    create_dnn, create_dnn_tiny, create_dnn_base,
    create_lenet5, create_resnet20,
    create_mobilenetv2, create_vit_tiny, create_simple_cnn,
)
from client_logs import write_evaluate_logs
from literature.set_parameters import he_parameters_to_model, he_packs_to_model
from literature import fit_ckks, fit_bfv, fit_batchcrypt, fit_fedphe, fit_plaintext, fit_bitnetflip, fit_nvnl,fit_nvnl_scout,fit_fedphe_scout, fit_sw, fit_rw
from  client_utils import flat_parameters
import math


from typing import Optional,Dict,Any
def estimate_ckks_sigma(context, v, n_clients, dim=1000, trials=50):
    """Estimates CKKS additive noise std-dev. Only meaningful for TenSEAL contexts."""
    import tenseal as ts
    v = np.array(v)
    errs = []
    for _ in range(trials):
        enc = ts.ckks_vector(context, v)
        for _ in range(n_clients):
            enc += v
        dec = np.array(enc.decrypt())
        errs.append(dec - (v * n_clients))
    sigma_ckks = np.std(np.concatenate(errs))
    return math.sqrt(n_clients) * sigma_ckks

class HEClient(fl.client.NumPyClient):
    def __init__(self, cid:int, niid:bool, dataset:str, num_clients:int, 
                 dirichlet_alpha:float, solution:str,percentage:float,technique:str,
                 key_config: Dict[str,Any], seed:int = 42,log_folder:str=''):
        
        self.cid              = int(cid)
        self.dataset          = dataset
        self.niid             = niid
        self.num_clients      = num_clients
        self.dirichlet_alpha  = dirichlet_alpha
        self.dataset_manager  = ManageDatasets(self.cid)
        self.solution         = str(solution).lower()
        self.weights_packs    = []
        self.technique        = technique
        self.percentage       = percentage
        self.dirichlet_alpha = dirichlet_alpha
        self.key_config = key_config
        self.sigma = None
                                       
        
        
        
        
        self.seed = seed 

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism() # type: ignore
        tf.config.threading.set_intra_op_parallelism_threads(1) # type: ignore
        tf.config.threading.set_inter_op_parallelism_threads(1) # type: ignore
        self.poly_modulus_degree = self.key_config.get("poly_modulus_degree")
        self.plain_modulus = self.key_config.get("plain_modulus")
        self.coeff_mod_bit_sizes = self.key_config.get("coeff_mod_bit_sizes")
        self.global_scale = self.key_config.get("global_scale")
        self.key_type = self.key_config.get("key_type")
        self.n_length = self.key_config.get("n_length")
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        
        self.x_train, self.y_train, self.x_test, self.y_test = load_data_flowerdataset(self)
        class_n_dict = {
            "AISNP/PathMNIST":9,
            "FASHION_MNIST":10,
            "randall-lab/medmnist":2,
            "CIFAR10":10,
            "tanganke/gtsrb":43
            
        }
        class_n = class_n_dict.get(self.dataset, 10)
        arch = os.environ.get("MODEL_ARCH", "").lower()

        # Ensure channel dim exists for conv/vit models when dataset is grayscale
        if self.x_train.ndim == 3:
            self.x_train = self.x_train[..., np.newaxis]
            self.x_test  = self.x_test[..., np.newaxis]

        shape = self.x_train.shape[1:]

        _model_factories = {
            "resnet20":    lambda: (create_resnet20(shape, class_n),    "resnet20"),
            "lenet5":      lambda: (create_lenet5(shape, class_n),      "lenet5"),
            "dnn":         lambda: (create_dnn(shape, class_n, seed),   "DNN"),
            "dnn_base":    lambda: (create_dnn_base(shape, class_n, seed), "DNN_base"),
            "dnn_tiny":    lambda: (create_dnn_tiny(shape, class_n, seed), "DNN_tiny"),
            "mobilenetv2": lambda: (create_mobilenetv2(shape, class_n), "mobilenetv2"),
            "vit_tiny":    lambda: (create_vit_tiny(shape, class_n),    "vit_tiny"),
            "simple_cnn":  lambda: (create_simple_cnn(shape, class_n),  "simple_cnn"),
        }

        if arch in _model_factories:
            self.model, self.model_name = _model_factories[arch]()
        else:
            self.model, self.model_name = create_dnn(shape, class_n, seed), "DNN"

        print("x_train:", self.x_train.shape)
        print("model:", self.model_name, "| input:", self.model.input_shape)
        self.config_solution()
        if self.homomorphic:
            self.context = self.get_client_context()
        dataset_folder = self.dataset.replace("/","_")
        init_weights = f"logs/models/init_weights/{self.model_name}/{dataset_folder}/init_weights_{cid}.weights.h5"
        os.makedirs(f"logs/models/init_weights/{self.model_name}/{dataset_folder}",exist_ok=True)
        if os.path.exists(init_weights):
            self.model.load_weights(init_weights)
            print("Load weights")
        elif self.solution == "plaintext":
            print("Init weights")
            self.model.save_weights(init_weights)
        else:
            raise Exception("Cannot start, generate plaintext first")
        
        
        
        # dataset_folder = self.dataset.replace("/","_")
        self.log_folder=log_folder
        self.len_shared_data  =  len(flat_parameters(self.model.get_weights()))
        print(f"Log folder: {self.log_folder}")        

        
        
    def get_client_context(self):
        if self.homomorphic_type == 'Paillier':
            with open('context/paillier.pkl', 'rb') as file:
                return pickle.load(file)
        elif self.homomorphic_type == 'CKKS':
            from encryption.ckks_backend import get_backend
            backend_name = self.key_config.get("ckks_backend") or os.environ.get("CKKS_BACKEND", "tenseal")
            return get_backend(backend_name).load_client_context("../context")
        else:  # BFV
            import tenseal as ts
            with open('../context/bfv_secret.pkl', 'rb') as file:
                return ts.context_from(pickle.load(file)["context"])

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
        elif str(self.solution).lower() == 'nvnl':
            self.homomorphic      = True
            self.homomorphic_type = 'CKKS'
            self.only_sum         = False
            self.packing          = True
        elif str(self.solution).lower() == 'fedphe-scout':
            self.homomorphic      = True
            self.homomorphic_type = 'CKKS'
            self.only_sum         = False
            self.packing          = True
        elif str(self.solution).lower() == 'nvnl-scout':
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
        elif str(self.solution).lower() == 'sw':
            self.homomorphic      = True
            self.homomorphic_type = 'CKKS'
            self.only_sum         = False
            self.packing          = True
        elif str(self.solution).lower() == 'rw':
            self.homomorphic      = True
            self.homomorphic_type = 'CKKS'
            self.only_sum         = False
            self.packing          = True
        
            
            

    def fit(self, parameters, config):
        
        if str(self.solution).lower() == 'ckks':
            fit_msg = fit_ckks(self, parameters, config)
        elif str(self.solution).lower() == 'bfv':
            fit_msg = fit_bfv(self, parameters, config)
        elif str(self.solution).lower() == 'batchcrypt':
            fit_msg = fit_batchcrypt(self, parameters, config)
        elif str(self.solution).lower() == 'fedphe':
            fit_msg = fit_fedphe(self, parameters, config)        
        elif str(self.solution).lower() == 'nvnl':
            fit_msg = fit_nvnl(self,parameters,config)
        elif str(self.solution).lower() == 'fedphe-scout':
            fit_msg = fit_fedphe_scout(self, parameters, config)        
        elif str(self.solution).lower() == 'nvnl-scout':
            fit_msg = fit_nvnl_scout(self,parameters,config)
        elif str(self.solution).lower() == 'plaintext':
            fit_msg = fit_plaintext(self, parameters, config)
        elif str(self.solution).lower() == "socas":
            fit_msg = fit_bitnetflip(self,parameters,config)
        elif str(self.solution).lower() == "sw":
            fit_msg = fit_sw(self,parameters,config)
        elif str(self.solution).lower() == "rw":
            fit_msg = fit_rw(self,parameters,config)
            
            
        print("TRYING SEND FIT MESSAGE")
        return self.model.get_weights(), len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        decypher_time_clock = time.time()   
        decypher_time_process = time.process_time()
        
        
        if len(config['he']) > 0 and self.homomorphic:
            if self.packing:
                he_packs_to_model(self, config)
            else:
                he_parameters_to_model(self, config)
            
        else:
            self.model.set_weights(parameters)
        decypher_time_process = time.process_time() - decypher_time_process
        decypher_time_clock = time.time() - decypher_time_clock
        loss, acc = self.model.evaluate(self.x_test, self.y_test)

        write_evaluate_logs(self, config['round'], loss, acc, decypher_time_clock, decypher_time_process)

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
        
# def main():
#     tf.keras.backend.clear_session()
#     keys_config:Dict[str,Any] = {}
    
#     keys_config['poly_modulus_degree'] = int(os.environ.get('POLY', 0))
#     keys_config['plain_modulus']       = int(os.environ.get('PLAIN_MODULUS', -1))
#     keys_config['coeff_mod_bit_sizes'] = list(map(int, os.environ.get('COEFFS', '').split(',')))
#     keys_config['global_scale']        = int(os.environ.get('SCALE', 0))
#     keys_config['key_type']            = os.environ.get('KEY_TYPE', None)
#     keys_config['n_length']            = int(os.environ.get('N_LENGTH', 0))
    
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     print(f"GPUS: {gpus}")
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
#     print(f"dirichlet_alpha {float(os.environ['NIID'])}")
#     percentage = float(os.environ['PERCENTAGE'])
#     niid = True if float(os.environ['NIID']) == 1 else False
#     client =  HEClient(
#                         cid             = int(os.environ['CID']), 
#                         niid            = niid,
#                         dirichlet_alpha = float(os.environ['NIID']) ,
#                         dataset         = os.environ["DATASET"], 
#                         num_clients     = int(os.environ['NCLIENTS']), 
                     
#                         solution        = str(os.environ['SOLUTION']),        
#                         percentage      = percentage,
#                         technique       = str(os.environ["TECHNIQUE"]),
#                         log_folder = str(os.environ.get("LOGFOLDER")),
                        
#                         key_config      = keys_config ,
#                         )
#     while True:
#         try:
#             fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], 
#                                         client=client)
#             break
#         except Exception as e:
#             print(f"Connection error: {e}. Retrying in 5 seconds...")
#             time.sleep(5)


# if __name__ == '__main__':
# 	main()


def main():
    tf.keras.backend.clear_session()
    keys_config:Dict[str,Any] = {}
    
    keys_config['poly_modulus_degree']  = int(os.environ.get('POLY', 0))
    keys_config['plain_modulus']        = int(os.environ.get('PLAIN_MODULUS', -1))
    keys_config['coeff_mod_bit_sizes']  = list(map(int, os.environ.get('COEFFS', '').split(',')))
    keys_config['global_scale']         = int(os.environ.get('SCALE', 0))
    keys_config['key_type']             = os.environ.get('KEY_TYPE', None)
    keys_config['n_length']             = int(os.environ.get('N_LENGTH', 0))
    keys_config['ckks_backend']         = os.environ.get('CKKS_BACKEND', 'tenseal')
    keys_config['simplify_N']           = int(os.environ.get('SIMPLIFY_N', 8192))
    keys_config['simplify_bits']        = int(os.environ.get('SIMPLIFY_BITS', 62))
    keys_config['simplify_scale_bits']  = int(os.environ.get('SIMPLIFY_SCALE_BITS', 60))
    keys_config['pycrypt_N']            = int(os.environ.get('PYCRYPT_N', 8192))
    keys_config['pycrypt_bits']         = int(os.environ.get('PYCRYPT_BITS', 60))
    keys_config['openfhe_ring_dim']     = int(os.environ.get('OPENFHE_RING_DIM', 8192))
    keys_config['openfhe_mult_depth']   = int(os.environ.get('OPENFHE_MULT_DEPTH', 1))
    keys_config['openfhe_scale_bits']   = int(os.environ.get('OPENFHE_SCALE_BITS', 50))
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"GPUS: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"dirichlet_alpha {float(os.environ['NIID'])}")
    percentage = float(os.environ['PERCENTAGE'])
    niid = True if float(os.environ['NIID']) == 1 else False
    
    client =  HEClient(
        cid             = int(os.environ['CID']), 
        niid            = niid,
        dirichlet_alpha = float(os.environ['NIID']) ,
        dataset         = os.environ["DATASET"], 
        num_clients     = int(os.environ['NCLIENTS']), 
        solution        = str(os.environ['SOLUTION']),        
        percentage      = percentage,
        technique       = str(os.environ["TECHNIQUE"]),
        log_folder      = str(os.environ.get("LOGFOLDER")),
        key_config      = keys_config,
    )
    
    # ---------------------------------------------------------
    # NOVO: Definindo o limite máximo de mensagem do gRPC para 1GB
    # ---------------------------------------------------------
    MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024 

    while True:
        try:
            fl.client.start_numpy_client(
                server_address=os.environ['SERVER_IP'], 
                client=client,
                grpc_max_message_length=MAX_MESSAGE_LENGTH  # <-- Parâmetro adicionado
            )
            break
        except Exception as e:
            print(f"Connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == '__main__':
    main()