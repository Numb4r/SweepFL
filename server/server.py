import flwr as fl
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import pickle
import tenseal as ts
import os
import numpy as np
from functools import reduce
import logging
logging.basicConfig(level=logging.DEBUG)
import time
import random
from typing import Dict, Any

from server_utils import align_packs, aggregate_packs_and_masks, serialize_packs, get_pack_mask_pair
from server_utils import aggregate_packs_and_mask, get_cyphered_parameters, aggregated_cyphered_parameters
from server_logs import write_agg_logs, write_eval_logs

import pickle as pk
from encryption.paillier import PaillierCipher


class HEServer(fl.server.strategy.FedAvg):
    def __init__(
        self,
        num_clients: int,
        dirichlet_alpha: float,
        dataset: str,
        fraction_fit: float,
        solution: str,
        key_config: Dict[str, Any],
        log_folder: str = "",
        experiment_name: str = "",
    ):
        self.num_clients      = num_clients
        self.dirichlet_alpha  = dirichlet_alpha
        self.dataset          = dataset
        self.agg_parameters   = ''
        self.agg_mask         = ''
        self.selection_time   = 0.0
        self.total_examples   = 0
        self.homomorphic_type = ''
        self.solution         = str(solution).lower()
        self.key_config       = key_config
        self.log_folder       = log_folder
        self.experiment_name  = experiment_name

        self.config_solution()
        self.poly_modulus_degree  = self.key_config.get("poly_modulus_degree")
        self.plain_modulus        = self.key_config.get("plain_modulus")
        self.coeff_mod_bit_sizes  = self.key_config.get("coeff_mod_bit_sizes")
        self.global_scale         = self.key_config.get("global_scale")
        self.key_type             = self.key_config.get("key_type")

        self.context = self.get_server_context()
        super().__init__(
            fraction_fit=fraction_fit,
            min_available_clients=num_clients,
            min_evaluate_clients=num_clients,
        )
        print("SERVER_READY")

    def config_solution(self):
        if self.solution == 'ckks':
            self.homomorphic = True; self.packing = False; self.onlysum = False; self.homomorphic_type = 'CKKS'
        elif self.solution == 'bfv':
            self.homomorphic = True; self.packing = False; self.onlysum = True;  self.homomorphic_type = 'BFV'
        elif self.solution == 'batchcrypt':
            self.homomorphic = True; self.packing = False; self.onlysum = True;  self.homomorphic_type = 'Paillier'
        elif self.solution in ('fedphe', 'nvnl', 'fedphe-scout', 'nvnl-scout', 'sw', 'rw'):
            self.homomorphic = True; self.packing = True;  self.onlysum = False; self.homomorphic_type = 'CKKS'
        elif self.solution == 'socas':
            self.homomorphic = True; self.packing = False; self.onlysum = False; self.homomorphic_type = 'CKKS'
        else:  # plaintext
            self.homomorphic = False; self.packing = False; self.onlysum = False; self.homomorphic_type = 'None'

    def get_server_context(self):
        if self.homomorphic_type == 'Paillier':
            with open('context/paillier.pkl', 'rb') as f:
                return pickle.load(f)
        elif self.homomorphic_type == 'CKKS':
            from encryption.ckks_backend import get_backend
            backend_name = self.key_config.get("ckks_backend") or os.environ.get("CKKS_BACKEND", "tenseal")
            return get_backend(backend_name).load_server_context("./context")
        else:  # BFV or plaintext
            with open('./context/bfv_server_key.pkl', 'rb') as f:
                return ts.context_from(pickle.load(f)["context"])

    def configure_fit(self, server_round, parameters, client_manager):
        selection_start = time.time()
        data2send = ''
        if len(self.agg_parameters) > 0:
            data2send = self.agg_parameters

        config = {
            'he':             data2send,
            'mask':           self.agg_mask,
            'round':          server_round,
            'total_examples': str(self.total_examples),
        }
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        self.selection_time = time.time() - selection_start
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregation_start = time.time()
        parameters_list   = []
        total_examples    = 0

        if self.homomorphic:
            if self.packing:
                from encryption.ckks_backend import get_backend as _get_backend
                _backend = _get_backend(self.key_config.get("ckks_backend"))
                pack_mask_pair, total_examples     = get_pack_mask_pair(self, results)
                aggregated_packs, aggregated_masks = aggregate_packs_and_mask(
                    pack_mask_pair, total_examples, self.onlysum,
                    backend=_backend, ctx=self.context,
                )
                aggregated_packs    = serialize_packs(aggregated_packs)
                self.agg_parameters = pickle.dumps(aggregated_packs)
                self.agg_mask       = pickle.dumps(aggregated_masks)
                self.total_examples = total_examples
            else:
                parameters_list, total_examples = get_cyphered_parameters(self, results)
                agg_parameters                  = aggregated_cyphered_parameters(self, parameters_list, total_examples)
                if self.homomorphic_type == 'Paillier':
                    self.agg_parameters = pickle.dumps(agg_parameters)
                else:
                    self.agg_parameters = agg_parameters.serialize()
                self.total_examples = total_examples

            aggregation_time = time.time() - aggregation_start
            write_agg_logs(
                self.log_folder, self.experiment_name,
                server_round, len(results), total_examples,
                self.selection_time, aggregation_time,
            )
            return [], {}

        else:  # plaintext
            for _, fit_res in results:
                parameters_list.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
                total_examples += int(fit_res.num_examples)

            agg_parameters      = aggregate(parameters_list)
            aggregation_time    = time.time() - aggregation_start
            self.total_examples = total_examples

            write_agg_logs(
                self.log_folder, self.experiment_name,
                server_round, len(results), total_examples,
                self.selection_time, aggregation_time,
            )
            return ndarrays_to_parameters(agg_parameters), {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        accuracies = [r.metrics['accuracy'] for _, r in results]
        loss_aggregated = weighted_loss_avg(
            [(r.num_examples, r.loss) for _, r in results]
        )
        mean_acc = sum(accuracies) / len(accuracies)

        print(f"Round {server_round} | loss={loss_aggregated:.4f} | acc={mean_acc:.4f}")

        write_eval_logs(
            self.log_folder, self.experiment_name,
            server_round, len(results), loss_aggregated, mean_acc,
        )
        return loss_aggregated, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []

        config = {
            'he':             self.agg_parameters,
            'round':          server_round,
            'mask':           self.agg_mask,
            'total_examples': str(self.total_examples),
        }
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, evaluate_ins) for client in clients]


def main():
    keys_config: Dict[str, Any] = {}

    keys_config['poly_modulus_degree']  = int(os.environ.get('POLY', 0))
    keys_config['plain_modulus']        = int(os.environ.get('PLAIN_MODULUS', -1))
    keys_config['coeff_mod_bit_sizes']  = list(map(int, os.environ.get('COEFFS', '').split(',')))
    keys_config['global_scale']         = int(os.environ.get('SCALE', 0))
    keys_config['key_type']             = os.environ.get('KEY_TYPE', None)
    keys_config['ckks_backend']         = os.environ.get('CKKS_BACKEND', 'tenseal')
    keys_config['simplify_N']           = int(os.environ.get('SIMPLIFY_N', 8192))
    keys_config['simplify_bits']        = int(os.environ.get('SIMPLIFY_BITS', 62))
    keys_config['simplify_scale_bits']  = int(os.environ.get('SIMPLIFY_SCALE_BITS', 60))
    keys_config['pycrypt_N']            = int(os.environ.get('PYCRYPT_N', 8192))
    keys_config['pycrypt_bits']         = int(os.environ.get('PYCRYPT_BITS', 60))
    keys_config['openfhe_ring_dim']     = int(os.environ.get('OPENFHE_RING_DIM', 8192))
    keys_config['openfhe_mult_depth']   = int(os.environ.get('OPENFHE_MULT_DEPTH', 1))
    keys_config['openfhe_scale_bits']   = int(os.environ.get('OPENFHE_SCALE_BITS', 50))
    keys_config['n_length']             = int(os.environ.get('N_LENGTH', 0))

    server = HEServer(
        num_clients     = int(os.environ['NCLIENTS']),
        dirichlet_alpha = float(os.environ.get('NIID', 1.0)),
        dataset         = os.environ.get('DATASET', ''),
        fraction_fit    = 1.0,
        solution        = str(os.environ['SOLUTION']),
        key_config      = keys_config,
        log_folder      = os.environ.get('LOGFOLDER', ''),
        experiment_name = os.environ.get('EXPERIMENT_NAME', ''),
    )

    fl.server.start_server(
        server_address=os.environ['SERVER_IP'],
        config=fl.server.ServerConfig(num_rounds=int(os.environ['NUM_ROUNDS'])),
        strategy=server,
    )


if __name__ == '__main__':
    main()
