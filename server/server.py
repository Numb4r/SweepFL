import flwr as fl
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import pickle
import tenseal as ts
import os
import sys
import json
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
        fraction_evaluate: float,
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

        self.start_round_offset  = 0
        self.initial_parameters  = None
        self.job_deadline         = int(os.environ.get("JOB_DEADLINE_EPOCH", "0"))
        self.checkpoint_threshold = int(os.environ.get("CHECKPOINT_THRESHOLD_SECS", "300"))
        self.total_target_rounds  = int(os.environ.get("NUM_ROUNDS", "0"))

        self.config_solution()
        self.poly_modulus_degree  = self.key_config.get("poly_modulus_degree")
        self.plain_modulus        = self.key_config.get("plain_modulus")
        self.coeff_mod_bit_sizes  = self.key_config.get("coeff_mod_bit_sizes")
        self.global_scale         = self.key_config.get("global_scale")
        self.key_type             = self.key_config.get("key_type")

        self.context = self.get_server_context()

        resume_logfolder = os.environ.get("RESUME_LOGFOLDER", "")
        if resume_logfolder:
            meta_path = f'logs/{resume_logfolder}/checkpoint/meta.json'
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.start_round_offset = int(meta.get('round', 0))
                print(f"[RESUME] Resuming from round {self.start_round_offset} ({resume_logfolder})")
                if not self.homomorphic:
                    server_ckpt = f'logs/{resume_logfolder}/checkpoint/server_agg.npz'
                    if os.path.exists(server_ckpt):
                        data = np.load(server_ckpt)
                        weights = [data[f'arr_{i}'].astype(np.float32) for i in range(len(data.files))]
                        self.initial_parameters = ndarrays_to_parameters(weights)
                        print(f"[RESUME] Loaded server checkpoint ({len(weights)} arrays)")
            else:
                print(f"[RESUME] No meta.json found at {meta_path}, starting fresh")

        # Número de clientes esperados por rodada para fit e evaluate.
        # max(1,...) garante pelo menos 1 cliente mesmo com fração muito pequena.
        n_part = max(1, round(num_clients * fraction_fit))
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=n_part,
            min_evaluate_clients=n_part,
            min_available_clients=num_clients,
            initial_parameters=self.initial_parameters,
        )
        print("SERVER_READY")

    def config_solution(self):
        if self.solution == 'ckks':
            self.homomorphic = True; self.packing = False; self.onlysum = False; self.homomorphic_type = 'CKKS'
        elif self.solution == 'bfv':
            self.homomorphic = True; self.packing = False; self.onlysum = True;  self.homomorphic_type = 'BFV'
        elif self.solution == 'batchcrypt':
            self.homomorphic = True; self.packing = False; self.onlysum = True;  self.homomorphic_type = 'Paillier'
        elif self.solution in ('fedphe', 'nvnl', 'fedphe-scout', 'nvnl-scout', 'sw', 'rw', 'topkrr', 'srr'):
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

    def _time_remaining(self):
        if not self.job_deadline:
            return float('inf')
        return self.job_deadline - time.time()

    def _write_incomplete_log(self, completed_round: int):
        target = self.total_target_rounds
        if target <= 0 or completed_round >= target:
            return
        remaining = target - completed_round
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        log_path = f'logs/{self.log_folder}/INCOMPLETE.log'
        with open(log_path, 'w') as f:
            f.write(
                f"INTERRUPTED : {ts}\n"
                f"Round       : {completed_round} / {target}\n"
                f"Remaining   : {remaining}\n"
                f"LOGFOLDER   : {self.log_folder}\n"
                f"Solution    : {self.solution}\n"
            )
        print(f"[INCOMPLETE] {log_path} (round {completed_round}/{target})")

    def configure_fit(self, server_round, parameters, client_manager):
        if self._time_remaining() < self.checkpoint_threshold:
            completed = (server_round - 1) + self.start_round_offset
            print(f"[CHECKPOINT] Time limit approaching ({self._time_remaining():.0f}s left), stopping gracefully")
            self._write_incomplete_log(completed)
            sys.exit(0)

        selection_start = time.time()
        data2send = ''
        if len(self.agg_parameters) > 0:
            data2send = self.agg_parameters

        global_round = server_round + self.start_round_offset
        config = {
            'he':             data2send,
            'mask':           self.agg_mask,
            'round':          global_round,
            'total_examples': str(self.total_examples),
        }
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        self.selection_time = time.time() - selection_start
        return [(client, fit_ins) for client in clients]

    def _save_checkpoint_meta(self, global_round):
        ckpt_folder = f'logs/{self.log_folder}/checkpoint'
        os.makedirs(ckpt_folder, exist_ok=True)
        meta = {'round': global_round, 'solution_type': self.homomorphic_type}
        with open(f'{ckpt_folder}/meta.json', 'w') as f:
            json.dump(meta, f)

    def aggregate_fit(self, server_round, results, failures):
        aggregation_start = time.time()
        parameters_list   = []
        total_examples    = 0
        global_round      = server_round + self.start_round_offset

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

            self._save_checkpoint_meta(global_round)
            aggregation_time = time.time() - aggregation_start
            write_agg_logs(
                self.log_folder, self.experiment_name,
                global_round, len(results), total_examples,
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

            self._save_checkpoint_meta(global_round)
            ckpt_folder = f'logs/{self.log_folder}/checkpoint'
            np.savez_compressed(
                f'{ckpt_folder}/server_agg',
                *[w.astype(np.float16) for w in agg_parameters],
            )
            write_agg_logs(
                self.log_folder, self.experiment_name,
                global_round, len(results), total_examples,
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
        global_round = server_round + self.start_round_offset

        print(f"Round {global_round} | loss={loss_aggregated:.4f} | acc={mean_acc:.4f}")

        write_eval_logs(
            self.log_folder, self.experiment_name,
            global_round, len(results), loss_aggregated, mean_acc,
        )
        return loss_aggregated, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []

        global_round = server_round + self.start_round_offset
        config = {
            'he':             self.agg_parameters,
            'round':          global_round,
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

    client_fraction = float(os.environ.get('CLIENT_FRACTION', 1.0))
    server = HEServer(
        num_clients       = int(os.environ['NCLIENTS']),
        dirichlet_alpha   = float(os.environ.get('NIID', 1.0)),
        dataset           = os.environ.get('DATASET', ''),
        fraction_fit      = client_fraction,
        fraction_evaluate = client_fraction,
        solution          = str(os.environ['SOLUTION']),
        key_config        = keys_config,
        log_folder        = os.environ.get('LOGFOLDER', ''),
        experiment_name   = os.environ.get('EXPERIMENT_NAME', ''),
    )

    remaining_rounds = max(1, int(os.environ['NUM_ROUNDS']) - server.start_round_offset)
    fl.server.start_server(
        server_address=os.environ['SERVER_IP'],
        config=fl.server.ServerConfig(num_rounds=remaining_rounds),
        strategy=server,
    )


if __name__ == '__main__':
    main()
