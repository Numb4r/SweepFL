from __future__ import annotations
from typing import Dict, Any, Optional
import os

def write_train_logs(self, rnd, loss, acc, model_size, times, qnt_packing=0,size_pack=0,sigma=None):

    k = ", ".join(self.key_config.keys())
    v = "-".join(map(str,self.key_config.values()))
    v_f = ",".join(map(str,self.key_config.values()))
    times_k = ", ".join(times.keys())
    times_v = ", ".join(map(str,times.values()))
    iid = "niid" if self.niid else 'iid'
    itens = self.key_config.values()
    # Convert dict_values to a list first
    itens = list(itens) 

    # Now you can use indexing
    itens[2] = str(itens[2]).replace(",", ";")
    
    v_f = ",".join(map(str,itens))
    
    # dataset = "PneumoMnist" if self.dataset=="randall-lab/medmnist" else self.dataset
    dataset = self.dataset.replace("/","_")
    
    
    filename = f'logs/{self.log_folder}/train.csv' 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'a') as file:
        if os.path.getsize(filename) == 0:
            file.write(f"round, cid, loss, accuracy, dataset, model_name, iid, dirichlet_alpha, model_size, {times_k}, schema,{k},packing,qnt_packing,size_pack,sigma\n")
        file.write(f"{rnd}, {self.cid}, {loss}, {acc}, {dataset}, {self.model_name} ,{iid},{self.dirichlet_alpha}, {model_size}, {times_v}, {self.homomorphic_type},{v_f},{self.packing},{qnt_packing},{size_pack},{sigma}\n")


def write_evaluate_logs(
    self: Self,
    rnd: int,
    loss: float,
    acc: float,
    decyfer_time_clock: float,
    decyfet_time_process: float
) -> None:


    k = ", ".join(self.key_config.keys())
    v = "-".join(map(str,self.key_config.values()))
    itens = self.key_config.values()
    # Convert dict_values to a list first
    itens = list(itens) 

    # Now you can use indexing
    itens[2] = str(itens[2]).replace(",", ";")
    v_f = ",".join(map(str,itens))
    

  
    iid = "niid" if self.niid else 'iid'
    # dataset = "PneumoMnist" if self.dataset=="randall-lab/medmnist" else self.dataset
    dataset = self.dataset.replace("/","_")
    filename = f'logs/{self.log_folder}/evaluate.csv' 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'a') as file:
        if os.path.getsize(filename) == 0:
            file.write(f"round, cid, loss, accuracy, dataset, model_name, iid,  dirichlet_alpha, decyfer_time_clock, decyfer_time_process,{k}\n")
        file.write(f"{rnd}, {self.cid}, {loss}, {acc}, {dataset}, {self.model_name}, {iid},{self.dirichlet_alpha},{decyfer_time_clock},{decyfet_time_process},{v_f}\n")