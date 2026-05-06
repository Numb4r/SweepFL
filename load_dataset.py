# Script rápido para baixar o dataset uma única vez
# from datasets import load_dataset
from flwr_datasets import FederatedDataset

import os
os.environ["HF_DATASETS_CACHE"] = "./client/data"
# load_dataset("MNIST")
FederatedDataset(dataset="MNIST")