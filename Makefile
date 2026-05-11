PYTHON := python3.10

configure:
	docker build -t flhe:server server/.
	docker build -t flhe:client client/.

run:
	docker compose -f FedPHE-FASHIONMNIST-10.yaml up

graph:
	$(PYTHON) generate_graphs.py 

keys:
	$(PYTHON) generate_keys.py

cleanlogs:
	rm -rf logs/

# Pre-train a model on a source dataset and save backbone weights.
# Weights are saved to logs/models/pretrained/<model>/<dataset>/pretrained.weights.h5
#
# Defaults: vit_tiny on cifar100, 100 epochs, batch 128, lr 3e-4, GPU 0
# Override any arg:
#   make pretrain EPOCHS=50 BATCH=256 GPU=1
#   make pretrain MODEL=resnet20 DATASET=cifar10
MODEL   ?= vit_tiny
DATASET ?= cifar100
EPOCHS  ?= 100
BATCH   ?= 128
LR      ?= 3e-4
GPU     ?= 0

pretrain:
	$(PYTHON) pretrain.py \
	    --model $(MODEL) \
	    --dataset $(DATASET) \
	    --epochs $(EPOCHS) \
	    --batch-size $(BATCH) \
	    --lr $(LR) \
	    --gpu $(GPU)