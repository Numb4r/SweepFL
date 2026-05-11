PYTHON := python3.10

SIF_DIR ?= .

# ── Docker ────────────────────────────────────────────────────────────────────
configure:
	docker build -t flhe:server -f server/Dockerfile .
	docker build -t flhe:client -f client/Dockerfile .

test-ckks:
	@echo "=== Testing ckks_project installation: client ==="
	docker run --rm -v "$$PWD/tests:/tests:ro" flhe:client python /tests/test_ckks_install.py
	@echo "=== Testing ckks_project installation: server ==="
	docker run --rm -v "$$PWD/tests:/tests:ro" flhe:server python /tests/test_ckks_install.py

# ── Singularity ───────────────────────────────────────────────────────────────
# Option A — build directly from .def files (no Docker required, pulls base from Hub):
singularity-configure:
	singularity build $(SIF_DIR)/flhe_client.sif client/singularity.def
	singularity build $(SIF_DIR)/flhe_server.sif server/singularity.def

# Option B — convert already-built Docker images (faster, requires local Docker images):
singularity-from-docker:
	singularity build $(SIF_DIR)/flhe_client.sif docker-daemon://flhe:client
	singularity build $(SIF_DIR)/flhe_server.sif docker-daemon://flhe:server

test-ckks-singularity:
	@echo "=== Testing ckks_project installation: client (singularity) ==="
	singularity exec --bind "$$PWD/tests:/tests:ro" $(SIF_DIR)/flhe_client.sif \
	    python /tests/test_ckks_install.py
	@echo "=== Testing ckks_project installation: server (singularity) ==="
	singularity exec --bind "$$PWD/tests:/tests:ro" $(SIF_DIR)/flhe_server.sif \
	    python /tests/test_ckks_install.py

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