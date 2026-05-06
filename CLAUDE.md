# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Slidhe** is a research project implementing privacy-preserving Federated Learning (FL) using Homomorphic Encryption (HE). Clients train local ML models and send encrypted gradients to an aggregation server. The system supports multiple HE schemes (CKKS, BFV, Paillier/BatchCrypt) and packet selection strategies (top-k, round-robin, sliding window).

## Commands

```bash
# Build Docker images (server and client) — required once before any experiment
make configure

# Generate HE keys (required before running experiments)
make keys
# or directly:
python3 generate_keys.py

# Run the full experiment suite (the normal way to run experiments)
./start_all.sh

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_server_utils.py

# Clean logs
make cleanlogs

# Generate graphs
make graph
```

### Linting and Formatting
```bash
black .
isort .
flake8 .
mypy .
```

## Architecture

### Experiment Execution Flow

`start_all.sh` is the top-level experiment runner. It:
1. Iterates over all combinations of HE config, solution, dataset, niid, seed, and percentage defined at the top of the script.
2. For each combination, calls `run_combo_keys` (Python) to regenerate keys, then calls `run_experiment_tmux`.
3. `run_experiment_tmux` launches a **tmux session `SERVER`** running `./run_server.sh` and one **tmux session `client-<i>`** per client running `./run_client.sh`.
4. Both scripts launch Docker containers: server as `fl_server-gpu` on port 9999; each client as `client-<i>` connecting to `host.docker.internal:9999`.
5. Clients are distributed across physical GPUs via `GPU_IDS` and `CLIENTS_PER_GPU` (e.g. 3 clients per GPU × 3 GPUs = 9 clients).
6. After the server session ends, logs are moved to `mv_logs/<LOGFOLDER>/` and `clean_logs.sh` is run.
7. A Telegram notification is sent on completion (token/chat hardcoded in the script).

The `LOGFOLDER` name encodes all experiment parameters: `<model>_<dataset>_<solution>_<technique>_<rounds>rounds_<clients>clients_seed<N>_niid<N>_percentage<N>_<poly>_<plain>_<nlength>_<coeffs>_<scale>_<keytype>_<timestamp>`.

### FL Communication Flow
1. **Server** (`server/server.py`: `HEServer`) extends `fl.server.strategy.FedAvg` via the [Flower](https://flower.ai/) framework.
2. **Clients** (`client/client.py`: `HEClient`) extend `fl.client.NumPyClient`.
3. Each round: server sends aggregated encrypted parameters → clients decrypt, train locally, re-encrypt their update, send back → server aggregates in ciphertext.

### Key Design Decisions

**Solution parameter** controls the entire encryption/aggregation path. Valid values: `ckks`, `bfv`, `batchcrypt`, `fedphe`, `nvnl`, `fedphe-scout`, `nvnl-scout`, `plaintext`, `sw`, `rw`. Both client and server call `config_solution()` to set `homomorphic`, `packing`, `onlysum`/`only_sum`, and `homomorphic_type` from this string.

**Packing vs. no packing**: When `packing=True`, model weights are flattened and chunked into CKKS vectors (packs) of size `poly_modulus_degree // 2`. Only a subset of packs is encrypted and transmitted, selected by a mask (top-k, round-robin, or sliding window). The server aggregates packs homomorfically and sends the aggregated packs+mask back to clients. When `packing=False`, all weights are concatenated into a single ciphertext.

**Context files** live in `context/`. CKKS keys: `nvnlc2.pkl` (client, with secret key) and `serverc2.pkl` (server, public only). BFV keys: `bfv_secret.pkl` and `bfv_server_key.pkl`. Paillier: `paillier.pkl`.

### Module Structure

- `client/client.py` — `HEClient`: orchestrates per-round training and dispatches to `fit_*` functions in `client/literature/`
- `client/literature/` — one `fit_<solution>.py` per HE scheme; each implements local train → encrypt → return fit message
- `client/literature/set_parameters.py` — decryption and model weight reconstruction on the client side
- `client/literature/fedphe.py` — also contains mask generation: `get_topk_mask`, `get_robin_round_mask`, `get_slide_window_mask`, `get_pondering_random_mask`
- `client/client_utils.py` — `packing`, `cypher_packs`, `decypher_packs`, `flat_packs`, `reshape_model`
- `client/models.py` — model factories: `create_dnn`, `create_lenet5`, `create_resnet20`
- `server/server.py` — `HEServer`: aggregate_fit dispatches to packing vs. non-packing path
- `server/server_utils.py` — HE aggregation helpers: `align_packs`, `aggregate_packs_and_masks`, `aggregate_packs_and_mask`, `get_pack_mask_pair`, `get_cyphered_parameters`, `aggregated_cyphered_parameters`
- `encryption/` and `server/encryption/` and `client/encryption/` — largely mirrored; `encrypt.py` defines base `Encrypt` class; `ckks.py`, `bfv.py`, `paillier.py` implement scheme-specific logic; `quantize.py` / `quantize_batchcrypt.py` handle quantization

### Important Constraints

- **Plaintext must run first** to generate and save initial model weights (`logs/models/init_weights/…`). All HE solutions load these weights at startup; they raise an exception if the file is absent.
- **CKKS parameter constraint**: `[30,60,30]` with `scale=2^60` is documented as incompatible with scalar multiplication (see `tests/test_server_utils.py` xfail). Production uses `[30,60,30]` with `scale=2^60` in `encryption/encrypt.py`—this is a known limitation; prefer `[60,40,40,60]` with `scale=2^40` for reliable scalar ops.
- **gRPC message size**: set to 1 GB (`MAX_MESSAGE_LENGTH = 1024*1024*1024`) in `client/client.py` to accommodate large ciphertexts.
- Python version is **3.10** (see `pyproject.toml` and `Makefile`).

### Environment Variables (Docker / runtime)
Clients and server are configured entirely via environment variables: `CID`, `NCLIENTS`, `SOLUTION`, `DATASET`, `NIID`, `PERCENTAGE`, `TECHNIQUE`, `SERVER_IP`, `NUM_ROUNDS`, `LOGFOLDER`, `POLY`, `PLAIN_MODULUS`, `COEFFS`, `SCALE`, `KEY_TYPE`, `N_LENGTH`.
