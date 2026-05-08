# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Scope

**IMPORTANT — literatura ignorada**: Ignore qualquer artigo ou referência que mencione a técnica **SOCAS** (pura ou combinada com outra técnica). Não sugerir, implementar, referenciar ou comparar com SOCAS em nenhuma circunstância.

## Project Overview

**Slidhe** is a research project implementing privacy-preserving Federated Learning (FL) using Homomorphic Encryption (HE). Clients train local ML models and send encrypted gradients to an aggregation server. The system supports multiple HE schemes (CKKS, BFV, Paillier/BatchCrypt) and packet selection strategies (top-k, round-robin, sliding window, pondering-random).

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
pytest tests/test_model_reconstruction.py

# Clean logs
make cleanlogs

# Generate graphs
make graph

# Sync to remote cluster
./sync_remote.sh
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
1. Iterates over all combinations of HE config, solution, dataset, model, niid, seed, and percentage defined at the top of the script using `MODELS=()`, `DATASETS=()`, `SOLUTIONS=()` arrays (fully decoupled — any model can run with any dataset).
2. For each combination, calls `generate_keys()` (inline Python) to regenerate keys, then calls `run_experiment()`.
3. `run_experiment()` launches a **tmux session `SERVER`** running `./run_server.sh` and one **tmux session `client-<i>`** per client running `./run_client.sh`.
4. Both scripts launch Docker containers: server as `fl_server-gpu` on port 9999; each client as `client-<i>` connecting to `host.docker.internal:9999`.
5. Clients are distributed across physical GPUs via `GPU_IDS` and `CLIENTS_PER_GPU` (e.g. 3 clients per GPU × 3 GPUs = 9 clients).
6. After the server session ends, client tmux sessions are killed. No mv_logs step — logs stay in `logs/<LOGFOLDER>/`.
7. A Telegram notification is sent on completion (token/chat hardcoded in the script).

The `LOGFOLDER` name encodes all experiment parameters: `<model>_<dataset>_<solution>_<technique>_<rounds>rounds_<clients>clients_seed<N>_niid<N>_percentage<N>_<poly>_<plain>_<nlength>_<coeffs>_<scale>_<keytype>_<timestamp>`.

**Experiment grouping**: Set `FOLDER="subfolder_name"` in `start_all.sh` to place all experiments under `logs/subfolder_name/`. Leave empty to write directly into `logs/`.

**Experiment label**: Set `EXPERIMENT_NAME="label"` to add a human-readable name to every CSV log row. Leave empty to auto-generate as `{model}_{dataset}_{solution}`.

**Model selection**: `MODEL_ARCH` env var selects the model factory (without `create_` prefix). If unset, defaults to `dnn`. Models are dataset-agnostic — the factory adapts input shape and num_classes at runtime based on the actual data loaded.

### FL Communication Flow
1. **Server** (`server/server.py`: `HEServer`) extends `fl.server.strategy.FedAvg` via the [Flower](https://flower.ai/) framework.
2. **Clients** (`client/client.py`: `HEClient`) extend `fl.client.NumPyClient`.
3. Each round: server sends aggregated encrypted parameters → clients decrypt, train locally, re-encrypt their update, send back → server aggregates in ciphertext.

### Key Design Decisions

**Solution parameter** controls the entire encryption/aggregation path. Valid values: `ckks`, `bfv`, `batchcrypt`, `fedphe`, `nvnl`, `fedphe-scout`, `nvnl-scout`, `plaintext`, `sw`, `rw`. Both client and server call `config_solution()` to set `homomorphic`, `packing`, `onlysum`/`only_sum`, and `homomorphic_type` from this string.

**Packing vs. no packing**: When `packing=True`, model weights are flattened and chunked into CKKS vectors (packs) of size `poly_modulus_degree // 2`. Only a subset of packs is encrypted and transmitted, selected by a mask (top-k, round-robin, sliding window, or pondering-random). The server aggregates packs homomorphically and sends the aggregated packs+mask back to clients. When `packing=False`, all weights are concatenated into a single ciphertext.

**CKKS backend abstraction**: `encryption/ckks_backend.py` (mirrored to `client/encryption/` and `server/encryption/`) defines a `CKKSBackend` ABC with methods `generate_and_save`, `load_client_context`, `load_server_context`, `encrypt`, `serialize`, `deserialize`, `add(ctx, ct1, ct2)`, `mul(ctx, ct, scalar)`, `decrypt`. Concrete backends:
- `TenSEALBackend` (default) — context files `nvnlc2.pkl` / `serverc2.pkl`
- `CKKSSimplifyBackend` — user's `ckks_python_simplify` lib; context file `ckks_simplify_ctx.pkl`
- `CKKSPythonBackend` — user's verbose `ckks_python` lib; files `ckks_python_sk.bin`, `ckks_python_pk.bin`, `ckks_python_params.pkl`
- `OpenFHEBackend` — files `openfhe_cc.bin`, `openfhe_sk.bin`, `openfhe_pk.bin`

Select backend via `CKKS_BACKEND` env var (set in `start_all.sh`). `get_backend(name)` is the factory, cached per process.

**Context files** live in `context/`. CKKS keys managed by the backend (see above). BFV keys: `bfv_secret.pkl` and `bfv_server_key.pkl`. Paillier: `paillier.pkl`.

### Module Structure

- `client/client.py` — `HEClient`: reads `MODEL_ARCH` env var to select model; falls back to `dnn` if unset; dispatches per-round training to `fit_*` functions in `client/literature/`
- `client/literature/` — one `fit_<solution>.py` per HE scheme; each implements local train → encrypt → return fit message with granular per-operation timing
- `client/literature/set_parameters.py` — decryption and model weight reconstruction on the client side; saves per-round compressed model snapshots (`np.savez_compressed` + float16, cid==0 only)
- `client/literature/fedphe.py` — top-k packing; also contains mask helpers: `get_topk_mask`, `get_robin_round_mask`, `get_slide_window_mask`, `get_pondering_random_mask`
- `client/literature/NVNL.py` — round-robin packing
- `client/literature/sw.py` — sliding-window packing (stride=0.5)
- `client/literature/rw.py` — pondering-random packing
- `client/literature/fedphe_scout.py`, `NVNL_scout.py` — scout variants with custom TF training loop (trust-region + gating vs sigma + EMA blending)
- `client/client_logs.py` — fixed-schema CSV logging (39-column `train.csv`, 15-column `evaluate.csv`)
- `client/client_utils.py` — `packing`, `cypher_packs`, `decypher_packs`, `flat_packs`, `reshape_model`
- `client/models.py` — model factories (see Models section below)
- `server/server.py` — `HEServer`: `aggregate_fit` dispatches to packing vs. non-packing path; calls `write_agg_logs`
- `server/server_logs.py` — server-side CSV logging: `write_agg_logs` (aggregation timing) and `write_eval_logs` (round evaluation)
- `server/server_utils.py` — HE aggregation helpers: `align_packs`, `aggregate_packs_and_masks`, `aggregate_packs_and_mask`, `get_pack_mask_pair`, `get_cyphered_parameters`, `aggregated_cyphered_parameters`
- `encryption/ckks_backend.py` — CKKS backend abstraction (mirrored to `client/encryption/` and `server/encryption/`)
- `encryption/` and `server/encryption/` and `client/encryption/` — largely mirrored; `encrypt.py` defines base `Encrypt` class; `ckks.py`, `bfv.py`, `paillier.py` implement scheme-specific logic; `quantize.py` / `quantize_batchcrypt.py` handle quantization
- `sync_remote.sh` — rsync to `yuri.faria@ssh.recod.ic.unicamp.br:~/keys` (excludes `.venv/`, `logs/`, `__pycache__`, `*.csv`, `.git/`)

### CSV Log Schema

**`train.csv`** (39 columns, one row per client per round):

| Group | Columns |
|---|---|
| Identity | `experiment_name`, `round`, `cid` |
| Config | `solution`, `schema`, `model`, `dataset`, `distribution`, `dirichlet_alpha` |
| Accuracy | `train_loss`, `train_acc` |
| Decrypt | `decrypt_time_s`, `decrypt_time_cpu` |
| Train | `train_time_s`, `train_time_cpu` |
| Flatten | `flatten_time_s`, `flatten_time_cpu` |
| Quantize | `quantize_time_s`, `quantize_time_cpu` (BFV / BatchCrypt only) |
| Batch padding | `batch_padding_time_s`, `batch_padding_time_cpu` (BatchCrypt only) |
| Pack | `pack_time_s`, `pack_time_cpu` (packing solutions only) |
| Mask select | `mask_select_time_s`, `mask_select_time_cpu` (packing solutions only) |
| Encrypt | `encrypt_time_s`, `encrypt_time_cpu` |
| Serialize | `serialize_time_s`, `serialize_time_cpu` |
| Payload | `payload_bytes`, `n_params` |
| Pack info | `n_packs`, `packs_sent`, `pack_size`, `sigma_ckks` |
| HE params | `poly_modulus_degree`, `coeff_mod_bits`, `scale_bits`, `plain_modulus` |

Columns unused by a technique are emitted as `0`. `coeff_mod_bits` uses semicolons as separator (e.g. `60;40;40;60`) to avoid CSV breakage. `scale_bits` stores the exponent (e.g. `40` for `2^40`).

**`evaluate.csv`** (15 columns): `experiment_name`, `round`, `cid`, `solution`, `schema`, `model`, `dataset`, `distribution`, `dirichlet_alpha`, `eval_loss`, `eval_acc`, `decrypt_time_s`, `poly_modulus_degree`, `coeff_mod_bits`, `scale_bits`.

**Server logs** (in `logs/<LOGFOLDER>/`):
- `server_agg.csv`: `experiment_name`, `round`, `n_clients`, `total_examples`, `selection_time_s`, `aggregation_time_s`
- `server_eval.csv`: `experiment_name`, `round`, `n_clients`, `mean_loss`, `mean_acc`

### Model Compression

Per-round model snapshots are saved as `logs/<LOGFOLDER>/model/round_<N>.npz` using `np.savez_compressed` with float16 weights. This reduces storage by ~54% vs float32. Only client 0 saves snapshots. All rounds are kept (not just the last).

Loading saved weights:
```python
data = np.load("round_N.npz")
weights = [data[f'arr_{i}'] for i in range(len(data.files))]  # numeric order, NOT sorted()
```

### Models (`client/models.py`)

All factories follow `create_<name>(input_shape, num_classes, ...)` and return a compiled Keras model. Models are **dataset-agnostic** — input shape and num_classes are derived from the actual data at runtime.

| Factory | Arch | ~Params |
|---|---|---|
| `create_resnet20` | ResNet-20 | 270 k |
| `create_lenet5` | LeNet-5 | 44 k |
| `create_dnn` / `create_dnn_base` | DNN 4-layer | 309 k |
| `create_dnn_tiny` | DNN 3-layer | 111 k |
| `create_mobilenetv2(alpha=0.5)` | MobileNetV2 | 460 k |
| `create_vit_tiny` | ViT patch=4 embed=64 depth=6 | 206 k |
| `create_simple_cnn` | Simple CNN | — |

`create_mobilenetv2` handles grayscale inputs (auto-repeats channel to RGB) and inputs smaller than 32×32 (auto-resizes). `create_vit_tiny` requires `h % patch_size == 0`.

To select a model at runtime set `MODEL_ARCH` to the factory name without the `create_` prefix (e.g. `MODEL_ARCH=vit_tiny`). If unset, defaults to `dnn`.

### Important Constraints

- **Plaintext must run first** to generate and save initial model weights (`logs/models/init_weights/<model_name>/<dataset>/…`). All HE solutions load these weights at startup; they raise an exception if the file is absent.
- **CKKS parameter constraint**: `[30,60,30]` with `scale=2^60` is incompatible with scalar multiplication (see `tests/test_server_utils.py` xfail). Prefer `[60,40,40,60]` with `scale=2^40` for reliable scalar ops.
- **gRPC message size**: set to 1 GB (`MAX_MESSAGE_LENGTH = 1024*1024*1024`) in `client/client.py` to accommodate large ciphertexts.
- **ckks_backend.py must be kept in sync** across `encryption/`, `client/encryption/`, and `server/encryption/` — each Docker container only mounts its own directory.
- **`np.load` ordering**: use `[data[f'arr_{i}'] for i in range(len(data.files))]`, never `sorted(data.files)` — lexicographic sort breaks shape order for 10+ layers.
- Python version is **3.10** (see `pyproject.toml` and `Makefile`).

### Environment Variables (Docker / runtime)
Clients and server are configured entirely via environment variables: `CID`, `NCLIENTS`, `SOLUTION`, `DATASET`, `NIID`, `PERCENTAGE`, `TECHNIQUE`, `SERVER_IP`, `NUM_ROUNDS`, `LOGFOLDER`, `POLY`, `PLAIN_MODULUS`, `COEFFS`, `SCALE`, `KEY_TYPE`, `N_LENGTH`, `MODEL_ARCH`, `EXPERIMENT_NAME`, `CKKS_BACKEND`.

CKKS backend-specific vars (only read when the relevant backend is active): `SIMPLIFY_N`, `SIMPLIFY_BITS`, `SIMPLIFY_SCALE_BITS` (ckks_python_simplify); `PYCRYPT_N`, `PYCRYPT_BITS` (ckks_python); `OPENFHE_RING_DIM`, `OPENFHE_MULT_DEPTH`, `OPENFHE_SCALE_BITS` (openfhe).
