# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Scope

**IMPORTANT — literatura ignorada**: Ignore qualquer artigo ou referência que mencione a técnica **SOCAS** (pura ou combinada com outra técnica). Não sugerir, implementar, referenciar ou comparar com SOCAS em nenhuma circunstância.

## Project Overview

**Slidhe** is a research project implementing privacy-preserving Federated Learning (FL) using Homomorphic Encryption (HE). Clients train local ML models and send encrypted gradients to an aggregation server. The system supports multiple HE schemes (CKKS, BFV, Paillier/BatchCrypt) and packet selection strategies (top-k, round-robin, sliding window, pondering-random).

**Research focus (June–August 2026)**: impact of NVNL (round-robin) sparsification on communication overhead and time, compared against FedPHE (top-k), SW (sliding window), and RW (pondering-random). Culminates in dissertation correction and journal submission. See `docs/plano_pesquisa.md` for the full research plan prompt (Claude Opus).

## Commands

```bash
# Build Docker images (server and client) — required once before any experiment
make configure

# Build Singularity images from .def files (pulls base from Docker Hub)
make singularity-configure

# Build Singularity images from already-built Docker images (faster)
make singularity-from-docker

# Generate HE keys (required before running experiments)
make keys
# or directly:
python3 generate_keys.py

# Run the full experiment suite (the normal way to run experiments)
./start_all.sh

# Run a themed experiment batch (preferred for grouped experiments)
bash experiments/my_theme.sh

# Smoke-test ckks_python_simplify backend: Docker + Singularity, nvnl + fedphe,
# 2 clients, 3 rounds, FashionMNIST, LeNet-5 (run from repo root)
bash experiments/tests/test_ckks_backend.sh

# Estimate wall-clock time for an experiment script (local machine)
python3 experiments/estimate_time.py experiments/my_theme.sh
# Estimate for a specific cluster partition
python3 experiments/estimate_time.py experiments/my_theme.sh --machine h100
# Show estimates for all partitions
python3 experiments/estimate_time.py experiments/my_theme.sh --all

# Submit an experiment to the SLURM cluster
# 1. Copy and edit the template:
cp sbatch/template.sh sbatch/my_experiment.sh
# 2. Submit:
sbatch sbatch/my_experiment.sh
# Monitor:
squeue -u $USER
tail -f logs/slurm_<jobid>.out

# Submit one independent SLURM job per model:dataset pair (parallel on cluster)
# Each job runs the full solution/niid/percentage/seed sweep for its single pair.
# The original sbatch script is unchanged — direct sbatch still runs all pairs in one job.
bash sbatch/submit_per_pair.sh sbatch/sparsification_percentage/sbatch_sparsification_percentage_srr.sh --dry-run
bash sbatch/submit_per_pair.sh sbatch/sparsification_percentage/sbatch_sparsification_percentage_srr.sh
bash sbatch/submit_per_pair.sh sbatch/client_frac/sbatch_client_frac_rw.sh --dry-run
# Works for all sbatch/<tipo>/sbatch_*.sh scripts that set EXPERIMENT_SCRIPT
# When a theme sweeps >1 seed, the fan-out is pair × seed and job names gain a
# _seed<N> suffix (e.g. seed_experiment: 2 pairs × 2 seeds = 4 jobs per technique).

# Submit the full seed_experiment batch (4 techniques × 2 pairs × 2 seeds = 16 jobs)
for t in NVNL rw fedphe sw; do
  bash sbatch/submit_per_pair.sh sbatch/seed_experiment/sbatch_seed_experiment_$t.sh --dry-run
done

# ── Checkpoint / Resume ───────────────────────────────────────────────────────

# Find all interrupted experiments, generate resume scripts, and submit via sbatch
./resume_all.sh
./resume_all.sh --dry-run            # list what would be submitted (no files created)
./resume_all.sh --generate-only      # generate resume.sh + resume_sbatch.sh without submitting
./resume_all.sh --merge-only         # merge completed (original + resumed) pairs, no submission
./resume_all.sh --partition p100     # override SLURM partition (default: l40s)
./resume_all.sh --time 1-00:00:00    # override wall-time limit (default: 2-00:00:00)
./resume_all.sh --logs-dir /path     # use a non-default logs directory

# Generate resume scripts only (lower-level; resume_all.sh calls this internally)
python3 experiments/generate_resume_jobs.py
python3 experiments/generate_resume_jobs.py --dry-run

# Merge original + resumed experiment folders in-place (called automatically by resume.sh)
python3 experiments/merge_resume.py                      # interactive y/N prompt
python3 experiments/merge_resume.py --dry-run            # show what would be merged
python3 experiments/merge_resume.py --yes                # skip prompt (used in SLURM/batch)
python3 experiments/merge_resume.py --logs-dir /path     # custom logs directory

# End-to-end resume smoke test (Docker, ~5 min: plaintext seed → nvnl 1 round → nvnl 2 rounds)
bash experiments/tests/test_resume_e2e.sh

# Checkpoint/resume unit tests (no Docker required; uses existing logs)
bash tests/test_checkpoint_resume.sh

# Merge smoke tests (no Docker required; creates synthetic log structures)
bash tests/test_merge_resume.sh

# Audit experiment status: done / partial / missing / wrong backend
python3 experiments/check_status.py                          # show partial + missing only
python3 experiments/check_status.py --all                    # show everything including done
python3 experiments/check_status.py --summary-only           # global summary table only
python3 experiments/check_status.py --scripts sparsification_percentage_sw.sh
python3 experiments/check_status.py --no-color               # plain text (for logs)
python3 experiments/check_status.py --logs-dir /path/to/logs # custom logs dir

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_server_utils.py
pytest tests/test_model_reconstruction.py
pytest tests/test_topkrr.py
pytest tests/test_srr.py
pytest tests/test_check_status.py
bash tests/test_merge_resume.sh

# Test ckks_project installation inside Docker containers
make test-ckks

# Test ckks_project installation inside Singularity containers
make test-ckks-singularity

# Clean logs
make cleanlogs

# Generate graphs
make graph

# Sync to remote cluster
./sync_remote.sh

# ── Standalone benchmarks (no Flower, no HE) ─────────────────────────────────

# MobileNetV2 + CIFAR-10: run in Docker (all batch sizes [16,32,64,128], 200 epochs each)
bash benchmark/run_docker.sh
bash benchmark/run_docker.sh --batch-size 32   # single batch size
bash benchmark/run_docker.sh --gpu 1           # select GPU
bash benchmark/run_docker.sh --no-gpu          # CPU mode

# ViT-Tiny + GTSRB: run in Docker
bash benchmark/run_docker_vit.sh
bash benchmark/run_docker_vit.sh --batch-size 64
bash benchmark/run_docker_vit.sh --gpu 1

# Submit MobileNetV2 benchmark to SLURM (job array: one task per batch size)
sbatch benchmark/sbatch_mobilenetv2.sh
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

**Solution parameter** controls the entire encryption/aggregation path. Valid values: `ckks`, `bfv`, `batchcrypt`, `fedphe`, `nvnl`, `fedphe-scout`, `nvnl-scout`, `plaintext`, `sw`, `rw`, `topkrr`, `srr`. Both client and server call `config_solution()` to set `homomorphic`, `packing`, `onlysum`/`only_sum`, and `homomorphic_type` from this string.

**Packing vs. no packing**: When `packing=True`, model weights are flattened and chunked into CKKS vectors (packs) of size `poly_modulus_degree // 2`. Only a subset of packs is encrypted and transmitted, selected by a mask (top-k, round-robin, sliding window, or pondering-random). The server aggregates packs homomorphically and sends the aggregated packs+mask back to clients. When `packing=False`, all weights are concatenated into a single ciphertext.

**CKKS backend abstraction**: `encryption/ckks_backend.py` (mirrored to `client/encryption/` and `server/encryption/`) defines a `CKKSBackend` ABC with methods `generate_and_save`, `load_client_context`, `load_server_context`, `encrypt`, `serialize`, `deserialize`, `add(ctx, ct1, ct2)`, `mul(ctx, ct, scalar)`, `decrypt`. Concrete backends:
- `TenSEALBackend` (default) — context files `nvnlc2.pkl` / `serverc2.pkl`
- `CKKSSimplifyBackend` — `ckks_python_simplify` lib; saves raw key bytes instead of pickling the C++ context: `ckks_simplify_params.pkl` (N/q/scale), `ckks_simplify_sk.bin`, `ckks_simplify_pk.bin`. Reconstructs `CKKSContext` via `__new__` on load to avoid pickling pybind11 objects.
- `CKKSPythonBackend` — user's verbose `ckks_python` lib; files `ckks_python_sk.bin`, `ckks_python_pk.bin`, `ckks_python_params.pkl`
- `OpenFHEBackend` — files `openfhe_cc.bin`, `openfhe_sk.bin`, `openfhe_pk.bin`

Select backend via `CKKS_BACKEND` env var (set in `start_all.sh`). `get_backend(name)` is the factory, cached per process.

**Host setup for `ckks_python_simplify`**: `generate_keys.py` runs on the host (not inside a container), so the library must also be installed locally. After `pip install ckks_project/` (which installs `ckks_federated` and `ckks_python`), create the alias symlink:
```bash
python3 -c "import site, os; sp=site.getsitepackages()[0]; os.symlink(f'{sp}/ckks_federated', f'{sp}/ckks_python_simplify')"
# or for user installs (~/.local):
ln -s ~/.local/lib/python3.10/site-packages/ckks_federated \
      ~/.local/lib/python3.10/site-packages/ckks_python_simplify
```

**Context files** live in `context/`. CKKS keys managed by the backend (see above). BFV keys: `bfv_secret.pkl` and `bfv_server_key.pkl`. Paillier: `paillier.pkl`.

### Module Structure

- `client/client.py` — `HEClient`: reads `MODEL_ARCH` env var to select model; falls back to `dnn` if unset; dispatches per-round training to `fit_*` functions in `client/literature/`
- `client/literature/` — one `fit_<solution>.py` per HE scheme; each implements local train → encrypt → return fit message with granular per-operation timing
- `client/literature/set_parameters.py` — decryption and model weight reconstruction on the client side; saves per-round compressed model snapshots (`np.savez_compressed` + float16, cid==0 only). `he_packs_to_model` / `he_parameters_to_model` time only the model-manipulation work (deserialize + decrypt + reshape) into `self.last_decrypt_clock` / `self.last_decrypt_cpu`; the snapshot + checkpoint `np.savez_compressed` writes run **after** the timer stops, so they are excluded from `decrypt_time_s`. Each `fit_*.py` (and `client.py::evaluate()`) reads `self.last_decrypt_*` rather than wrapping the call in its own timer.
- `client/literature/fedphe.py` — top-k packing; also contains mask helpers: `get_topk_mask`, `get_robin_round_mask`, `get_slide_window_mask`, `get_pondering_random_mask`, `get_spread_rr_mask`
- `client/literature/NVNL.py` — round-robin packing
- `client/literature/sw.py` — sliding-window packing (stride=0.5)
- `client/literature/rw.py` — pondering-random packing
- `client/literature/topkrr.py` — hybrid packing: top-k for rounds `< RR_START_ROUND`, then round-robin for remaining rounds; imports `get_topk_mask` and `get_robin_round_mask` from `fedphe.py`; RR offset starts at 0 on the first RR round so window begins at pack index 0
- `client/literature/srr.py` — spread round-robin packing: k=ceil(n*pct) packs selected per round, evenly spaced with stride=n//k; offset advances by 1 each round; uses `get_spread_rr_mask` from `fedphe.py`; no extra env vars (uses `PERCENTAGE` like NVNL)
- `client/literature/fedphe_scout.py`, `NVNL_scout.py` — scout variants with custom TF training loop (trust-region + gating vs sigma + EMA blending)
- `client/client_logs.py` — fixed-schema CSV logging (41-column `train.csv`, 15-column `evaluate.csv`)
- `client/client_utils.py` — `packing`, `cypher_packs`, `decypher_packs`, `flat_packs`, `reshape_model`
- `client/models.py` — model factories (see Models section below)
- `server/server.py` — `HEServer`: `aggregate_fit` dispatches to packing vs. non-packing path; calls `write_agg_logs`
- `server/server_logs.py` — server-side CSV logging: `write_agg_logs` (aggregation timing) and `write_eval_logs` (round evaluation)
- `server/server_utils.py` — HE aggregation helpers: `align_packs`, `aggregate_packs_and_masks`, `aggregate_packs_and_mask`, `get_pack_mask_pair`, `get_cyphered_parameters`, `aggregated_cyphered_parameters`
- `encryption/ckks_backend.py` — CKKS backend abstraction (mirrored to `client/encryption/` and `server/encryption/`)
- `encryption/` and `server/encryption/` and `client/encryption/` — largely mirrored; `encrypt.py` defines base `Encrypt` class; `ckks.py`, `bfv.py`, `paillier.py` implement scheme-specific logic; `quantize.py` / `quantize_batchcrypt.py` handle quantization
- `sync_remote.sh` — rsync to `yuri.faria@ssh.recod.ic.unicamp.br:~/keys` (excludes `.venv/`, `logs/`, `logs/**`, `__pycache__`, `*.csv`, `.git/`); `logs/**` is excluded in addition to `logs/` to prevent `--delete` from removing experiment results inside `logs/` on the cluster
- `resume_all.sh` — top-level resume launcher: runs `generate_resume_jobs.py`, submits all generated `resume_sbatch.sh` files via `sbatch`; flags: `--dry-run`, `--generate-only`, `--merge-only`, `--partition`, `--time`, `--logs-dir`; `--merge-only` skips submission and runs `merge_resume.py` interactively
- `experiments/lib.sh` — auto-dispatcher: sources `lib_slurm.sh` when `SLURM_JOB_ID` is set, otherwise sources `lib_tmux.sh`; experiment scripts always `source lib.sh` and call `run_theme` — no changes needed per environment
- `experiments/lib_tmux.sh` — tmux backend (local/Docker/Singularity): launches server as tmux session `SERVER` and one session per client; waits for server session to exit, then kills client sessions; contains `_find_resume_checkpoint` and `_write_incomplete_log` helpers
- `experiments/lib_slurm.sh` — SLURM backend: server + clients run as background processes (no tmux); GPU_IDS derived automatically from `SLURM_GPUS_ON_NODE`; waits for server PID, then kills client PIDs and waits for port 9999 to free; contains same `_find_resume_checkpoint`, `_write_incomplete_log` helpers as lib_tmux.sh plus `JOB_DEADLINE_EPOCH` computation
- `experiments/generate_resume_jobs.py` — scans `logs/` for `INCOMPLETE.log` files, checks if already completed (skips), deduplicates by config fingerprint (picks best checkpoint), generates `resume.sh` + `resume_sbatch.sh` inside each interrupted experiment folder; the generated `resume.sh` automatically calls `merge_resume.py --yes` after `run_theme` so the merge happens at the end of every resumed experiment; outputs sbatch paths to stdout, info to stderr
- `experiments/merge_resume.py` — merges original + resumed experiment folders in-place: appends CSV rows from all resumed folders into the original, copies model snapshots, replaces checkpoint with the latest, updates `experiment_details.json` with `MERGED_FROM`/`MERGED_AT`, removes `INCOMPLETE.log`; archives all non-original folders to `<parent>/archive/` (safety backup of original also kept there); handles chains of N resumes; skips already-merged originals (idempotent); flags: `--dry-run`, `--yes`/`--auto`, `--logs-dir`
- `experiments/tests/test_resume_e2e.sh` — end-to-end Docker resume smoke test: plaintext seed → nvnl 1-round (creates checkpoint) → nvnl 2-round (detects checkpoint, resumes); verifies experiment_details.json, server_eval.csv round numbering, `[RESUME]` in logs, checkpoint updated
- `experiments/sparsification_percentage/sparsification_percentage_topkrr.sh` — topkrr experiment; same sweep as other sparsification scripts; sets `RR_START_ROUND=100` (switches at round 100 out of 200)
- `sbatch/sparsification_percentage/sbatch_sparsification_percentage_topkrr.sh` — SLURM job for the topkrr sweep (l40s, 3 GPUs, 9 clients)
- `experiments/sparsification_percentage/sparsification_percentage_srr.sh` — srr experiment; same sweep as other sparsification scripts
- `sbatch/sparsification_percentage/sbatch_sparsification_percentage_srr.sh` — SLURM job for the srr sweep (l40s, 3 GPUs, 9 clients)
- `experiments/client_frac/client_frac_{fedphe,NVNL,sw,rw,srr,topkrr,plaintext}.sh` — client participation fraction sweep for all 7 solutions; `FOLDER="client_frac"` (shared); `CLIENT_FRACTIONS=(0.25 0.5 0.75)`, `PERCENTAGES=(0.1 0.5)`, `NIIDS=(0.1 1)`, resnet20+mobilenetv2, 20 clients/GPU × 3 GPUs = 60 clients; topkrr sets `RR_START_ROUND=100`; plaintext runs first (generates init_weights, 12 experiments); packing solutions run 24 experiments each
- `sbatch/client_frac/sbatch_client_frac_{fedphe,NVNL,sw,rw,srr,topkrr,plaintext}.sh` — SLURM jobs for client_frac sweep (l40s, 3 GPUs, 32 CPUs, 4-day wall time, `--mem=0`)
- `experiments/seed_experiment/seed_experiment_{NVNL,rw,fedphe,sw}.sh` — multi-seed repetition of the sparsification sweep so results can be reported as mean ± spread; `FOLDER="seed_experiment"` (shared), `SEEDS=(43 44)`, `PERCENTAGES=(0.1 0.3 0.5 0.7 0.9)`, `NIIDS=(0.1 1)`, resnet20+mobilenetv2 on CIFAR10, 3 clients/GPU × 3 GPUs = 9 clients, 200 rounds. Config is otherwise identical to `sparsification_percentage`, so seed-43/44 runs stay directly comparable to the existing seed-42 ones. 40 experiments per technique, 160 total. **No plaintext script** — since `client.py` now lets any solution create `init_weights`, the techniques bootstrap their own; analyses needing a plaintext baseline must source it from `logs/sparsification_percentage/`.
- `sbatch/seed_experiment/sbatch_seed_experiment_{NVNL,rw,fedphe,sw}.sh` — SLURM jobs for the seed_experiment sweep (l40s, 3 GPUs, 12 CPUs, `--mem=64G`, 2-day wall time). Submit via `submit_per_pair.sh` → 4 jobs per technique (pair × seed), 10 experiments each.
- `experiments/standalone/template.sh` — copy this to create a new themed batch; uses model↔dataset pairs format
- `experiments/tests/test_ckks_backend.sh` — smoke-test for the `ckks_python_simplify` backend (3 phases: plaintext seed → Docker HE → Singularity HE)
- `tests/test_checkpoint_resume.sh` — unit tests for checkpoint/resume logic (6 tests): verifies checkpoint files present, `_find_resume_checkpoint` for completed/incomplete runs, Python round offset math, `lib_slurm.sh` path, artificial partial checkpoint selection
- `tests/test_merge_resume.sh` — smoke tests for `merge_resume.py` (31 tests, no Docker): dry-run, basic merge (2+2 rounds), CSV row counts, model snapshots, checkpoint update, `experiment_details.json` metadata, archive behaviour, idempotency, 3-folder chain merge, conflict detection
- `experiments/estimate_time.py` — estimates wall-clock time for an experiment script by parsing its config variables and applying per-machine speed factors; supports `--machine <partition>`, `--all`, and `--margin <factor>`; machine table sourced from `SLURM.md`
- `experiments/ESTIMATION_METHODOLOGY.md` — documents the time estimation model (GPU-bound train time, CPU-bound HE time, scaling formulas)
- `sbatch/standalone/template.sh` — SLURM job template; copy and edit `--partition`, `--gres`, `--time`, and `EXPERIMENT_SCRIPT` then submit with `sbatch`
- `sbatch/submit_per_pair.sh` — launcher that reads `EXPERIMENT_SCRIPT` from any `sbatch/<tipo>/sbatch_*.sh`, enumerates its `MODEL_DATASET_PAIRS`, and submits one independent SLURM job per pair (via `--export=ALL,MODEL_DATASET_PAIR=<pair>`); supports `--dry-run`; backward-compat: direct `sbatch sbatch/<tipo>/sbatch_*.sh` still runs all pairs in one job
- `sbatch/<tipo>/` — SLURM job scripts grouped by experiment type: `client_frac/`, `sparsification_percentage/`, `standalone/`
- `SLURM.md` — cluster partition reference table: partition name, nodes, GPU model, max wall time, GPU Speed factor, CPU Speed factor (used by `estimate_time.py`)
- `benchmark/train_mobilenetv2_cifar10.py` — standalone MobileNetV2 + CIFAR-10 benchmark; loads CIFAR-10 from `benchmark/cifar-10-python.tar.gz` (local) or falls back to `tf.keras` download; trains 200 epochs; accepts `--batch-size N`; results in `benchmark/results/`
- `benchmark/train_vit_gtsrb.py` — standalone ViT-Tiny + GTSRB benchmark; loads `tanganke/gtsrb` via HuggingFace `datasets`, cached in `benchmark/data/`; resizes images to 32×32; 43 classes; results in `benchmark/results_vit_gtsrb/`
- `benchmark/run_docker.sh` — runs MobileNetV2 benchmark in `flhe:client` container; mounts `benchmark/` and `client/`; flags: `--batch-size N`, `--gpu ID`, `--no-gpu`, `--dry-run`; pre-creates `results/` with `chmod 777` to avoid root-ownership issues
- `benchmark/run_docker_vit.sh` — same as above for ViT-Tiny + GTSRB; also mounts `benchmark/data/` as HuggingFace cache (`HF_DATASETS_CACHE`)
- `benchmark/sbatch_mobilenetv2.sh` — SLURM job array (`--array=0-3`) on `l40s`; maps task ID → batch size `[16,32,64,128]`; one GPU per task; results in `benchmark/results/`
- `benchmark/cifar-10-python.tar.gz` — local copy of the official CIFAR-10 Python dataset; loaded directly without extraction via `tarfile`+`pickle`
- `tests/test_topkrr.py` — unit tests for the `topkrr` mask logic (46 tests, no GPU required): top-k phase, round-robin phase, phase transition at `RR_START_ROUND`, coverage guarantee, edge cases
- `tests/test_srr.py` — unit tests for the `srr` mask logic (34 tests, no GPU required): count, binary mask, uniform spacing, offset advance, coverage (when n%k==0), wrap-around, disjoint windows, edge cases
- `experiments/check_status.py` — experiment audit tool: discovers all `experiments/**/*.sh` scripts (recursive) that call `run_theme`, enumerates every expected experiment, scans `logs/` for matching folders, and reports status per experiment; statuses: `✅ done`, `⚠️ partial`, `❌ missing`, `🔀 wrong backend` (CKKS_BACKEND in `experiment_details.json` doesn't match the script's `CKKS_BACKEND`); default shows only partial/missing — use `--all` to include done; columns: Solution, Model, Dataset, niid, %pct, seed, clients, poly, coeffs, scale, backend, Status, Rounds; `classify()` returns `(status, max_round, logfolder, found_backend)` 4-tuple; log matching uses `*clients` glob wildcard to tolerate client-count drift; backward-compat: folders without `experiment_details.json` are accepted regardless of backend
- `tests/test_check_status.py` — 44 unit tests for `check_status.py` (no Docker required): `TestParseBashScript` (10), `TestEnumerateExpected` (5), `TestGetMaxRound` (5), `TestClassify` (10, includes backend-verification cases), `TestDiscoverScripts` (7), `TestCLI` (7)

### CSV Log Schema & Model Compression

See `docs/log_schema.md` for the full column list, `decrypt_time_s` scope note, model snapshot path convention, and payload bytes caveat.

Key constraints (also in Important Constraints below):
- `decrypt_time_s` bundles deserialize+decrypt+reshape — not comparable to `encrypt_time_s`.
- Model save path inside containers: always `f'logs/{self.log_folder}/model'` (Docker CWD is `/`).
- `np.load` ordering: `[data[f'arr_{i}'] for i in range(len(data.files))]`, never `sorted(data.files)`.

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

`create_mobilenetv2` handles grayscale inputs (auto-repeats channel to RGB) and inputs smaller than 96×96 (auto-resizes to 96×96). At 32×32 (e.g. CIFAR10) the stride-2 layers collapse feature maps to 1×1 before GAP, preventing learning — the 96×96 threshold ensures a 3×3 final feature map. `create_vit_tiny` requires `h % patch_size == 0`.

To select a model at runtime set `MODEL_ARCH` to the factory name without the `create_` prefix (e.g. `MODEL_ARCH=vit_tiny`). If unset, defaults to `dnn`.

---

## Analysis Studies

**Convention**: `studies/` is only for generating graphs and tables — it never runs experiments.

- **`studies/general/`** — generic analysis suite (CÓDIGO): `01_*.py` … `23_*.py`, `load_data.py`, `run_all.sh`. The same suite works for every experiment type (auto-discovers models, datasets, solutions from logs).
- **`studies/<nome>/`** — OUTPUT for each experiment type: `figures/`, `tables/`, `notebooks/`. Created automatically by `run_all.sh`.

See `docs/analysis_studies.md` for per-script details (scripts 10–23, `load_data.py` API, coverage formulas).

**Comparison rule**: always compare techniques at the same sparsification percentage. Plaintext baseline must appear in every percentage-specific figure. Never mix percentages across techniques in the same plot.

```bash
# Run all analysis scripts for an experiment type (reads logs/<nome>/, writes studies/<nome>/)
bash studies/general/run_all.sh sparsification_percentage
bash studies/general/run_all.sh client_frac
# or from studies/general/:
cd studies/general && ./run_all.sh client_frac

# Pass a custom logs dir:
bash studies/general/run_all.sh client_frac --logs-dir /path/to/logs/client_frac

# Run individual scripts (LOGS_DIR env var sets the logs source):
cd studies/general
LOGS_DIR=../../logs/client_frac python3 01_accuracy_curves.py  # convergence grid + final-accuracy bar
LOGS_DIR=../../logs/client_frac python3 02_solution_comparison.py   # heatmap + grouped bar across solutions
LOGS_DIR=../../logs/client_frac python3 03_overhead_breakdown.py    # timing breakdown (encrypt/decrypt/train)
LOGS_DIR=../../logs/client_frac python3 04_communication.py         # payload per round
LOGS_DIR=../../logs/client_frac python3 05_niid_impact.py           # paired bar + delta heatmap per niid
LOGS_DIR=../../logs/client_frac python3 06_memory_profile.py        # RAM usage over rounds
LOGS_DIR=../../logs/client_frac python3 07_convergence_speed.py     # rounds to threshold + AUC bar
LOGS_DIR=../../logs/client_frac python3 08_efficiency_frontier.py   # Pareto scatter: accuracy vs payload/round
LOGS_DIR=../../logs/client_frac python3 09_summary_table.py         # CSV + LaTeX summary table
LOGS_DIR=../../logs/client_frac python3 10_total_data_convergence.py  # full model coverage rounds (NVNL vs SW)
LOGS_DIR=../../logs/client_frac python3 13_overhead_vs_accuracy.py    # communication time overhead vs accuracy + stacked bar
LOGS_DIR=../../logs/client_frac python3 14_neurons_vs_accuracy.py     # cumulative neurons transmitted vs accuracy + neurons/round bar
LOGS_DIR=../../logs/client_frac python3 15_bytes_vs_maxacc.py         # bytes vs accuracy, techniques only, best-technique max-acc target
LOGS_DIR=../../logs/client_frac python3 16_overhead_vs_maxacc.py      # comm overhead vs accuracy, techniques only, best-technique max-acc target
LOGS_DIR=../../logs/client_frac python3 17_coverage_unique.py         # observed unique-pack coverage per round (sawtooth, no duplicate counting)
LOGS_DIR=../../logs/client_frac python3 18_coverage_volume.py         # observed volume-based coverage per round (sawtooth, with duplicate counting)
LOGS_DIR=../../logs/client_frac python3 19_coverage_accuracy.py       # coverage (study 17) + accuracy per round on dual Y axes
LOGS_DIR=../../logs/client_frac python3 20_pack_frequency.py          # how often each pack index was transmitted (horizontal bar)
LOGS_DIR=../../logs/client_frac python3 21_layer_frequency.py         # how often each model layer was transmitted (horizontal bar, needs snapshots)
LOGS_DIR=../../logs/client_frac python3 22_layer_heatmap.py           # 2D heatmap: Y=layer, X=percentage, color=send_pct; scales for large models
LOGS_DIR=../../logs/client_frac python3 23_pack_heatmap.py            # 2D heatmap: Y=pack index, X=percentage, color=send_pct (like study 22 but for packs)
```

### Checkpoint / Resume System

Every experiment saves rolling checkpoints so interrupted runs (SLURM wall-time, SIGKILL, graceful time-limit shutdown) can be resumed automatically.

**Files saved per round** (inside `logs/<LOGFOLDER>/checkpoint/`):

| File | Written by | Content |
|---|---|---|
| `client_{cid}.npz` | each client after every round | float16 model weights for that client |
| `server_agg.npz` | server after `aggregate_fit` (plaintext only) | float16 aggregated weights |
| `meta.json` | server after `aggregate_fit` | `{"round": N, "solution_type": "..."}` |

**When an experiment stops before completing all rounds**, the shell `_write_incomplete_log()` helper (present in `lib_tmux.sh`, `lib_slurm.sh`, `start_all.sh`) writes `logs/<LOGFOLDER>/INCOMPLETE.log` with timestamp, round/total, remaining, and all config fields. The Python server also writes it proactively before `sys.exit(0)` in the graceful time-limit shutdown path.

**Automatic resume detection**: before launching each experiment, `run_theme` calls `_find_resume_checkpoint`. It globs all folders with the same config (any `NUM_ROUNDS`, any timestamp) and picks the one with the highest incomplete checkpoint round (i.e. `round < NUM_ROUNDS`). If found, sets `RESUME_LOGFOLDER` and `RESUME_ROUND`; both are propagated to server and clients and recorded in `experiment_details.json` as `RESUMED_FROM` / `RESUME_ROUND`.

**Resume behaviour by solution type**:
- *HE (packing + non-packing)*: clients load `checkpoint/client_{cid}.npz` on startup. Server starts with `agg_parameters=''`, so round 1 sends an empty `config['he']`; clients skip HE decryption (guard: `if len(config['he']) > 0`) and train directly on checkpoint weights. Rounds continue from `start_round_offset + 1`.
- *Plaintext*: server loads `checkpoint/server_agg.npz` and passes it as `initial_parameters` to FedAvg so clients receive the correct weights in round 1.

**Round offset**: `server.start_round_offset = meta['round']`. All `config['round']` values sent to clients and all CSV rows use `server_round + start_round_offset` so global round numbers are continuous across resume boundaries.

**Graceful time-limit shutdown** (SLURM only): `lib_slurm.sh` computes `JOB_DEADLINE_EPOCH` via `squeue` and passes it to the server container. In `configure_fit`, if `_time_remaining() < CHECKPOINT_THRESHOLD_SECS` (default 300 s), the server writes `INCOMPLETE.log` and calls `sys.exit(0)`. The checkpoint from the previous round is already saved.

**Generating resume jobs**: `experiments/generate_resume_jobs.py` scans `logs/` for `INCOMPLETE.log` files, skips any whose config already has a completed run (`meta.json round >= NUM_ROUNDS`), deduplicates to the best checkpoint per config, and writes `resume.sh` + `resume_sbatch.sh` inside each interrupted experiment folder. The generated `resume.sh` always calls `python3 experiments/merge_resume.py --yes` after `run_theme` so the merge is automatic. `resume_all.sh` calls the generator and submits all output sbatch files.

**Post-resume merge**: when a resumed experiment finishes, `experiments/merge_resume.py` merges the two folders **in-place** into the original: CSV rows from the resumed folder are appended and sorted by round, `model/round_*.npz` snapshots are copied, the checkpoint is replaced with the latest one, `experiment_details.json` gains `MERGED_FROM` / `MERGED_AT`, and `INCOMPLETE.log` is deleted. The resumed folder(s) are **moved to `<parent>/archive/`**; a safety backup of the original is also placed there before any modification. Running the script a second time is a no-op (idempotent). To merge manually after experiments already ran: `./resume_all.sh --merge-only` (interactive) or `python3 experiments/merge_resume.py --dry-run` (preview only).

### Convenção: experiment_details.json

Toda env var de experimento nova adicionada ao `run_client.sh` **deve** ser incluída também no bloco `experiment_details.json` em `experiments/lib_tmux.sh` e `experiments/lib_slurm.sh`. Variáveis de infraestrutura (`HF_*`, `TF_*`, `TORCH_*`, `CUDA_*`, `CID`, `LOGFOLDER`) são exceção — não pertencem ao JSON.

`experiment_details.json` now also contains `RESUMED_FROM` (relative path of the source checkpoint folder, empty string if fresh start) and `RESUME_ROUND` (checkpoint round loaded, 0 if fresh).

### Important Constraints

- **Initial model weights** live at `logs/models/init_weights/<model_name>/<dataset>/seed<N>/init_weights_<cid>.weights.h5` — the path is **seed-scoped**, so each seed gets its own init. Every solution loads them at startup and **any solution may create them** if absent (`client/client.py`), so plaintext no longer has to run first. Creation uses a per-process temp file + `os.replace`, because several jobs for the same (model, dataset, seed) can race: each client is a fresh process that calls `tf.random.set_seed(seed)` before building the model, so concurrent creators write bit-identical bytes (verified for resnet20 across two processes). Consequence: forgetting to run plaintext is no longer a loud failure — a run that self-inits looks the same as one that loaded plaintext-generated weights.
- **CKKS parameter constraint**: `[30,60,30]` with `scale=2^60` is incompatible with scalar multiplication (see `tests/test_server_utils.py` xfail). Prefer `[60,40,40,60]` with `scale=2^40` for reliable scalar ops.
- **gRPC message size**: set to 1 GB (`MAX_MESSAGE_LENGTH = 1024*1024*1024`) in `client/client.py` to accommodate large ciphertexts.
- **ckks_backend.py must be kept in sync** across `encryption/`, `client/encryption/`, and `server/encryption/` — each Docker container only mounts its own directory.
- **`np.load` ordering**: use `[data[f'arr_{i}'] for i in range(len(data.files))]`, never `sorted(data.files)` — lexicographic sort breaks shape order for 10+ layers.
- **Model save path inside containers**: Docker CWD is `/`; logs volume is mounted at `/logs`. Always use `f'logs/{self.log_folder}/model'` — never `f'{self.log_folder}/model'`. The latter writes to ephemeral container storage and the file is lost when the container exits. (`set_parameters.py` and `plaintext.py`)
- Python version is **3.10** (see `pyproject.toml` and `Makefile`).
- **GPU access in Docker**: `run_client.sh` uses `--gpus="device=$GPU_ID"`, which requires the `nvidia-persistenced` daemon to be running (`sudo systemctl start nvidia-persistenced`). Set `USE_GPU=false` to skip the GPU flag and run clients in CPU mode — useful for local smoke-tests or machines without a working nvidia-persistenced socket. `lib.sh` and `start_all.sh` propagate this variable to all client tmux sessions.
- **Singularity images** (`flhe_client.sif`, `flhe_server.sif`) are built from Docker images via `make singularity-from-docker` or from scratch via `make singularity-configure`. Both `.sif` files must exist at `SIF_DIR` (default: repo root) before running `RUNTIME=singularity` experiments. Singularity clients connect to `localhost:9999` by default (no Docker bridge needed).

### Environment Variables (Docker / runtime)
Clients and server are configured entirely via environment variables: `CID`, `NCLIENTS`, `SOLUTION`, `DATASET`, `NIID`, `PERCENTAGE`, `TECHNIQUE`, `SERVER_IP`, `NUM_ROUNDS`, `LOGFOLDER`, `POLY`, `PLAIN_MODULUS`, `COEFFS`, `SCALE`, `KEY_TYPE`, `N_LENGTH`, `MODEL_ARCH`, `EXPERIMENT_NAME`, `CKKS_BACKEND`, `RR_START_ROUND`.

CKKS backend-specific vars (only read when the relevant backend is active): `SIMPLIFY_N`, `SIMPLIFY_BITS`, `SIMPLIFY_SCALE_BITS` (ckks_python_simplify); `PYCRYPT_N`, `PYCRYPT_BITS` (ckks_python); `OPENFHE_RING_DIM`, `OPENFHE_MULT_DEPTH`, `OPENFHE_SCALE_BITS` (openfhe).

`USE_GPU` (optional, default `true`): set to `false` in `run_client.sh` to omit the `--gpus` flag — clients run in CPU mode. Automatically set by `experiments/tests/test_ckks_backend.sh` when `/run/nvidia-persistenced/socket` is absent.

`BATCH_SIZE` (optional, default `4`): training batch size passed to `model.fit()` in all `fit_*.py` files. Set to `32` or higher when using MobileNetV2 — BatchNorm is unstable with batch_size=4.

`CLIENT_FRACTION` (optional, default `1.0`, **server-only**): fraction of available clients selected per round for both training (`fraction_fit`) and evaluation (`fraction_evaluate`). Supports `0.5` (50%) and `0.25` (25%). Sampling is random and independent between fit and evaluate each round (standard Flower `client_manager.sample` behaviour). The server still waits for all `NCLIENTS` to connect before starting (`min_available_clients = NCLIENTS`), but only `round(NCLIENTS * CLIENT_FRACTION)` are sampled per round. **Default 1.0 = all clients**, reproducing the behaviour of all existing `sparsification_percentage` experiments. When `CLIENT_FRACTION` ≠ 1.0, the LOGFOLDER gets a `cf<frac>_` token between `{KEY_TYPE}_` and the timestamp, so partial-participation runs are stored separately and `_find_resume_checkpoint` cannot cross-resume with full-participation runs. Propagated by `lib_tmux.sh`, `lib_slurm.sh`, `run_server.sh`; logged in `experiment_details.json`. Set in the experiment script or as an env prefix: `CLIENT_FRACTION=0.5 bash experiments/foo.sh`.

`CLIENT_FRACTIONS` (array, sweep dimension): bash array defined in themed experiment scripts to sweep `CLIENT_FRACTION` across multiple values in a single run, analogous to `PERCENTAGES`. `run_theme()` in `lib_tmux.sh`/`lib_slurm.sh` defaults to `(1.0)` when the array is absent (backward-compat: all existing scripts run as before). Example: `CLIENT_FRACTIONS=(0.25 0.5 0.75)` in `client_frac_*.sh` scripts. Each value in the array becomes the scalar `CLIENT_FRACTION` for one iteration of the inner loop. `check_status.py` reads `CLIENT_FRACTIONS` (array) when enumerating expected experiments; old scripts with scalar `CLIENT_FRACTION=<val>` are mapped to `CLIENT_FRACTIONS=(<val>)` automatically.

`RR_START_ROUND` (optional, default `200`, `topkrr` only): the FL round at which the mask strategy switches from top-k to round-robin. Rounds `< RR_START_ROUND` use top-k by magnitude; rounds `>= RR_START_ROUND` use round-robin with window offset zeroed at the transition. Set in the experiment script (e.g. `RR_START_ROUND=100` for a 200-round run). Propagated by `lib_tmux.sh`, `lib_slurm.sh`, and `run_client.sh`; logged in `experiment_details.json`.

`RESUME_LOGFOLDER` (optional): path relative to `logs/` of the experiment folder whose checkpoint to resume from. Set automatically by `_find_resume_checkpoint` in `run_theme`; passed to both server and client containers. When set, the server loads `checkpoint/meta.json` to compute `start_round_offset` and (for plaintext) `checkpoint/server_agg.npz` as `initial_parameters`; each client loads `checkpoint/client_{cid}.npz` as its starting weights.

`JOB_DEADLINE_EPOCH` (optional, default `0`): Unix timestamp of the SLURM job's wall-time deadline. Computed in `lib_slurm.sh` via `squeue`. Passed to the server container; used by `_time_remaining()` to trigger graceful shutdown before the kill signal.

`CHECKPOINT_THRESHOLD_SECS` (optional, default `300`): seconds before `JOB_DEADLINE_EPOCH` at which the server stops gracefully (writes `INCOMPLETE.log` and exits). Configurable per experiment script.

`MODEL_DATASET_PAIR` (optional): when set in the environment of a themed experiment script (e.g. `MODEL_DATASET_PAIR=lenet5:FASHION_MNIST bash experiments/foo.sh`), `run_theme()` replaces the script's `MODEL_DATASET_PAIRS` array with this single entry, restricting the run to that one pair. All other sweeps (solution, niid, percentage, seed) are unaffected. Set automatically by `sbatch/submit_per_pair.sh` via `--export`.

`PRINT_PAIRS_ONLY` (optional): when set (any non-empty value), `run_theme()` prints the (possibly `MODEL_DATASET_PAIR`-overridden) pairs to stdout and returns immediately — no tmux, no docker, no key generation. Used internally by `sbatch/submit_per_pair.sh` to enumerate pairs on the login node before submitting jobs.
