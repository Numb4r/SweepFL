# SweepFL

SweepFL runs privacy-preserving federated learning (FL) with homomorphic encryption (HE): clients train local
models and send encrypted updates to an aggregation server. It compares full-model references (plaintext FedAvg,
CKKS, BFV, BatchCrypt) with four partial packet-selection techniques (Top-k, Random, Round Robin, Sliding Window).

The analysis lives in `studies/journal/`: the scripts that produce the manuscript figures and tables. This README
covers how to run the experiments on SLURM, how to generate the figures, which logs those scripts read, and which
techniques they handle. `logs/` keeps only what they open (`studies/journal/used_logs.txt`).

Contents: [1. SLURM](#1-run-the-experiments-on-slurm) · [2. Figures and tables](#2-generate-figures-and-tables) ·
[3. Logs in use](#3-logs-in-use) · [4. Techniques](#4-available-techniques)

## 1. Run the experiments on SLURM

SLURM is the main way to run experiments. Each `sbatch/<type>/sbatch_*.sh` calls an `experiments/<type>/*.sh`
(variable `EXPERIMENT_SCRIPT`), which uses `lib_slurm.sh`. Jobs run under Singularity (`RUNTIME=singularity`,
`SIF_DIR` defaults to `~/keys`).

```bash
# 1. images: build with Docker locally, then convert to .sif
#    (flhe_client.sif and flhe_server.sif must be in SIF_DIR on the cluster)
make configure
make singularity-from-docker

# 2. send the repository to the cluster (rsync to ~/keys; excludes logs/)
./sync_remote.sh

# 3. on the cluster, from ~/keys, submit one job per (model:dataset pair x seed):
bash sbatch/submit_per_pair.sh sbatch/sparsification_percentage/sbatch_sparsification_percentage_NVNL.sh --dry-run
bash sbatch/submit_per_pair.sh sbatch/sparsification_percentage/sbatch_sparsification_percentage_NVNL.sh
# or a single job that runs all pairs in sequence:
sbatch sbatch/sparsification_percentage/sbatch_sparsification_percentage_NVNL.sh

# 4. monitor
squeue -u $USER
tail -f logs/slurm_<jobid>.out

# 5. runs cut by the time limit: generate and submit the resumes (the merge is automatic at the end)
./resume_all.sh --dry-run
./resume_all.sh

# 6. bring the logs back from the cluster (rsync from ~/keys; never deletes local-only files)
./fetch_logs_from_remote.sh
```

The `sbatch_*.sh` scripts cannot be run with `bash`: they abort outside a SLURM job.

### Which jobs produce the logs in use

| Folder in `logs/` | sbatch scripts (`sbatch/<type>/`) | Resources (`#SBATCH`) |
|---|---|---|
| `sparsification_percentage` | `sbatch_sparsification_percentage_{plaintext,ckks,bfv,batchcrypt,fedphe,rw,NVNL,sw}.sh` | 3 GPUs, 12 CPUs, 64 GB. Time: 9 h (plaintext, fedphe, rw, NVNL, sw), 1 day (ckks, bfv), 2 days (batchcrypt) |
| `client_frac` | `sbatch_client_frac_{plaintext,fedphe,rw,NVNL,sw}.sh` | 3 GPUs, 32 CPUs, `--mem=0`, 4 days |
| `seed_experiment_v2` | `sbatch_seed_experiment_{fedphe,rw,NVNL,sw}.sh` | 3 GPUs, 12 CPUs, 64 GB, 2 days |
| `range_fix` | `sbatch_range_fix_{fedphe,rw,nvnl,sw}.sh` | 3 GPUs, 12 CPUs, 64 GB, 6 h |

GPU ids are taken from `SLURM_GPUS_ON_NODE`. The number of clients is `GPUs x CLIENTS_PER_GPU`
(3 per GPU gives 9 clients; 20 per GPU gives 60 clients in `client_frac`).

## 2. Generate figures and tables

`clientfrac_en.py` was reconstructed from bytecode on 2026-11-11 (validated against Table B.1, 180/180 cells).

```bash
./studies/journal/run_all.sh   # everything; about 8 min; output in studies/journal/output/ (byte-reproducible PDFs)
```

A single script can also be run alone. Those that use `load_data` need `LOGS_DIR`:

```bash
cd studies/journal
export LOGS_DIR=../../logs/sparsification_percentage
python3 27_three_panel.py
python3 accuracy_vs_budget_en.py
```

`FIG_OUT=/path` redirects the figures (default `studies/journal/output/figures`). `LOGS=/path` replaces the log folder of the
`*_en.py` scripts (default `logs/sparsification_percentage`; `clientfrac_en.py` uses `logs/client_frac`).

### Output and logs read by each script

| Script (in `studies/journal/`) | Output in `studies/journal/output/` | Log folders (runs opened) | Files read |
|---|---|---|---|
| `27_three_panel.py` | `figures/27_three_panel/{model}_{dataset}_niid{n}_p{pct}.{png,pdf}` (50 figures; the paper uses ResNet-20, p = 10%) | sparsification_percentage (309) | server_eval, server_agg, train, evaluate, mask |
| `accuracy_per_round_en.py` | `figures/accuracy-per-round.pdf` | sparsification_percentage (51) | server_eval, train |
| `starvation_vs_packets_en.py` | `figures/starvation-vs-packets.pdf` | sparsification_percentage (69), range_fix (2) | mask, server_eval, train |
| `accuracy_vs_budget_en.py` | `figures/accuracy-vs-budget.pdf` | sparsification_percentage (132), range_fix (6) | server_eval, train |
| `seed_curves_en.py` | `figures/seed-curves.pdf`, `seed_curves.md` (mean ± std per seed; basis of Table C.1) | sparsification_percentage (8), range_fix (3), seed_experiment_v2 (12) | server_eval, train |
| `downlink_en.py` | `figures/downlink-resnet20.pdf` | sparsification_percentage (29) | mask, server_eval, train |
| `mask_agreement.py --plot` | `figures/mask-agreement.pdf` | sparsification_percentage (40) | mask |
| `clientfrac_en.py` | `figures/clientfrac.pdf` | client_frac (54) | mask, server_eval, train |
| `paper_numbers.py` | `paper_numbers.md` (numbers cited in the text; data of Tables A.1 and B.1) | sparsification_percentage (309), client_frac (96), range_fix (6) | mask, server_agg, server_eval, train |
| `29_seed_variance.py --out output/seed_variance` | `seed_variance/` (mean, std and variance per seed, ResNet-20 and MobileNetV2) | sparsification_percentage (309), seed_experiment_v2 (31), range_fix (8) | server_agg, server_eval, train, evaluate |
| `check_pendencies.py` | terminal output (missing cells of Section 8) | sparsification_percentage (148), client_frac (6), range_fix (6) | mask, server_eval, train |
| `pack_heatmap_en.py` | `figures/pack-heatmap-p10.pdf` | sparsification_percentage (25) | mask |
| `20_pack_frequency_individual.py` | `figures/20_pack_frequency_individual/` | sparsification_percentage (280) | mask, train |
| `15_bytes_vs_maxacc_pt.py`, `26_union_coverage_downlink_pt.py`, `28_download_vs_maxacc_pt.py` | `figures/capitulo5_pt/`, `tables/` (Portuguese labels, used in the dissertation) | sparsification_percentage (309 / 280 / 309) | server_eval, server_agg, train, evaluate, mask |

Support modules (no output): `load_data.py` (log reading and deduplication), `paper_style.py`,
`capitulo5_style.py`, `solution_style.py`.

`accuracy_vs_budget_en.py` and `starvation_vs_packets_en.py` read the MobileNetV2 cells at p = 70% and 90% for
Round Robin, Sliding Window and Top-k from `range_fix`. Random uses `sparsification_percentage`. `range_fix` reran
those cells after the CKKS overflow fix.

## 3. Logs in use

`studies/journal/used_logs.txt` lists the 450 run folders that the scripts open. `studies/journal/deleted_logs_2026-11-11.txt` lists what was
removed from `logs/` (`logs/models` was kept).

| Folder | Runs | Model:dataset | Techniques | Clients | Seeds | `niid` | p (`percentage`) | `cf` |
|---|---|---|---|---|---|---|---|---|
| `sparsification_percentage` | 309 | `lenet5:FASHION_MNIST` (60), `dnn_tiny:FASHION_MNIST` (60), `vit_tiny:tanganke_gtsrb` (60), `resnet20:CIFAR10` (64), `mobilenetv2:CIFAR10` (65) | plaintext, ckks, bfv, batchcrypt, fedphe, rw, nvnl, sw | 9 | 42 | 0.1, 1 | -1 (references), 0.1, 0.3, 0.5, 0.7, 0.9 | – |
| `client_frac` | 102 | `resnet20:CIFAR10` (51), `mobilenetv2:CIFAR10` (51) | plaintext, fedphe, rw, nvnl, sw | 60 | 42 | 0.1, 1 | -1, 0.1, 0.5 | 0.25, 0.5, 0.75 |
| `seed_experiment_v2` | 31 | `resnet20:CIFAR10` (15), `mobilenetv2:CIFAR10` (16) | fedphe, rw, nvnl, sw | 9 | 43, 44 | 1 | 0.1, 0.7 | – |
| `range_fix` | 8 | `mobilenetv2:CIFAR10` | fedphe, rw, nvnl, sw | 9 | 42 | 1 | 0.7, 0.9 | – |

Every run has 200 rounds and `poly_modulus_degree` 8192.

Run folder name:
`<model>_<dataset>_<technique>_none_<rounds>rounds_<clients>clients_seed<N>_niid<alpha>_percentage<p>_<poly>_..._[cf<f>_]<timestamp>`.
The scripts filter by prefix (`<model>_<dataset>_<technique>_`) and by tokens (`_niid1_`, `_percentage0.1_`,
`_cf0.5_`). When a configuration has more than one run, `load_data.config_runs` treats each complete folder as one run
and stitches resume fragments together.

`niid`: the scripts treat `niid1` as Dirichlet alpha = 1 (the partition plotted in the figures) and `niid0.1` as the
IID reference (comment in `accuracy_vs_budget_en.py`: "Dirichlet alpha = 1; the niid0.1 folders ran IID";
`CONDS` in `paper_numbers.py`).

Files read in each folder: `server_eval.csv` (accuracy per round), `train.csv` (timings, `n_params`),
`mask.csv` (`round`, `packs_sent`, `selected_packs`), `server_agg.csv`, `evaluate.csv`.

## 4. Available techniques

Keys used in run folder names and in `paper_style.py`.

| Key | Label | Type | `percentage` | Experiment (`experiments/sparsification_percentage/`) | CKKS backend |
|---|---|---|---|---|---|
| `plaintext` | FedAvg | reference, full model, unencrypted | -1 | `..._plaintext.sh` | – |
| `ckks` | CKKS | reference, full model, encrypted | -1 | `..._ckks.sh` | `tenseal` |
| `bfv` | BFV | reference, full model, encrypted | -1 | `..._bfv.sh` | `tenseal` |
| `batchcrypt` | BatchCrypt | reference, full model (Paillier) | -1 | `..._batchcrypt.sh` | `tenseal` |
| `fedphe` | Top-k (FedPHE) | partial packet selection, per client | 0.1 to 0.9 | `..._fedphe.sh` | `ckks_python_simplify` |
| `rw` | Random | partial packet selection | 0.1 to 0.9 | `..._rw.sh` | `ckks_python_simplify` |
| `nvnl` | Round Robin | partial packet selection, rotation | 0.1 to 0.9 | `..._NVNL.sh` | `ckks_python_simplify` |
| `sw` | Sliding Window | partial packet selection, sliding window | 0.1 to 0.9 | `..._sw.sh` | `ckks_python_simplify` |

`ps.ORDER` fixes the legend order: plaintext, ckks, bfv, batchcrypt, fedphe, rw, nvnl, sw. The reference
techniques appear with `percentage` -1 in the logs.

Bytes sent per round (`27_three_panel.py`), one rule for every technique:

| Technique | Upload bytes per client and round |
|---|---|
| fedphe, rw, nvnl, sw | `packs_sent x (2 * poly * l * 8)`, with l = 1 |
| ckks (full model) | `ceil(n_params / (poly/2)) x (2 * poly * 1 * 8)` |
| bfv (full model) | `ceil(n_params / poly) x (2 * poly * (l - 1) * 8)` |
| batchcrypt | `ceil(n_params / 50) x (2 * n_length / 8)` |
| plaintext | `n_params x 4` (float32) |

`packs_sent` comes from `mask.csv`, not `train.csv`: `client_logs.py` truncates with `int()` where the mask uses
`ceil()`.

Number of packets `P` per model (used in figure titles): LeNet-5 16, DNN 28, ViT-Tiny 52, ResNet-20 67,
MobileNetV2 176.
