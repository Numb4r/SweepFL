# studies/journal

Figuras e números do manuscrito do journal, gerados a partir de `logs/`. Os scripts vieram de
`journal-nvnl/figures/scripts/` e rodam neste repositório. O `clientfrac_en.py` foi reconstruído do
bytecode em 2026-11-11 (validado contra a Tabela B.1, 180/180 células).

O conteúdo abaixo vem do que os scripts desta pasta leem e imprimem. A pasta `logs/` mantém só o que eles abrem
(`used_logs.txt`).

## 1. Executar os experimentos pelo SLURM

O SLURM é a forma principal de execução. Cada `sbatch/<tipo>/sbatch_*.sh` chama um
`experiments/<tipo>/*.sh` (variável `EXPERIMENT_SCRIPT`), que usa o `lib_slurm.sh`. Os jobs rodam em Singularity
(`RUNTIME=singularity`, `SIF_DIR` padrão `~/keys`).

```bash
# 1. imagens: Docker local, depois .sif (flhe_client.sif e flhe_server.sif precisam estar em SIF_DIR no cluster)
make configure
make singularity-from-docker

# 2. enviar o repositório ao cluster (rsync para ~/keys; exclui logs/)
./sync_remote.sh

# 3. no cluster, submeter (a partir de ~/keys). Um job por (par modelo:dataset × seed):
bash sbatch/submit_per_pair.sh sbatch/sparsification_percentage/sbatch_sparsification_percentage_NVNL.sh --dry-run
bash sbatch/submit_per_pair.sh sbatch/sparsification_percentage/sbatch_sparsification_percentage_NVNL.sh
# ou um único job que roda todos os pares em sequência:
sbatch sbatch/sparsification_percentage/sbatch_sparsification_percentage_NVNL.sh

# 4. acompanhar
squeue -u $USER
tail -f logs/slurm_<jobid>.out

# 5. runs interrompidos por limite de tempo: gerar e submeter os resumes (o merge é automático ao final)
./resume_all.sh --dry-run
./resume_all.sh

# 6. trazer os logs do cluster para cá (rsync de ~/keys; nunca apaga arquivos locais)
./fetch_logs_from_remote.sh
```

Os scripts `sbatch_*.sh` não podem ser executados com `bash`: eles abortam fora de um job SLURM.

### Quais jobs geram os logs usados

| Pasta em `logs/` | Scripts sbatch (`sbatch/<tipo>/`) | Recursos (`#SBATCH`) |
|---|---|---|
| `sparsification_percentage` | `sbatch_sparsification_percentage_{plaintext,ckks,bfv,batchcrypt,fedphe,rw,NVNL,sw}.sh` | 3 GPUs, 12 CPUs, 64 GB. Tempo: 9 h (plaintext, fedphe, rw, NVNL, sw), 1 dia (ckks, bfv), 2 dias (batchcrypt) |
| `client_frac` | `sbatch_client_frac_{plaintext,fedphe,rw,NVNL,sw}.sh` | 3 GPUs, 32 CPUs, `--mem=0`, 4 dias |
| `seed_experiment_v2` | `sbatch_seed_experiment_{fedphe,rw,NVNL,sw}.sh` | 3 GPUs, 12 CPUs, 64 GB, 2 dias |
| `range_fix` | `sbatch_range_fix_{fedphe,rw,nvnl,sw}.sh` | 3 GPUs, 12 CPUs, 64 GB, 6 h |

Os GPUs são detectados por `SLURM_GPUS_ON_NODE`. O número de clientes é `GPUs × CLIENTS_PER_GPU`
(3 por GPU → 9 clientes; 20 por GPU → 60 clientes em `client_frac`).

## 2. Gerar figuras e tabelas

```bash
cd studies/journal
./run_all.sh          # tudo; ~8 min; saída em output/ (PDFs byte-reproduzíveis via SOURCE_DATE_EPOCH)
```

Também é possível rodar um script isolado. Os que usam `load_data` precisam de `LOGS_DIR`:

```bash
export LOGS_DIR=../../logs/sparsification_percentage
python3 27_three_panel.py
python3 accuracy_vs_budget_en.py
```

`FIG_OUT=/caminho` redireciona as figuras (padrão `output/figures`). `LOGS=/caminho` substitui a pasta de logs dos
scripts `*_en.py` (padrão `logs/sparsification_percentage`; `clientfrac_en.py` usa `logs/client_frac`).

### Saídas e logs lidos por cada script

| Script | Saída em `output/` | Pastas de logs (execuções abertas) | Arquivos lidos |
|---|---|---|---|
| `27_three_panel.py` | `figures/27_three_panel/{modelo}_{dataset}_niid{n}_p{pct}.{png,pdf}` (50 figuras; a do paper é ResNet-20, p = 10%) | sparsification_percentage (309) | server_eval, server_agg, train, evaluate, mask |
| `accuracy_per_round_en.py` | `figures/accuracy-per-round.pdf` | sparsification_percentage (51) | server_eval, train |
| `starvation_vs_packets_en.py` | `figures/starvation-vs-packets.pdf` | sparsification_percentage (69), range_fix (2) | mask, server_eval, train |
| `accuracy_vs_budget_en.py` | `figures/accuracy-vs-budget.pdf` | sparsification_percentage (132), range_fix (6) | server_eval, train |
| `seed_curves_en.py` | `figures/seed-curves.pdf`, `seed_curves.md` (média ± desvio por seed; base da Tabela C.1) | sparsification_percentage (8), range_fix (3), seed_experiment_v2 (12) | server_eval, train |
| `downlink_en.py` | `figures/downlink-resnet20.pdf` | sparsification_percentage (29) | mask, server_eval, train |
| `mask_agreement.py --plot` | `figures/mask-agreement.pdf` | sparsification_percentage (40) | mask |
| `clientfrac_en.py` | `figures/clientfrac.pdf` | client_frac (54) | mask, server_eval, train |
| `paper_numbers.py` | `paper_numbers.md` (números citados e dados das Tabelas A.1 e B.1) | sparsification_percentage (309), client_frac (96), range_fix (6) | mask, server_agg, server_eval, train |
| `29_seed_variance.py --out output/seed_variance` | `seed_variance/` (média, desvio e variância por seed, ResNet-20 e MobileNetV2) | sparsification_percentage (309), seed_experiment_v2 (31), range_fix (8) | server_agg, server_eval, train, evaluate |
| `check_pendencies.py` | impressão no terminal (células faltando da Seção 8) | sparsification_percentage (148), client_frac (6), range_fix (6) | mask, server_eval, train |
| `pack_heatmap_en.py` | `figures/pack-heatmap-p10.pdf` | sparsification_percentage (25) | mask |
| `20_pack_frequency_individual.py` | `figures/20_pack_frequency_individual/` | sparsification_percentage (280) | mask, train |
| `15_bytes_vs_maxacc_pt.py`, `26_union_coverage_downlink_pt.py`, `28_download_vs_maxacc_pt.py` | `figures/capitulo5_pt/`, `tables/` (rótulos em português, usados na dissertação) | sparsification_percentage (309 / 280 / 309) | server_eval, server_agg, train, evaluate, mask |

Módulos de apoio (não geram saída): `load_data.py` (leitura e deduplicação dos logs), `paper_style.py`,
`capitulo5_style.py`, `solution_style.py`.

`accuracy_vs_budget_en.py` e `starvation_vs_packets_en.py` leem de `range_fix` as células MobileNetV2 com
p = 70% e 90% para Round Robin, Sliding Window e Top-k. O Random usa `sparsification_percentage`. Os dois scripts
fazem isso porque o `range_fix` reexecutou essas células depois da correção do overflow do CKKS.

## 3. Logs usados

`used_logs.txt` lista as 450 pastas de execução que os scripts abrem. `deleted_logs_2026-11-11.txt` lista o que
foi removido de `logs/` (mais `logs/models`, mantido).

| Pasta | Execuções | Modelo:dataset | Técnicas | Clientes | Seeds | `niid` | p (`percentage`) | `cf` |
|---|---|---|---|---|---|---|---|---|
| `sparsification_percentage` | 309 | `lenet5:FASHION_MNIST` (60), `dnn_tiny:FASHION_MNIST` (60), `vit_tiny:tanganke_gtsrb` (60), `resnet20:CIFAR10` (64), `mobilenetv2:CIFAR10` (65) | plaintext, ckks, bfv, batchcrypt, fedphe, rw, nvnl, sw | 9 | 42 | 0.1, 1 | -1 (referências), 0.1, 0.3, 0.5, 0.7, 0.9 | – |
| `client_frac` | 102 | `resnet20:CIFAR10` (51), `mobilenetv2:CIFAR10` (51) | plaintext, fedphe, rw, nvnl, sw | 60 | 42 | 0.1, 1 | -1, 0.1, 0.5 | 0.25, 0.5, 0.75 |
| `seed_experiment_v2` | 31 | `resnet20:CIFAR10` (15), `mobilenetv2:CIFAR10` (16) | fedphe, rw, nvnl, sw | 9 | 43, 44 | 1 | 0.1, 0.7 | – |
| `range_fix` | 8 | `mobilenetv2:CIFAR10` | fedphe, rw, nvnl, sw | 9 | 42 | 1 | 0.7, 0.9 | – |

Todos os runs têm 200 rodadas e `poly_modulus_degree` 8192.

Nome da pasta de execução:
`<modelo>_<dataset>_<técnica>_none_<rodadas>rounds_<clientes>clients_seed<N>_niid<α>_percentage<p>_<poly>_..._[cf<f>_]<timestamp>`.
Os scripts filtram por prefixo (`<modelo>_<dataset>_<técnica>_`) e por tokens (`_niid1_`, `_percentage0.1_`, `_cf0.5_`).
Quando uma configuração tem mais de uma execução, `load_data.config_runs` trata cada pasta completa como um run e
costura fragmentos de resume.

`niid`: os scripts usam `niid1` como Dirichlet α = 1 (a partição das figuras) e `niid0.1` como a referência IID
(comentário em `accuracy_vs_budget_en.py`: "Dirichlet alpha = 1; the niid0.1 folders ran IID";
`CONDS` em `paper_numbers.py`).

Arquivos lidos em cada pasta: `server_eval.csv` (acurácia por rodada), `train.csv` (tempos, `n_params`),
`mask.csv` (`round`, `packs_sent`, `selected_packs`), `server_agg.csv`, `evaluate.csv`.

## 4. Técnicas disponíveis

Chaves usadas nos nomes das pastas e em `paper_style.py`.

| Chave | Rótulo | Tipo | `percentage` | Experimento (`experiments/sparsification_percentage/`) | Backend CKKS |
|---|---|---|---|---|---|
| `plaintext` | FedAvg | referência, modelo completo sem cifra | -1 | `..._plaintext.sh` | – |
| `ckks` | CKKS | referência, modelo completo cifrado | -1 | `..._ckks.sh` | `tenseal` |
| `bfv` | BFV | referência, modelo completo cifrado | -1 | `..._bfv.sh` | `tenseal` |
| `batchcrypt` | BatchCrypt | referência, modelo completo (Paillier) | -1 | `..._batchcrypt.sh` | `tenseal` |
| `fedphe` | Top-k (FedPHE) | seleção parcial de pacotes, por cliente | 0.1–0.9 | `..._fedphe.sh` | `ckks_python_simplify` |
| `rw` | Random | seleção parcial de pacotes | 0.1–0.9 | `..._rw.sh` | `ckks_python_simplify` |
| `nvnl` | Round Robin | seleção parcial de pacotes, rodízio | 0.1–0.9 | `..._NVNL.sh` | `ckks_python_simplify` |
| `sw` | Sliding Window | seleção parcial de pacotes, janela deslizante | 0.1–0.9 | `..._sw.sh` | `ckks_python_simplify` |

`ps.ORDER` fixa a ordem de legenda: plaintext, ckks, bfv, batchcrypt, fedphe, rw, nvnl, sw. Os `percentage`
dos experimentos com referências aparecem como -1 nos logs.

Contagem de bytes enviados por rodada (`27_three_panel.py`), usada para todas as técnicas:

| Técnica | Bytes de upload por cliente e rodada |
|---|---|
| fedphe, rw, nvnl, sw | `packs_sent × (2 · poly · ℓ · 8)`, com ℓ = 1 |
| ckks (modelo completo) | `ceil(n_params / (poly/2)) × (2 · poly · 1 · 8)` |
| bfv (modelo completo) | `ceil(n_params / poly) × (2 · poly · (ℓ − 1) · 8)` |
| batchcrypt | `ceil(n_params / 50) × (2 · n_length / 8)` |
| plaintext | `n_params × 4` (float32) |

`packs_sent` vem de `mask.csv`, não de `train.csv`: o `client_logs.py` trunca com `int()` onde a máscara usa `ceil()`.

Número de pacotes `P` por modelo (usado nos títulos das figuras): LeNet-5 16, DNN 28, ViT-Tiny 52, ResNet-20 67,
MobileNetV2 176.
