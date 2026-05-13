# Metodologia de Estimativa de Tempo — `estimate_time.py`

## 1. Motivação

O script `estimate_time.py` estima o tempo total de execução de um lote de experimentos FL
antes de submetê-los ao cluster, permitindo escolher a partição SLURM adequada e evitar
surpresas de timeout ou subutilização de recursos.

---

## 2. Modelo de Dois Componentes

O tempo de um round FL tem dois componentes com gargalos distintos:

| Componente | Acelerador | Dominante quando |
|---|---|---|
| **Treino** (forward + backprop + atualização) | **GPU** | Sempre — ~163s na referência |
| **HE** (pack + encrypt + serialize + decrypt) | **CPU** | GPU muito rápida (ex: H100 ~6s/round) |

Em GPUs lentas (Speed ≤ 10), o treino domina e a CPU é irrelevante.
Em GPUs muito rápidas (Speed > 20, ex: H100), o treino cai para ~6–8 s/round e o overhead HE (~0,5 s/round na referência) passa a representar 6–8% do total — ainda pequeno, mas modelado para consistência.

---

## 3. Dados de Referência

### 3.1 Referência de treino (GPU-bound)

Única medição empírica disponível (laptop, CPU pura, 12 threads):

| Modelo    | Dataset | Clientes | Batch | Épocas | Tempo/round |
|-----------|---------|----------|-------|--------|-------------|
| resnet20  | CIFAR10 | 2        | 4     | 1      | **163 s**   |

Inclui: forward pass, backprop, atualização de pesos, métricas, overhead do Flower.

### 3.2 Referência de HE (CPU-bound)

Estimado a partir das colunas `encrypt_time_s`, `serialize_time_s`, `decrypt_time_s` dos logs `train.csv` (tenseal, poly_degree=8192, resnet20):

```
T_he_ref = 0,50 s   # encrypt + serialize + decrypt por cliente por round
                     # inclui margem sobre os ~0,15–0,30 s medidos
```

### 3.3 Parâmetros de referência

```
T_train_ref  = 163 s         # treino, por cliente por round
T_he_ref     = 0,50 s        # HE, por cliente por round
P_ref        = 274 042       # parâmetros do resnet20
D_ref        = 50 000        # amostras de treino CIFAR10
N_ref        = 2             # clientes na referência
B_ref        = 4             # batch size
poly_ref     = 8 192         # poly_modulus_degree de referência
chunk_ref    = poly_ref / 2  = 4 096   # floats por pack CKKS
n_chunks_ref = ceil(P_ref / chunk_ref) = ceil(274 042 / 4 096) ≈ 67
```

---

## 4. Modelo de Escalonamento

### 4.1 Treino de um único cliente (GPU-bound)

O tempo de treino escala com dois fatores independentes:

**a) Número de parâmetros** — custo de forward+backward por amostra:
```
α_params = P_modelo / P_ref
```

**b) Passos por época** — quantas vezes o custo se repete:
```
steps     = ceil((D_dataset / N_clientes) / B)
α_steps   = steps / steps_ref
T_train_1 = T_train_ref × α_params × α_steps
```

### 4.2 HE de um único cliente (CPU-bound)

As operações HE (pack, encrypt, serialize, decrypt) escalam com o número de chunks CKKS:
```
n_chunks  = ceil(P_modelo / chunk_size)    chunk_size = poly_degree / 2
T_he_1    = T_he_ref × (n_chunks / n_chunks_ref)
```

Essa relação linear é válida porque cada chunk é um ciphertext independente — o custo
de encrypt/decrypt é constante por chunk.

### 4.3 Paralelismo entre clientes

`GPU_IDS` define as GPUs disponíveis e `CLIENTS_PER_GPU` os clientes por GPU:
```
N_clientes = |GPU_IDS| × CLIENTS_PER_GPU
```

**Clientes em GPUs diferentes** rodam em paralelo — não acumulam tempo.

**Clientes na mesma GPU** adotamos hipótese conservadora de **execução sequencial**:
```
T_round_local = CLIENTS_PER_GPU × (T_train_1 + T_he_1)
```

### 4.4 Ajuste pela máquina alvo

O `SLURM.md` define dois fatores por partição:

- `GPU_Speed` (`S_gpu`): velocidade de treino relativa a esta máquina
- `CPU_Speed` (`S_cpu`): velocidade de operações single-thread relativa a esta máquina

O tempo de round na máquina alvo é:
```
T_round_machine = CLIENTS_PER_GPU × (T_train_1 / S_gpu + T_he_1 / S_cpu)
```

Os fatores foram estimados a partir de:
- **GPU_Speed**: FLOPs de pico relativo + benchmarks públicos (ResNet-50 ImageNet)
- **CPU_Speed**: desempenho single-thread relativo (clock × IPC), relevante para operações
  de criptografia que são majoritariamente single-threaded (TenSEAL / numpy / pybind11)

Para partições com nós heterogêneos (ex: `l40s` tem Threadripper em dl-01–03 e Xeon Silver
em dl-05), o `CPU_Speed` usa o valor do nó mais lento (conservador).

### 4.5 Tempo total do experimento

Cada experimento individual adiciona um overhead de setup (geração de chaves, pull de
imagem Docker, startup do container Flask):

```
SETUP_OVERHEAD = 90 s   # estimado conservadoramente
```

O tempo de um experimento é:

```
T_experimento = NUM_ROUNDS × T_round_maquina + SETUP_OVERHEAD
```

O tempo total do lote é a soma sequencial de todos os experimentos (o script de
`start_all.sh` / `lib.sh` roda experimentos em série, um por vez):

```
T_total = Σ T_experimento_i
```

### 4.6 Margem de segurança

O tempo estimado é multiplicado por um fator de margem (default `1.5 = +50%`) para
absorver:

- Variações de carga no cluster (jobs de outros usuários na mesma GPU)
- Overhead de encrypt/decrypt (< 0,5 s por round, absorvido aqui)
- Rounds de avaliação (evaluate) que não foram separados do treino na estimativa
- Variações de convergência (épocas mais lentas por gradientes maiores)
- Overhead de Flower (gRPC, serialização de parâmetros, checagens internas)

```
T_com_margem = T_total × margem
```

---

## 5. Contagem de Experimentos

O script lê os arrays do script bash via regex (sem executar o shell):

```bash
SOLUTIONS=( nvnl sw rw fedphe )
PERCENTAGES=( 0.1 0.3 0.5 0.7 )
NIIDS=( 0.1 )
SEEDS=( 42 )
CKKS_CONFIGS=( "ckks 8192 60" )
KEY_TYPES=( None )
MODEL_DATASET_PAIRS=( "dnn_tiny:FASHION_MNIST" "lenet5:FASHION_MNIST" "mobilenetv2:CIFAR10" )
```

O número de experimentos é o produto cartesiano de todas as combinações. Soluções sem
packing (ex: `plaintext`, `batchcrypt`, `ckks`) **não** iteram sobre `PERCENTAGES` —
apenas uma variação é contada.

Soluções de packing reconhecidas:
```
fedphe, nvnl, sw, rw, fedphe-scout, nvnl-scout
```

---

## 6. Leitura do SLURM.md

O `SLURM.md` é o ponto central de configuração das máquinas. O script lê a tabela
markdown identificando as colunas pelo nome do header (case-insensitive, tolerante a
acentuação). Para adicionar uma nova máquina basta acrescentar uma linha na tabela:

```markdown
| **nova-gpu** | dl-11 | RTX 4090 | 5 dias | 15.0 | Nota |
```

Nenhuma modificação em `estimate_time.py` é necessária.

A coluna `Speed` é obrigatória para estimativas por máquina. O campo `Tempo máx` (formato
`N dias` ou `N days`) é usado para verificar se a estimativa cabe dentro do limite da
partição.

---

## 7. Calibração com Runs Reais

Após executar um experimento em uma máquina do cluster, o tempo real por round pode ser
lido da coluna `train_time_s` do `train.csv`. Para calibrar, anote o fator de speedup
medido e atualize a coluna `Speed` no `SLURM.md`:

```
S_real = T_local_estimado / T_cluster_medido
```

Por exemplo, se no local estimamos 163 s/round e no `l40s` medimos 14 s/round:

```
S_l40s = 163 / 14 ≈ 11.6  →  ajustar para 12.0 (arredondado)
```

O comando futuro para calibração automática (não implementado ainda):

```bash
python3 experiments/estimate_time.py --calibrate logs/meu_experimento/
```

---

## 8. Limitações Conhecidas

| Limitação | Impacto | Mitigação |
|-----------|---------|-----------|
| Somente resnet20/CIFAR10 medido empiricamente | Escalonamento por n_params é aproximado | Calibrar com 1 run real de cada modelo-alvo |
| Scaling linear por n_params | ViT tem atenção quadrática por sequência | Subestime ViT em ≈20–40%; use `--margin 2.0` |
| Clientes por GPU: hipótese sequencial | Pode superestimar se há paralelismo CUDA real | Conservador — preferível a subestimar |
| `GPU_Speed` estimado por FLOPs, não medido | Erro de ±30–50% no tempo absoluto | Calibrar após 1 run real no cluster |
| `CPU_Speed` estimado por clock/IPC | Erro de ±20–30%; Xeon vs Threadripper varia muito | Calibrar medindo `encrypt_time_s` nos logs |
| `T_he_ref = 0,50 s` é tenseal | ckks_python_simplify/openfhe podem ser 5–50× mais lentos | Para backend HE pesado use `--margin 2.0` |
| Partições mistas (l40s): usa pior nó | Pode superestimar em 20–30% para jobs que caiam em dl-01–03 | Conservador; aceitar |
| Setup overhead fixo (90 s) | Varia com carga e tamanho da imagem | Aumentar margem para clusters ocupados |
| Experimentos sequenciais | Se rodar múltiplos `start_all.sh` em paralelo, tempo real ÷ N | Dividir estimativa pelo número de jobs paralelos |

---

## 9. Exemplo de Uso

```bash
# Estimativa rápida para todas as máquinas
python3 experiments/estimate_time.py experiments/sparsification_percentage.sh --all

# Apenas para h100, com margem 2×
python3 experiments/estimate_time.py experiments/sparsification_percentage.sh --machine h100 --margin 2.0

# Breakdown detalhado
python3 experiments/estimate_time.py experiments/sparsification_percentage.sh --all --verbose
```

Saída de exemplo para `sparsification_percentage.sh` (48 experimentos, 50 rounds cada):

```
  Máquina      GPU                  GPU Spd CPU Spd      Sem margem      COM MARGEM  vs limite
  l40s         L40S 46GB               12.0     0.8     16h 22m 37s      1d 33m 56s  OK (26% do limite)
  h100         H100 NVL                25.0     1.3      8h 36m 46s     12h 55m 10s  OK (27% do limite)
  a5000        RTX A5000               10.0     0.7     19h 21m 31s    1d 5h 2m 16s  OK (40% do limite)
```

Note que l40s tem `CPU_Speed=0.8` (Xeon Silver no dl-05) contra `GPU_Speed=12.0` —
a CPU lenta prolonga o tempo HE, impactando mais do que parece à primeira vista.
