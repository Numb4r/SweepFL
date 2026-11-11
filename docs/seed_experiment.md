# seed_experiment: auditoria da varredura de agosto e o novo escopo (v2)

Data: 2026-09-23.

## Resumo

- A varredura de agosto (`logs/seed_experiment/`) terminou as 160 runs, mas **23 delas
  colapsaram para NaN**, com a acurácia caindo a 0,100. A causa é o estouro de faixa do
  CKKS de primo único (`docs/ckks_range_overflow.md`). Confirmei o mecanismo em duas delas.
- O escopo foi reduzido a **α=1, p ∈ {0.1, 0.7}, seeds 43/44** e será **rerodado inteiro com a
  correção** (`MAGG_D_REF`) em `logs/seed_experiment_v2/`. São 32 runs. Os logs de agosto não
  foram alterados.
- O estudo `studies/general/29_seed_variance.py` gera a média, o desvio-padrão e a variância
  da acurácia entre as seeds 42/43/44 para cada técnica com packing.

## 1. A varredura de agosto (`logs/seed_experiment/`)

Configuração: `SEEDS=(43 44)`, `NIIDS=(0.1 1)`, `PERCENTAGES=(0.1 0.3 0.5 0.7 0.9)`,
resnet20 e mobilenetv2 em CIFAR-10, 9 clientes, 200 rodadas, técnicas sweepfl, sw, rw e fedphe
(4 × 2 × 2 × 2 × 5 = 160 runs).

- `check_status.py`: 160/160 com 200 rodadas; nenhuma foi retomada de checkpoint.
- As runs foram feitas com o código anterior à correção `MAGG_D_REF`: o cliente cifrava
  `w·D_i`, e a soma do servidor chega a ~50 000·w.
- **23/160 têm NaN em `server_eval.csv`**:

| modelo | técnica | seed | α | p | rodadas NaN | 1º NaN | acc final |
|---|---|---|---|---|---|---|---|
| mobilenetv2 | fedphe | 43 | 0.1 | 0.1 | 130 | 60 | 0.100 |
| mobilenetv2 | fedphe | 43 | 0.1 | 0.5 | 52 | 127 | 0.100 |
| mobilenetv2 | fedphe | 43 | 0.1 | 0.7 | 11 | 159 | 0.558 |
| mobilenetv2 | fedphe | 44 | 0.1 | 0.1 | 64 | 96 | 0.100 |
| mobilenetv2 | fedphe | 44 | 0.1 | 0.5 | 26 | 148 | 0.100 |
| mobilenetv2 | fedphe | 44 | 1 | 0.7 | 1 | 200 | 0.100 |
| mobilenetv2 | sweepfl | 43 | 0.1 | 0.1 | 6 | 195 | 0.251 |
| mobilenetv2 | sweepfl | 43 | 0.1 | 0.3 | 10 | 172 | 0.481 |
| mobilenetv2 | sweepfl | 43 | 0.1 | 0.5 | 65 | 121 | 0.100 |
| mobilenetv2 | sweepfl | 43 | 0.1 | 0.7 | 56 | 133 | 0.100 |
| mobilenetv2 | sweepfl | 43 | 0.1 | 0.9 | 14 | 178 | 0.711 |
| mobilenetv2 | rw | 43 | 0.1 | 0.7 | 13 | 174 | 0.595 |
| mobilenetv2 | rw | 43 | 0.1 | 0.9 | 21 | 147 | 0.704 |
| mobilenetv2 | sw | 43 | 0.1 | 0.3 | 33 | 155 | 0.100 |
| mobilenetv2 | sw | 43 | 0.1 | 0.5 | 31 | 142 | 0.100 |
| mobilenetv2 | sw | 43 | 0.1 | 0.7 | 32 | 150 | 0.100 |
| mobilenetv2 | sw | 43 | 0.1 | 0.9 | 49 | 134 | 0.100 |
| mobilenetv2 | sw | 44 | 0.1 | 0.5 | 18 | 167 | 0.100 |
| resnet20 | fedphe | 43 | 0.1 | 0.1 | 46 | 155 | 0.100 |
| resnet20 | fedphe | 43 | 0.1 | 0.3 | 31 | 170 | 0.100 |
| resnet20 | fedphe | 43 | 0.1 | 0.5 | 37 | 164 | 0.100 |
| resnet20 | fedphe | 44 | 0.1 | 0.1 | 63 | 138 | 0.100 |
| resnet20 | fedphe | 44 | 0.1 | 0.5 | 59 | 142 | 0.100 |

O problema é bem mais amplo que na seed 42, onde só as 5 células corrigidas por
`range_fix` quebraram. Em α=0.1 (Dirichlet de fato, 21 dos 23 casos) ele aparece até em
p=0.1 e também no resnet20.

### Confirmação da causa (probe)

Usei `experiments/tests/ckks_range_probe.py` sobre os snapshots `model/round_*.npz` baixados
do cluster para uma pasta temporária:

- **resnet20 fedphe s43 α0.1 p0.1:** a razão coeficiente / q/(2Δ) do pacote 57 sobe de
  0,60 (rodada 140) a 0,996 (rodada 153). Na 154 o pacote estoura (mín. −488), e o NaN
  começa na 155. O pacote 57 contém os kernels de `conv2d_19`/`conv2d_20` e as
  estatísticas de `batch_normalization_19`. Aqui o estouro corrompe também pesos
  treináveis, não só BN.
- **mobilenetv2 fedphe s44 α1 p0.7:** a razão chega a 0,965 no pacote 172 (o mesmo pacote
  `Conv_1_bn` da seed 42), e na rodada 200 aparece o deslocamento de −88, que é a
  assinatura 2²²/ΣD ≈ 84.

## 2. Novo escopo: `seed_experiment_v2`

| | agosto | v2 |
|---|---|---|
| `FOLDER` | `seed_experiment` | `seed_experiment_v2` |
| `NIIDS` | (0.1 1) | **(1)** |
| `PERCENTAGES` | (0.1 0.3 0.5 0.7 0.9) | **(0.1 0.7)** |
| `SEEDS` | (43 44) | (43 44) |
| código | antes de `MAGG_D_REF` | com `MAGG_D_REF` + `range.csv` |
| runs | 160 | **32** |

- **Por que só α=1:** o `niid0.1` da seed 42 (`sparsification_percentage`) é IID por causa do
  rótulo antigo (Readme §15.1). Portanto, só α=1 permite juntar as seeds 42, 43 e 44.
- **Por que rerodar tudo:** das 64 células de agosto no novo escopo, só 1 quebrou. Mesmo
  assim, rerodar deixa as seeds 43/44 inteiras com o mesmo código e com `range.csv`.
- **Scripts:** `experiments/seed_experiment/seed_experiment_{SweepFL,rw,fedphe,sw}.sh`, editados no
  lugar. O cabeçalho registra a configuração antiga. O sbatch não mudou.
- **Estimativa** (`estimate_time.py --all --slurm SLURM.md`, l40s): 5h03m por técnica se as 8 runs
  rodarem em sequência (7h34m com margem). Com `submit_per_pair.sh` são 4 jobs por técnica
  (par × seed), com 2 runs cada, 16 jobs no total, dentro do limite de 2 dias do sbatch.
- `check_status.py --scripts seed_experiment_*.sh`: 32 esperadas, 32 faltando.

Submissão (no cluster, depois de `./sync_remote.sh`):
```bash
for t in SweepFL rw fedphe sw; do bash sbatch/submit_per_pair.sh sbatch/seed_experiment/sbatch_seed_experiment_$t.sh; done
```

## 3. Análise: estudo 29

`studies/general/29_seed_variance.py` roda fora do `run_all.sh`, porque junta 3 pastas:

| fonte | conteúdo |
|---|---|
| `logs/seed_experiment_v2` | seeds 43/44 |
| `logs/sparsification_percentage` | seed 42 e o baseline FedAvg (plaintext) |
| `logs/range_fix` | seed 42 rerodada com a correção; substitui a célula antiga de mesma chave (mobilenetv2 p=0.7) |

- **Métrica:** a acurácia final é a média de `mean_acc` nas últimas 10 rodadas. A tabela traz
  também o valor da rodada 200.
- **Runs com NaN** nas últimas 10 rodadas são descartadas e listadas na saída. Uma run quebrada
  nunca entra na média sem aviso.
- **Saída** (`studies/seed_experiment_v2/`):
  - `figures/29_seed_variance/bars_{model}_p{pct}`: média ± desvio por técnica, com um ponto
    por seed e a linha do FedAvg.
  - `figures/29_seed_variance/curves_{model}_p{pct}`: curva média com faixa de ±desvio e a
    curva do FedAvg.
  - `tables/29_seed_variance.{csv,tex}`: n_seeds, seeds, média, desvio (ddof=1), variância,
    mínimo e máximo.
- Há uma figura por (modelo, p), seguindo a regra de nunca misturar porcentagens.

```bash
python3 studies/general/29_seed_variance.py
# teste com os logs antigos:
python3 studies/general/29_seed_variance.py --seed-dir logs/seed_experiment --out /tmp/s29
```

### Resultado preliminar (logs de agosto, só para validar o script)

Rodei o script com `--seed-dir logs/seed_experiment`. Esses números vêm do código anterior
à correção e **não são o resultado final**. Duas runs foram descartadas por NaN
(mobilenetv2 fedphe p0.7, seeds 42 e 44).

| modelo | p | técnica | n | média | desvio |
|---|---|---|---|---|---|
| resnet20 | 0.1 | SweepFL / FedPHE / SW / RW | 3 | 0.615 / 0.246 / 0.586 / 0.537 | 0.021 / 0.026 / 0.017 / 0.028 |
| resnet20 | 0.7 | SweepFL / FedPHE / SW / RW | 3 | 0.771 / 0.770 / 0.769 / 0.768 | 0.006 / 0.004 / 0.006 / 0.013 |
| mobilenetv2 | 0.1 | SweepFL / FedPHE / SW / RW | 3 | 0.646 / 0.292 / 0.630 / 0.585 | 0.015 / 0.052 / 0.007 / 0.018 |
| mobilenetv2 | 0.7 | SweepFL / FedPHE / SW / RW | 3 / 1 / 3 / 3 | 0.751 / 0.747 / 0.753 / 0.752 | 0.002 / — / 0.013 / 0.006 |

Leitura preliminar: em p=0.1 a ordem SweepFL > SW > RW ≫ FedPHE supera o desvio entre seeds
(0,01–0,03). Em p=0.7 as 4 técnicas ficam dentro de ~1 desvio umas das outras.

## 4. Resultados v2

A preencher quando as 32 runs e as 8 do `range_fix` terminarem. Critério: nenhuma run
descartada por NaN, e `max(coef_ratio) < 1` em todos os `range.csv`.
