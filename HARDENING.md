# HARDENING — Garantias antes do sweep final do NVNL

Data: 2026-06-29  
Autor: auditoria automática + modificações mínimas.

---

## H1 — MAGG_RENORM, CKKS_BACKEND, CLIENT_FRACTION nos CSVs

### Modificação realizada

`client/client_logs.py`:
- Três colunas adicionadas ao final de `_TRAIN_HEADER` (linha 66 antes / 67 depois):
  `magg_renorm,ckks_backend,client_fraction`
- `write_train_logs()` (linhas 167-170 novas): lê as três variáveis do ambiente no
  momento da escrita do log e as inclui em cada linha de `train.csv`.

```
magg_renorm     = os.environ.get("MAGG_RENORM", "1")
ckks_backend    = os.environ.get("CKKS_BACKEND", "tenseal")
client_fraction = os.environ.get("CLIENT_FRACTION", "1.0")
```

Todo `train.csv` gerado daqui em diante registra as três flags por linha.

### Scan de CSVs existentes contaminados por MAGG_RENORM=0

Varredura em todos os `experiment_details.json` dentro de `logs/`:

| Categoria | Count | Resultado |
|-----------|-------|-----------|
| Com `"MAGG_RENORM": 0` explícito | **0** | Nenhum experimento ativo rodou com MAGG_RENORM=0 |
| Com `"MAGG_RENORM": 1` explícito | **440** | Todos os experimentos recentes (client_frac etc.) |
| Sem campo `MAGG_RENORM` — em `archive/` | **287** | Experimentos antigos (maio 2026, pre-fix). Backend era `ckks_python_simplify`; comportamento efetivo equivale a MAGG_RENORM=0 (pré-escala por 1/K, sem renorm pós). Ver nota abaixo. |
| Sem campo `MAGG_RENORM` — fora de `archive/` (ativos) | **85** | Todos em `benchmark_resources/`, todos `plaintext` — MAGG_RENORM não se aplica. |

**Nota sobre os 287 experimentos de arquivo:** Esses experimentos foram rodados com
código anterior à introdução da variável `MAGG_RENORM`. O comportamento na época era
equivalente a `MAGG_RENORM=0` (sem renorm per-pack). Para `nvnl` e `sw` (máscaras
determinísticas, CLIENT_FRACTION=1.0), o viés era nulo. Para `fedphe` e `rw` (máscaras
data-dependent/aleatórias), havia viés de divisão. **Esses resultados já estão em
`archive/` e NÃO devem entrar na dissertação** — os experimentos corretos (com
MAGG_RENORM=1) são os experimentos de `client_frac` e futuros sweeps finais.

**Conclusão H1:** Nenhum resultado ativo contaminado por MAGG_RENORM=0. Dissertação segura.

---

## H2 — Parâmetros efetivos de execução

Todos os valores abaixo são confirmados a partir dos `experiment_details.json` dos
experimentos ativos (440 experimentos com MAGG_RENORM explícito).

### MAGG_RENORM

**Valor efetivo: `1` (padrão correto)**

- Fonte de leitura: `run_client.sh:89` (Docker) e `:140` (Singularity):
  ```bash
  -e MAGG_RENORM=${MAGG_RENORM:-1}
  ```
  Default no shell: `${MAGG_RENORM:-1}` → valor 1 quando não definido externamente.
- Fonte de escrita no JSON: `experiments/lib_tmux.sh:165` e `experiments/lib_slurm.sh:220`:
  ```json
  "MAGG_RENORM": ${MAGG_RENORM:-1}
  ```
- Verificado nos 440 experimentos recentes: todos têm `"MAGG_RENORM": 1`.
- Único script que define `MAGG_RENORM=0` explicitamente:
  `experiments/tests/test_magg_ab.sh` (script de teste A/B, não usado na dissertação).

### CKKS_BACKEND

**Valor efetivo: `ckks_python_simplify` (519 de 525 experimentos ativos)**

- Distribuição nos experimentos ativos: `ckks_python_simplify`=519, `tenseal`=6.
- Os 6 com `tenseal` são experimentos antigos ou de sanity-check.
- Fonte: `run_client.sh:89` → `-e CKKS_BACKEND=${CKKS_BACKEND:-tenseal}` (default seria
  tenseal, mas os scripts de experimento definem `CKKS_BACKEND=ckks_python_simplify`
  explicitamente via `lib_tmux.sh:176`).
- Implicação: nos experimentos da dissertação, aplica-se o **caminho Simplify + MAGG_RENORM=1**:
  clientes pré-escalam por D_i, servidor faz soma pura, cliente divide por Σ_{S_k} D_i.

### CLIENT_FRACTION

**Valor efetivo: varia por experimento (0.25, 0.5, 0.75 no sweep `client_frac`; 1.0 em `sparsification_percentage`)**

- Lido em `server.py:297`: `float(os.environ.get('CLIENT_FRACTION', 1.0))`
- Propagado por `lib_tmux.sh:195` e `lib_slurm.sh` via `CLIENT_FRACTION='${CLIENT_FRACTION:-1.0}'`
- Nos sweeps `sparsification_percentage` (sweep principal do NVNL): CLIENT_FRACTION=1.0
  → todos os K clientes participam de todas as rodadas → S_k = K para NVNL/SW (sem viés
  mesmo em caminho legado).
- No sweep `client_frac`: CLIENT_FRACTION ∈ {0.25, 0.5, 0.75} → K_sel < K clientes por
  rodada → MAGG_RENORM=1 é necessário para dividir corretamente por Σ_{S_k} D_i.

### Número de packs — todos os clientes têm o mesmo?

**Sim.** Todos os clientes de um experimento compartilham o mesmo `MODEL_ARCH`, `DATASET`
e `poly_modulus_degree` (`POLY`=8192 nos experimentos recentes). O número de packs é
determinado por:
```
n_packs = ceil(total_params / chunk_size) + 1
chunk_size = poly_modulus_degree // 2 = 4096
```
Como `total_params` e `poly_modulus_degree` são iguais para todos os clientes, `n_packs`
é idêntico para todos.

Fonte: `client_utils.py:packing()` (linha 15-36) e `server_utils.py:align_packs()` (alinha
por máscara de tamanho `len(mask)`, não por tamanho do pack individual).

---

## H3 — Cobertura por pack (coverage.csv)

### Modificações realizadas

**`server/server_logs.py`:**
- Nova função `write_coverage_logs()` (linhas 67-90).
- Novo cabeçalho `_COVERAGE_HEADER = "experiment_name,round,solution,pack_idx,n_contributors,n_clients_round"`.
- Escreve em `logs/<LOGFOLDER>/coverage.csv`, uma linha por pack por rodada.

**`server/server.py`:**
- Importa `write_coverage_logs` (linha 19).
- Em `aggregate_fit()`, após `get_pack_mask_pair()`:
  - Computa `contributor_counts[k] = |S_k|` = número de clientes que enviaram pack k,
    determinado por `wm[k] > 0` (máscara binária × D_i, texto claro).
  - Chama `write_coverage_logs()` após a agregação, antes do `write_agg_logs()`.

### Schema de coverage.csv

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `experiment_name` | str | Label do experimento |
| `round` | int | Rodada global (com offset de resume) |
| `solution` | str | nvnl / fedphe / sw / rw / etc. |
| `pack_idx` | int | Índice do pack (0 … n_packs-1) |
| `n_contributors` | int | Quantos clientes enviaram este pack nesta rodada |
| `n_clients_round` | int | Total de clientes que responderam nesta rodada |

### Como usar para análise (H3 aceite)

```python
import pandas as pd
df = pd.read_csv("logs/<LOGFOLDER>/coverage.csv")

# Frequência absoluta de cada pack (quantas rodadas foi enviado)
freq = df.groupby("pack_idx")["n_contributors"].sum()

# Uniformidade por rodada: desvio-padrão de n_contributors por round
uniformity = df.groupby("round")["n_contributors"].std()

# Para NVNL: espera-se n_contributors constante (= n_clients_round) por round
# Para FedPHE: espera-se alta variância entre pack_idx dentro de cada round
```

Gráficos esperados:
- **NVNL**: barra horizontal plana para todos os packs — cobertura uniforme.
- **FedPHE**: packs de alta norma com barra alta, packs de baixa norma com barra baixa —
  confirma o argumento da dissertação de que NVNL tem cobertura mais equitativa.

---

## H4 — Inventário completo de leituras de MAGG_RENORM

13 pontos no codebase leem ou propagam a variável:

### Pré-escala no cliente (6 arquivos)

| Arquivo | Linha | Efeito se MAGG_RENORM=0 |
|---------|-------|-------------------------|
| `client/literature/NVNL.py` | 58 | Pré-divide por K em vez de multiplicar por D_i |
| `client/literature/fedphe.py` | 58 | Idem — cria viés para top-k data-dependent |
| `client/literature/sw.py` | 58 | Idem — sem viés para SW determinístico com CF=1.0 |
| `client/literature/rw.py` | 58 | Idem — cria viés para masks aleatórias |
| `client/literature/srr.py` | 63 | Idem — sem viés para SRR determinístico com CF=1.0 |
| `client/literature/topkrr.py` | 66 | Idem — viés na fase top-k, sem viés na fase RR com CF=1.0 |

### Renormalização pós-decifragem (1 arquivo)

| Arquivo | Linha | Efeito se MAGG_RENORM=0 |
|---------|-------|-------------------------|
| `client/literature/set_parameters.py` | 115 | Pula a divisão por `aggregated_mask[k]`; resultado bruto em vez de média correta |

### Propagação para containers e JSON (6 lugares)

| Arquivo | Linha | Papel |
|---------|-------|-------|
| `run_client.sh` | 89 | Docker: `-e MAGG_RENORM=${MAGG_RENORM:-1}` |
| `run_client.sh` | 140 | Singularity: `--env MAGG_RENORM=...` |
| `experiments/lib_tmux.sh` | 165 | Grava em `experiment_details.json` |
| `experiments/lib_tmux.sh` | 215 | Passa para container via tmux session |
| `experiments/lib_slurm.sh` | 220 | Grava em `experiment_details.json` |
| `experiments/lib_slurm.sh` | 275 | Passa para container via SLURM env |

### O que acontece com MAGG_RENORM=0 (resumo)

**Backend `ckks_python_simplify` (padrão dos experimentos):**
- Pré-escala: `w_i / K` (divide por número total de clientes, não por D_i)
- Servidor: soma pura (`_pure_sum=True` para CKKSSimplifyBackend)
- Cliente: sem divisão pós-decrypt
- Resultado por pack k: `Σ_{S_k} w_i / K`
- Viés: subestimação por fator `|S_k|/K` para packs com |S_k| < K

**Único arquivo de experimento que usa MAGG_RENORM=0:**
`experiments/tests/test_magg_ab.sh` (script de teste A/B, fases 1 e 3).
Não é usado em nenhum sweep de dissertação.

**Decisão de remover o caminho legado:** a cargo do pesquisador após revisão deste relatório.

---

## Resumo executivo

| Item | Status |
|------|--------|
| H1 — MAGG_RENORM nos CSVs | ✅ Implementado em `client_logs.py` |
| H1 — Scan de resultados contaminados | ✅ Zero experimentos ativos com MAGG_RENORM=0 |
| H1 — 287 experimentos de arquivo pré-fix | ⚠️ Equivalentes a MAGG_RENORM=0; descartáveis; já em `archive/` |
| H2 — MAGG_RENORM efetivo | ✅ 1 em todos os experimentos ativos |
| H2 — CKKS_BACKEND efetivo | ✅ `ckks_python_simplify` (519/525 ativos) |
| H2 — CLIENT_FRACTION efetivo | ✅ 1.0 no sweep principal; 0.25/0.5/0.75 no sweep client_frac |
| H2 — Packs uniformes entre clientes | ✅ Sim (mesmo modelo e poly_modulus_degree) |
| H3 — coverage.csv instrumentation | ✅ Implementado em `server_logs.py` + `server.py` |
| H4 — Inventário MAGG_RENORM | ✅ 13 pontos documentados |


# TAREFA — Execução final do bench_flhe (fechamento do critério 7.7)

## Contexto
Todas as fases do HANDOFF.md (0 a 7) já estão implementadas e validadas no
RESULTS_SUMMARY.md atual, que foi gerado em modo --quick. Falta apenas a execução
completa do critério 7.7. Esta tarefa NÃO altera código de harness nem de backend.
Se algo exigir mudança de código para rodar, PARAR e reportar antes de alterar.

## Pré-checagem (antes de rodar)
1. Limpar de results/ qualquer CSV de execução --quick anterior, OU garantir que a
   agregação final filtra por uma marca de execução completa. Não misturar quick
   com completo no mesmo agg_*.csv.
2. Confirmar que MODEL_SIZES inclui [512, 50_000, 250_000] em bench_flhe.py.

## Execução
Rodar bench_flhe SEM --quick para as três libs, todas as configs curated,
com MODEL_SIZES = [512, 50_000, 250_000], K in {10, 50}, R in {1, 5}.

## Validação obrigatória (a parte que mais importa)
1. As TRÊS libs produzem resultado válido para model_size=250_000 — sem
   RuntimeError, sem célula [FAIL]. Em particular o OpenFHE, que antes da Fase 0.5
   falhava para model > N/2. Para 250k com 4096 slots, n_chunks ≈ 62.
   Se o OpenFHE falhar em 250k, PARAR e reportar — não mascarar com except.
2. Rodar a sanidade 7.3 nos novos números: uplink do custom em [1.0x, 1.3x] do
   teórico, e custom <= tenseal minimo por criptograma.
3. Comparar os números de TAMANHO e LATÊNCIA do sweep completo com os do --quick:
   eles devem ser essencialmente iguais (são determinísticos por config, não
   dependem de K nem de R). Se divergirem de forma relevante, reportar — indica
   que algo no harness depende de K quando não deveria.
4. Erro de agregação: reportar o MAE do custom em K=50, R=5 (o pior caso de
   acúmulo). Confirmar que continua na ordem de 1e-7 ou menor.

## Entregável
Atualizar RESULTS_SUMMARY.md com:
- Tabela de uplink/downlink incluindo a coluna model=250k (hoje só vai até 50k).
- Linha de erro para K=50, R=5.
- Uma frase confirmando que tamanho/latência do completo batem com o --quick.
PARAR ao final e aguardar revisão.