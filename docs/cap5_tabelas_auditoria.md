# Auditoria das tabelas do Capítulo 5

Data: 2026-07-30. Escopo: tabelas do cap. 5 corrigidas; prosa **não** editada (lista de contradições na seção 4).

---

## 1. Os dois defeitos encontrados

### Defeito 1 — `tab:results_legacy` media grandezas diferentes na mesma coluna

A hipótese levantada na revisão (coluna diferente, divisor esquecido, ou `p` diferente nas linhas de envio parcial) **não se confirma**. A fórmula é uniforme nas seis linhas:

```
TX Bytes (MB) = média(payload_bytes por rodada, entre clientes) × 200 / 1e6
```

As 12 células publicadas reproduzem com razão 1,000 sob essa única fórmula. O problema é a grandeza que `payload_bytes` registra:

| linhas | o que `payload_bytes` mede |
| :--- | :--- |
| FedAvg, BFV, CKKS, BatchCrypt | o *payload* serializado real |
| FedPHE, NVNL-FL | **apenas os metadados de índice de pacote/máscara** (~72 B por pack) |

Já documentado em `docs/log_schema.md:53` ("Payload bytes caveat") e contornado em `studies/general/27_three_panel.py:130` (`_est_wire_bytes`). A tabela, porém, estava escrita à mão dentro de `resultados.tex` e nunca foi regenerada depois dessa correção.

Verificação direta nos logs:

| par | `packs_sent` | `payload_bytes/rodada` | conferência |
| :--- | ---: | ---: | :--- |
| ResNet-20, `p=90%` | 60 | 4392 B | `(60+1) × 72` |
| LeNet-5, `p=90%` | 14 | 1080 B | `(14+1) × 72` |

Valor correto, pelo mesmo estimador que já alimenta as figuras de três painéis (`packs_sent × 2N·8`, com `N=8192` → 131 072 B por pacote cifrado da biblioteca do cap. 4):

| par | publicado | correto | fator |
| :--- | ---: | ---: | ---: |
| ResNet-20 / CIFAR-10 | 0,88 MB | **1572,86 MB** | 1787× |
| LeNet-5 / FashionMNIST | 0,22 MB | **367,00 MB** | 1668× |

Isso reconcilia a tabela com a previsão independente do cap. 4 (criptograma 2,6× a 4,0× menor): a vantagem sobre o CKKS passa a ser **2,85×** (ResNet-20) e **2,91×** (LeNet-5).

### Defeito 2 — SweepFL contado duas vezes (não detectado na revisão)

`logs/sparsification_percentage` tem 309 pastas para 240 configs. **60 configs possuem duas execuções completas independentes** (rodadas 1–200 em ambas), **50 delas SweepFL**. `load_data.load_all` concatenava todas as pastas, e os estudos 15/16 somam a métrica por cliente dentro de cada rodada — o SweepFL entrava duas vezes.

Medido antes da correção: SweepFL com 18 linhas por rodada (9 clientes × 2 pastas) contra 9 de `sw`/`rw`/`fedphe`. Toda célula SweepFL de `tab:fedphe_bytes` caía para exatamente 0,500 do valor publicado ao deduplicar; nenhuma célula não-SweepFL se movia.

O erro era **adversarial à tese**: inflava justamente o custo da técnica proposta.

Há ainda **7 configs que são fragmentos de resume não mesclados** (rodadas disjuntas, `merge_resume.py` nunca rodou). Nesses, concatenar é correto — a união é o experimento real. Um deles (`resnet20/sweepfl/p=0.9/niid1`) tem uma tentativa abandonada de 1–123 ao lado da cadeia 1–107 + 108–200, então "manter a pasta mais recente" ou "manter a mais longa" perdem rodadas. `studies/general/move_duplicates.sh` usa "manter a mais recente" e **destruiria 4 desses pares** — não usar.

`logs/client_frac`: as 108 "duplicatas" são pastas sem `train.csv`. Nenhuma tabela de `client_frac` foi afetada (verificado: contagem de linhas idêntica antes e depois).

---

## 2. O que foi corrigido

**`studies/general/load_data.py`** — `load_all`, `load_masks` e `count_experiments` passam por `_experiment_folders`, que agrupa pastas por config (nome sem o timestamp) e escolhe o subconjunto disjunto de maior cobertura de rodadas: fragmentos de resume são mantidos, reruns sobrepostos são descartados. 309 → 249 pastas em `sparsification_percentage`; `client_frac` inalterado. Cobertura em `tests/test_load_data_dedup.py` (9 testes).

**`dissertation/capitulo5/gerar_tabelas_fedphe.py`** — duas tabelas que estavam escritas à mão dentro de `resultados.tex` passam a ser geradas: `build_results_legacy_table()` e `build_final_results_consolidation_table()`. Foi exatamente o *hardcode* que permitiu ao Defeito 1 sobreviver à correção do `payload_bytes`.

**`dissertation/capitulo5/resultados.tex`** — os dois blocos de tabela inline viram `\input{...}`.

### Tabelas alteradas

`tab:results_legacy` — 23 de 120 células mudaram. As 32 células de BFV, CKKS e BatchCrypt, todas as acurácias e todos os TX Bytes das linhas não-empacotadas reproduzem **exatamente** o publicado.

| linha | coluna | publicado | corrigido | causa |
| :--- | :--- | ---: | ---: | :--- |
| FedPHE / NVNL-FL (4 linhas) | TX Bytes, FashionMNIST | 0,22 | **367,00** | Defeito 1 |
| FedPHE / NVNL-FL (4 linhas) | TX Bytes, CIFAR-10 | 0,88 | **1572,86** | Defeito 1 |
| NVNL-FL | Tempo / Cifragem / Descifragem | — | ±0,3 % a 5,1 % | Defeito 2 |
| FedAvg | Tempo (4 células) | — | −1,6 % a +1,2 % | Defeito 2 (10 configs `plaintext` duplicados) |

`tab:fedphe_bytes` e `tab:fedphe_overhead` — só a linha SweepFL mudou em cada par; nenhuma célula de FedPHE, SW ou RW se moveu. O pico do SweepFL mudou de rodada porque antes a curva era a média de duas execuções distintas:

| tabela | par / cenário | publicado | corrigido |
| :--- | :--- | ---: | ---: |
| bytes | ResNet-20, IID | 73,86 @188, 1104,28 MB | **72,72 @196, 575,73 MB** |
| bytes | ResNet-20, não-IID | 75,37 @170, 998,11 MB | **75,79 @182, 534,45 MB** |
| bytes | LeNet-5, IID | 87,82 @188, 220,86 MB | **87,78 @156, 91,55 MB** |
| bytes | LeNet-5, não-IID | 88,03 @73, 85,20 MB | **88,07 @134, 78,58 MB** |
| overhead | ResNet-20, IID | 1567,31 s | **815,49 s** |
| overhead | ResNet-20, não-IID | 1454,52 s | **794,77 s** |
| overhead | LeNet-5, IID | 130,17 s | **54,29 s** |
| overhead | LeNet-5, não-IID | 51,56 s | **47,58 s** |

`tab:study27_summary` — só as 5 linhas SweepFL; FedPHE inalterado.

`tab:final_results_consolidation` — só as 30 células de tempo do SweepFL (≤5,1 %). Todas as 66 acurácias, todos os *loss* e todas as células de CKKS e FedPHE reproduzem exatamente.

`tab:fedphe_downlink`, `tab:cf_bytes`, `tab:cf_downlink` — **inalterados**, como esperado (usam médias e razões).

---

## 3. Verificação

| checagem | resultado |
| :--- | :--- |
| linhas/rodada por config, todo o dataset | máximo 9 (era 18 no SweepFL) |
| `logs/client_frac` antes vs. depois | idêntico nos 4 CSVs |
| `pytest tests/` | 342 passaram, 1 skip, 1 xfail |
| gerador executado duas vezes | idempotente, zero diferenças |
| colunas LaTeX das duas tabelas novas | 12 em todas as linhas; `table`/`tabular` balanceados |

---

## 4. Trechos da prosa que passam a contradizer as tabelas

Nada abaixo foi editado.

### `resultados.tex:60` — o parágrafo mais afetado

Afirma: *"o NVNL-FL e o FedPHE já transmitem uma fração ínfima do que os esquemas de modelo completo exigem (0,22 MB no FashionMNIST e 0,88 MB no CIFAR-10, contra 1069,70 MB e 4479,34 MB do CKKS)"*.

Com os valores corrigidos, em `p=90%`:

| referência | ResNet-20 | LeNet-5 |
| :--- | :--- | :--- |
| vs. CKKS | 2,85× mais barato | 2,91× mais barato |
| vs. BFV | 1,53× mais barato | 1,54× mais barato |
| vs. FedAvg | **3,20× mais caro** | **3,35× mais caro** |
| vs. BatchCrypt | **2,77× mais caro** | **2,87× mais caro** |

"Fração ínfima" não se sustenta em `p=90%`. A vantagem real vem da esparsificação, não da cifragem: em `p=10%` (`resultados.tex:62`) as técnicas parciais ficam em 0,79 MB por rodada contra 2,46 MB do FedAvg — essa afirmação **continua válida**, pois já usava o estimador correto.

### `resultados.tex:187` — custo em bytes até a acurácia máxima

- *"15,40 MB contra 998,11 MB do NVNL-FL no caso não-IID de ResNet-20"* → **534,45 MB**.
- *"42,64% do FedPHE contra 73,86% do NVNL-FL"* (IID) → SweepFL **72,72%**.
- *"75,37% do NVNL-FL"* (não-IID) → **75,79%**.
- *"84,50 a 87,25% para o FedPHE contra 87,82 a 88,03% para o NVNL-FL"* (LeNet-5) → SweepFL **87,78 a 88,07%**.

### `resultados.tex:200` — custo em tempo até a acurácia máxima

- *"689,33 s contra 1454,52 s do NVNL-FL"* → **689,33 s contra 794,77 s**: 15 % mais caro, não 2,1×.
- *"para acurácia 0,31 ponto percentual inferior"* → **0,73 ponto percentual**.
- *"o SW é a mais barata em três dos quatro cenários"* → **continua verdadeiro**, mas a exceção muda: *"a exceção é LeNet-5/FashionMNIST IID, em que o RW é marginalmente mais barato que o SW (56,29 s contra 62,72 s)"* → agora quem é mais barata ali é o **NVNL-FL (54,29 s)**, à frente do RW (56,29 s) e do SW (62,72 s).
- *"O SW sustenta acurácia sempre dentro de 1,53 ponto percentual do NVNL-FL"* → dentro de **0,73** ponto percentual.
- *"o RW … de 0,59 a 4,85 pontos percentuais abaixo do NVNL-FL"* → **0,63 a 3,71** pontos percentuais (maior diferença segue em ResNet-20 IID).

### `resultados.tex:176` — onde a vantagem do SweepFL se inverte

Afirma: *"chega a se inverter em p≥70% (MobileNetV2 em p=70% e p=90%; ResNet-20 em p=90%)"*.

Pela tabela corrigida, as inversões passam a ser **ResNet-20 em p=50% e p=90%; MobileNetV2 em p=90%**:

| par | p=10 | p=30 | p=50 | p=70 | p=90 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet-20 | SweepFL | SweepFL | **FedPHE** (76,49 vs 76,24) | SweepFL | **FedPHE** (79,57 vs 78,68) |
| MobileNetV2 | SweepFL | SweepFL | SweepFL | SweepFL (78,68 vs 76,85) | **FedPHE** (77,15 vs 76,47) |

MobileNetV2 em `p=70%` deixa de ser uma inversão. A frase seguinte (*"nos demais três pares o NVNL-FL mantém vantagem nas cinco porcentagens"*) permanece correta.

### Sem alteração

`resultados.tex:30` (razões 9,8×/5,2× e 9,1×/4,9× do CKKS/BFV vs. FedAvg; 1,2× do BatchCrypt; 40 min vs. 5 h) e `resultados.tex:62` (hierarquia por rodada) usam apenas linhas não-empacotadas ou o estimador correto — conferidos, seguem válidos. `resultados.tex:124` (0,44 vs. 0,73 em `p=30%`; 0,76 vs. 0,76 em `p=50%`) também.

---

## 5. Figuras — regeradas

As 22 figuras que `resultados.tex` referencia foram regeradas com os dados deduplicados por
**`dissertation/capitulo5/gerar_figuras_cap5.py`** (novo). O script deriva o manifesto dos
`\includegraphics` do próprio `.tex`, roda cada estudo com as saídas redirecionadas em memória
para um staging e publica só os PDFs referenciados em `images/nvnl-slidhe/` e `figures/`.
Nada é escrito dentro de `studies/`.

Comparação raster a raster (via `pdftoppm`, porque o PDF carrega timestamp), antes vs. depois:
**13 mudaram, 9 idênticas, 0 divergências** em relação ao previsto na seção 1.

| estudo | figuras | resultado |
| :--- | ---: | :--- |
| 27 três painéis | 10 | mudaram |
| 15 bytes (`p030`) | 1 | mudou |
| 16 overhead (`p030`) | 1 | mudou |
| 28 download (`p030`) | 1 | mudou |
| 20 frequência de pacote | 4 | idênticas |
| 26 cobertura / inflação | 4 | idênticas |
| 15 bytes (`cf0.5`) | 1 | idêntica |

Os rótulos das figuras passaram a bater com as tabelas corrigidas:

| figura | antes | depois | tabela |
| :--- | ---: | ---: | :--- |
| 15 bytes, ResNet-20 não-IID | `75.4% R170 998.1MB` | `75.8% R182 534.4MB` | `tab:fedphe_bytes` |
| 16 overhead, ResNet-20 não-IID | `75.4% R170 1455s` | `75.8% R182 795s` | `tab:fedphe_overhead` |

Nas duas, os rótulos de SW (`75.1% R164`), RW (`72.8% R182`) e FedPHE (`48.0% R6`) ficaram
inalterados, como esperado.

O script também publica o resumo do Estudo 26 em `data/26_union_coverage/<tema>.csv`, agora lido
por `gerar_tabelas_fedphe.py` (com fallback para `studies/<tema>/tables/`), de modo que
`dissertation/` seja reprodutível sem rodar nada em `studies/`. Confirmado que as três colunas
consumidas (`client_cov_pct`, `union_cov_pct`, `inflation`) são idênticas às antigas em todas as
200 linhas — só `clients_per_round` muda, em 2 linhas — e que as 8 tabelas saem **inalteradas**
após a troca.

### Reflexo na prosa (segue sem edição, conforme decidido)

| local | prosa | figura regerada |
| :--- | ---: | ---: |
| `:78` ResNet-20 IID `p=10%` | 62,7 % | 63,3 % |
| `:100` MobileNetV2 IID `p=50%` | 74,8 % | 75,2 % |
| `:100` MobileNetV2 IID `p=10%` | 65,9 % | 66,8 % |
| `:187` ResNet-20 não-IID `p=30%` | 75,37 % | 75,79 % |

Os valores de FedPHE citados nos mesmos trechos (24,2 %, 65,8 %, 24,1 %, 48,03 %) estão inalterados.
