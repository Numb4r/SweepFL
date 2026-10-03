# Estudo de confiança entre seeds — MobileNetV2 · CIFAR10 · α=1

Compara FedPHE (top-k), NVNL (round-robin), SW (sliding window) e RW (pondering-random) com **3 seeds (42, 43, 44)**, p ∈ {0.1, 0.7}, 9 clientes, 200 rodadas. FedAvg plaintext (seed 42) é a referência.

Scripts (em `studies/general/`, rodar com `python3.10`, que é o Python com matplotlib/pandas):

| Script | O que faz |
|---|---|
| `31_seed_confidence.py` | média, desvio, CV, IC95% (t de Student, n=3), min–max, separação entre técnicas |
| `32_seed_confidence_mask_cause.py` | explica a diferença: cobertura de packs, norma × frequência, tipo de parâmetro, acc. de treino × servidor |
| `seed_conf_common.py` | localiza as runs (range_fix > seed_experiment_v2 > sparsification_percentage; só runs completas) |

Métrica: acurácia do servidor, média das últimas 10 rodadas. Tabelas em `tables/`, figuras em `figures/`.

## 1. Resultado (tabela completa em `tables/31_confidence.md`)

| p | Técnica | média | desvio | CV % | IC95% | mín–máx |
|---|---|---|---|---|---|---|
| 0.1 | **FedPHE** | 0.277 | 0.055 | **19.9** | [0.140, 0.414] | 0.242–0.341 |
| 0.1 | NVNL | 0.645 | 0.010 | 1.5 | [0.621, 0.670] | 0.635–0.654 |
| 0.1 | SW | 0.623 | 0.015 | 2.4 | [0.586, 0.660] | 0.607–0.636 |
| 0.1 | RW | 0.583 | 0.018 | 3.1 | [0.537, 0.628] | 0.566–0.602 |
| 0.7 | FedPHE | 0.753 | 0.005 | 0.6 | [0.741, 0.764] | 0.750–0.758 |
| 0.7 | NVNL | 0.752 | 0.004 | 0.5 | [0.743, 0.761] | 0.748–0.756 |
| 0.7 | SW | 0.757 | 0.009 | 1.2 | [0.734, 0.780] | 0.746–0.764 |
| 0.7 | RW | 0.751 | 0.013 | 1.7 | [0.719, 0.784] | 0.736–0.760 |

FedAvg plaintext (seed 42): 0.761.

Leitura com n=3 (`tables/31_pairwise.csv`):
- **p=0.7: as quatro técnicas são estatisticamente indistinguíveis** (diferenças ≤ 0.5 p.p., ICs sobrepostos) e todas ficam a ~1 p.p. do FedAvg. Qualquer ranking nesse p seria ruído.
- **p=0.1: FedPHE é 31–37 p.p. pior que todas as outras**, com faixas min–max e ICs disjuntos. Essa conclusão é robusta: o *pior* seed das outras técnicas (RW 0.566) está muito acima do *melhor* do FedPHE (0.341).
- p=0.1, entre NVNL, SW e RW: NVNL > SW por 2.3 p.p. (não separável com n=3); NVNL > RW por 6.3 p.p. (faixas disjuntas, ICs ainda se tocam). A ordem NVNL ≥ SW ≥ RW se repete nos 3 seeds em média, mas é uma evidência fraca.
- A variância entre seeds do FedPHE p=0.1 é 5–10× maior que a das demais (CV 19.9% vs 1.5–3.1%), puxada pelo seed 43 (0.341 vs 0.242/0.248).

Figuras: `31_bars_confidence`, `31_curves_seeds`, `31_cv`.

## 2. Por que o FedPHE em p=0.1 é tão pior e instável

### 2.1 A máscara congela: só ~17% do modelo é sincronizado

(`tables/32_coverage.csv`, `figures/32_coverage`)

| Técnica | Packs distintos enviados em 200 rodadas (de 176) | Packs enviados por todos em toda rodada | % dos envios nos 18 packs mais frequentes |
|---|---|---|---|
| FedPHE seeds 42/43/44 | 32 / 29 / 30 (16–18%) | 9 / 10 / 9 | 95–96% |
| NVNL | 176 (100%) | 0 | 10.5% |
| SW | 176 (100%) | 0 | 11.0% |
| RW | 176 (100%) | 0 | 13.6–14.6% |

Com p=0.1 cada rodada envia 18 de 176 packs. NVNL/SW/RW passam por todos os packs; o top-k manda praticamente os mesmos ~29 packs durante as 200 rodadas. Os parâmetros dos outros ~147 packs **nunca são agregados**: em kernels convolucionais, só 13–14% dos parâmetros estão em packs já enviados (`tables/32_kind_coverage.csv`); o bias da camada final (10 parâmetros) nunca é enviado.

### 2.2 Por que o top-k escolhe sempre os mesmos packs

`client/literature/fedphe.py` chama `get_topk_mask(packed_parameters, ...)`, que ordena os packs pela **norma L2 dos próprios pesos**, não pela norma da atualização. Pesos absolutos mudam pouco entre rodadas, então o ranking quase não muda. Além disso, a magnitude é dominada por um tipo de parâmetro (`tables/32_kind_magnitude.csv`, round 100):

| Tipo | \|w\| médio | \|w\| máx | norma L2 total |
|---|---|---|---|
| kernel conv/dense | 0.10 | 0.87 | 104 |
| BN gamma | 0.96 | 1.43 | 93 |
| BN beta | 0.11 | 0.48 | 14 |
| BN moving_mean | 0.14 | 3.4 | 30 |
| **BN moving_variance** | **10.8** | **79** | **1908** |

A `moving_variance` do BatchNorm tem valores ~100× maiores que os kernels. Os packs que contêm esses valores têm norma L2 muito alta e ficam fixos no topo do ranking. Em `figures/32_norm_vs_frequency` a frequência de envio é um degrau em função da norma: packs com norma ≳ 60–80 são enviados em ~100% das rodadas, os demais em ~0%. Os 18 packs de maior norma coincidem com os 18 mais enviados em 16/17/18 dos casos (seeds 42/43/44); Spearman norma×frequência ≈ 0.64–0.66 (`tables/32_norm_vs_selection.csv`).

Atenção ao limite dessa evidência: usei o snapshot do round 100 (modelo agregado, float16, cliente 0) como proxy das normas que os clientes usaram; a classificação por tipo assume a ordem padrão do Keras (`gamma, beta, moving_mean, moving_variance`). O padrão degrau é claro, mas a atribuição fina "este pack é de moving_variance" é inferida, não medida pack a pack. Não testei trocar o critério de seleção para a norma da atualização; seria o experimento que fecha a causalidade.

### 2.3 Consequência: clientes sobreajustam e o modelo global não aprende

(`figures/32_train_vs_server`)

Para o FedPHE, a acurácia de treino dos clientes chega a ~0.97–1.0 em torno da rodada 55–100, enquanto a acurácia do servidor fica em ~0.25–0.34. Com a maior parte dos pesos nunca sincronizada, cada cliente treina uma parte privada do modelo sobre os dados do seu shard; o modelo global combina um pedaço agregado com pedaços que divergem entre clientes. No NVNL, a acurácia do servidor acompanha a de treino (0.64 vs 0.99 de treino no fim, e ainda subindo). A acurácia do FedPHE p=0.1 atinge o pico nas rodadas ~54–62 (seeds 42 e 44) e **cai** depois, o que é compatível com esse processo, mas o mecanismo exato (divergência dos pesos não sincronizados) é interpretação minha e não foi medido diretamente.

### 2.4 Por que a variância entre seeds é grande

Os conjuntos de packs enviados quase não diferem entre seeds (`tables/32_seed_sensitivity.csv`): 28 packs são comuns aos três seeds; a união tem 33. O seed 43 (o melhor, 0.341) não envia nenhum pack que os outros não enviem, exceto o pack 10 (4 envios no total). O seed 42 envia os packs 6 e 22 (303 envios no total, entre rodadas e clientes) e tem o pior resultado (0.242), mas o seed 44, que não os envia, fica em 0.248. Ou seja, **a diferença de acurácia entre seeds não é explicada pela máscara**. O que sobra é a dinâmica de treino num regime degenerado (≈83% do modelo sem sincronização): pequenas diferenças de inicialização/partição levam a trajetórias bem diferentes (o seed 43 atinge o pico só na rodada 142, os outros em 54–62). Com n=3, o IC95% do FedPHE p=0.1 é [0.14, 0.41], o que mostra que a média é pouco confiável — o que é robusto é o *nível baixo*, não o valor exato.

NVNL/SW/RW, ao varrer todos os packs, mantêm todos os parâmetros sincronizados de tempos em tempos; o resultado é que pequenas diferenças de seed não se amplificam (CV 1.5–3.1%).

### 2.5 Por que em p=0.7 o problema some

Com 70% dos packs por rodada (124 de 176), o top-k inclui quase todos os packs mesmo mantendo o ranking fixo; cobertura e sincronização deixam de ser o gargalo e as quatro técnicas empatam no nível do FedAvg.

## 3. Ressalvas

- n=3 por célula: IC95% com t de Student (2 g.l.) é largo. Diferenças < ~3 p.p. não são resolvíveis aqui.
- As runs de seed 42 com p=0.1 vêm de `logs/sparsification_percentage` e são **anteriores** à correção `MAGG_D_REF` (coluna `post_fix` em `tables/31_runs.csv`); as de p=0.7 (seed 42) vêm de `range_fix`. Os seeds 43/44 são pós-correção. O FedPHE p=0.1 do seed 42 se comporta como os outros (0.242), então a correção não parece mudar a conclusão, mas os seeds não são 100% homogêneos.
- FedAvg plaintext só existe para seed 42.
- Apenas MobileNetV2/CIFAR10/α=1; não generalizar para outros modelos (BN com moving_variance grande é específico de arquiteturas com BatchNorm).

## 4. Reproduzir

```bash
cd studies/general
python3.10 31_seed_confidence.py
python3.10 32_seed_confidence_mask_cause.py
```
