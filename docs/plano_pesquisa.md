# Prompt para Claude Opus — Plano de Trabalho de Pesquisa (Junho–Agosto 2026)

## Instrução

Você é um assistente especializado em pesquisa acadêmica em Aprendizado Federado e Criptografia Homomórfica. Sua tarefa é organizar e redigir um **documento de plano de trabalho de pesquisa** estruturado, com base no contexto detalhado abaixo.

O documento deve cobrir:
1. **O que está sendo proposto** — hipótese central e contribuição científica
2. **O que já foi feito** — implementações, experimentos e resultados obtidos
3. **Próximos passos** — cronograma detalhado de junho a agosto de 2026
4. **O que se espera alcançar** — metas finais: correção da dissertação de mestrado e publicação de journal

O documento deve ter tom acadêmico, ser escrito em **português**, com seções bem delimitadas e um cronograma mensal com marcos concretos. Ao final, inclua também um **abstract em inglês** (máx. 250 palavras) para uso no journal.

---

## Contexto do Projeto

### Nome e Escopo

**Projeto Slidhe** — Sistema de Aprendizado Federado (FL) com preservação de privacidade via Criptografia Homomórfica (HE). Clientes treinam modelos locais e enviam gradientes cifrados a um servidor de agregação. O foco da pesquisa é a técnica de **esparsificação NVNL** (seleção de pacotes por round-robin) e seu impacto no overhead de comunicação e tempo total de treinamento.

### Hipótese Central

A esparsificação por round-robin (NVNL) oferece um balanço superior entre acurácia final e overhead de comunicação/tempo comparado às estratégias top-k (FedPHE), sliding-window (SW) e pondering-random (RW), especialmente em cenários non-IID com alta heterogeneidade de dados.

### Técnicas Implementadas e Comparadas

| Técnica | Estratégia de Seleção | Característica Principal |
|---|---|---|
| **plaintext** | Baseline (sem cifração) | Referência de acurácia e velocidade |
| **NVNL** | Round-robin sequencial | Cobertura uniforme, determinística |
| **FedPHE** | Top-k por magnitude | Seleciona gradientes mais relevantes, fixos por round |
| **SW** | Sliding window (stride 50%) | Cobertura progressiva com sobreposição 50% |
| **RW** | Pondering-random | Seleção ponderada aleatória |

### Backend de Criptografia

O sistema suporta múltiplos backends CKKS, selecionáveis via variável de ambiente `CKKS_BACKEND`:

| Backend | Descrição | Uso no projeto |
|---|---|---|
| **ckks_python_simplify** (padrão) | Implementação própria em C++ com NTT O(N log N), adição/multiplicação escalar homomórficas e serialização binária eficiente | Backend principal dos experimentos; permite controle fino de parâmetros |
| **TenSEAL** | Biblioteca Python de HE sobre Microsoft SEAL; madura e amplamente usada em FL | Referência consolidada na literatura de FL com HE |
| **OpenFHE** | Biblioteca C++ multi-esquema (CKKS, BFV, BGV) com Python bindings | Referência de alto desempenho; usada para validação comparativa |

- **Parâmetros CKKS atuais**: N=8192, BITS=62, SCALE=40 (estabilidade comprovada em 9 clientes, 200 rodadas)
- Problema histórico de overflow com SCALE=60 resolvido e documentado em `CKKS_PARAMS.md`
- A existência de múltiplos backends permite avaliar se os resultados de comunicação/overhead são reprodutíveis em implementações independentes

### Infraestrutura Experimental

- **Framework FL**: Flower (flwr)
- **Orquestração**: tmux (local/Docker) e SLURM (cluster HPC com partições l40s, h100, p100)
- **Clientes**: 9 clientes (3 GPUs × 3 clientes), distribuição Dirichlet não-IID (α ∈ {0.1, 1.0})
- **Modelos testados**: ResNet-20, LeNet-5, DNN, DNN-tiny, MobileNetV2 (α=0.5), ViT-Tiny
- **Datasets**: CIFAR-10, FashionMNIST, GTSRB
- **Porcentagens de esparsificação**: 10%, 30%, 50%, 70%, 90%
- **Rodadas por experimento**: 200
- **Sistema de checkpoint/resume**: checkpoints por rodada com retomada automática de experimentos interrompidos no cluster

### O Que Já Foi Implementado

#### Implementações Completas

1. **5 estratégias de seleção de pacotes**: plaintext, NVNL, FedPHE, SW, RW — todas com cifragem CKKS e integração Flower
2. **Backend CKKS customizado em C++** com 11 suítes de testes unitários (>1000 assertivas)
3. **Sistema de checkpoint/resume** robusto: checkpoints float16 por rodada, detecção automática de experimentos incompletos, geração e submissão de jobs de retomada via SLURM
4. **Pipeline de análise**: 14 scripts de análise cobrindo acurácia, overhead de comunicação, tempo, memória RAM, eficiência de cobertura do modelo e fronteira de Pareto
5. **Logging estruturado**: CSV de 41 colunas por cliente por rodada (`train.csv`), logs do servidor (`server_agg.csv`, `server_eval.csv`), snapshots do modelo em float16

#### Estudos de Análise Implementados (`studies/sparsification_percentage/`)

| Script | O que analisa |
|---|---|
| 01_accuracy_curves.py | Curvas de convergência e acurácia final por técnica |
| 02_solution_comparison.py | Heatmap e bar chart comparando todas as técnicas |
| 03_overhead_breakdown.py | Breakdown de tempo: cifração, decifração, treinamento |
| 04_communication.py | Payload por rodada (bytes) |
| 05_niid_impact.py | Impacto do grau de heterogeneidade non-IID |
| 06_memory_profile.py | Perfil de uso de RAM ao longo das rodadas |
| 07_convergence_speed.py | Rodadas para atingir limiar + AUC |
| 08_efficiency_frontier.py | Fronteira de Pareto: acurácia × payload/rodada |
| 09_summary_table.py | Tabela resumo CSV e LaTeX |
| 10_total_data_convergence.py | Rodadas até cobertura total do modelo (por técnica) |
| 11_data_vs_coverage.py | Bytes cumulativos × cobertura do modelo (NVNL vs SW) |
| 12_bytes_vs_accuracy.py | Custo em bytes para atingir 80% de acurácia |
| 13_overhead_vs_accuracy.py | Tempo de overhead de comunicação × acurácia |
| 14_neurons_vs_accuracy.py | Parâmetros transmitidos × acurácia |

#### Resultados Conhecidos

- Slidhe mantém **92–96% de acurácia** (cenário non-IID) com apenas **30% de esparsificação**
- NVNL garante cobertura completa do modelo em `ceil(n_packs / packs_sent)` rodadas (determinístico)
- SW cobre apenas metade do território novo por rodada comparado a NVNL (custo 2× maior)

### O Que Está Em Aberto / Próximos Passos Esperados

_(Preencha conforme sua análise dos resultados já obtidos)_

- Consolidar experimentos com todos os modelos e datasets nas 5 porcentagens de esparsificação
- Executar estudos 13 e 14 com dados completos do cluster
- Análise estatística de significância entre técnicas
- Identificar cenários onde cada técnica é superior
- Redigir seções da dissertação com os resultados
- Preparar journal (provável IEEE ou similar) com as contribuições principais

---

## Estrutura Esperada do Documento

Por favor, gere o documento com as seguintes seções:

### 1. Resumo Executivo (½ página)
Síntese do projeto, hipótese, estado atual e metas do período junho–agosto.

### 2. Proposta de Pesquisa
- Motivação e problema
- Hipótese central
- Contribuições científicas esperadas
- Posicionamento frente à literatura (não mencionar SOCAS em nenhuma circunstância)

### 3. Trabalho Realizado
- Implementações concluídas
- Infraestrutura experimental (Flower, Docker, SLURM, múltiplos backends CKKS)
- Experimentos executados
- Resultados preliminares conhecidos

### 4. Plano de Trabalho: Junho–Agosto 2026

#### Junho 2026 — Consolidação Experimental
- Semana 1–2: [preencher]
- Semana 3–4: [preencher]
- Marco: [preencher]

#### Julho 2026 — Análise e Escrita
- Semana 1–2: [preencher]
- Semana 3–4: [preencher]
- Marco: [preencher]

#### Agosto 2026 — Correção da Dissertação e Submissão do Journal
- Semana 1–2: [preencher]
- Semana 3–4: [preencher]
- Marco: Dissertação corrigida + Journal submetido

### 5. Metas e Resultados Esperados
- Acurácia mínima alvo por técnica e dataset
- Overhead de comunicação esperado
- Contribuição para o journal (hipótese sobre NVNL vs concorrentes)
- Formato e alvo de publicação do journal

### 6. Riscos e Mitigações
- Riscos técnicos (cluster, experimentos longos, overflow CKKS, estabilidade de backends)
- Riscos de prazo

### 7. Abstract (inglês, máx. 250 palavras)
Para uso na submissão do journal.

---

## Instruções Finais para o Opus

- Preencha os campos `[preencher]` com base no contexto acima e no seu julgamento sobre o que é natural para este estágio de pesquisa de mestrado
- Onde não houver informação suficiente, formule perguntas específicas ao pesquisador em um bloco **"Perguntas para o Pesquisador"** ao final do documento
- Mantenha tom acadêmico mas direto
- O cronograma deve ser realista para um estudante de mestrado em final de curso (12–16 semanas efetivas de trabalho)
- Destaque claramente quais contribuições são **originais** deste trabalho vs. o que é adaptação ou uso de trabalhos existentes (TenSEAL, OpenFHE, Flower)
- A dissertação é o produto principal; o journal deve derivar dos resultados mais sólidos e generalizáveis
- Ao citar frameworks de terceiros (TenSEAL, OpenFHE), posicione-os como referências estabelecidas em relação às quais o backend customizado é comparado
