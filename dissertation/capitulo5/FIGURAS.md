# Manifesto de figuras — Capítulo 5 (parte experimental refeita)

Rastreabilidade de cada figura que entra em `resultados.tex`:
figura final → estudo de origem → comando de geração → label LaTeX.

## Workflow

1. Identificar o estudo em `studies/general/` (01–26; variantes `_individual` existem para 12–23 e 25).
2. Se precisar separar um grid combinado (01–08, 26) ou mudar formato, criar/ajustar
   variante `NN_*_individual.py` em `studies/general/` seguindo o padrão existente
   (`load_data.py`, backend Agg, salvar PNG+PDF).
3. Inserir o bloco `\begin{figure}` em `resultados.tex` com o path
   `sessoes/capitulo5/images/nvnl-slidhe/<subdir>/<arquivo>.pdf`, e mapear
   `<subdir>` para o módulo do estudo em `STUDY_BY_DIR`/`MODULE_OUTPUTS`
   (`gerar_figuras_cap5.py`) caso o diretório ainda não exista lá.
4. Gerar e publicar:
   ```bash
   cd dissertation/capitulo5
   python3 gerar_figuras_cap5.py --dry-run   # confere o manifesto
   python3 gerar_figuras_cap5.py             # gera e publica
   ```
5. Registrar a figura nas tabelas abaixo.

`gerar_figuras_cap5.py` lê os `\includegraphics` de `resultados.tex` (fonte da
verdade — não há lista fixa), roda cada estudo com as saídas redirecionadas para
`.build_figuras/` e publica **só** os PDFs referenciados em
`images/nvnl-slidhe/` e `figures/`. O tema de log vem do nome do arquivo:
sufixo `_cf<frac>` → `logs/client_frac`, senão `logs/sparsification_percentage`.
Nada é escrito dentro de `studies/`. Ele também publica o CSV de resumo do
Estudo 26 em `data/26_union_coverage/<tema>.csv`, consumido por
`gerar_tabelas_fedphe.py`.

Passo 2 continua valendo quando o formato de uma figura precisa mudar: a
variante ainda é criada em `studies/general/`; só a geração e a cópia saíram de lá.

**Regras** (CLAUDE.md): comparar técnicas sempre na mesma porcentagem; plaintext
como baseline em toda figura por porcentagem; nunca misturar porcentagens de
técnicas diferentes no mesmo gráfico.

Na dissertação as figuras vivem em `images/nvnl-slidhe/` — ao copiar para lá,
manter os nomes desta pasta e os paths dos `\includegraphics`.

### Revisão editorial (2026-07-21, pedido do autor)

`resultados.tex` reescrito integralmente com as diretrizes: sem "campanha"
(→ "conjunto de experimentos"), sem travessão como pontuação, sem
"sweep"/"grid" (→ "varredura"/"grade"), especificação condensada só em
`sec:partial_setup` (descrição dos 3 painéis movida para lá; parênteses
"(N_c=9, T=200, ...)" removidos do corpo), sem menções à "dissertação
anterior" (exceção: referências à hipótese da análise de seleção,
`sec:analysis_selection`), foco em ResNet-20/MobileNetV2 sobre CIFAR-10,
registro sóbrio (sem "eloquente" etc.), subsubsection "Extensão ao sweep
atual" **fundida** em `sec:sparsification_results` (labels
`sec:sparsification_full_model_baselines` e `sec:sparsification_current_sweep`
mantidos como aliases na subseção fundida), e "Estudo N" removido da prosa e
de todas as legendas (figuras e tabelas — as legendas em `tables/*.tex` foram
editadas manualmente; **se as tabelas forem regeneradas por
`gerar_tabelas_fedphe.py`, as legendas voltam a conter "(Estudo N, tema X)"
e precisam ser limpas de novo ou o gerador ajustado**). Figuras da seção de
esparsificação reorganizadas: resnet20 p030, resnet20 p010, mobilenetv2 p010,
**mobilenetv2 p050 (nova inserção, `fig:sparsification_p50_mobilenet`,
espelho atualizado)** e lenet5 p030 como contraponto de tarefa simples.

| Arquivo (figures/27_three_panel/) | Estudo | Label | Status |
|---|---|---|---|
| `mobilenetv2_CIFAR10_niid1_p050.pdf` | 27 | `fig:sparsification_p50_mobilenet` (painel superior) | inserida (2026-07-21) |
| `mobilenetv2_CIFAR10_niid0.1_p050.pdf` | 27 | `fig:sparsification_p50_mobilenet` (painel inferior) | inserida (2026-07-21) |

## Figuras

### Reestruturação da Seção 5.2 (2026-07-20): Figuras 5.4/5.5 e Tabela 5.2 antigas como "primeira parte"

Os experimentos ckks/bfv/batchcrypt full-model do protocolo atual não
concluíram a tempo. Por pedido do autor, `sec:sparsification_results` foi
reestruturada em duas subsubsections:

1. **`sec:sparsification_full_model_baselines`** ("Comparação com os
   baselines cifrados de modelo completo (dados da dissertação anterior)")
   — movida para o **topo** da subseção e expandida com o material completo
   já produzido na dissertação anterior: protocolo antigo ($N_c=10$,
   $T=50$), Tabela~`tab:results_legacy` (já existia), as **Figuras 5.4 e
   5.5 antigas** (`fig:partial_comparison_mnist`,
   `fig:partial_comparison_fashiomnist` — copiadas de
   `Yuri-Disserta-o-Masters/images/nvnl-slidhe/three/{MNIST,FASHION_MNIST}_p30_iid{1.0,0.1}.pdf`
   para `dissertation/capitulo5/images/nvnl-slidhe/three/`, ambas com as 6
   técnicas: Plaintext, BFV, BatchCrypt, CKKS, NVNL 30%, FedPHE 30%) e a
   **Tabela 5.2 antiga** portada como `tab:final_results_consolidation`
   (dados estáticos, hardcoded, sem script — grid CKKS/NVNL{30,50,80%}/
   FedPHE{30,50,80%} × {CIFAR10,FashionMNIST,MNIST,GTSRB} × {IID,não-IID},
   conferida número a número contra o arquivo antigo via `diff`).
2. **`sec:sparsification_current_sweep`** ("Extensão ao \textit{sweep}
   atual: mais arquiteturas e granularidade percentual") — nova
   subsubsection contendo o que antes era o topo sem título da subseção
   (`fig:sparsification_cifar10`/`fig:sparsification_fashionmnist` em
   $p=30\%$, `fig:sparsification_p10_*` em $p=10\%$,
   `tab:study27_summary`) — inalterado em conteúdo, só realocado e com um
   parágrafo de transição novo.

`sec:fedphe_direct_comparison` (Estudos 15/16/20/26) segue por último,
inalterada em conteúdo (só corrigida uma referência cruzada de "subseção
anterior" para `\ref{sec:sparsification_full_model_baselines}`, já que
agora há uma subsubsection intermediária).

| Arquivo (images/nvnl-slidhe/three/) | Tema | Label | Status |
|---|---|---|---|
| `MNIST_p30_iid1.0.pdf` | dissertação anterior ($N_c=10$, $T=50$) | `fig:partial_comparison_mnist` (subfigura IID) | inserida |
| `MNIST_p30_iid0.1.pdf` | idem | `fig:partial_comparison_mnist` (subfigura não-IID) | inserida |
| `FASHION_MNIST_p30_iid1.0.pdf` | idem | `fig:partial_comparison_fashiomnist` (subfigura IID) | inserida |
| `FASHION_MNIST_p30_iid0.1.pdf` | idem | `fig:partial_comparison_fashiomnist` (subfigura não-IID) | inserida |

### Estudo 27 — 3 painéis (acurácia×rodadas / MB×técnica / overhead×técnica)

`studies/general/27_three_panel.py` (novo script, criado nesta tarefa) recria as
figuras "three" do capítulo antigo com os dados do `sparsification_percentage`:
uma figura por (modelo, base, niid, porcentagem), com 3 painéis lado a lado:
acurácia por rodada, MB de \textit{upload} médio por rodada, e overhead de
processamento (treino+cifra+decifra, barra empilhada). Plota automaticamente os
baselines full-model (plaintext, ckks, bfv, batchcrypt) quando presentes nos
logs. **Atualização 2026-07-21**: os reruns ckks/bfv/batchcrypt do protocolo
atual concluíram (exceto batchcrypt mobilenetv2/CIFAR10 iid1, ainda rodando —
os dados parciais já dão visão estável do resultado e serão atualizados ao
final); as 50 figuras foram regeneradas (15:13) **com as curvas dos quatro
baselines full-model nos três painéis** e recopiadas para
`dissertation/capitulo5/figures/27_three_panel/` e para o espelho
`images/nvnl-slidhe/27_three_panel/` (as 8 já inseridas no .tex). O texto de
`sec:sparsification_full_model_baselines` e `sec:sparsification_current_sweep`
foi complementado (2026-07-21) para ler os baselines nas figuras: hierarquia
de payload (CKKS ~9×, BFV ~5×, BatchCrypt ~1,2× vs FedAvg), tempo/rodada do
BatchCrypt (93,1 s resnet20; 222,5 s mobilenetv2), degradação monotônica de
BFV/BatchCrypt no resnet20 (erro de quantização acumulado) e o hiato NVNL vs
full-model em p10 (~10 p.p. no mobilenetv2, fecha em p50).

Comando:
```bash
cd studies/general
LOGS_DIR=../../logs/sparsification_percentage python3 27_three_panel.py
```

Todas as 50 combinações (5 pares modelo/base × 2 niid × 5 porcentagens) foram
geradas e copiadas para `dissertation/capitulo5/figures/27_three_panel/`.
Quatro foram inseridas em `resultados.tex` (2026-07-19): as duas originais
(`fig:sparsification_cifar10`, `fig:sparsification_fashionmnist`, $p=30\%$)
mais duas novas em $p=10\%$ — o orçamento onde a distância NVNL-FL vs.
FedPHE é máxima (`fig:sparsification_p10_resnet`: gap de $42,9$pp;
`fig:sparsification_p10_mobilenet`: gap de $28,7$pp, critério explícito
"maior impacto do NVNL" pedido pelo autor). As demais 44 combinações **não**
entram como figura — ficam resumidas em `tables/tab_study27_summary.tex`
(acurácia final NVNL vs. FedPHE, 5 pares × 5 porcentagens, não-IID),
gerada por `build_study27_summary_table()` em `gerar_tabelas_fedphe.py`.
Regenerar o script `27_three_panel.py` quando os baselines full-model
terminarem de rodar — nenhuma mudança de código é necessária, os baselines
aparecerão automaticamente nos painéis.

| Arquivo (figures/27_three_panel/) | Tema | Estudo | Comando | Label | Status |
|---|---|---|---|---|---|
| `resnet20_CIFAR10_niid1_p030.pdf` | sparsification_percentage | 27 | acima | `fig:sparsification_cifar10` (painel superior) | inserida |
| `resnet20_CIFAR10_niid0.1_p030.pdf` | sparsification_percentage | 27 | acima | `fig:sparsification_cifar10` (painel inferior) | inserida |
| `lenet5_FASHION_MNIST_niid1_p030.pdf` | sparsification_percentage | 27 | acima | `fig:sparsification_fashionmnist` (painel superior) | inserida |
| `lenet5_FASHION_MNIST_niid0.1_p030.pdf` | sparsification_percentage | 27 | acima | `fig:sparsification_fashionmnist` (painel inferior) | inserida |
| `resnet20_CIFAR10_niid1_p010.pdf` | sparsification_percentage | 27 | acima | `fig:sparsification_p10_resnet` (painel superior) | inserida |
| `resnet20_CIFAR10_niid0.1_p010.pdf` | sparsification_percentage | 27 | acima | `fig:sparsification_p10_resnet` (painel inferior) | inserida |
| `mobilenetv2_CIFAR10_niid1_p010.pdf` | sparsification_percentage | 27 | acima | `fig:sparsification_p10_mobilenet` (painel superior) | inserida |
| `mobilenetv2_CIFAR10_niid0.1_p010.pdf` | sparsification_percentage | 27 | acima | `fig:sparsification_p10_mobilenet` (painel inferior) | inserida |
| `{dnn_tiny,lenet5,mobilenetv2,resnet20,vit_tiny}_{FASHION_MNIST,CIFAR10,tanganke_gtsrb}_niid{0.1,1}_p{010,030,050,070,090}.pdf` (42 restantes) | sparsification_percentage | 27 | acima | — | disponível, resumido em `tab:study27_summary` |

### Estudos 15, 16, 20, 22, 26 — variantes PT-BR separadas por config (`_pt.py`)

Cinco novos scripts em `studies/general/` recriam os estudos 15, 16, 20, 22 e
26 (as duas figuras: curvas de cobertura + inflação de downlink) totalmente
em português, com nomes de solução corretos (FedAvg, NVNL, FedPHE, SW, RW —
via `studies/general/solution_style.py`, novo módulo compartilhado). No tema
`sparsification_percentage` (fração de clientes sempre 100%) o rótulo de
fração de clientes é omitido; no tema `client_frac` ele aparece no título e
no nome do arquivo (`_cf0.25`, `_cf0.5`, `_cf0.75`).

Os estudos 15 e 16 têm uma figura por configuração completa (modelo, base,
niid, porcentagem[, fração de clientes]) e comparam **apenas as quatro
técnicas de envio parcial estruturado (\textit{packing})** — NVNL, FedPHE,
SW, RW (`data[data["solution"].isin(sstyle.PACKING_SOLS)]`, 2026-07-21).
O FedAvg (plaintext) já havia sido excluído por pedido do autor
(2026-07-18); **CKKS, BFV e BatchCrypt também foram excluídos** (mesmo
quando presentes nos logs, o que já ocorre para algumas combinações do
\textit{sweep} atual) — a métrica "bytes/tempo acumulado até a acurácia
máxima" só faz sentido comparativo entre técnicas que fazem
\textit{packing}; HE de modelo completo não é um recorte de comparação
pretendido nestes dois estudos. **Lembrar disso ao escrever texto novo a
partir dessas figuras**: Estudos 15/16 = só NVNL/FedPHE/SW/RW, nunca
CKKS/BFV/BatchCrypt/FedAvg. A comparação com CKKS/BFV/BatchCrypt/FedAvg
continua coberta em outro lugar do capítulo (§`sec:sparsification_full_model_baselines`,
Tabelas `tab:results_legacy`/`tab:final_results_consolidation`).

O estudo 26 tem granularidade assimétrica entre suas duas figuras
(2026-07-18): as **curvas de cobertura da união** (`26_union_coverage_curves_pt/`)
voltaram a ser uma figura por (modelo, base, niid, porcentagem[, fração de
clientes]) — mesmo padrão de 15/16/20/22 — enquanto as **barras de inflação
de downlink** (`26_downlink_inflation_pt/`) continuam com **todas as
porcentagens combinadas na mesma figura** por (modelo, base, niid[, fração de
clientes]), grupos de barras por porcentagem no eixo X, igual ao estudo 26
original em inglês. Os estudos 20, 22 e 26 dependem de `mask.csv` (só gerado
pelas técnicas de empacotamento: NVNL, SW, RW, FedPHE) — os baselines
full-model nunca aparecerão neles, por design.

Comando (mesmo padrão para os 5 scripts):
```bash
cd studies/general
LOGS_DIR=../../logs/sparsification_percentage python3 15_bytes_vs_maxacc_pt.py
LOGS_DIR=../../logs/client_frac python3 15_bytes_vs_maxacc_pt.py
# idem para 16_overhead_vs_maxacc_pt.py, 20_pack_frequency_pt.py,
# 22_layer_heatmap_pt.py, 26_union_coverage_downlink_pt.py
```

**Pasta dedicada**: todas as figuras destes 5 scripts caem sob um subdiretório
próprio `capitulo5_pt/` — separado dos ~30 estudos em inglês (`01_*`…`27_*`)
que não serão usados na dissertação. Path completo:
`studies/{sparsification_percentage,client_frac}/figures/capitulo5_pt/<estudo>_pt/`.

**Importante — separação física por tema**: os scripts sempre escrevem no
diretório temporário compartilhado `studies/general/output/figures/`. Rodar
os comandos acima para os dois temas em sequência sem mover o resultado entre
uma rodada e outra mistura os arquivos dos dois temas na mesma pasta (o
sufixo `_cf` no nome do arquivo distingue visualmente, mas não separa
fisicamente). Depois de gerar para um tema, copiar
`studies/general/output/figures/capitulo5_pt/<estudo>_pt/` para
`studies/<nome_do_tema>/figures/capitulo5_pt/<estudo>_pt/` (apagando o
conteúdo antigo dessas pastas de destino antes) e só então rodar para o outro
tema — ou usar `bash studies/general/run_all.sh <nome_do_tema>`, que já faz
essa cópia/mover automaticamente ao final de cada execução.

| Pasta (figures/capitulo5_pt/) | Estudo | sparsification_percentage | client_frac | Status |
|---|---|---|---|---|
| `15_bytes_maxacc_pt/` | 15 | 100 arquivos (50 png+pdf) | 48 arquivos (24 png+pdf) | gerado (sem FedAvg) |
| `16_overhead_maxacc_pt/` | 16 | 100 | 48 | gerado (sem FedAvg) |
| `20_pack_frequency_pt/` | 20 | 400 (200 png+pdf) | 192 (96 png+pdf) | gerado |
| `22_layer_heatmap_pt/` | 22 | 400 | 192 | gerado |
| `26_union_coverage_curves_pt/` | 26 (curvas) | 100 (50 png+pdf) | 48 (24 png+pdf) | gerado (1 por porcentagem) |
| `26_downlink_inflation_pt/` | 26 (barras) | 20 (10 png+pdf) | 24 (12 png+pdf) | gerado (porcentagens combinadas) |

Correção aplicada (2026-07-17): nos estudos 20 e 22 o rótulo vertical da
colorbar estava sendo cortado pela borda da figura (`bbox_inches="tight"` sem
margem suficiente para a fonte grande). Corrigido reduzindo a fonte do rótulo/
ticks da colorbar (`cbar.set_label(..., fontsize=6.5)`,
`cbar.ax.tick_params(labelsize=6)`) e aumentando `pad_inches` de `0.02` para
`0.08` no `savefig`. Todas as figuras destes dois estudos foram regeneradas.

### Comparação direta com o FedPHE (Estudos 15, 16, 20, 26) — inserida em `resultados.tex` (2026-07-18)

Nova subsubsection `sec:fedphe_direct_comparison`, logo após a comparação com
os baselines full-model (§`sec:sparsification_full_model_baselines`), usando
$p=30\%$ e os mesmos dois pares já usados no resto do capítulo
(ResNet-20/CIFAR-10, LeNet-5/FashionMNIST), tema `sparsification_percentage`
apenas. Descrita como um experimento progressivo: custo em bytes (15) → custo
em tempo (16) → mecanismo de seleção por pacote (20) → custo de downlink
(26). Arquivos copiados de
`studies/sparsification_percentage/figures/capitulo5_pt/` para
`dissertation/capitulo5/figures/capitulo5_pt/` (mantendo os nomes originais):

| Arquivo (figures/capitulo5_pt/) | Estudo | Label | Status |
|---|---|---|---|
| `15_bytes_maxacc_pt/resnet20_CIFAR10_niid0.1_p030.pdf` | 15 | `fig:fedphe_bytes` | inserida |
| `15_bytes_maxacc_pt/lenet5_FASHION_MNIST_niid0.1_p030.pdf` | 15 | — (citada na Tabela `tab:fedphe_bytes`, não incluída como figura) | disponível |
| `16_overhead_maxacc_pt/resnet20_CIFAR10_niid0.1_p030.pdf` | 16 | `fig:fedphe_overhead` | inserida |
| `16_overhead_maxacc_pt/lenet5_FASHION_MNIST_niid0.1_p030.pdf` | 16 | — (citada na Tabela `tab:fedphe_overhead`, não incluída como figura) | disponível |
| `20_pack_frequency_pt/resnet20_CIFAR10_fedphe_niid0.1_p030.pdf` | 20 | `fig:fedphe_pack_freq` (painel superior) | inserida |
| `20_pack_frequency_pt/resnet20_CIFAR10_nvnl_niid0.1_p030.pdf` | 20 | `fig:fedphe_pack_freq` (painel inferior) | inserida |
| `26_union_coverage_curves_pt/resnet20_CIFAR10_niid0.1_p030.pdf` | 26 (curvas) | `fig:fedphe_union_coverage` | inserida |
| `26_union_coverage_curves_pt/lenet5_FASHION_MNIST_niid0.1_p030.pdf` | 26 (curvas) | — | disponível |
| `26_downlink_inflation_pt/resnet20_CIFAR10_niid0.1.pdf` | 26 (barras) | `fig:fedphe_inflation` | inserida |
| `26_downlink_inflation_pt/lenet5_FASHION_MNIST_niid0.1.pdf` | 26 (barras) | — | disponível |

As demais combinações (outros pares modelo/base, IID, outras porcentagens)
permanecem disponíveis nos diretórios de origem, não copiadas para o
capítulo — aguardando ditado do autor para uma eventual extensão futura
desta subsubsection.

### Efeito da participação parcial (Estudos 15, 20, 26, tema `client_frac`) — inserida em `resultados.tex` (2026-07-19)

Nova subsubsection `sec:client_frac_fedphe_comparison`, substituindo os
`% TODO` que preenchiam `sec:client_frac_results`, usando ResNet-20/CIFAR-10,
não-IID ($\alpha=0{,}1$), variando $f_c\in\{0,25;\,0,5;\,0,75\}$. Mesma
estrutura de experimento progressivo da subsubsection irmã
(§`sec:fedphe_direct_comparison`), agora com foco no argumento de que a
variação de quais clientes participam por rodada pode inflar o
\textit{download} do servidor muito além do \textit{upload} individual — o
NVNL-FL e o SW mantêm inflação de \textit{downlink} $1,00\times$ em toda
combinação de $p$/$f_c$ testada (Tabela `tab:cf_downlink`), enquanto o RW
chega a $\sim\!7,9\times$ e o FedPHE a $\sim\!1,95\times$. Estudo 16 citado
apenas em prosa (sem figura nova, para não repetir uma figura quase idêntica
à do Estudo 15). Arquivos copiados de
`studies/client_frac/figures/capitulo5_pt/` para
`dissertation/capitulo5/figures/capitulo5_pt/`:

| Arquivo (figures/capitulo5_pt/) | Estudo | Label | Status |
|---|---|---|---|
| `15_bytes_maxacc_pt/resnet20_CIFAR10_niid0.1_p010_cf0.5.pdf` | 15 | `fig:cf_bytes` | inserida |
| `20_pack_frequency_pt/resnet20_CIFAR10_fedphe_niid0.1_p010_cf0.5.pdf` | 20 | `fig:cf_pack_freq` (painel superior) | inserida |
| `20_pack_frequency_pt/resnet20_CIFAR10_nvnl_niid0.1_p010_cf0.5.pdf` | 20 | `fig:cf_pack_freq` (painel inferior) | inserida |
| `26_union_coverage_curves_pt/resnet20_CIFAR10_niid0.1_p010_cf0.5.pdf` | 26 (curvas) | `fig:cf_union_coverage` | inserida |
| `26_downlink_inflation_pt/resnet20_CIFAR10_niid0.1_cf0.75.pdf` | 26 (barras) | `fig:cf_inflation` | inserida |

As demais combinações (mobilenetv2/CIFAR10, IID, $p=50\%$ como figura,
demais $f_c$) permanecem disponíveis em `studies/client_frac/figures/capitulo5_pt/`,
não copiadas para o capítulo — os números de $p=50\%$ são citados apenas em
prosa (Tabela `tab:cf_downlink` já cobre os dois percentuais).

### Estudo 28 (novo, 2026-07-22; corrigido 2026-07-22; RW excluído 2026-07-22) — download acumulado vs. acurácia, experimento completo, correlaciona com o Estudo 26

`studies/general/28_download_vs_maxacc_pt.py`: mesmo formato visual dos
Estudos 15/16 (curva acurácia × custo acumulado + painel lateral por
solução, legenda única embaixo), mas o eixo de custo é o **download
acumulado** em vez do upload — reaproveita (via import, sem duplicar) a
cobertura da união por rodada já calculada em
`26_union_coverage_downlink_pt.py::build_series_df`. Fórmula:

```
download_bytes_rodada = (union_cov_pct/100 × n_params × 4) × n_clientes_na_rodada
```

**Correção (2026-07-22)**: a primeira versão copiava do Estudo 15 o
recorte da curva no **pico de acurácia** e o resumo de "acurácia máxima" no
painel lateral — mas isso não faz sentido aqui: o objetivo do Estudo 28 é
correlacionar com o Estudo 26, que sempre reporta a série **completa**
(todas as rodadas), sem recorte. Corrigido: a curva principal agora vai até
a última rodada registrada de cada solução (sem `_clip_at_peak`); o painel
lateral resume **acurácia final / rodada final / download total
acumulado** (`_final_row`/`_final_acc`/`_final_round`/`_cost_at_final`,
helpers locais novos — pegam a última linha da série, não o argmax de
acurácia); a linha "melhor" passa a comparar acurácia final entre soluções.
Catálogo (148 arquivos) regenerado e recopiado para os 4 locais abaixo
após a correção.

**RW excluído (2026-07-22, pedido do autor)**: `STUDY_SOLS =
sstyle.PACKING_SOLS - {"rw"}` (constante nova no script, usada no filtro de
`main()` em vez de `sstyle.PACKING_SOLS` diretamente). Escopo do estudo
passa a ser NVNL/FedPHE/SW apenas. **Justificativa a desenvolver no texto
da dissertação** (linha argumentativa indicada pelo autor, não escrita
ainda): a comparação deste estudo deve ficar restrita a métodos de seleção
que não sejam inteiramente aleatórios — NVNL (round-robin determinístico),
FedPHE (Top-$k$ por magnitude, adaptativo) e SW (janela deslizante
determinística) seguem alguma estrutura; RW sorteia os pacotes de forma
independente a cada rodada, então incluí-lo nessa comparação específica
mediria principalmente o efeito da aleatoriedade, não da estrutura de
seleção. Catálogo (148 arquivos, agora só 3 soluções por figura)
regenerado e recopiado para os 4 locais abaixo após a exclusão.

Isso liga diretamente os dois estudos: a razão `download_acumulado /
upload_acumulado` ao longo do treino se aproxima do fator `inflation` já
calculado no Estudo 26. **Validação feita** (config resnet20/CIFAR10,
não-IID, $p=30\%$, comparando com `26_union_coverage_downlink_summary_pt.csv`):
FedPHE (1,32 vs. inflação 1,26), SW (1,05 vs. 1,00) e RW (1,76 vs. 1,68)
bateram razoavelmente bem (a pequena diferença é esperada: o Estudo 26 usa
média por rodada após aquecimento, aqui é razão dos totais acumulados no
round final). **NVNL não bateu** (razão 0,53 quando o esperado é ~1,00) —
não é um bug do Estudo 28 (que usa `mask.csv` deduplicado, igual ao Estudo
26), mas sim um **bug pré-existente nos Estudos 15/16**: `train.csv` tem
linhas duplicadas de reruns para a solução `nvnl` em **todas** as 50
combinações de `sparsification_percentage` (2× o número de linhas esperado
— confirmado por contagem, não afeta fedphe/sw/rw, só nvnl), e
`compute_cumulative_bytes`/`compute_cumulative_overhead` (Estudos 15/16) não
deduplicam antes de somar por rodada, ao contrário do Estudo 26/28 que já
deduplicam `mask.csv` por `(config, round, cid)`. **Efeito prático**: os
números de "MB acumulado"/"overhead acumulado" do NVNL já publicados em
`tab:fedphe_bytes`/`tab:fedphe_overhead` (e no texto de
`sec:fedphe_direct_comparison`) estão provavelmente ~2× inflados. Ainda não
corrigido — ver aviso separado ao autor.

**Inserido em `resultados.tex` (2026-07-21)**: novo
`\paragraph{Custo acumulado de \textit{download} (Estudo 28).}` em
`sec:fedphe_direct_comparison`, após o parágrafo do Estudo 26 (intro da
subsubsection atualizada de "quatro" para "cinco lentes"; Síntese idem). A
justificativa da exclusão do RW foi desenvolvida no texto conforme a linha
argumentativa indicada pelo autor (comparação restrita a seleções com
estrutura; RW mediria só o efeito da aleatoriedade, já quantificado pela
inflação 1,68× do Estudo 26). Números citados em prosa de outras configs
(sempre mesma porcentagem entre técnicas): resnet20 niid0.1 p010
(256,7 vs 205,6 MB; 23,7% vs 66,6%), p050 (1.914,3 vs 998,5 MB, acurácias
equivalentes 76,5–77,5%); mobilenetv2 niid1 p010 (551,1 vs 527,7 MB; 24,1%
vs 65,9%), p050 (3.683,7 vs 2.579,9 MB; 65,8% vs 74,8%). A razão
download/upload da figura inserida (776,8/616,7 ≈ 1,26) bate com o fator de
inflação do FedPHE no Estudo 26 (validação cruzada citada no texto).

| Arquivo (figures/capitulo5_pt/) | Estudo | Label | Status |
|---|---|---|---|
| `28_download_maxacc_pt/resnet20_CIFAR10_niid0.1_p030.pdf` | 28 | `fig:fedphe_download_cum` | inserida |
| demais 147 arquivos (2 temas) | 28 | — (citados em prosa: resnet20 p010/p050 niid0.1, mobilenetv2 p010/p050 niid1) | disponíveis |

Gerado para os dois temas (`sparsification_percentage`: 50
figuras; `client_frac`: 24 figuras). **Catálogo completo já copiado**
(2026-07-22, mesmo precedente do Estudo 27 — copiar tudo, não só um
recorte) para:
- `studies/sparsification_percentage/figures/capitulo5_pt/28_download_maxacc_pt/` (100 arquivos)
- `studies/client_frac/figures/capitulo5_pt/28_download_maxacc_pt/` (48 arquivos)
- `dissertation/capitulo5/figures/capitulo5_pt/28_download_maxacc_pt/` (148 arquivos, os dois temas juntos)
- `dissertation/capitulo5/images/nvnl-slidhe/capitulo5_pt/28_download_maxacc_pt/` (148 arquivos, espelho local usado pelos `\includegraphics`)

Disponível para escolha de figuras específicas quando o texto for escrito,
em conjunto com o Estudo 26.

## Tabelas

| Arquivo (tables/) | Tema | Estudo | Label | Status |
|---|---|---|---|---|
| _(tabela `tab:results_legacy` inserida diretamente em resultados.tex — dados da dissertação anterior, $N_c=10$, $T=50$, 6 técnicas incl. FedPHE, portada de `Yuri-Disserta-o-Masters/sessoes/capitulo5/resultados.tex` em 2026-07-19, ver nota no .tex)_ | sparsification_percentage | — | `tab:results_legacy` | base reaproveitada — substituir quando ckks/bfv/batchcrypt do sweep atual terminarem (renomeada de `tab:results_placeholder`) |
| _(tabela `tab:final_results_consolidation` inserida diretamente em resultados.tex — dados estáticos da dissertação anterior, $N_c=10$, $T=50$, CKKS/NVNL{30,50,80%}/FedPHE{30,50,80%} × 4 bases, portada de `Yuri-Disserta-o-Masters/sessoes/capitulo5/resultados.tex` (antiga Tabela 5.2) em 2026-07-20, conferida número a número via `diff`)_ | sparsification_percentage | — | `tab:final_results_consolidation` | base reaproveitada — substituir quando ckks/bfv/batchcrypt do sweep atual terminarem |
| `tab_fedphe_bytes.tex` | sparsification_percentage | 15 | `tab:fedphe_bytes` | inserida (`\input`), gerada por `dissertation/capitulo5/gerar_tabelas_fedphe.py` |
| `tab_fedphe_overhead.tex` | sparsification_percentage | 16 | `tab:fedphe_overhead` | inserida (`\input`), idem |
| `tab_fedphe_downlink.tex` | sparsification_percentage | 26 | `tab:fedphe_downlink` | inserida (`\input`), idem (filtra `26_union_coverage_downlink_summary_pt.csv`) |
| `tab_cf_bytes.tex` | client_frac | 15 | `tab:cf_bytes` | inserida (`\input`), idem |
| `tab_cf_downlink.tex` | client_frac | 26 | `tab:cf_downlink` | inserida (`\input`), idem (filtra `studies/client_frac/tables/26_union_coverage_downlink_summary_pt.csv`) |
| `tab_study27_summary.tex` | sparsification_percentage | 27 | `tab:study27_summary` | inserida (`\input`), gerada por `build_study27_summary_table()` — resume as 42 combinações de `27_three_panel` não inseridas como figura |
