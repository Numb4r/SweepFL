# Estouro de faixa do CKKS de primo único (MobileNetV2, niid1, p ≥ 0.7)

Data: 2026-09-22. Backend: `ckks_python_simplify`, `N=8192`, `SIMPLIFY_BITS=62`, `SIMPLIFY_SCALE_BITS=40`.

## Resumo

As 5 runs quebradas de `sparsification_percentage` (MobileNetV2/CIFAR-10, α=1, seed 42:
nvnl/sw/rw/fedphe p=0.9 e fedphe p=0.7) sofreram **estouro modular na soma agregada do
servidor**. Com `MAGG_RENORM=1` o cliente cifra `w·D_i`. O servidor soma os 9
ciphertexts, então o plaintext decifrado é `Σ_i D_i·w_i ≈ 50 000·w`. Quando um
**coeficiente do polinômio** codificado passa de q/(2Δ) ≈ 2²¹, a decifragem erra por
exatamente q/Δ = 2²² em todos os slots daquele pacote. Depois da renormalização por
M_agg = 50 000, isso vira **±83,9 em cada peso do pacote**.

A hipótese inicial está certa quanto ao mecanismo, mas precisa de duas correções:

1. **O limite vale para a soma, não para o cliente.** O servidor faz soma pura
   (`server/server_utils.py`, `_pure_sum`). O limiar relevante é `ΣD·|w|`, não `D_i·|w|`.
2. **O limite vale para coeficientes, não para slots.** O encoder usa a imersão canônica
   (`ckks_project/src/ckks/encoder.cpp`), com `|a_k| ≤ (2/N)·Σ_j |z_j|`. Uma coordenada
   isolada grande é diluída por N/2 e não estoura. O que estoura é um pacote com **média**
   alta, porque um vetor constante v codifica no polinômio constante v. O limite
   por coordenada de `CKKS_PARAMS.md` é suficiente, mas não necessário: a razão por slot
   passou de 1 já na rodada 60 sem quebrar nada.

O pacote culpado é sempre o **172** (de 176). Ele contém `Conv_1_bn/moving_mean`,
`Conv_1_bn/moving_variance` (1280 canais, média ≈ 132 na rodada 103) e o início de
`dense/kernel`. Como só as estatísticas móveis de BN são usadas exclusivamente na
inferência, o treino continua saudável (o treino usa estatísticas do batch, e o momentum
de BN recompõe `moving_*` durante a época local). Só a avaliação do agregado quebra.

## Evidência

Snapshots `model/round_*.npz` (cid 0, agregado decifrado) baixados do cluster para uma
pasta temporária (nada em `logs/` foi alterado). Script: `experiments/tests/ckks_range_probe.py`.
Ele reconstrói `ΣD·w` a partir do snapshot da rodada r, que aproxima o plaintext da rodada r+1.

**Pesos na quebra (nvnl p=0.9, rodada 103 → 104):**

| array | r103 | r104 |
|---|---|---|
| `Conv_1_bn/moving_variance` média / mín | 132,1 / 21,2 | 50,0 / **−65,1** (351 negativos) |
| `Conv_1_bn/moving_mean` mín | — | −89,1 |
| `dense/kernel` mín | — | −85,4 |

Variância negativa gera NaN em `sqrt(var+ε)` na inferência, e a acurácia vai a 0,100.
Os deslocamentos de −82 a −89 batem com 2²²/50 000 = 83,9.

**Razão do pior coeficiente para q/(2Δ) (pacote 172):**

| rodada | nvnl p0.9 α=1 (quebrou) | nvnl p0.5 α=1 (ok) | nvnl p0.9 α=0.1 (ok) |
|---|---|---|---|
| 60 | 0.238 | 0.254 | 0.213 |
| 72 | 0.283 | — | — |
| 73 (resume) | **0.455** | — | — |
| 90 | 0.770 | 0.363 | 0.289 |
| 100 | 0.925 | 0.404 | 0.315 |
| 103 | **0.993** → quebra em 104 | — | — |
| 114 | 1.022 (quebra de novo) | — | — |
| 124 | 1.338 | — | — |
| 130 | 0.975 | 0.525 | 0.399 |

A razão sobe monotonicamente até 0,993 na rodada 103, e a rodada 104 quebra. O plaintext
real da rodada 104 é `Σ D_i·w_i` após o treino local, um pouco maior que o estimado a
partir do snapshot 103. As rodadas 105–130 oscilam em torno de 1 e alternam quebras e
recuperações, o mesmo padrão de `server_eval.csv`.

**Confirmação com a biblioteca real** (`--verify-round 103`): o pacote 172 da rodada 103
foi cifrado como 9 partes, somado homomorficamente e decifrado.
- como está (razão 0,993): erro máx. 4,8e-7, sem estouro.
- 5% maior (≈ uma rodada de deriva): erro máx. **4,194e6 = q/Δ**, ou 83,9 por peso depois de /ΣD.

## Fator agravante: resume

As 4 runs p=0.9 que quebraram são **runs retomadas** (`RESUME_ROUND` 72, 85, 85, 92).
Na fronteira do resume a razão salta de 0,283 para 0,455, e depois cresce cerca de 3×
mais rápido (≈0,015/rodada contra ≈0,004 antes). O checkpoint guarda só os pesos em
float16. O estado do Adam de cada cliente é zerado no resume. Isso é plausível como
causa do salto, mas **não foi isolado**. A fedphe p=0.7, que não foi retomada, também
quebra, só que tarde (rodada 196). Portanto o estouro acontece sem resume, e o resume
só o antecipa. A correção abaixo resolve os dois casos. O reset do Adam no resume
continua sendo uma questão separada.

## Correção adotada: pré-escala pública `MAGG_D_REF = 1024`

`client/client_utils.py::MAGG_D_REF`. O cliente cifra `w·D_i/1024`
(`client/literature/{NVNL,sw,rw,fedphe,srr,topkrr,*_scout}.py`), e
`client/literature/set_parameters.py` divide por `M_agg/1024`. Como 1024 é uma constante
pública e potência de 2, o FedAvg resultante é idêntico e a divisão em float é exata.

| Opção | Bytes por ciphertext | Custo |
|---|---|---|
| Subir `SIMPLIFY_BITS` | 62 já é o teto de um primo em uint64 (`llround` em int64 no encoder). Mais faixa exige um 2º primo (RNS): 131 072 → 262 144 B, expansão 8× → 16×, ~2× no tempo HE. | Muda a contabilidade do artigo. |
| Baixar `SIMPLIFY_SCALE_BITS` 40 → 30 | inalterado | 1024× de folga, precisão ~2⁻³⁰ relativa, mas muda um parâmetro da família em relação às células p ≤ 0.5. |
| **Pré-escala `D_i/1024` (adotada)** | **inalterado** | Razão ~1000× menor (0,001 no pior caso observado). Erro por peso ~1e-11 → ~7e-9 (medido em `tests/test_range_logs.py`), abaixo do ulp float32 de pesos ~0,1. Parâmetros CKKS idênticos à família. Nenhum custo de tempo. |
| BN à parte (claro ou local, FedBN) | menor | Muda o método, e BN em claro vaza informação. Muda também a contabilidade de bytes. |

`D_round` (ΣD da rodada) foi descartado: o cliente só conheceria esse valor depois da
rodada 1, e uma constante pública resolve o mesmo problema sem mudar o protocolo.

Resultado do probe com a correção (`--prescale 1024`) nas mesmas rodadas: razão 0,001
em 103, 114, 124 e 130.

**Teste ponta a ponta (Docker, CPU, resnet20, 2 clientes, 3 rodadas, nvnl p=0.1):** o A/B no mesmo dia com `MAGG_D_REF=1` (idêntico ao código antigo) e `=1024` deu acurácia 0,361/0,412/0,426 contra 0,355/0,422/0,438. A diferença fica dentro da variação entre execuções: uma segunda run com 1024 deu 0,448 na rodada 3. As runs de maio do mesmo smoke test chegavam a ~0,66 na rodada 3. Essa diferença também aparece com `MAGG_D_REF=1` e, portanto, **não vem da correção** (candidatos: init seed-scoped e outras mudanças desde maio; não investigado). Saídas em `logs/test_ckks_backend/ab_dref_{1,1024}/`.

**Comparabilidade:** para as células que nunca estouraram (p ≤ 0.5, e p=0.7 exceto
fedphe), a correção só muda o arredondamento do CKKS na ordem de 1e-8 por peso. As runs
antigas continuam válidas.

## Instrumentação: `range.csv`

Uma linha por (rodada, cliente), escrita por `client_logs.write_range_logs` fora das
janelas cronometradas. Só é gravada com `ckks_python_simplify`. O `train.csv` não muda.

| coluna | significado |
|---|---|
| `d_i`, `bound` | tamanho local; q/(2·scale) |
| `max_abs_w` | max\|w\| antes de ×D_i/D_REF |
| `max_abs_wd`, `argmax_pack`, `slot_ratio` | max\|slot\| nos pacotes enviados, índice do pacote e razão para `bound` |
| `coef_max`, `coef_argmax_pack`, `coef_ratio` | o mesmo, em coeficientes codificados (a razão que realmente estoura, ≥ 1) |
| `per_pack_coef` | max\|coef\| de todos os pacotes (`;`). Somando as linhas dos 9 clientes obtém-se um limite superior para a soma do servidor. |

## Limitação conhecida (para o artigo)

Com um único primo de 62 bits e Δ=2⁴⁰, a faixa decifrável da **soma** do servidor é
|a_k| < 2²¹ por coeficiente do polinômio. Em slots, isso equivale a ≈ 2²¹ para a média
do pacote e a ≈ 2²¹·N/2 para uma coordenada isolada. Sem pré-escala, o protocolo com peso
por D_i estoura quando ΣD·(média do pacote) chega a 2²¹, o que acontece com
MobileNetV2/CIFAR-10 (ΣD = 50 000) quando `moving_variance` de `Conv_1_bn` passa de ~130.
A pré-escala desloca esse limite por um fator de 1024. Ele continua existindo, a
~2³¹/ΣD na média do pacote.

## Rerun: `experiments/range_fix/`

`range_fix_{nvnl,sw,rw,fedphe}.sh`: mobilenetv2:CIFAR10, α=1, p ∈ {0.7, 0.9}, seed 42,
9 clientes, 200 rodadas, `CLIENT_FRACTION=1.0`, com os parâmetros CKKS da família. São 8 runs,
com SLURM em `sbatch/range_fix/sbatch_range_fix_*.sh` (l40s, 3 GPUs, `--time=0-6:00:00`).
Estimativa (`estimate_time.py --all --slurm SLURM.md`): 1h15m por script no l40s sem
margem, 1h53m com margem de 1,5×. O limite de 6 h existe para evitar resume, porque
todas as falhas em p=0.9 ocorreram em runs retomadas.

Aceitação: nas 8 runs, `server_eval.csv` tem 200 linhas, todas com `mean_loss` finito,
nenhuma `mean_acc == 0.100`, e `max(coef_ratio) < 1` em `range.csv`.

### Tabela de comparação (acurácia final na rodada 200, α=1)

Células atuais da família (`logs/sparsification_percentage/`). A coluna `range_fix` será
preenchida quando as runs terminarem.

| técnica | p=0.1 | p=0.3 | p=0.5 | p=0.7 (antiga) | p=0.9 (antiga) | p=0.7 range_fix | p=0.9 range_fix |
|---|---|---|---|---|---|---|---|
| nvnl | 0.6496 | 0.7146 | 0.7515 | 0.7638 | 0.1000 (NaN) | — | — |
| sw | 0.6065 | 0.7032 | 0.7406 | 0.7552 | 0.1000 (NaN) | — | — |
| rw | 0.5999 | 0.6989 | 0.7313 | 0.7551 | 0.1000 (NaN) | — | — |
| fedphe | 0.2414 | 0.5729 | 0.6578 | 0.1000 (NaN) | 0.1000 (NaN) | — | — |
| plaintext | 0.7629 | | | | | | |
