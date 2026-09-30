# CLAUDE.md — Verificação: FedPHE (agregação segura, sem seleção de clientes)

## Objetivo deste arquivo

Este documento é a **especificação de referência** para auditar uma implementação já
existente do FedPHE restrita ao pilar de **agregação segura**. Ele NÃO pede para
construir do zero. A tarefa do agente é: ler o código existente, mapeá-lo contra as
invariantes e equações abaixo, rodar os testes de correção e apontar onde a
implementação diverge do paper.

Paper de referência: Li et al., "FedPHE: A Secure and Efficient Federated Learning via
Packed Homomorphic Encryption", IEEE TDSC, vol. 22 no. 5, 2025. As seções/equações/tabelas
citadas abaixo (ex.: "Eq. 8", "Fig. 5", "Tab. IV") são desse paper.

## Escopo: o que está DENTRO e o que está FORA

DENTRO (precisa existir e estar correto):

- CKKS packing: empacotar B valores de plaintext em 1 ciphertext (Sec. III-B, Fig. 4).
- Sparsificação no nível do pacote (pack-level top-k) por norma L2 (Sec. III-B, Fig. 5b, Tab. IV).
- Máscara binária por pacote `M_i` e alinhamento de pacotes no servidor.
- Agregação no ciphertext (ponderada ou não, ver "Pesos" abaixo).
- Decriptação, deobfuscação (se houver) e unpack no cliente.
- (Opcional, camada de segurança) obfuscação por secret sharing (Sec. III-B, Eq. 4-5, 8-9).

FORA (não deve ser implementado nem auditado aqui; se existir no código, marcar como fora de escopo):

- Sketching com LSH, matriz de mapeamento `M`, sketches `h_i` (Sec. III-C, Alg. 3).
- Vetores de perturbação `V_c`, `V_{s,i}`.
- Clustering, gap statistic, K-means.
- Prioridade/seleção do cliente mais rápido `F_i^t` (Eq. 12).
- Qualquer coisa do Algoritmo 3.

Consequência da remoção da seleção: o conjunto de participantes é **todos os clientes**
(ou um subconjunto fixo/aleatório, se o código quiser, mas a *lógica de seleção* está fora
de escopo). O peso de contribuição `p_i` da Eq. 7 depende de similaridade de sketches, que
não existe mais. Ver "Pesos de agregação".

## Pipeline de referência por rodada (cliente i, servidor PS)

Cliente (após treino local, obtido o update `g_i` achatado para vetor de tamanho d):

1. Particionar `g_i` em K pacotes de tamanho B, com K = ceil(d / B). Último pacote com zero-padding.
2. Para cada pacote k, calcular a norma L2 `||pack_k||_2`.
3. Construir máscara `M_i ∈ {0,1}^K`: 1 para os `ceil(ζ·K)` pacotes de maior norma L2, 0 nos demais.
4. (Opcional, segurança) obfuscar os pacotes mantidos somando `Δ_i` (ver "Obfuscação").
5. Encriptar com CKKS (chave pública) apenas os pacotes com máscara 1 → ciphertext `C_i`
   (tipicamente um ciphertext CKKS por pacote mantido).
6. Enviar `{C_i, M_i}` ao PS. A máscara vai em **plaintext**.

Servidor (PS):

7. Para cada índice de pacote k, agregar somente sobre os clientes que mantiveram k (M_i[k]=1):
   `C[k] = Σ_{i: M_i[k]=1} (peso_i ⊗ C_i[k])`.
8. Computar a massa de pesos por pacote `M_agg[k] = Σ_{i: M_i[k]=1} p_i`.
9. Despachar `{C, M_agg}` (e o fator global de obfuscação R, se houver) a todos os clientes.

Cliente (recepção):

10. Decriptar `P = D(C, sk)`; se houver obfuscação, `P = P − R`.
11. Renormalizar por pacote: `g[k] = P[k] / M_agg[k]` para pacotes com `M_agg[k] > 0`.
12. Unpack (descartando o zero-padding) → update global `g`; aplicar `w ← w − η·g`
    (ou usar `g` diretamente como modelo agregado, conforme a convenção do seu loop).

## Matemática que precisa fechar

Agregação ponderada (Eq. 8-9), com obfuscação:

```
E(g_global) = Σ_i E(p_i) ⊗ E(g_i + Δ_i)
            = Σ_i E(p_i·g_i + r_i + r_{d,i})        # se Δ_i = (r_i + r_{d,i}) / p_i
            = E( Σ_i p_i·g_i + r + r_d )
            = E( Σ_i p_i·g_i + R ),    com R = r + r_d
```

O ponto sutil e fácil de errar: **`Δ_i = (r_i + r_{d,i}) / p_i`**. A divisão pelo peso é
justamente o que faz a obfuscação somar para R de forma independente dos pesos, depois da
multiplicação por `E(p_i)`. Se o código obfusca com `Δ_i = r_i + r_{d,i}` (sem dividir) e
ainda assim faz multiplicação ponderada, o `R` não cancela.

## Pesos de agregação (substituição da Eq. 7)

Sem sketches, a Eq. 7 (peso por similaridade de Jaccard dos sketches) não é computável.
Substituir por uma das opções e deixar explícito qual está em uso:

- (recomendado) peso por tamanho de dados, estilo FedAvg (Eq. 3): `p_i = D_i / Σ_j D_j`.
- (baseline trivial) não ponderado: `p_i = 1/|participantes|`.

Decisão de multiplicação no PS (impacta CKKS):

- Como no paper, ct×ct: `E(p_i) ⊗ C_i`. Mantém o peso encriptado, consome 1 nível de mult.
- Mais simples e válido no escopo reduzido, pt×ct: `p_i · C_i` (PS conhece `p_i`). Mais barato
  em profundidade. Se usar essa via, a obfuscação pode usar `Δ_i = r_i + r_{d,i}` sem a divisão
  por peso, **desde que** a renormalização por `M_agg` seja coerente. Verificar consistência
  entre a forma do `Δ_i` e a via de multiplicação escolhida.

## Obfuscação por secret sharing (camada opcional de segurança)

Só auditar se o objetivo inclui defesa contra clientes honest-but-curious. Caso contrário,
marcar como não implementado por escolha.

- Shamir `(th, |S|)` (Eq. 4-5): cada cliente recebe `r_i`, constrói polinômio de grau `th−1`
  com termo constante `r_i`, divide em shares, e recupera `r = Σ_i r_i` por interpolação de Lagrange.
- Camada aditiva do KDC: `r_d` distribuído como `r_{d,i}`; cada cliente combina `Δ_i` no obfuscar.
- Fator global `R = r + r_d`, subtraído após a decriptação.

## Checklist de invariantes (auditar uma a uma)

1. `B == poly_modulus_degree / 2` (slots = N/2). No paper N=8192 ⇒ B=4096.
2. `K = ceil(d / B)`; último pacote com zero-padding; unpack trunca o padding.
3. Sparsificação é **por pacote**, não por escalar. `M_i ∈ {0,1}^K`, uma entrada por pacote.
   Norma L2 calculada por pacote inteiro. Mantém exatamente `ceil(ζ·K)` pacotes.
4. A máscara é enviada em plaintext; o PS alinha pacotes por índice usando as máscaras.
   A sparsificação NÃO acontece antes do packing (ver Fig. 5a: isso quebra o alinhamento por
   máscaras de coordenadas inconsistentes; é o erro que o FedPHE evita).
5. A agregação por pacote soma apenas sobre clientes que mantiveram aquele pacote. O conjunto
   de contribuintes varia por pacote.
6. Renormalização por pacote: dividir o agregado pelo `M_agg[k] = Σ_{contribuintes} p_i`
   (ou pela contagem, se não ponderado). Sem isso, pacotes mantidos por poucos clientes ficam
   subponderados e a agregação fica enviesada para os clientes densos.
7. Obfuscação ponderada (se ativa): `Δ_i = (r_i + r_{d,i}) / p_i`, `R = r + r_d`. A subtração
   de R cancela exatamente em aritmética exata e até o ruído do CKKS na prática.
8. Escala CKKS: todos os ciphertexts no mesmo `scale` antes de somar; rescale após cada
   multiplicação; níveis (modulus chain) compatíveis antes de adicionar.
9. Profundidade multiplicativa: a via ct×ct consome 1 nível. O `coeff_modulus` precisa ter
   primos suficientes para o nível + rescale, senão a decriptação degrada silenciosamente.
10. `scale` de encoding consistente em todas as encriptações da rodada.

## Testes de correção (rodar e reportar)

- T1 (round-trip): pack → encrypt → decrypt → unpack de um tensor aleatório. Erro absoluto
  máximo < ε (dependente do scale, ex.: 1e-3). Falha aqui indica problema de scale/encoding,
  não de sparsificação.
- T2 (oráculo em plaintext): computar a agregação ponderada em float (numpy) e comparar com
  o resultado decriptado do FedPHE. Erro relativo dentro da tolerância do CKKS.
- T3 (sparsificação): conferir que exatamente `ceil(ζ·K)` pacotes são mantidos, que são os de
  maior norma L2, e que pacotes descartados não entram na agregação.
- T4 (obfuscação cancela): mesmo input, comparar resultado com e sem obfuscação. A diferença
  deve ser da ordem do ruído, não da ordem de `r`.
- T5 (normalização por máscara): caso em que dois clientes mantêm pacotes disjuntos. Cada
  pacote do resultado deve ser igual ao valor (ponderado) do único contribuinte daquele pacote,
  e não dividido por N. É o teste que pega o esquecimento da renormalização por `M_agg`.
- T6 (ponta a ponta): MNIST + LeNet-5, Dirichlet α=1, ζ=0.7. Acurácia dentro de ~1.45% do
  baseline plaintext (envelope reportado no paper, Sec. IV-B e Tab. IV linha "L2 norm"). Rodadas
  de convergência comparáveis à linha L2-norm da Tab. IV.

## Anti-padrões a sinalizar se encontrados no código

- Sparsificar no nível escalar antes do packing (Fig. 5a). Quebra o alinhamento no PS.
- Top-k por magnitude de coeficiente individual em vez de norma L2 do pacote.
- Falta de renormalização por pacote (`M_agg`), gerando agregação enviesada.
- Scales/níveis CKKS incompatíveis cujo erro de decriptação é confundido com "ruído de sparsificação".
- Obfuscação somada mas não dividida pelo peso, enquanto ainda se faz multiplicação ponderada ct×ct.
- Reuso dos mesmos fatores de obfuscação entre rodadas (problema de segurança), ou não atualizar
  chaves a cada `τ` rodadas, se o seu modelo de ameaça previr isso.
- Implementar qualquer parte de sketch/seleção (fora de escopo): se existir, isolar ou remover.

## Parâmetros do paper (para reprodução do T6)

- N = 10 clientes; 25% stragglers (irrelevante sem seleção, mas mantém o setup do paper).
- poly_modulus_degree = 8192; packing size = 4096.
- batch size = 64; η = 1e-3 (1e-2 para CIFAR-100).
- Datasets/modelos: MNIST/LeNet-5, FMNIST/CNN (2 conv + 1 fc), CIFAR-10/ResNet-20, CIFAR-100/ResNet-32.
- ζ sugerido: 10% para MNIST/FMNIST, 70% para CIFAR-10/100 (Tab. VI).

## Formato do relatório de auditoria esperado

Para cada item do checklist e cada teste, reportar: PASS / FAIL / N/A, o trecho de código
correspondente (arquivo:linha), e em caso de FAIL a correção mínima sugerida. Priorizar
correções de correção numérica (scale/nível CKKS, renormalização, `Δ_i`) antes de otimizações.