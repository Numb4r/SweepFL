# Seleção de Parâmetros CKKS — ckks_python_simplify

## Parâmetros controlados

| Variável de ambiente | Significado |
|---|---|
| `SIMPLIFY_N` | Dimensão do anel (ring dimension). Controla segurança e tamanho do slot: `pack_size = N/2` |
| `SIMPLIFY_BITS` | log₂(q) — tamanho do módulo coeficiente em bits |
| `SIMPLIFY_SCALE_BITS` | log₂(scale) — bits de precisão fracionária |

---

## A equação fundamental

O CKKS da biblioteca `ckks_python_simplify` usa um **único módulo** q de `SIMPLIFY_BITS` bits.
Cada valor plaintext `m` é codificado como `round(m × scale)` antes da criptografia.
Para que a descriptografia produza o valor correto, o coeficiente codificado deve caber em q/2:

```
|m × scale| < q/2
⟹ |m| < q / (2 × scale)
⟹ range por ciphertext = 2^(BITS − SCALE_BITS − 1)
```

### Exemplo com os parâmetros antigos (BITS=62, SCALE=60)

```
range = 2^(62 − 60 − 1) = 2^1 = ±2  por ciphertext
```

### Exemplo com os parâmetros atuais (BITS=62, SCALE=40)

```
range = 2^(62 − 40 − 1) = 2^21 = ±2,097,152  por ciphertext
```

---

## Impacto da agregação federada

O servidor soma os ciphertexts de todos os clientes em uma única operação homomorfica:

```
ct_agg = ct₁ + ct₂ + ... + ct_C
```

Cada `ct_i` carrega um valor codificado `mᵢ × scale`. A soma resultante carrega
`(m₁ + m₂ + ... + m_C) × scale`. Para descriptografar corretamente, a soma deve caber em q/2:

```
C × max_|w| × scale < q/2
⟹ log₂(C × max_|w|) < BITS − SCALE_BITS − 1
⟹ BITS − SCALE_BITS ≥ ceil(log₂(C × max_|w|)) + 1
```

### Por que o colapso ocorria na rodada ~16–25 (experimentos resnet20)

Com BITS=62, SCALE=60 e 9 clientes, o limite de peso por cliente era:

```
max_|w|_cliente = 2^(62−60−1) / 9 ≈ ±0.22
```

Os pesos iniciais do ResNet20 (inicialização He) têm magnitude ~0.01–0.1 — dentro do limite.
Após ~15–25 rodadas de treino, alguns pesos crescem além de |w| > 0.22, causando overflow
modular silencioso na agregação: o servidor descriptografa lixo, a loss explode, e o modelo
colapsa para accuracy de ~10% (baseline aleatório do CIFAR-10).

O `fedphe` (top-k por magnitude) selecionava exatamente os pesos de **maior magnitude**,
acelerando a chegada ao limite. O `sw` (janela deslizante com stride=0.5) cobria os pesos
grandes depois de algumas rodadas e colapsava alguns rounds antes por conta da sobreposição dupla.

---

## Equações de projeto

Dado:
- **Precisão relativa** ε (e.g., ε = 10⁻⁶ → SCALE_BITS = ceil(log₂(1/ε)) = 20)
- **Magnitude máxima dos pesos** M (e.g., M = 5 para ResNet20 convergido)
- **Número de clientes** C

```
SCALE_BITS  ≥  ceil(log₂(1/ε))
BITS        ≥  SCALE_BITS + ceil(log₂(C × M)) + 1 + margem_segurança
```

Recomenda-se `margem_segurança = 3–5 bits`.

### Tabela de referência — variando clientes e magnitude (SCALE=40)

| Clientes (C) | max_|w| | BITS mínimo | Configuração recomendada |
|---|---|---|---|
| 2   | 5 | 40 + ceil(log₂(10))  + 5 = 49 | BITS=52 |
| 9   | 5 | 40 + ceil(log₂(45))  + 5 = 51 | BITS=54 |
| 18  | 5 | 40 + ceil(log₂(90))  + 5 = 52 | BITS=55 |
| 50  | 5 | 40 + ceil(log₂(250)) + 5 = 53 | BITS=56 |
| 100 | 5 | 40 + ceil(log₂(500)) + 5 = 54 | BITS=57 |

---

## Crescimento do número de clientes

### Comportamento logarítmico

O número de clientes entra como fator linear na soma dos ciphertexts, mas como fator
**logarítmico** no requisito de bits. Dobrar o número de clientes exige apenas +1 bit:

| Clientes (C) | log₂(5C) | BITS mínimo (SCALE=40, M=5) | Bits extras vs C=1 |
|---|---|---|---|
| 1     | 2.3  | 48 | — |
| 2     | 3.3  | 49 | +1 |
| 9     | 5.5  | 51 | +3 |
| 18    | 6.5  | 52 | +4 |
| 50    | 7.9  | 53 | +5 |
| 100   | 8.9  | 54 | +6 |
| 500   | 11.3 | 57 | +9 |
| 1 000 | 12.3 | 58 | +10 |
| 10 000| 15.6 | 61 | +13 |

**Regra prática**: para cada 10× no número de clientes, BITS cresce ~3–4 bits.
Escalar de 9 → 10.000 clientes exige apenas ~10 bits extras.

### Número máximo de clientes suportado (C_max)

Para uma configuração fixa, o limite é:

```
C_max = floor( 2^(BITS − SCALE_BITS − 1) / M )
```

| Configuração | max_|w|=1 | max_|w|=5 | max_|w|=10 |
|---|---|---|---|
| BITS=62, SCALE=60 (antiga) | 1 | **0 — falha** | **0 — falha** |
| BITS=62, SCALE=40 (atual)  | 2,097,152 | 419,430 | 209,715 |
| BITS=62, SCALE=20          | 2.1 × 10⁹ | 4.3 × 10⁸ | 2.1 × 10⁸ |

A configuração antiga não suportava nem 9 clientes com pesos acima de 0.22.
A configuração atual suporta centenas de milhares de clientes com pesos típicos de ML.

### Interação com a estratégia de seleção de packs

Cada solução tem uma sensibilidade diferente ao número de clientes, porque o que é transmitido
ao servidor determina quais valores são somados:

| Solução | Critério de seleção | Sensibilidade ao C |
|---|---|---|
| `fedphe` | top-k por magnitude — envia os **maiores** pesos | **Alta**: maximiza o valor somado, pior caso para overflow |
| `sw` | janela deslizante (stride=0.5×) — cobre todos após N rodadas | Média |
| `sweepfl` | round-robin uniforme | Baixa: distribui a variação ao longo das rodadas |
| `rw` | aleatório ponderado | Baixa/média |
| `topkrr` | top-k nas primeiras rodadas, depois round-robin | Alta nas primeiras rodadas |
| `plaintext` | sem criptografia | Nenhuma |

Para `fedphe`, o dimensionamento deve usar M = max peso global do modelo (não a média),
pois o top-k irá inevitavelmente selecionar esses valores.

### Efeito do não-IID no magnitude dos pesos

Com dados não-IID (niid < 1), os gradientes locais divergem entre clientes. Isso não causa
overflow diretamente, mas pode **aumentar a norma dos pesos** ao longo das rodadas, pois
gradientes conflitantes empurram o modelo para regiões de maior magnitude. Para dimensionamento
conservador com dados não-IID:

```
M_efetivo ≈ M_iid × (1 + α)
```

onde α é um fator empírico (~0.3–0.5 para niid=0.1, ~0 para niid=1).

---

## Configuração atual do projeto

```
SIMPLIFY_N=8192        → pack_size = 4096 slots por pack
SIMPLIFY_BITS=62       → q ≈ 2^62
SIMPLIFY_SCALE_BITS=40 → scale = 2^40, precisão ≈ 10^{-12}
```

### Verificação

```
range por ciphertext     = 2^(62 − 40 − 1) = 2^21 = ±2,097,152
range por cliente (C=9)  = ±232,960
```

ResNet20 (CIFAR-10): pesos tipicamente em [−5, 5] após convergência → **SEGURO** com margem de ~46.000×.

---

## Trade-offs práticos

| Parâmetro | Aumentar | Diminuir |
|---|---|---|
| `SIMPLIFY_SCALE_BITS` | Mais precisão, **menos range** | Menos precisão, mais range |
| `SIMPLIFY_BITS` | Mais range e precisão, ciphertext **maior e mais lento** | Mais rápido, menos headroom |
| `SIMPLIFY_N` | Mais segurança, mais slots, **operações mais lentas** | Menos segurança, pack menor |

Notas:
- SCALE_BITS=40 representa precisão de ~10⁻¹², muito além do necessário para float32 (~10⁻⁷).
- SCALE_BITS=23–30 é suficiente para reproduzir float32 com fidelidade total.
- Com BITS=62 e SCALE=40, há 22 bits de headroom — generoso o suficiente para não impor restrições
  práticas ao número de clientes ou magnitude dos pesos em qualquer experimento deste projeto.

---

## Quando regenerar as chaves

As chaves CKKS (`ckks_simplify_params.pkl`, `ckks_simplify_sk.bin`, `ckks_simplify_pk.bin`) são
geradas com N, q e scale fixos. Qualquer mudança em `SIMPLIFY_N`, `SIMPLIFY_BITS` ou
`SIMPLIFY_SCALE_BITS` invalida as chaves existentes:

```bash
python3 generate_keys.py
```

---

## Sinais de overflow (diagnóstico)

| Sintoma em `server_eval.csv` | Causa provável |
|---|---|
| Loss explode (×100–×1000) em uma rodada | Wrap modular na descriptografia da agregação |
| Loss = NaN nas rodadas seguintes | Pesos corrompidos propagados entre rounds |
| Accuracy trava em ~10% (CIFAR-10 de 10 classes) | Modelo efetivamente aleatório |
| Nenhum erro reportado pelos clientes | Overflow é silencioso — não gera exceção |

O colapso segue uma ordem determinística pela estratégia de seleção:
1. `fedphe` / `topkrr` (top-k) — colapsa primeiro (seleciona pesos maiores)
2. `sw` (janela deslizante) — colapsa logo após (stride duplo nas bordas)
3. `sweepfl` / `rw` / `srr` — mais tarde (distribuição mais uniforme dos valores)
4. `plaintext` — nunca colapsa (sem criptografia)
