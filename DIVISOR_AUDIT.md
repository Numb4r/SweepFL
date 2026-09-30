# DIVISOR_AUDIT — Semântica do divisor de média sob upload mascarado

Auditoria read-only. Nenhum código foi modificado.  
Data: 2026-06-29  
Escopo: esquemas **nvnl**, **fedphe**, **rw**, **sw** (todos com `packing=True`, `only_sum=False`, `homomorphic_type='CKKS'`).

---

## Q1 — Onde e como a divisão acontece

### Configuração de `only_sum` (cliente e servidor)

`client/client.py:231–270` — `config_solution()`:

```
fedphe → only_sum = False, packing = True
nvnl   → only_sum = False, packing = True
sw     → only_sum = False, packing = True
rw     → only_sum = False, packing = True
```

`server/server.py:105` — `config_solution()`:

```python
elif self.solution in ('fedphe', 'nvnl', ..., 'sw', 'rw', ...):
    self.homomorphic = True; self.packing = True; self.onlysum = False; ...
```

Para os quatro esquemas: `only_sum / onlysum = False` sem exceção.

---

### Passo 1 — Pré-escala no cliente (ANTES da cifragem)

Todos os quatro `fit_*.py` contêm o mesmo bloco (citado aqui a partir de `NVNL.py:57-67`; linhas idênticas em `fedphe.py:57-67`, `sw.py:57-67`, `rw.py:57-67`):

```python
_ckks_backend = self.key_config.get("ckks_backend") or os.environ.get("CKKS_BACKEND", "tenseal")
_magg_renorm  = int(os.environ.get("MAGG_RENORM", "1"))
if self.only_sum or (_ckks_backend == "ckks_python_simplify" and _magg_renorm):
    # Caminho B — pré-pondera por D_i (tamanho local do dataset)
    flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
elif _ckks_backend == "ckks_python_simplify":
    # Caminho C — legado MAGG_RENORM=0: pré-divide por K
    flatted_parameters = np.array(flatted_parameters) / self.num_clients
# Caminho A — TenSEAL: nenhuma pré-escala (raw w_i)
```

Como `only_sum = False` para todos os quatro, os três caminhos são:

| Caminho | Condição de entrada | O que o cliente envia (por pack ativo k) |
|---------|---------------------|------------------------------------------|
| **A — TenSEAL** | `CKKS_BACKEND ≠ ckks_python_simplify` (padrão) | `w_i` bruto |
| **B — Simplify + MAGG_RENORM=1** | `CKKS_BACKEND=ckks_python_simplify` e `MAGG_RENORM=1` (default) | `D_i × w_i` |
| **C — Simplify + MAGG_RENORM=0** | `CKKS_BACKEND=ckks_python_simplify` e `MAGG_RENORM=0` | `w_i / K` |

---

### Passo 2 — Agregação no servidor (sob cifragem)

`server/server_utils.py:68-83` — `get_pack_mask_pair()`:

```python
weighted_mask = np.array(mask) * dataset_size   # → D_i ou 0 por pack
total_examples += dataset_size                  # → Σ D_j (todos participantes)
```

`server/server_utils.py:86-118` — `aggregate_packs_and_mask()`:

```python
_pure_sum = onlysum or isinstance(backend, CKKSSimplifyBackend)
# Para TenSEAL:  _pure_sum = False
# Para Simplify: _pure_sum = True   (independente de MAGG_RENORM)

for packs, mask in pack_mask_pair:
    mask = mask / total_examples          # → D_i/D_total por slot

    for idx_pack, p in enumerate(packs):
        if p == 0: continue               # pack não enviado por este cliente

        if _pure_sum:                     # Simplify: soma pura
            new_term = p
            aggregated_masks[idx_pack] += mask[idx_pack] * total_examples  # → += D_i
        else:                             # TenSEAL: pondera antes de somar
            new_term = backend.mul(ctx, p, float(mask[idx_pack]))           # p × (D_i/D_total)
            aggregated_masks[idx_pack] += mask[idx_pack]                    # += D_i/D_total
```

Resultado após varrer todos os clientes (S_k = conjunto de clientes que enviaram pack k):

| Caminho | `aggregated_packs[k]` (cifrado) | `aggregated_masks[k]` (texto claro) |
|---------|---------------------------------|--------------------------------------|
| **A — TenSEAL** | Σ_{i∈S_k} w_i × (D_i/D_total) | Σ_{i∈S_k} D_i / D_total |
| **B — Simplify/MAGG_RENORM=1** | Σ_{i∈S_k} D_i × w_i | Σ_{i∈S_k} D_i |
| **C — Simplify/MAGG_RENORM=0** | Σ_{i∈S_k} w_i / K | Σ_{i∈S_k} D_i |

O servidor serializa e envia `aggregated_packs` (cifrado) e `aggregated_masks` (pickle, texto claro) aos clientes (`server/server.py:196-197`).

---

### Passo 3 — Renormalização no cliente (APÓS a decifragem)

`client/literature/set_parameters.py:115-121` — `he_packs_to_model()`:

```python
_magg_renorm = int(os.environ.get("MAGG_RENORM", "1"))
for idx_mask, m in enumerate(aggredated_mask):
    if m > 0:
        if self.only_sum or _magg_renorm:          # → True se MAGG_RENORM=1
            packed_parameters[idx_mask] = np.array(decyphered_pack.pop(0)) / m
        else:                                       # MAGG_RENORM=0: sem divisão
            packed_parameters[idx_mask] = decyphered_pack.pop(0)
```

O divisor aplicado é `m = aggregated_masks[k]`, que varia por pack (ver tabela acima).

---

### Resultado final por pack k, por caminho

| Caminho | Valor final para pack k |
|---------|-------------------------|
| **A — TenSEAL + MAGG_RENORM=1** | [Σ_{S_k} w_i·D_i/D_total] / [Σ_{S_k} D_i/D_total] = **Σ_{S_k} w_i·D_i / Σ_{S_k} D_i** |
| **B — Simplify + MAGG_RENORM=1** | [Σ_{S_k} D_i·w_i] / [Σ_{S_k} D_i] = **Σ_{S_k} w_i·D_i / Σ_{S_k} D_i** |
| **C — Simplify + MAGG_RENORM=0** | Σ_{S_k} w_i / K *(sem divisão pós)* |
| **TenSEAL + MAGG_RENORM=0** | Σ_{S_k} w_i·D_i / D_total *(sem divisão pós)* |

**Resposta à Q1:**  
- Com `MAGG_RENORM=1` (padrão): a divisão ocorre **pós-decifragem no cliente**, em `set_parameters.py:119`. O divisor é **por pack** (`aggregated_masks[k]`), não uma constante global.  
- Com `MAGG_RENORM=0` (legado): a divisão pós-decifragem **não ocorre**; a "divisão" foi feita na pré-escala por K (Simplify) ou por D_total (TenSEAL, implícita nos pesos da soma).

---

## Q2 — O divisor casa com o número real de contribuintes?

### Com MAGG_RENORM=1 (padrão, ambos os backends)

Divisor para pack k = Σ_{i∈S_k} D_i (ou proporcional a isso), onde S_k = conjunto dos clientes que REALMENTE enviaram o pack k nessa rodada.

O resultado é a média FedAvg ponderada pelos dados **apenas sobre os contribuintes reais de k**:

```
w̄_k = Σ_{i∈S_k} D_i · w_i / Σ_{i∈S_k} D_i
```

✅ **O divisor casa exatamente com o número ponderado de contribuintes reais.**  
Clientes que não enviaram pack k (mask[k]=0) não entram nem no numerador (ciphertext) nem no denominador (aggregated_mask[k]).

### Com MAGG_RENORM=0 (legado)

**Caminho C (Simplify):** Cliente divide por K (total de clientes). Servidor soma as pré-escalas. Resultado para pack k: Σ_{i∈S_k} w_i / K.  
Se |S_k| < K (apenas alguns clientes enviaram k), a média fica **subestimada** por fator |S_k|/K.

**Caminho A (TenSEAL):** Servidor pondera por D_i/D_total. Sem divisão pós. Resultado: Σ_{i∈S_k} w_i·D_i/D_total.  
Subestimação por fator Σ_{S_k} D_i / D_total.

❌ **MAGG_RENORM=0 produz média enviesada para packs com |S_k| < K.**

---

## Q3 — O servidor conhece a contagem por pacote?

**Sim.** O `aggregated_masks` é calculado em **texto claro** no servidor (`server_utils.py:103-108`), acumulando o peso D_i de cada cliente ativo por pack. Além disso, cada máscara individual chega ao servidor como texto claro: `cypher_packs()` em `client_utils.py:38-45` cifra apenas os valores dos packs selecionados; a máscara binária é enviada em `pickle.dumps(mask)` via `fit_msg` (`NVNL.py:119`).

O servidor portanto conhece:
- A máscara individual de cada cliente (texto claro).
- O `aggregated_masks[k]` = peso acumulado em texto claro por pack.

Esse `aggregated_masks` é serializado em `server.py:197` (`self.agg_mask = pickle.dumps(aggregated_masks)`) e retorna aos clientes via `config['mask']`, onde é usado como divisor (`set_parameters.py:119`).

---

## Q4 — O round-robin garante contagem fixa por pack?

### NVNL (`get_robin_round_mask`, `NVNL.py:149-152`)

```python
def get_robin_round_mask(round, packs, size_window, percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
    return get_slide_window_mask(round, packs, size_window, size_window)
```

A função depende apenas de `round` e `len(packs)`. Para um dado round, **todos os clientes com o mesmo número de packs produzem a mesma máscara binária** — é determinística e independente dos pesos locais.

Consequência:
- Se TODOS os K clientes participam da rodada (CLIENT_FRACTION=1.0), S_k é sempre {todos K} para todo pack ativo k.
- Se CLIENT_FRACTION < 1.0 e K_sel clientes são amostrados, todos K_sel enviam os mesmos packs → S_k = {todos K_sel} para todo pack ativo.

**A contagem por pack é fixa e igual ao número de clientes participantes.** Não flutua entre packs dentro de uma rodada.

Com MAGG_RENORM=0 (legado) e CLIENT_FRACTION=1.0: Σ_{S_k} w_i / K = (K · ŵ) / K = ŵ → sem viés. Com CLIENT_FRACTION < 1.0 e K_sel < K: Σ_{K_sel} w_i / K → viés por K_sel/K.

### SW (`get_slide_window_mask` com stride=0.5, `sw.py:78-81`)

Mesma análise: máscara determinística por round e tamanho. Todos os participantes de uma rodada enviam os mesmos packs → S_k fixo e igual a todos os selecionados.

### FedPHE (`get_topk_mask`, `fedphe.py:121-141`)

A seleção é baseada na **norma L2 dos packs locais** (`np.linalg.norm`). Packs diferentes por clientes distintos → S_k varia por pack e por cliente dentro de uma mesma rodada.

**Nenhuma garantia de contagem fixa.** Packs com alta norma em muitos clientes acumulam mais contribuintes do que packs de baixa norma.

### RW (`get_pondering_random_mask`, `rw.py:81-84`)

Seleção explicitamente probabilística por cliente. S_k é aleatório por pack e por cliente.

**Nenhuma garantia de contagem fixa.** Alta variância no número de contribuintes por pack entre rodadas.

---

## Q5 — Consistência entre os quatro esquemas

### Os quatro usam o mesmo código de agregação e divisão?

**Sim, 100%.** Os quatro esquemas compartilham:

| Componente | Arquivo | Linhas | Compartilhado? |
|------------|---------|--------|----------------|
| Bloco de pré-escala | `NVNL.py`, `fedphe.py`, `sw.py`, `rw.py` | 57-67 (cada) | Código **idêntico** copiado |
| Agregação no servidor | `server_utils.py` | 86-118 | Função **única** compartilhada |
| Renormalização pós-decrypt | `set_parameters.py` | 110-121 | Função **única** compartilhada |

A única diferença entre os quatro esquemas é a função de seleção de máscara chamada dentro de `fit_*()`. O pipeline de divisão/média é completamente unificado.

### Tabela-resumo: divisor por esquema e caminho

| Esquema | Máscara | Pré-escala (cliente) | Divisor pós-decrypt | Divisor varia por pack? | Risco de viés |
|---------|---------|----------------------|----------------------|------------------------|---------------|
| **nvnl** | round-robin determinístico | Caminho A/B/C | `aggregated_masks[k]` = Σ_{S_k} D_i (MAGG_RENORM=1) | Não¹ | Não (MAGG_RENORM=1) |
| **sw** | janela deslizante determinístico | Caminho A/B/C | `aggregated_masks[k]` = Σ_{S_k} D_i (MAGG_RENORM=1) | Não¹ | Não (MAGG_RENORM=1) |
| **fedphe** | top-k por norma local | Caminho A/B/C | `aggregated_masks[k]` = Σ_{S_k} D_i (MAGG_RENORM=1) | **Sim** | Não (MAGG_RENORM=1) / **Sim** (MAGG_RENORM=0) |
| **rw** | ponderado aleatório | Caminho A/B/C | `aggregated_masks[k]` = Σ_{S_k} D_i (MAGG_RENORM=1) | **Sim** | Não (MAGG_RENORM=1) / **Sim** (MAGG_RENORM=0) |

¹ Não varia dentro de uma rodada para CLIENT_FRACTION=1.0; pode diferir se clientes tiverem números de packs distintos (improvável com modelo fixo).

---

## Tabela final de auditoria

| Esquema | Onde divide | Divisor usado (MAGG_RENORM=1) | Divisor varia por coord./pack? | Risco de viés (MAGG_RENORM=1) |
|---------|-------------|-------------------------------|-------------------------------|-------------------------------|
| **nvnl** | pós-decrypt no cliente (`set_parameters.py:119`) | Σ_{S_k} D_i / Σ D_j (TenSEAL) ou Σ_{S_k} D_i (Simplify) | Não (máscara uniforme entre clientes) | **Não** |
| **sw** | pós-decrypt no cliente (`set_parameters.py:119`) | Σ_{S_k} D_i / Σ D_j (TenSEAL) ou Σ_{S_k} D_i (Simplify) | Não (máscara uniforme entre clientes) | **Não** |
| **fedphe** | pós-decrypt no cliente (`set_parameters.py:119`) | Σ_{S_k} D_i / Σ D_j (TenSEAL) ou Σ_{S_k} D_i (Simplify) | **Sim** (top-k data-dependent) | **Não** |
| **rw** | pós-decrypt no cliente (`set_parameters.py:119`) | Σ_{S_k} D_i / Σ D_j (TenSEAL) ou Σ_{S_k} D_i (Simplify) | **Sim** (aleatório por cliente) | **Não** |

> **Nota sobre MAGG_RENORM=0 (legado):** O risco de viés passa a **Sim** para fedphe e rw (divisor fixo K ou D_total, não por pack). Para nvnl e sw com CLIENT_FRACTION=1.0, o viés é nulo mesmo no caminho legado (S_k = K em todos os packs ativos), mas passa a existir com CLIENT_FRACTION < 1.0.

> **MAGG_RENORM padrão:** A variável de ambiente `MAGG_RENORM` tem default `1` em ambos cliente (`set_parameters.py:115`) e cliente (cada `fit_*.py:58`), portanto o caminho correto é o ativo por default em todos os experimentos atuais.

---

*Fim do relatório. Nenhuma modificação de código foi realizada.*
