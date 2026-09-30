# Porte da API de Serialização do `CKKSContext` para `ckks_project/`

**Data de criação:** 2026-06-22  
**Fonte da mudança:** `~/Documentos/CKKS_python/ckks_federated/__init__.py` (commit `b7adf9f "staged"`, 22/jun/2026)  
**Alvo a atualizar:** `keys/ckks_project/ckks_federated/__init__.py`

---

## O que foi feito em `~/Documentos/CKKS_python`

A API de **serialização/persistência** do `CKKSContext` foi implementada na biblioteca
canônica (`~/Documentos/CKKS_python`). O objetivo é permitir:

- **Persistir e recarregar chaves em disco** (`save` / `load`) sem gerar novas chaves, usando
  um layout de 3 arquivos: `params.pkl`, `pk.bin`, `sk.bin`.
- **Reconstruir um contexto a partir de bytes já em memória** (`from_keys`), sem chamar
  `KeyGenerator` — necessário quando o servidor recebe apenas a chave pública via rede.
- **Pickle completo** (`__getstate__` / `__setstate__`): suporte a multiprocessing,
  serialização Flower e qualquer outro mecanismo que use `pickle` internamente.

Estas funções são usadas por `CKKSSimplifyBackend` em `encryption/ckks_backend.py`
(método `load_client_context` / `load_server_context`).

---

## Estado atual das cópias (confirmado em 22/jun/2026)

| Cópia | Linhas | Situação |
|---|---|---|
| `~/Documentos/CKKS_python/ckks_federated/__init__.py` | **204** | ✅ atualizada (fonte) |
| `keys/ckks_project/ckks_federated/__init__.py` | 130 | ❌ desatualizada (alvo) |
| `~/.local/lib/python3.10/site-packages/ckks_federated/__init__.py` | 130 | ❌ instalada antiga |
| `~/Documentos/CKKS_python/build/lib.../ckks_federated/__init__.py` | 130 | ❌ build antigo |
| `~/Área de Trabalho/ckks_project/ckks_federated/__init__.py` | 130 | ❌ cópia antiga |

A diferença é **puramente aditiva**: nada foi removido ou alterado na versão antiga —
apenas os 2 imports e os 5 métodos abaixo foram adicionados.

---

## Delta exato a aplicar

### 1. Adicionar imports no topo do arquivo

**Localização:** antes de `import numpy as np` (linha 1 da versão antiga).

```python
import os
import pickle
```

### 2. Adicionar 5 métodos dentro de `CKKSContext`

**Localização:** dentro da classe `CKKSContext`, imediatamente **após** o final de
`__init__` (linha 29 da versão antiga, após `self.evaluator = ...`) e **antes** de
`make_public`. Inserir exatamente o bloco abaixo, mantendo a indentação de 4 espaços:

```python
    @classmethod
    def from_keys(cls, N: int, q: int, scale: int,
                  pk_bytes: bytes, sk_bytes: bytes = None,
                  public_only: bool = False) -> "CKKSContext":
        """Reconstrói um CKKSContext a partir de chaves serializadas, sem gerar novas."""
        ctx = cls.__new__(cls)
        ctx.N = N
        ctx.q = q
        ctx.scale = scale
        ctx.bits = q.bit_length()
        ctx.pk = ckks_python.PublicKey.from_bytes(pk_bytes)
        ctx.encoder = ckks_python.Encoder(N, q, scale)
        ctx.encryptor = ckks_python.Encryptor(N, q, ctx.pk)
        ctx.evaluator = ckks_python.Evaluator(N, q)
        if public_only or sk_bytes is None:
            ctx.sk = None
            ctx.decryptor = None
        else:
            ctx.sk = ckks_python.SecretKey.from_bytes(sk_bytes)
            ctx.decryptor = ckks_python.Decryptor(N, q, ctx.sk)
        return ctx

    def save(self, path: str) -> None:
        """Salva chaves e parâmetros em disco (3 arquivos em `path/`)."""
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump({"N": self.N, "q": self.q, "scale": self.scale}, f)
        with open(f"{path}/pk.bin", "wb") as f:
            f.write(self.pk.to_bytes())
        if self.sk is not None:
            with open(f"{path}/sk.bin", "wb") as f:
                f.write(self.sk.to_bytes())

    @classmethod
    def load(cls, path: str, public_only: bool = False) -> "CKKSContext":
        """Carrega contexto do disco. `public_only=True` para o servidor (sem SK)."""
        with open(f"{path}/params.pkl", "rb") as f:
            p = pickle.load(f)
        pk_bytes = open(f"{path}/pk.bin", "rb").read()
        sk_path = f"{path}/sk.bin"
        sk_bytes = None if public_only or not os.path.exists(sk_path) else open(sk_path, "rb").read()
        return cls.from_keys(
            N=p["N"], q=p["q"], scale=p["scale"],
            pk_bytes=pk_bytes, sk_bytes=sk_bytes,
            public_only=public_only,
        )

    def __getstate__(self):
        return {
            "N":     self.N,
            "q":     self.q,
            "scale": self.scale,
            "sk":    self.sk.to_bytes() if self.sk is not None else None,
            "pk":    self.pk.to_bytes(),
        }

    def __setstate__(self, state):
        self.N = state["N"]
        self.q = state["q"]
        self.scale = state["scale"]
        self.bits = state["q"].bit_length()
        self.pk = ckks_python.PublicKey.from_bytes(state["pk"])
        self.encoder = ckks_python.Encoder(self.N, self.q, self.scale)
        self.encryptor = ckks_python.Encryptor(self.N, self.q, self.pk)
        self.evaluator = ckks_python.Evaluator(self.N, self.q)
        if state["sk"] is not None:
            self.sk = ckks_python.SecretKey.from_bytes(state["sk"])
            self.decryptor = ckks_python.Decryptor(self.N, self.q, self.sk)
        else:
            self.sk = None
            self.decryptor = None

```

---

## Prompt para o Claude Code aplicar esta mudança

> **Contexto:** A biblioteca `ckks_federated` (dentro de `keys/ckks_project/`) está
> desatualizada. A versão canônica com a API de serialização está em
> `~/Documentos/CKKS_python/ckks_federated/__init__.py` (204 linhas). A cópia em
> `keys/ckks_project/ckks_federated/__init__.py` tem 130 linhas e está **sem** os
> métodos `from_keys`, `save`, `load`, `__getstate__`, `__setstate__`.
>
> **Tarefa:**
>
> 1. Leia `keys/ckks_project/ckks_federated/__init__.py`.
> 2. Adicione `import os` e `import pickle` no topo do arquivo (antes de `import numpy`).
> 3. Dentro de `CKKSContext`, logo após o final do método `__init__` (após a linha
>    `self.evaluator = ckks_python.Evaluator(self.N, self.q)`) e antes de `make_public`,
>    insira os 5 métodos exatamente como estão em
>    `docs/ckks_serialization_port.md` (bloco "Delta exato a aplicar").
> 4. Após editar, execute:
>    ```bash
>    pip install --user keys/ckks_project/
>    # recriar o symlink ckks_python_simplify → ckks_federated (conforme CLAUDE.md):
>    python3 -c "import site, os; sp=site.getsitepackages()[0]; target=f'{sp}/ckks_python_simplify'; \
>      os.path.exists(target) and os.remove(target); os.symlink(f'{sp}/ckks_federated', target)"
>    # ou, para install --user:
>    ln -sf ~/.local/lib/python3.10/site-packages/ckks_federated \
>            ~/.local/lib/python3.10/site-packages/ckks_python_simplify
>    ```
> 5. Verifique:
>    ```bash
>    python3 -c "from ckks_federated import CKKSContext; \
>      print(hasattr(CKKSContext,'save'), hasattr(CKKSContext,'from_keys'), \
>            hasattr(CKKSContext,'load'), hasattr(CKKSContext,'__getstate__'))"
>    # deve imprimir: True True True True
>    wc -l keys/ckks_project/ckks_federated/__init__.py
>    # deve ser 204 (ou próximo — whitespace pode variar ±1)
>    ```
> 6. (Opcional) rode o teste de serialização da lib canônica:
>    ```bash
>    python3 ~/Documentos/CKKS_python/test_py_serialization.py
>    ```

---

## Nota sobre as mudanças da camada FL (Slidhe)

As entradas em `ckks_change_logs/` (datas 2026-05-11) **não dizem respeito à lib
`ckks_project/`** — são mudanças na camada de abstração FL do Slidhe
(`encryption/ckks_backend.py` e `client/literature/`), que já foram aplicadas no repo
keys. Esta mudança aqui é independente e trata exclusivamente do `__init__.py` da lib.
