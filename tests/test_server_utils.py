"""
Testes para aggregate_packs_and_masks com vetores cifrados CKKS reais (TenSEAL).
Executados em múltiplas configurações de parâmetros CKKS.

Fluxo simulado:
  cliente  → ts.ckks_vector(ctx, values).serialize()   bytes cifrados
  servidor ← ts.ckks_vector_from(ctx, bytes)            align_packs entrega isso
  servidor →  aggregate_packs_and_masks(...)
  cliente  ← enc_vec.decrypt()                          verifica resultado

Configurações testadas
──────────────────────
  [60,40,40,60] scale=2^40  →  config de produção (server/encryption/ckks.py)
  [60,30,30,60] scale=2^30  →  config mais compacta, mesma estrutura de primos

Configuração INCOMPATÍVEL documentada com xfail
──────────────────────────────────────────────
  [30,60,30] scale=2^60  →  scale já está no limite do primo de 60 bits;
  qualquer multiplicação escalar (usada em aggregate_packs_and_masks) lança
  ValueError: scale out of bounds.
"""

import sys
import os
import pytest
import numpy as np
import tenseal as ts
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.server_utils import aggregate_packs_and_masks

# ── Configurações CKKS compatíveis com scalar multiplication ──────────────────
# Regra: o primo especial (primeiro) deve ser >= aos primos de computação.
# O scale deve ser ≈ 2^(bits_primo_computação) para que scale² caiba no primo.
CKKS_CONFIGS = [
    dict(
        id="[60,40,40,60]_scale2^40",
        poly_mod=8192,
        coeff_bits=[60, 40, 40, 60],
        scale=2**40,
        atol=1e-3,          # ruído típico < 1e-4 para valores pequenos
    ),
    dict(
        id="[60,30,30,60]_scale2^30",
        poly_mod=8192,
        coeff_bits=[60, 30, 30, 60],
        scale=2**30,
        atol=1e-2,          # primos menores → ruído acumula até ~6e-3 com 3 clientes
    ),
]

CKKSBundle = namedtuple("CKKSBundle", ["context", "atol", "id"])


@pytest.fixture(scope="module", params=CKKS_CONFIGS, ids=lambda c: c["id"])
def ctx(request):
    """
    Fixture parametrizado: todos os testes rodam uma vez para cada config CKKS.
    O id aparece no nome do teste: test_...[[60,40,40,60]_scale2^40], etc.
    """
    cfg = request.param
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=cfg["poly_mod"],
        coeff_mod_bit_sizes=cfg["coeff_bits"],
    )
    context.global_scale = cfg["scale"]
    return CKKSBundle(context=context, atol=cfg["atol"], id=cfg["id"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def enc(ctx: CKKSBundle, values):
    """
    Cifra → serializa → desserializa.
    Replica o que align_packs entrega: ts.ckks_vector_from(self.context, bytes)
    """
    serialized = ts.ckks_vector(ctx.context, list(values)).serialize()
    return ts.ckks_vector_from(ctx.context, serialized)


def dec(enc_vec) -> np.ndarray:
    """Decifra vetor CKKS (contexto precisa ter a chave secreta)."""
    return np.array(enc_vec.decrypt())


# ── Testes: primeiro cliente (aggregated_packs vazio) ─────────────────────────

class TestPrimeiraAgregacao:
    """Caminho is_empty=True — append nos acumuladores."""

    def test_pack_escalado_pelo_peso(self, ctx):
        """enc(v) * w  →  dec deve ser ≈ v * w."""
        values, weight = [1.0, 2.0, 3.0], 10.0
        agg_packs, agg_mask = [], []

        aggregate_packs_and_masks(
            [enc(ctx, values)], np.array([weight]), agg_packs, agg_mask
        )

        np.testing.assert_allclose(
            dec(agg_packs[0]), np.array(values) * weight, atol=ctx.atol
        )

    def test_mascara_armazenada_sem_cifrar(self, ctx):
        """weighted_mask é plaintext — deve ser preservada exatamente."""
        weight = 15.0
        agg_packs, agg_mask = [], []

        aggregate_packs_and_masks(
            [enc(ctx, [1.0])], np.array([weight]), agg_packs, agg_mask
        )

        assert agg_mask[0] == pytest.approx(weight)

    def test_multiplas_camadas_appended(self, ctx):
        """Cada elemento de aligned_pack gera um entry separado em agg_packs."""
        layer0, layer1, weight = [1.0, 2.0], [3.0, 4.0], 5.0
        agg_packs, agg_mask = [], []

        aggregate_packs_and_masks(
            [enc(ctx, layer0), enc(ctx, layer1)],
            np.array([weight, weight]),
            agg_packs, agg_mask,
        )

        assert len(agg_packs) == 2
        assert len(agg_mask)  == 2
        np.testing.assert_allclose(dec(agg_packs[0]), np.array(layer0) * weight, atol=ctx.atol)
        np.testing.assert_allclose(dec(agg_packs[1]), np.array(layer1) * weight, atol=ctx.atol)

    def test_posicao_inativa_fica_zero(self, ctx):
        """aligned_pack[i]=0 (int) com weighted_mask[i]=0 → posição inativa."""
        agg_packs, agg_mask = [], []
        aggregate_packs_and_masks([0], np.array([0]), agg_packs, agg_mask)
        assert agg_packs[0] == 0
        assert agg_mask[0] == 0


# ── Testes: segundo cliente em diante (aggregated_packs não vazio) ─────────────

class TestAgregacaoAcumulada:
    """Caminho is_empty=False — adição homomórfica entre clientes."""

    def _prime(self, ctx, values, weight):
        """Inicializa acumuladores com o primeiro cliente."""
        agg_packs, agg_mask = [], []
        aggregate_packs_and_masks(
            [enc(ctx, values)], np.array([weight]), agg_packs, agg_mask
        )
        return agg_packs, agg_mask

    def test_acumula_pack_cifrado(self, ctx):
        """dec(resultado) deve ser ≈ v1*w1 + v2*w2."""
        agg_packs, agg_mask = self._prime(ctx, [1.0, 2.0], 10.0)

        aggregate_packs_and_masks(
            [enc(ctx, [3.0, 4.0])], np.array([20.0]), agg_packs, agg_mask
        )

        # [1*10+3*20, 2*10+4*20] = [70, 100]
        np.testing.assert_allclose(dec(agg_packs[0]), [70.0, 100.0], atol=ctx.atol)

    def test_acumula_mascara_plaintext(self, ctx):
        agg_packs, agg_mask = self._prime(ctx, [1.0], 10.0)
        aggregate_packs_and_masks([enc(ctx, [1.0])], np.array([20.0]), agg_packs, agg_mask)
        assert agg_mask[0] == pytest.approx(30.0)

    def test_acumula_multiplas_camadas(self, ctx):
        """Cada camada acumula independentemente."""
        agg_packs, agg_mask = [], []
        aggregate_packs_and_masks(
            [enc(ctx, [1.0]), enc(ctx, [2.0])],
            np.array([10.0, 10.0]),
            agg_packs, agg_mask,
        )
        aggregate_packs_and_masks(
            [enc(ctx, [3.0]), enc(ctx, [4.0])],
            np.array([20.0, 20.0]),
            agg_packs, agg_mask,
        )

        np.testing.assert_allclose(dec(agg_packs[0]), [70.0],  atol=ctx.atol)  # 1*10+3*20
        np.testing.assert_allclose(dec(agg_packs[1]), [100.0], atol=ctx.atol)  # 2*10+4*20


# ── Testes: múltiplos clientes — Federated Averaging com HE ───────────────────

class TestFedAvgCKKS:
    """Simula rodada completa de FedAvg: N clientes → agregação homomórfica."""

    def test_tres_clientes_soma_ponderada(self, ctx):
        clients = [([1.0], 10.0), ([2.0], 20.0), ([3.0], 30.0)]
        agg_packs, agg_mask = [], []
        for values, weight in clients:
            aggregate_packs_and_masks(
                [enc(ctx, values)], np.array([weight]), agg_packs, agg_mask
            )

        # 1*10 + 2*20 + 3*30 = 140;  mask = 60
        np.testing.assert_allclose(dec(agg_packs[0]), [140.0], atol=ctx.atol)
        assert agg_mask[0] == pytest.approx(60.0)

    def test_media_ponderada_correta(self, ctx):
        """(soma ponderada) / (soma da máscara) deve ser a média ponderada."""
        clients = [([10.0], 10.0), ([20.0], 20.0)]
        agg_packs, agg_mask = [], []
        for values, weight in clients:
            aggregate_packs_and_masks(
                [enc(ctx, values)], np.array([weight]), agg_packs, agg_mask
            )

        # (10*10 + 20*20) / 30 = 500/30 ≈ 16.667
        avg = dec(agg_packs[0]) / agg_mask[0]
        np.testing.assert_allclose(avg, [500.0 / 30.0], atol=ctx.atol)

    def test_pesos_iguais_resulta_media_simples(self, ctx):
        values_list = [1.0, 5.0, 9.0]
        agg_packs, agg_mask = [], []
        for v in values_list:
            aggregate_packs_and_masks(
                [enc(ctx, [v])], np.array([1.0]), agg_packs, agg_mask
            )

        avg = dec(agg_packs[0]) / agg_mask[0]
        np.testing.assert_allclose(avg, [np.mean(values_list)], atol=ctx.atol)

    def test_multiplas_camadas_dois_clientes(self, ctx):
        """Cada camada agrega independentemente."""
        clients = [
            ([1.0, 2.0], [3.0, 4.0], 10.0),
            ([5.0, 6.0], [7.0, 8.0], 20.0),
        ]
        agg_packs, agg_mask = [], []
        for l0, l1, w in clients:
            aggregate_packs_and_masks(
                [enc(ctx, l0), enc(ctx, l1)],
                np.array([w, w]),
                agg_packs, agg_mask,
            )

        # camada 0: [1*10+5*20, 2*10+6*20] = [110, 140]
        np.testing.assert_allclose(dec(agg_packs[0]), [110.0, 140.0], atol=ctx.atol)
        # camada 1: [3*10+7*20, 4*10+8*20] = [170, 200]
        np.testing.assert_allclose(dec(agg_packs[1]), [170.0, 200.0], atol=ctx.atol)

    def test_resultado_serializavel(self, ctx):
        """Vetor agregado deve sobreviver a um ciclo serialize/deserialize."""
        agg_packs, agg_mask = [], []
        aggregate_packs_and_masks(
            [enc(ctx, [3.0, 6.0])], np.array([10.0]), agg_packs, agg_mask
        )

        reloaded = ts.ckks_vector_from(ctx.context, agg_packs[0].serialize())
        np.testing.assert_allclose(dec(reloaded), [30.0, 60.0], atol=ctx.atol)


# ── Testes: contrato de retorno e mutação in-place ────────────────────────────

class TestContrato:

    def test_retorna_mesmas_referencias(self, ctx):
        agg_packs, agg_mask = [], []
        ret_packs, ret_mask = aggregate_packs_and_masks(
            [enc(ctx, [1.0])], np.array([1.0]), agg_packs, agg_mask
        )
        assert ret_packs is agg_packs
        assert ret_mask  is agg_mask

    def test_comprimento_igual_ao_weighted_mask(self, ctx):
        n = 4
        agg_packs, agg_mask = [], []
        aggregate_packs_and_masks(
            [enc(ctx, [float(i)]) for i in range(n)],
            np.ones(n) * 5.0,
            agg_packs, agg_mask,
        )
        assert len(agg_packs) == n
        assert len(agg_mask)  == n


# ── xfail: config [30,60,30] scale=2^60 não suporta scalar multiplication ─────

@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason=(
        "Config [30,60,30] com scale=2^60 é incompatível com scalar multiplication: "
        "scale já está no limite do primo de 60 bits — scale² estoura o orçamento. "
        "Para usar essa estrutura de primos, escale para 2^29 ou menos, "
        "mas a precisão numérica fica degradada (erro ~1e6x maior)."
    ),
)
def test_xfail_30_60_30_scale60_scalar_mul_incompativel():
    """Documenta que [30,60,30] + scale=2^60 falha na multiplicação escalar."""
    ctx_incomp = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[30, 60, 30])
    ctx_incomp.global_scale = 2**60

    v = ts.ckks_vector(ctx_incomp, [1.0, 2.0])
    v_loaded = ts.ckks_vector_from(ctx_incomp, v.serialize())
    v_loaded * 10.0  # ValueError: scale out of bounds
