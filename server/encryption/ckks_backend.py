"""
CKKS backend abstraction layer.

Select backend via env var CKKS_BACKEND (or pass name to get_backend()):
  tenseal              — TenSEAL / SEAL (default, current behaviour)
  ckks_python_simplify — Custom simplified lib  (CKKSContext / CKKSTensor)
  ckks_python          — Custom verbose lib     (KeyGenerator / Evaluator / …)
  openfhe              — OpenFHE via openfhe-python

All backends expose the same interface so the rest of the codebase never
imports tenseal (or any other lib) directly for CKKS operations.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Abstract interface
# ══════════════════════════════════════════════════════════════════════════════

class CKKSBackend(ABC):

    @abstractmethod
    def generate_and_save(self, params: dict, context_dir: str) -> None:
        """Generate key pair and persist client (secret) + server (public) contexts."""

    @abstractmethod
    def load_client_context(self, context_dir: str) -> Any:
        """Load context that holds the secret key (encrypt + decrypt)."""

    @abstractmethod
    def load_server_context(self, context_dir: str) -> Any:
        """Load context with public key only (homomorphic ops, no decrypt)."""

    @abstractmethod
    def encrypt(self, ctx: Any, vector) -> Any:
        """Encrypt a real-valued 1-D array/list. Returns an opaque ciphertext."""

    @abstractmethod
    def serialize(self, ct: Any) -> bytes:
        """Serialize a ciphertext to bytes for network/disk transmission."""

    @abstractmethod
    def deserialize(self, ctx: Any, data: bytes) -> Any:
        """Reconstruct a ciphertext from bytes."""

    @abstractmethod
    def add(self, ctx: Any, ct1: Any, ct2: Any) -> Any:
        """Non-destructive homomorphic addition of two ciphertexts."""

    @abstractmethod
    def mul(self, ctx: Any, ct: Any, scalar: float) -> Any:
        """Non-destructive multiplication of a ciphertext by a plaintext scalar."""

    @abstractmethod
    def decrypt(self, ctx: Any, ct: Any) -> list:
        """Decrypt a ciphertext. ctx must carry the secret key."""


# ══════════════════════════════════════════════════════════════════════════════
#  TenSEAL  (default — preserves existing behaviour)
# ══════════════════════════════════════════════════════════════════════════════

class TenSEALBackend(CKKSBackend):

    _CLIENT_FILE = "tenseal_client.pkl"
    _SERVER_FILE = "tenseal_server.pkl"

    def generate_and_save(self, params: dict, context_dir: str) -> None:
        import tenseal as ts
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=params.get("poly_modulus_degree", 8192),
            coeff_mod_bit_sizes=params.get("coeff_mod_bit_sizes", [30, 60, 30]),
        )
        ctx.global_scale = params.get("global_scale", 2 ** 60)
        os.makedirs(context_dir, exist_ok=True)
        with open(f"{context_dir}/{self._CLIENT_FILE}", "wb") as f:
            pickle.dump({"context": ctx.serialize(save_secret_key=True)}, f)
        with open(f"{context_dir}/{self._SERVER_FILE}", "wb") as f:
            pickle.dump({"context": ctx.serialize()}, f)

    def load_client_context(self, context_dir: str) -> Any:
        import tenseal as ts
        with open(f"{context_dir}/{self._CLIENT_FILE}", "rb") as f:
            return ts.context_from(pickle.load(f)["context"])

    def load_server_context(self, context_dir: str) -> Any:
        import tenseal as ts
        with open(f"{context_dir}/{self._SERVER_FILE}", "rb") as f:
            return ts.context_from(pickle.load(f)["context"])

    def encrypt(self, ctx: Any, vector) -> Any:
        import tenseal as ts
        return ts.ckks_vector(ctx, list(vector))

    def serialize(self, ct: Any) -> bytes:
        return ct.serialize()

    def deserialize(self, ctx: Any, data: bytes) -> Any:
        import tenseal as ts
        return ts.ckks_vector_from(ctx, data)

    def add(self, ctx: Any, ct1: Any, ct2: Any) -> Any:
        return ct1 + ct2

    def mul(self, ctx: Any, ct: Any, scalar: float) -> Any:
        return ct * scalar

    def decrypt(self, ctx: Any, ct: Any) -> list:
        return list(ct.decrypt())


# ══════════════════════════════════════════════════════════════════════════════
#  ckks_python_simplify  (CKKSContext / CKKSTensor)
# ══════════════════════════════════════════════════════════════════════════════

class CKKSSimplifyBackend(CKKSBackend):
    # CKKSContext holds C++ pybind11 objects that cannot be pickled directly.
    # We persist the raw key bytes + scalar params and reconstruct the context on load.
    _PARAMS_FILE = "ckks_simplify_params.pkl"
    _SK_FILE     = "ckks_simplify_sk.bin"
    _PK_FILE     = "ckks_simplify_pk.bin"

    def generate_and_save(self, params: dict, context_dir: str) -> None:
        import ckks_python
        N          = params.get("simplify_N", 8192)
        bits       = params.get("simplify_bits", 62)
        scale_bits = params.get("simplify_scale_bits", 60)
        q          = ckks_python.generate_prime(bits, N)
        scale      = 2 ** scale_bits

        keygen = ckks_python.KeyGenerator(N, q)
        sk     = keygen.gen_secret_key()
        pk     = keygen.gen_public_key(sk)

        os.makedirs(context_dir, exist_ok=True)
        with open(f"{context_dir}/{self._PARAMS_FILE}", "wb") as f:
            pickle.dump({"N": N, "q": q, "scale": scale}, f)
        with open(f"{context_dir}/{self._SK_FILE}", "wb") as f:
            f.write(sk.to_bytes())
        with open(f"{context_dir}/{self._PK_FILE}", "wb") as f:
            f.write(pk.to_bytes())

    def _reconstruct(self, context_dir: str, with_sk: bool) -> Any:
        import ckks_python
        from ckks_python_simplify import CKKSContext
        with open(f"{context_dir}/{self._PARAMS_FILE}", "rb") as f:
            p = pickle.load(f)
        N, q, scale = p["N"], p["q"], p["scale"]

        # Bypass CKKSContext.__init__ (which would regenerate keys) and
        # manually wire in the persisted keys.
        ctx = CKKSContext.__new__(CKKSContext)
        ctx.N     = N
        ctx.q     = q
        ctx.scale = scale
        ctx.pk    = ckks_python.PublicKey.from_bytes(
            open(f"{context_dir}/{self._PK_FILE}", "rb").read()
        )
        ctx.encryptor = ckks_python.Encryptor(N, q, ctx.pk)
        ctx.encoder   = ckks_python.Encoder(N, q, scale)
        ctx.evaluator = ckks_python.Evaluator(N, q)

        if with_sk:
            ctx.sk        = ckks_python.SecretKey.from_bytes(
                open(f"{context_dir}/{self._SK_FILE}", "rb").read()
            )
            ctx.decryptor = ckks_python.Decryptor(N, q, ctx.sk)
        else:
            ctx.sk        = None
            ctx.decryptor = None

        return ctx

    def load_client_context(self, context_dir: str) -> Any:
        return self._reconstruct(context_dir, with_sk=True)

    def load_server_context(self, context_dir: str) -> Any:
        # Server must never decrypt — load without secret key.
        return self._reconstruct(context_dir, with_sk=False)

    def encrypt(self, ctx: Any, vector) -> Any:
        from ckks_python_simplify import CKKSTensor
        return CKKSTensor(ctx, list(vector))

    def serialize(self, ct: Any) -> bytes:
        return ct.serialize()

    def deserialize(self, ctx: Any, data: bytes) -> Any:
        from ckks_python_simplify import CKKSTensor
        return CKKSTensor(ctx, data)

    def add(self, ctx: Any, ct1: Any, ct2: Any) -> Any:
        from ckks_python_simplify import CKKSTensor
        result = CKKSTensor(ctx, ct1.serialize())
        result += ct2
        return result

    def mul(self, ctx: Any, ct: Any, scalar: float) -> Any:
        import ckks_python
        from ckks_python_simplify import CKKSTensor

        # Single-level CKKS has no rescaling.  Only exact operation is
        # integer-coefficient multiplication: k * c mod q → decoded k*m.
        # Fractional scalars must be handled via the pre-division protocol:
        # clients divide weights by num_clients before encryption; the server
        # then uses pure homomorphic addition only (sum(w_i/K) = average).
        k = round(scalar)
        if k <= 0:
            raise NotImplementedError(
                f"CKKSSimplifyBackend.mul: scalar={scalar} rounds to k={k}. "
                "Fractional scalars are not supported by single-level CKKS. "
                "Use the pre-division protocol: divide weights by num_clients "
                "on the client before encryption and aggregate with pure sums."
            )
        if k == 1:
            return ct

        N, q = ctx.N, ctx.q
        new_cts = []
        for raw_ct in ct.ciphertexts:
            c0 = (raw_ct.c0.astype(int) * k % q).astype(np.uint64)
            c1 = (raw_ct.c1.astype(int) * k % q).astype(np.uint64)
            new_cts.append(ckks_python.Ciphertext(
                ckks_python.Poly(N, q, c0),
                ckks_python.Poly(N, q, c1),
            ))
        return CKKSTensor(ctx, new_cts, original_size=ct.original_size)

    def decrypt(self, ctx: Any, ct: Any) -> list:
        return list(ct.decrypt())


# ══════════════════════════════════════════════════════════════════════════════
#  ckks_python  (verbose — KeyGenerator / Evaluator / Encryptor / Decryptor)
# ══════════════════════════════════════════════════════════════════════════════

class CKKSPythonBackend(CKKSBackend):

    _SK_FILE     = "ckks_python_sk.bin"
    _PK_FILE     = "ckks_python_pk.bin"
    _PARAMS_FILE = "ckks_python_params.pkl"

    @staticmethod
    def _import():
        import sys
        build_path = os.environ.get("CKKS_PYTHON_BUILD_PATH", "build")
        if build_path not in sys.path:
            sys.path.insert(0, build_path)
        import ckks_python
        return ckks_python

    def generate_and_save(self, params: dict, context_dir: str) -> None:
        mod   = self._import()
        N     = params.get("pycrypt_N", 8192)
        bits  = params.get("pycrypt_bits", 60)
        scale = params.get("pycrypt_scale", 2 ** 40)
        q     = mod.generate_prime(bits, N)

        keygen = mod.KeyGenerator(N, q)
        sk     = keygen.gen_secret_key()
        pk     = keygen.gen_public_key(sk)

        os.makedirs(context_dir, exist_ok=True)
        with open(f"{context_dir}/{self._SK_FILE}", "wb") as f:
            f.write(sk.to_bytes())
        with open(f"{context_dir}/{self._PK_FILE}", "wb") as f:
            f.write(pk.to_bytes())
        with open(f"{context_dir}/{self._PARAMS_FILE}", "wb") as f:
            pickle.dump({"N": N, "q": q, "scale": scale}, f)

    def _load_params(self, context_dir: str) -> dict:
        with open(f"{context_dir}/{self._PARAMS_FILE}", "rb") as f:
            return pickle.load(f)

    def load_client_context(self, context_dir: str) -> dict:
        mod = self._import()
        p   = self._load_params(context_dir)
        with open(f"{context_dir}/{self._SK_FILE}", "rb") as f:
            sk = mod.SecretKey.from_bytes(f.read())
        with open(f"{context_dir}/{self._PK_FILE}", "rb") as f:
            pk = mod.PublicKey.from_bytes(f.read())
        return {"N": p["N"], "q": p["q"], "scale": p["scale"], "sk": sk, "pk": pk, "_mod": mod}

    def load_server_context(self, context_dir: str) -> dict:
        mod = self._import()
        p   = self._load_params(context_dir)
        with open(f"{context_dir}/{self._PK_FILE}", "rb") as f:
            pk = mod.PublicKey.from_bytes(f.read())
        return {"N": p["N"], "q": p["q"], "scale": p["scale"], "sk": None, "pk": pk, "_mod": mod}

    def encrypt(self, ctx: dict, vector) -> Any:
        mod       = ctx["_mod"]
        encoder   = mod.Encoder(ctx["N"], ctx["q"], ctx["scale"])
        encryptor = mod.Encryptor(ctx["N"], ctx["q"], ctx["pk"])
        pt        = mod.Plaintext(encoder.encode(np.array(vector, dtype=np.float64)))
        return encryptor.encrypt(pt)

    def serialize(self, ct: Any) -> bytes:
        return ct.to_bytes()

    def deserialize(self, ctx: dict, data: bytes) -> Any:
        return ctx["_mod"].Ciphertext.from_bytes(data)

    def add(self, ctx: dict, ct1: Any, ct2: Any) -> Any:
        mod = ctx["_mod"]
        return mod.Evaluator(ctx["N"], ctx["q"]).add(ct1, ct2)

    def mul(self, ctx: dict, ct: Any, scalar: float) -> Any:
        mod       = ctx["_mod"]
        evaluator = mod.Evaluator(ctx["N"], ctx["q"])
        if not hasattr(evaluator, "mul_scalar"):
            raise NotImplementedError(
                "ckks_python Evaluator does not expose mul_scalar. "
                "Weighted FedAvg (onlysum=False) is not supported by this backend."
            )
        return evaluator.mul_scalar(ct, scalar)

    def decrypt(self, ctx: dict, ct: Any) -> list:
        mod       = ctx["_mod"]
        slots     = ctx["N"] // 2
        encoder   = mod.Encoder(ctx["N"], ctx["q"], ctx["scale"])
        decryptor = mod.Decryptor(ctx["N"], ctx["q"], ctx["sk"])
        pt        = decryptor.decrypt(ct)
        poly      = mod.Poly(ctx["N"], ctx["q"], pt.poly)
        return list(np.real(encoder.decode(poly))[:slots])


# ══════════════════════════════════════════════════════════════════════════════
#  OpenFHE  (openfhe-python bindings)
# ══════════════════════════════════════════════════════════════════════════════

class OpenFHEBackend(CKKSBackend):

    _CC_FILE = "openfhe_cc.bin"
    _SK_FILE = "openfhe_sk.bin"
    _PK_FILE = "openfhe_pk.bin"

    def generate_and_save(self, params: dict, context_dir: str) -> None:
        import openfhe
        p = openfhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(params.get("openfhe_mult_depth", 1))
        p.SetScalingModSize(params.get("openfhe_scale_bits", 50))
        p.SetRingDim(params.get("openfhe_ring_dim", 8192))

        cc = openfhe.GenCryptoContextCKKSRNS(p)
        cc.Enable(openfhe.PKESchemeFeature.PKE)
        cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
        keys = cc.KeyGen()

        os.makedirs(context_dir, exist_ok=True)
        openfhe.SerializeToFile(f"{context_dir}/{self._CC_FILE}", cc,          openfhe.BINARY)
        openfhe.SerializeToFile(f"{context_dir}/{self._SK_FILE}", keys.secretKey, openfhe.BINARY)
        openfhe.SerializeToFile(f"{context_dir}/{self._PK_FILE}", keys.publicKey,  openfhe.BINARY)

    def _load_cc(self, context_dir: str) -> Any:
        import openfhe
        cc, ok = openfhe.DeserializeCryptoContext(f"{context_dir}/{self._CC_FILE}", openfhe.BINARY)
        if not ok:
            raise RuntimeError(f"OpenFHE: failed to load crypto context from {context_dir}")
        return cc

    def load_client_context(self, context_dir: str) -> dict:
        import openfhe
        cc      = self._load_cc(context_dir)
        sk, _   = openfhe.DeserializeSecretKey(f"{context_dir}/{self._SK_FILE}", openfhe.BINARY)
        pk, _   = openfhe.DeserializePublicKey( f"{context_dir}/{self._PK_FILE}", openfhe.BINARY)
        return {"cc": cc, "sk": sk, "pk": pk}

    def load_server_context(self, context_dir: str) -> dict:
        import openfhe
        cc    = self._load_cc(context_dir)
        pk, _ = openfhe.DeserializePublicKey(f"{context_dir}/{self._PK_FILE}", openfhe.BINARY)
        return {"cc": cc, "sk": None, "pk": pk}

    def encrypt(self, ctx: dict, vector) -> Any:
        cc = ctx["cc"]
        pt = cc.MakeCKKSPackedPlaintext(list(vector))
        return cc.Encrypt(ctx["pk"], pt)

    def serialize(self, ct: Any) -> bytes:
        import openfhe
        return openfhe.Serialize(ct, openfhe.BINARY)

    def deserialize(self, ctx: dict, data: bytes) -> Any:
        import openfhe
        ct, ok = openfhe.DeserializeCiphertext(data, openfhe.BINARY)
        if not ok:
            raise RuntimeError("OpenFHE: failed to deserialize ciphertext")
        return ct

    def add(self, ctx: dict, ct1: Any, ct2: Any) -> Any:
        return ctx["cc"].EvalAdd(ct1, ct2)

    def mul(self, ctx: dict, ct: Any, scalar: float) -> Any:
        return ctx["cc"].EvalMult(ct, scalar)

    def decrypt(self, ctx: dict, ct: Any) -> list:
        cc = ctx["cc"]
        pt = cc.Decrypt(ctx["sk"], ct)
        return list(pt.GetRealPackedValue())


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

_BACKENDS: dict = {
    "tenseal":              TenSEALBackend,
    "ckks_python_simplify": CKKSSimplifyBackend,
    "ckks_python":          CKKSPythonBackend,
    "openfhe":              OpenFHEBackend,
}

_cache: dict = {}


def get_backend(name: Optional[str] = None) -> CKKSBackend:
    """Return the backend singleton for the given name (or CKKS_BACKEND env var).

    Calling get_backend() multiple times with the same name returns the same instance.
    """
    key = name or os.environ.get("CKKS_BACKEND", "tenseal")
    if key not in _cache:
        if key not in _BACKENDS:
            raise ValueError(
                f"Unknown CKKS_BACKEND '{key}'. Valid options: {list(_BACKENDS)}"
            )
        _cache[key] = _BACKENDS[key]()
    return _cache[key]
