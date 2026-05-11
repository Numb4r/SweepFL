"""
Verifies that ckks_project was correctly installed in the container.

Checks:
  1. ckks_python      (C++ extension) is importable
  2. ckks_federated   (Python wrapper) is importable
  3. ckks_python_simplify (alias used by ckks_backend.py) is importable
  4. CKKSContext can be instantiated and basic ops work
  5. CKKSTensor encrypt → serialize → deserialize → homomorphic add → decrypt
     round-trip with numerical error < 1e-10

Run inside a container:
    python /tests/test_ckks_install.py
Or via pytest:
    pytest tests/test_ckks_install.py -v
"""

import sys
import numpy as np


# ── 1. Low-level C extension ──────────────────────────────────────────────────

def test_ckks_python_importable():
    import ckks_python
    assert hasattr(ckks_python, "generate_prime"), \
        "ckks_python missing generate_prime"
    assert hasattr(ckks_python, "KeyGenerator"), \
        "ckks_python missing KeyGenerator"
    print("  [OK] ckks_python importable")


# ── 2. High-level Python package ─────────────────────────────────────────────

def test_ckks_federated_importable():
    from ckks_federated import CKKSContext, CKKSTensor
    print("  [OK] ckks_federated importable")


# ── 3. ckks_python_simplify alias ────────────────────────────────────────────

def test_ckks_python_simplify_importable():
    from ckks_python_simplify import CKKSContext, CKKSTensor
    print("  [OK] ckks_python_simplify importable")


# Parameters matching real-world backend defaults (SIMPLIFY_N / SIMPLIFY_BITS / SIMPLIFY_SCALE_BITS)
_N, _BITS, _SCALE_BITS = 8192, 62, 60


# ── 4. CKKSContext instantiation ─────────────────────────────────────────────

def test_context_creation():
    from ckks_federated import CKKSContext
    ctx = CKKSContext(N=_N, bits=_BITS, scale_bits=_SCALE_BITS)
    assert ctx.N == _N
    assert ctx.encoder  is not None
    assert ctx.encryptor is not None
    assert ctx.decryptor is not None
    assert ctx.evaluator is not None
    print(f"  [OK] CKKSContext created (N={_N}, bits={_BITS}, scale_bits={_SCALE_BITS})")


# ── 5. Full round-trip: encrypt → serialize → deserialize → add → decrypt ────

def test_encrypt_add_decrypt():
    from ckks_federated import CKKSContext, CKKSTensor

    ctx = CKKSContext(N=_N, bits=_BITS, scale_bits=_SCALE_BITS)

    data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    data2 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)

    t1 = CKKSTensor(ctx, data1)
    t2 = CKKSTensor(ctx, data2)

    # Simulate network transfer: serialize → deserialize
    t1_recv = CKKSTensor(ctx, t1.serialize())
    t2_recv = CKKSTensor(ctx, t2.serialize())

    # Homomorphic addition
    t1_recv += t2_recv

    result = t1_recv.decrypt()
    expected = data1 + data2
    max_err = np.max(np.abs(result - expected))

    assert max_err < 1e-10, \
        f"Numerical error {max_err:.2e} exceeds 1e-10. result={result}, expected={expected}"

    print(f"  [OK] encrypt+add+decrypt round-trip (max_err={max_err:.2e})")


# ── 6. ckks_python_simplify alias uses the same classes ──────────────────────

def test_simplify_alias_same_api():
    from ckks_python_simplify import CKKSContext, CKKSTensor

    ctx = CKKSContext(N=_N, bits=_BITS, scale_bits=_SCALE_BITS)
    data = [1.0, 2.0]
    t = CKKSTensor(ctx, data)
    result = t.decrypt()
    max_err = max(abs(result[i] - data[i]) for i in range(len(data)))
    assert max_err < 1e-10, f"ckks_python_simplify round-trip error: {max_err:.2e}"
    print(f"  [OK] ckks_python_simplify round-trip (max_err={max_err:.2e})")


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    test_ckks_python_importable,
    test_ckks_federated_importable,
    test_ckks_python_simplify_importable,
    test_context_creation,
    test_encrypt_add_decrypt,
    test_simplify_alias_same_api,
]

if __name__ == "__main__":
    print(f"Python {sys.version}")
    print("=" * 60)
    failures = []
    for fn in TESTS:
        print(f"Running {fn.__name__} …")
        try:
            fn()
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failures.append((fn.__name__, exc))
    print("=" * 60)
    if failures:
        print(f"FAILED: {len(failures)}/{len(TESTS)}")
        for name, err in failures:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print(f"All {len(TESTS)} tests passed.")
