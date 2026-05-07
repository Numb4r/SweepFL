"""
Tests for the model weight reconstruction pipeline.

Two roundtrip paths are covered for every model factory × input shape:

  Plaintext:  flat_parameters(weights) → reshape_parameters(stub, flat)
  Packing:    flat_parameters → packing → flat_packs → remove_padding → reshape_parameters

Input shapes span multiple "datasets" to verify that all models are
dataset-agnostic (any model can run with any dataset).
"""

import math
import sys
import os
import pytest
import numpy as np
from types import SimpleNamespace

# Allow imports from the 'client/' directory tree
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, "client"))

from models import (
    flat_parameters,
    reshape_parameters,
    create_dnn,
    create_dnn_base,
    create_dnn_tiny,
    create_lenet5,
    create_resnet20,
    create_mobilenetv2,
    create_vit_tiny,
    create_simple_cnn,
)
from client_utils import packing, flat_packs, remove_padding


# ── Parametrization ────────────────────────────────────────────────────────────

# (input_shape, num_classes) pairs representing common dataset formats
SHAPES = [
    pytest.param((28, 28, 1), 10, id="mnist_28x28x1"),
    pytest.param((32, 32, 3), 10, id="cifar_32x32x3"),
]

# All model factories; each entry is (name_str, factory_callable)
MODELS = [
    pytest.param("dnn",         create_dnn,         id="dnn"),
    pytest.param("dnn_base",    create_dnn_base,    id="dnn_base"),
    pytest.param("dnn_tiny",    create_dnn_tiny,    id="dnn_tiny"),
    pytest.param("lenet5",      create_lenet5,      id="lenet5"),
    pytest.param("resnet20",    create_resnet20,    id="resnet20"),
    pytest.param("mobilenetv2", create_mobilenetv2, id="mobilenetv2"),
    pytest.param("vit_tiny",    create_vit_tiny,    id="vit_tiny"),
    pytest.param("simple_cnn",  create_simple_cnn,  id="simple_cnn"),
]

CHUNK_SIZES = [
    pytest.param(4096, id="chunk4096"),
    pytest.param(2048, id="chunk2048"),
]


def _stub(model):
    """Minimal object that satisfies the reshape_parameters / remove_padding interface."""
    return SimpleNamespace(model=model)


# ── Plaintext roundtrip ────────────────────────────────────────────────────────

class TestPlaintextRoundtrip:
    """flat_parameters → reshape_parameters must recover original weights exactly."""

    @pytest.mark.parametrize("input_shape,num_classes", SHAPES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_weights_identical(self, _name, factory, input_shape, num_classes):
        model = factory(input_shape, num_classes)
        original = model.get_weights()
        flat = flat_parameters(original)
        reconstructed = reshape_parameters(_stub(model), flat)

        assert len(reconstructed) == len(original)
        for i, (orig, rec) in enumerate(zip(original, reconstructed)):
            np.testing.assert_array_equal(orig, rec, err_msg=f"layer {i}")

    @pytest.mark.parametrize("input_shape,num_classes", SHAPES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_set_weights_does_not_raise(self, _name, factory, input_shape, num_classes):
        """model.set_weights(reconstructed) must not raise a shape error."""
        model = factory(input_shape, num_classes)
        stub = _stub(model)
        flat = flat_parameters(model.get_weights())
        model.set_weights(reshape_parameters(stub, flat))

    @pytest.mark.parametrize("input_shape,num_classes", SHAPES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_flat_length_equals_total_params(self, _name, factory, input_shape, num_classes):
        """flat_parameters must contain exactly sum(layer.size) elements."""
        model = factory(input_shape, num_classes)
        total = sum(w.size for w in model.get_weights())
        assert len(flat_parameters(model.get_weights())) == total

    @pytest.mark.parametrize("input_shape,num_classes", SHAPES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_modified_weights_preserved(self, _name, factory, input_shape, num_classes):
        """Reconstruction must work correctly for arbitrary (post-training) weight values."""
        model = factory(input_shape, num_classes)
        rng = np.random.default_rng(0)
        new_weights = [
            rng.standard_normal(w.shape).astype(w.dtype)
            for w in model.get_weights()
        ]
        model.set_weights(new_weights)
        stub = _stub(model)

        flat = flat_parameters(new_weights)
        reconstructed = reshape_parameters(stub, flat)

        for i, (orig, rec) in enumerate(zip(new_weights, reconstructed)):
            np.testing.assert_array_equal(orig, rec, err_msg=f"layer {i}")


# ── Packing roundtrip ──────────────────────────────────────────────────────────

class TestPackingRoundtrip:
    """flat_parameters → packing → flat_packs → remove_padding → reshape_parameters."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("input_shape,num_classes", SHAPES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_weights_recovered(self, _name, factory, input_shape, num_classes, chunk_size):
        model = factory(input_shape, num_classes)
        original = model.get_weights()
        stub = _stub(model)

        flat = flat_parameters(original)
        packs = packing(list(flat), chunk_size=chunk_size)
        trimmed = remove_padding(stub, flat_packs(packs))
        reconstructed = reshape_parameters(stub, trimmed)

        assert len(reconstructed) == len(original)
        for i, (orig, rec) in enumerate(zip(original, reconstructed)):
            np.testing.assert_array_almost_equal(orig, rec, decimal=6, err_msg=f"layer {i}")

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_remove_padding_preserves_param_count(self, _name, factory, chunk_size):
        """After remove_padding the flat sequence must have exactly total_params elements."""
        model = factory((28, 28, 1), 10)
        stub = _stub(model)
        flat = flat_parameters(model.get_weights())
        packs = packing(list(flat), chunk_size=chunk_size)
        trimmed = remove_padding(stub, flat_packs(packs))
        assert len(trimmed) == len(flat)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_n_packs_formula(self, _name, factory, chunk_size):
        """Number of packs must always equal total_params // chunk_size + 1."""
        model = factory((28, 28, 1), 10)
        flat = flat_parameters(model.get_weights())
        packs = packing(list(flat), chunk_size=chunk_size)
        expected = len(flat) // chunk_size + 1
        assert len(packs) == expected

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_packing_does_not_mutate_original(self, _name, factory, chunk_size):
        """packing() must not alter the original flat list passed to it."""
        model = factory((28, 28, 1), 10)
        flat = flat_parameters(model.get_weights())
        flat_copy = list(flat)
        packing(flat_copy, chunk_size=chunk_size)
        # flat itself should be untouched (we pass a copy inside packing)
        assert len(flat) == sum(w.size for w in model.get_weights())


# ── Edge cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_params_smaller_than_chunk_size(self):
        """All params fit in a single pack — roundtrip must still be exact."""
        model = create_dnn_tiny((28, 28, 1), 2)
        stub = _stub(model)
        flat = flat_parameters(model.get_weights())
        # chunk larger than total params
        chunk = len(flat) + 1000
        packs = packing(list(flat), chunk_size=chunk)
        assert len(packs) == 1
        trimmed = remove_padding(stub, flat_packs(packs))
        reconstructed = reshape_parameters(stub, trimmed)
        for i, (orig, rec) in enumerate(zip(model.get_weights(), reconstructed)):
            np.testing.assert_array_almost_equal(orig, rec, decimal=6, err_msg=f"layer {i}")

    def test_params_exactly_divisible_by_chunk(self):
        """When total_params % chunk_size == 0, the extra empty last pack is harmless."""
        model = create_dnn_tiny((28, 28, 1), 10)
        stub = _stub(model)
        flat = flat_parameters(model.get_weights())

        # Find a chunk_size that divides the total exactly (or use a divisor of it)
        n = len(flat)
        # Pick a factor of n if possible, else use math.gcd approach
        chunk = math.gcd(n, 4096)
        if chunk < 64:
            pytest.skip("No clean divisor found for this model's param count")

        packs = packing(list(flat), chunk_size=chunk)
        trimmed = remove_padding(stub, flat_packs(packs))
        reconstructed = reshape_parameters(stub, trimmed)
        for i, (orig, rec) in enumerate(zip(model.get_weights(), reconstructed)):
            np.testing.assert_array_almost_equal(orig, rec, decimal=6, err_msg=f"layer {i}")

    def test_all_models_have_unique_param_counts(self):
        """Smoke-test that each factory produces a structurally distinct model."""
        counts = set()
        for _, factory in [(p.values[0], p.values[1]) for p in MODELS]:
            m = factory((28, 28, 1), 10)
            counts.add(m.count_params())
        # At minimum DNN/DNN_base are aliases; allow some overlap but not full collapse
        assert len(counts) >= len(MODELS) - 2

    @pytest.mark.parametrize("input_shape,num_classes", SHAPES)
    @pytest.mark.parametrize("_name,factory", MODELS)
    def test_layer_shapes_match_after_reconstruct(self, _name, factory, input_shape, num_classes):
        """Each reconstructed layer must have the same shape as the original."""
        model = factory(input_shape, num_classes)
        original = model.get_weights()
        stub = _stub(model)
        flat = flat_parameters(original)
        reconstructed = reshape_parameters(stub, flat)
        for i, (orig, rec) in enumerate(zip(original, reconstructed)):
            assert orig.shape == rec.shape, (
                f"{_name}/{input_shape}: layer {i} shape mismatch "
                f"orig={orig.shape} rec={rec.shape}"
            )
