"""Dynamic autodiff contracts for pure ``advect.index_update`` nodes."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from advect import grad, jacobian, jvp, vjp
from advect.autodiff._ephemeral import trace_call
from advect.autodiff.rules.array_family.vjp import reductions_indexing

_INDEX = (slice(1, None), slice(1, 3))


def _update(base: np.ndarray[Any, Any], replacement: np.ndarray[Any, Any]) -> np.ndarray:
    result = base.copy()
    result[_INDEX] = replacement
    return result


def test_augmented_basic_slice_lowers_to_one_additive_index_update() -> None:
    def update(base: np.ndarray[Any, Any], increment: np.ndarray[Any, Any]) -> np.ndarray:
        result = base.copy()
        result[1:-1] += increment
        return result

    base = np.arange(6.0)
    increment = np.array([0.25, -0.5, 0.75, 1.0])
    trace = trace_call(
        update,
        args=(base, increment),
        kwargs={},
        argnums=(0, 1),
        argnames=None,
        reverse_only=True,
    )

    try:
        assert trace.tape.op_names.count("advect.index_update") == 1
        assert "advect.getitem" not in trace.tape.op_names
        stats = trace.tape.stats()
        assert stats["reverse_pruned"] is True
        assert stats["retained_value_count"] == 0
    finally:
        trace.tape.release_payloads()

    cotangent = np.linspace(-0.4, 0.6, base.size)
    _value, pullback = vjp(update, argnums=(0, 1))(base, increment)
    base_grad, increment_grad = pullback(cotangent)
    assert_allclose(base_grad, cotangent)
    assert_allclose(increment_grad, cotangent[1:-1])


def test_index_update_jvp_overwrites_base_tangent_and_broadcasts_replacement() -> None:
    base = np.arange(12.0).reshape(3, 4)
    replacement = np.array([[20.0], [30.0]])
    base_tangent = np.linspace(-0.5, 0.6, 12).reshape(3, 4)
    replacement_tangent = np.array([[1.5], [-2.0]])

    value, tangent = jvp(_update, argnums=(0, 1))(
        base,
        replacement,
        tangents=(base_tangent, replacement_tangent),
    )

    expected_value = _update(base, replacement)
    expected_tangent = base_tangent.copy()
    expected_tangent[_INDEX] = replacement_tangent
    assert_allclose(value, expected_value)
    assert_allclose(tangent, expected_tangent)


def test_index_update_jvp_zeros_region_for_inactive_replacement() -> None:
    replacement = np.array([[20.0], [30.0]])

    def update_with_constant(base: np.ndarray[Any, Any]) -> np.ndarray:
        return _update(base, replacement)

    base = np.arange(12.0).reshape(3, 4)
    base_tangent = np.linspace(-0.5, 0.6, 12).reshape(3, 4)
    _value, tangent = jvp(update_with_constant)(base, tangents=base_tangent)

    expected = base_tangent.copy()
    expected[_INDEX] = 0.0
    assert_allclose(tangent, expected)


def test_index_update_dynamic_grad_reduces_broadcast_replacement() -> None:
    def loss(base: np.ndarray[Any, Any], replacement: np.ndarray[Any, Any]) -> np.ndarray:
        updated = _update(base, replacement)
        return np.sum(updated * updated)

    base = np.arange(12.0).reshape(3, 4) / 10.0
    replacement = np.array([[0.25], [-0.75]])
    base_grad, replacement_grad = grad(loss, argnums=(0, 1))(base, replacement)

    output_cotangent = 2.0 * _update(base, replacement)
    expected_base = output_cotangent.copy()
    expected_base[_INDEX] = 0.0
    expected_replacement = np.sum(output_cotangent[_INDEX], axis=1, keepdims=True)
    assert_allclose(base_grad, expected_base)
    assert_allclose(replacement_grad, expected_replacement)


def test_index_update_grad_restores_extra_leading_singleton_dimensions() -> None:
    def loss(base: np.ndarray[Any, Any], replacement: np.ndarray[Any, Any]) -> np.ndarray:
        result = base.copy()
        result[:3] = replacement
        return np.sum(result * result)

    base = np.array([0.2, -0.4, 0.7, 1.1])
    replacement = np.array([[0.3, -0.6, 0.8]])
    base_grad, replacement_grad = grad(loss, argnums=(0, 1))(base, replacement)

    assert_allclose(base_grad, np.array([0.0, 0.0, 0.0, 2.2]))
    assert replacement_grad.shape == replacement.shape
    assert_allclose(replacement_grad, 2.0 * replacement)


def test_index_update_complex_real_adjoint_dot_product() -> None:
    base = np.array(
        [
            [0.2 + 0.4j, -0.1 + 0.3j, 0.7 - 0.2j, 0.5 + 0.1j],
            [0.6 - 0.2j, 0.4 + 0.8j, -0.3 + 0.5j, 0.9 - 0.7j],
            [-0.5 + 0.6j, 0.2 - 0.9j, 0.1 + 0.3j, -0.8 + 0.4j],
        ],
        dtype=np.complex64,
    )
    replacement = np.array([[0.25], [-0.75]], dtype=np.float32)
    base_tangent = np.array(
        [
            [0.1 - 0.2j, 0.3 + 0.1j, -0.4 + 0.2j, 0.5 - 0.3j],
            [0.6 + 0.4j, -0.2 + 0.7j, 0.8 - 0.1j, -0.5 + 0.2j],
            [0.4 - 0.6j, 0.9 + 0.3j, -0.7 + 0.5j, 0.2 - 0.8j],
        ],
        dtype=np.complex64,
    )
    replacement_tangent = np.array([[0.6], [-0.4]], dtype=np.float32)
    cotangent = np.array(
        [
            [0.3 + 0.5j, -0.2 + 0.1j, 0.7 - 0.4j, 0.6 + 0.2j],
            [0.1 - 0.8j, 0.9 + 0.3j, -0.5 + 0.6j, 0.2 - 0.7j],
            [-0.4 + 0.9j, 0.8 - 0.2j, 0.5 + 0.4j, -0.3 + 0.1j],
        ],
        dtype=np.complex64,
    )

    _value, output_tangent = jvp(_update, argnums=(0, 1))(
        base,
        replacement,
        tangents=(base_tangent, replacement_tangent),
    )
    _value, pullback = vjp(_update, argnums=(0, 1))(base, replacement)
    base_grad, replacement_grad = pullback(cotangent)

    expected_base_grad = cotangent.copy()
    expected_base_grad[_INDEX] = 0.0
    expected_replacement_grad = np.sum(
        np.real(cotangent[_INDEX]),
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    assert base_grad.dtype == np.dtype(np.complex64)
    assert replacement_grad.dtype == np.dtype(np.float32)
    assert_allclose(base_grad, expected_base_grad)
    assert_allclose(replacement_grad, expected_replacement_grad)

    lhs = np.real(np.vdot(cotangent, output_tangent))
    rhs = np.real(np.vdot(base_grad, base_tangent) + np.vdot(replacement_grad, replacement_tangent))
    assert_allclose(lhs, rhs, rtol=2e-6, atol=2e-6)


def test_index_update_complex_to_real_cast_has_real_linear_adjoint() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    replacement = np.array([[0.25 + 0.8j], [-0.75 - 0.3j]], dtype=np.complex64)
    base_tangent = np.linspace(-0.5, 0.6, 12, dtype=np.float32).reshape(3, 4)
    replacement_tangent = np.array([[0.6 - 0.9j], [-0.4 + 0.2j]], dtype=np.complex64)
    cotangent = np.linspace(0.2, 1.3, 12, dtype=np.float32).reshape(3, 4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        value, output_tangent = jvp(_update, argnums=(0, 1))(
            base,
            replacement,
            tangents=(base_tangent, replacement_tangent),
        )
        _value, pullback = vjp(_update, argnums=(0, 1))(base, replacement)
        base_grad, replacement_grad = pullback(cotangent)

    expected_value = base.copy()
    expected_value[_INDEX] = np.real(replacement)
    expected_tangent = base_tangent.copy()
    expected_tangent[_INDEX] = np.real(replacement_tangent)
    expected_base_grad = cotangent.copy()
    expected_base_grad[_INDEX] = 0.0
    expected_replacement_grad = np.sum(
        cotangent[_INDEX],
        axis=1,
        keepdims=True,
    ).astype(np.complex64)

    assert_allclose(value, expected_value)
    assert_allclose(output_tangent, expected_tangent)
    assert_allclose(base_grad, expected_base_grad)
    assert_allclose(replacement_grad, expected_replacement_grad)
    lhs = np.real(np.vdot(cotangent, output_tangent))
    rhs = np.real(np.vdot(base_grad, base_tangent) + np.vdot(replacement_grad, replacement_tangent))
    assert_allclose(lhs, rhs, rtol=2e-6, atol=2e-6)


def test_index_update_vjp_remains_traceable_for_jvp_of_grad() -> None:
    def loss(base: np.ndarray[Any, Any], replacement: np.ndarray[Any, Any]) -> np.ndarray:
        updated = _update(base, replacement)
        return np.sum(updated * updated)

    base = np.arange(12.0).reshape(3, 4) / 10.0
    replacement = np.array([[0.25], [-0.75]])
    base_tangent = np.linspace(-0.5, 0.6, 12).reshape(3, 4)
    replacement_tangent = np.array([[0.6], [-0.4]])

    gradient, hvp = jvp(grad(loss, argnums=(0, 1)), argnums=(0, 1))(
        base,
        replacement,
        tangents=(base_tangent, replacement_tangent),
    )
    base_grad, replacement_grad = gradient
    base_hvp, replacement_hvp = hvp

    output = _update(base, replacement)
    expected_base_grad = 2.0 * output
    expected_base_grad[_INDEX] = 0.0
    expected_replacement_grad = np.sum(2.0 * output[_INDEX], axis=1, keepdims=True)
    output_tangent = base_tangent.copy()
    output_tangent[_INDEX] = replacement_tangent
    expected_base_hvp = 2.0 * output_tangent
    expected_base_hvp[_INDEX] = 0.0
    expected_replacement_hvp = np.sum(
        2.0 * output_tangent[_INDEX],
        axis=1,
        keepdims=True,
    )

    assert_allclose(base_grad, expected_base_grad)
    assert_allclose(replacement_grad, expected_replacement_grad)
    assert_allclose(base_hvp, expected_base_hvp)
    assert_allclose(replacement_hvp, expected_replacement_hvp)


def test_basic_getitem_pullback_remains_traceable_in_its_cotangent() -> None:
    source = np.array([1.0, 2.0, 3.0, 4.0])
    _value, pullback = vjp(lambda value: value[1:3])(source)
    cotangent = np.array([2.0, 3.0])
    cotangent_tangent = np.array([5.0, 7.0])

    gradient, gradient_tangent = jvp(pullback)(
        cotangent,
        tangents=cotangent_tangent,
    )

    assert_allclose(gradient, np.array([0.0, 2.0, 3.0, 0.0]))
    assert_allclose(gradient_tangent, np.array([0.0, 5.0, 7.0, 0.0]))


def test_advanced_getitem_pullback_propagates_scatter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_scatter(*_args: object) -> None:
        raise ValueError("provider scatter failed")

    namespace = SimpleNamespace(
        add=SimpleNamespace(at=fail_scatter),
        asarray=np.asarray,
        zeros_like=np.zeros_like,
    )
    monkeypatch.setattr(reductions_indexing, "xp", namespace)

    with pytest.raises(ValueError, match="provider scatter failed"):
        reductions_indexing._vjp_getitem(
            np.array([1.0]),
            np.array([1.0]),
            g=np.array([2.0]),
            index=np.array([0]),
        )


def test_stencil_augmented_slice_supports_jvp_of_grad() -> None:
    def stencil_step(field: np.ndarray[Any, Any]) -> np.ndarray:
        result = field.copy()
        laplacian = result[2:] - 2.0 * result[1:-1] + result[:-2]
        result[1:-1] += 0.25 * laplacian
        return result

    def stencil_loss(field: np.ndarray[Any, Any]) -> np.ndarray:
        updated = stencil_step(field)
        return np.sum(updated * updated)

    field = np.array([0.2, -0.4, 0.7, 1.1, -0.3, 0.8])
    tangent = np.array([0.5, -0.2, 0.1, 0.6, -0.7, 0.3])
    value, output_tangent = jvp(stencil_step)(field, tangents=tangent)
    direct_gradient = grad(stencil_loss)(field)
    gradient, hvp = jvp(grad(stencil_loss))(field, tangents=tangent)

    basis = np.eye(field.size)
    transform = np.stack([stencil_step(column) for column in basis.T], axis=1)
    normal = transform.T @ transform
    assert_allclose(value, transform @ field)
    assert_allclose(output_tangent, transform @ tangent)
    assert_allclose(direct_gradient, 2.0 * normal @ field)
    assert_allclose(gradient, 2.0 * normal @ field)
    assert_allclose(hvp, 2.0 * normal @ tangent)


def test_stencil_jacobian_uses_bounded_multi_seed_reverse() -> None:
    def stencil_step(field: np.ndarray[Any, Any]) -> np.ndarray:
        result = field.copy()
        laplacian = result[2:] - 2.0 * result[1:-1] + result[:-2]
        result[1:-1] += 0.25 * laplacian
        return result

    field = np.array([0.2, -0.4, 0.7, 1.1, -0.3, 0.8])
    basis = np.eye(field.size)
    expected = np.stack([stencil_step(column) for column in basis.T], axis=1)

    assert_allclose(jacobian(stencil_step)(field), expected)
