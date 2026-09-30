"""Dynamic autodiff contracts for pure ``advect.index_update`` nodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

from advect import grad, hessian, hvp, jacobian, jvp, vjp
from advect.autodiff._ephemeral import trace_call
from advect.autodiff.rules.array_family.vjp import reductions_indexing


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


@pytest.mark.parametrize("value", [3.0, np.array(3.0)], ids=["python-float", "0-d"])
@pytest.mark.parametrize(
    ("index", "derivative"),
    [(..., 8.0), (None, 8.0), (False, 6.0), ((False, ...), 6.0)],
    ids=["ellipsis", "new-axis", "false", "false-ellipsis"],
)
def test_index_update_set_mode_differentiates_a_rank_zero_base(
    value: Any, index: Any, derivative: float
) -> None:
    # A rank-0 base keeps its element when the index, like False, selects nothing.
    def replace(v: Any) -> Any:
        updated = v.copy()
        updated[index] = v + 1.0
        return updated * updated

    assert float(jvp(replace)(value, tangents=1.0)[1]) == derivative
    assert float(grad(replace)(value)) == derivative
    assert float(vjp(replace)(value)[1](1.0)) == derivative
    assert float(hessian(replace)(value)) == 2.0


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


def test_advanced_indexing_hessian_reports_first_order_boundary() -> None:
    """Higher-order advanced-index pullbacks fail with the public boundary."""
    value = np.arange(12.0).reshape(3, 4)
    tangent = np.full_like(value, 0.25)
    index = np.array([2, 0])

    def gradient_function(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        _output, pullback = vjp(lambda y: y[index, 1:3])(x)
        try:
            return pullback(np.ones((2, 2)))
        finally:
            pullback.close()

    with pytest.raises(
        NotImplementedError,
        match="Higher-order pullbacks for advanced indexing",
    ):
        jvp(gradient_function)(value, tangents=tangent)


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
    assert_allclose(jacobian(stencil_step)(field), transform)


@pytest.mark.parametrize("mode", ["set", "add"])
def test_array_api_index_update_jvp_copies_without_a_copy_method(mode: str) -> None:
    """Array API arrays have no ``copy()``; the tangent update stays portable."""
    value = np.array([0.3, -1.2, 0.7])
    tangent = np.array([1.0, 2.0, 3.0])

    def update(v: Any) -> Any:
        result = v.__array_namespace__().zeros_like(v) if mode == "set" else v * 1.0
        if mode == "set":
            result[0:2] = v[1:3] * 2.0
        else:
            result[0:2] += v[1:3] * 2.0
        return result

    _output, output_tangent = jvp(update)(
        strict.asarray(value, dtype=strict.float64),
        tangents=strict.asarray(tangent, dtype=strict.float64),
    )

    expected = np.zeros(3) if mode == "set" else tangent.copy()
    expected[0:2] = expected[0:2] + 2.0 * tangent[1:3]
    assert_allclose(np.asarray(output_tangent), expected)


def test_array_api_sort_hvp_scatters_through_a_traced_index_update() -> None:
    value = np.array([0.3, -1.2, 0.7])
    direction = np.array([1.0, 2.0, 3.0])

    def loss(v: Any) -> Any:
        namespace = v.__array_namespace__()
        return namespace.sum(namespace.sort(v) ** 3)

    _gradient, product = hvp(loss)(
        strict.asarray(value, dtype=strict.float64),
        vectors=strict.asarray(direction, dtype=strict.float64),
    )

    assert_allclose(np.asarray(product), 6.0 * value * direction)
