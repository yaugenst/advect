"""Contracts for the explicitly supported NumPy ufunc methods."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect.numpy._support_contract import numpy_support_declarations
from advect_numpy_tests._assertions import assert_spellings_agree
from advect_numpy_tests._support_case_families import support_cases

_DIFFERENTIABLE = {
    declaration.callable
    for declaration in numpy_support_declarations()
    if declaration.kind == "ufunc_method" and declaration.has_derivatives
}
_OUTER_CASES = {
    case.callable.removeprefix("numpy."): case
    for case in support_cases()
    if case.callable.endswith(".outer") and case.callable in _DIFFERENTIABLE
}


@given(
    name=st.sampled_from(sorted(_OUTER_CASES)),
    left_shape=hnp.array_shapes(min_dims=0, max_dims=2, min_side=1, max_side=3),
    right_shape=hnp.array_shapes(min_dims=0, max_dims=2, min_side=1, max_side=3),
)
@example(name="floor_divide.outer", left_shape=(3,), right_shape=(2,))
@example(name="remainder.outer", left_shape=(3,), right_shape=(2,))
@example(name="multiply.outer", left_shape=(4,), right_shape=(4,))
def test_differentiable_outer_forms_are_broadcast_calls(
    name: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> None:
    case = _OUTER_CASES[name]
    ufunc = getattr(np, name.removesuffix(".outer"))
    operands = tuple(
        np.resize(np.asarray(spec.data, dtype=spec.dtype), shape)
        for spec, shape in zip(case.inputs, (left_shape, right_shape), strict=True)
    )
    argnums = (case.derivative_argnums or ((0, 1),))[-1]
    argnums = tuple(index for index in argnums if operands[index].dtype.kind == "f")

    def broadcast(left: Any, right: Any) -> Any:
        return ufunc(np.reshape(left, np.shape(left) + (1,) * np.ndim(right)), right)

    assert_spellings_agree(
        ufunc.outer, broadcast, operands, argnums=argnums, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_outer_accepts_a_python_sequence_operand(side: str) -> None:
    value = np.array([0.5, 1.5])
    sequence = [1.0, 2.0, 3.0]

    def outer(x: Any) -> Any:
        return np.add.outer(sequence, x) if side == "left" else np.add.outer(x, sequence)

    primal, tangent = ad.jvp(outer)(value, tangents=np.ones_like(value))
    program = ad.stage(outer, specs=(ad.ArraySpec(value.shape, value.dtype),))

    np.testing.assert_allclose(primal, outer(value))
    np.testing.assert_allclose(tangent, np.ones_like(primal))
    np.testing.assert_allclose(program(value), outer(value))


def test_supported_ufunc_methods_functionalize_out_and_nondefault_controls() -> None:
    value = np.array([[0.7, 1.2, 1.8], [1.1, 0.8, 1.4]])
    direction = np.array([[0.2, -0.3, 0.5], [-0.1, 0.4, 0.25]])
    mask = np.array([[True, False, True], [False, True, True]])

    def reduced(x: Any) -> Any:
        destination = np.zeros_like(np.sum(x, axis=1))
        result = np.add.reduce(
            x,
            axis=1,
            dtype=np.float64,
            out=destination,
            keepdims=False,
            initial=0.25,
            where=mask,
        )
        assert result is destination
        return destination

    reduced_value, reduced_tangent = ad.jvp(reduced)(value, tangents=direction)
    np.testing.assert_allclose(
        reduced_value,
        np.add.reduce(value, axis=1, initial=0.25, where=mask),
    )
    np.testing.assert_allclose(reduced_tangent, np.sum(np.where(mask, direction, 0), axis=1))

    def accumulated(x: Any) -> Any:
        destination = np.zeros_like(x)
        result = np.add.accumulate(x, axis=1, dtype=np.float64, out=destination)
        assert result is destination
        return destination

    accumulated_value, accumulated_tangent = ad.jvp(accumulated)(
        value,
        tangents=direction,
    )
    np.testing.assert_allclose(accumulated_value, np.add.accumulate(value, axis=1))
    np.testing.assert_allclose(accumulated_tangent, np.add.accumulate(direction, axis=1))

    vector = value[0]

    def outer(x: Any) -> Any:
        destination = np.zeros((x.size, x.size), dtype=x.dtype, like=x)
        result = np.multiply.outer(x, x, out=destination, casting="same_kind")
        assert result is destination
        return destination

    outer_value, outer_tangent = ad.jvp(outer)(vector, tangents=direction[0])
    np.testing.assert_allclose(outer_value, np.multiply.outer(vector, vector))
    np.testing.assert_allclose(
        outer_tangent,
        np.multiply.outer(direction[0], vector) + np.multiply.outer(vector, direction[0]),
    )


@pytest.mark.parametrize("operation", [np.add, np.multiply])
def test_accumulate_rejects_a_0d_input_as_numpy_does(operation: np.ufunc) -> None:
    # np.cumsum and np.cumprod read a 0-d input as a vector; accumulate does not.
    value = np.asarray(1.5)
    with pytest.raises(TypeError, match="cannot accumulate on a scalar"):
        operation.accumulate(value)
    with pytest.raises(TypeError, match="cannot accumulate on a scalar"):
        ad.jvp(operation.accumulate)(value, tangents=np.ones_like(value))
    with pytest.raises(TypeError, match="cannot accumulate on a scalar"):
        ad.stage(operation.accumulate, specs=(ad.ArraySpec((), "float64"),))


@pytest.mark.parametrize("method", ["reduceat", "at"])
def test_unsupported_methods_fail_by_explicit_method_name(method: str) -> None:
    value = np.arange(4.0)

    def apply(x: Any) -> Any:
        if method == "reduceat":
            return np.add.reduceat(x, [0, 2])
        np.add.at(x, [0, 2], 1.0)
        return x

    with pytest.raises(ad.TracingError, match=rf"numpy\.add\.{method}"):
        ad.jvp(apply)(value, tangents=np.ones_like(value))

    with pytest.raises(ad.TracingError, match=rf"numpy\.add\.{method}"):
        ad.stage(apply, specs=(ad.ArraySpec(value.shape, value.dtype),))


@pytest.mark.parametrize("keyword", ["axes", "axis", "keepdims"])
def test_generalized_ufunc_controls_fail_by_parameter_name(keyword: str) -> None:
    matrix = np.eye(2)

    with pytest.raises(ad.TracingError, match=keyword):
        ad.jvp(lambda x: np.matmul(x, x, **{keyword: None}))(
            matrix,
            tangents=np.ones_like(matrix),
        )
