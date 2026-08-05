"""Contracts for the explicitly supported NumPy ufunc methods."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, strategies as st

import advect as ad


@pytest.mark.parametrize(
    ("method", "operation", "expected"),
    [
        ("reduce", np.add, lambda x: np.sum(x, axis=1)),
        ("reduce", np.multiply, lambda x: np.prod(x, axis=1)),
        ("accumulate", np.add, lambda x: np.cumsum(x, axis=1)),
        ("accumulate", np.multiply, lambda x: np.cumprod(x, axis=1)),
    ],
)
def test_reduction_methods_match_equivalent_numpy_functions(
    method: str,
    operation: np.ufunc,
    expected: Any,
) -> None:
    value = np.array([[0.7, 1.2, 1.8], [1.1, 0.8, 1.4]])
    direction = np.array([[0.2, -0.3, 0.5], [-0.1, 0.4, 0.25]])

    def apply(x: Any) -> Any:
        return getattr(operation, method)(x, axis=1, dtype=np.float64)

    primal, tangent = ad.jvp(apply)(value, tangents=direction)
    epsilon = 1e-6
    finite_difference = (
        expected(value + epsilon * direction) - expected(value - epsilon * direction)
    ) / (2 * epsilon)

    np.testing.assert_allclose(primal, expected(value))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-6, atol=2e-6)


@given(
    left_rows=st.integers(min_value=1, max_value=4),
    right_columns=st.integers(min_value=1, max_value=4),
)
def test_binary_outer_matches_broadcasted_call(
    left_rows: int,
    right_columns: int,
) -> None:
    left = np.linspace(0.3, 1.2, left_rows)
    right = np.linspace(-0.7, 0.8, right_columns)
    left_tangent = np.linspace(0.1, 0.4, left_rows)
    right_tangent = np.linspace(-0.2, 0.3, right_columns)

    primal, tangent = ad.jvp(lambda x, y: np.multiply.outer(x, y), argnums=(0, 1))(
        left,
        right,
        tangents=(left_tangent, right_tangent),
    )

    np.testing.assert_allclose(primal, np.multiply.outer(left, right))
    np.testing.assert_allclose(
        tangent,
        np.multiply.outer(left_tangent, right) + np.multiply.outer(left, right_tangent),
    )


@pytest.mark.parametrize(
    ("operation", "partial"),
    [
        (np.floor_divide, 0.0),
        (np.remainder, 1.0),
    ],
    ids=["floor-divide", "remainder"],
)
def test_nonsmooth_outer_left_jvp_has_the_primal_shape(
    operation: np.ufunc,
    partial: float,
) -> None:
    left = np.array([0.25, 1.25, 2.5])
    right = np.array([0.7, 1.5])
    left_tangent = np.array([0.1, -0.2, 0.3])

    primal, tangent = ad.jvp(lambda value: operation.outer(value, right))(
        left,
        tangents=left_tangent,
    )

    np.testing.assert_allclose(primal, operation.outer(left, right))
    assert tangent.shape == primal.shape
    expected = np.broadcast_to(partial * left_tangent[:, None], primal.shape)
    np.testing.assert_allclose(tangent, expected)


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


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda x: np.add.reduce(x, axis=1), lambda x: np.sum(x, axis=1)),
        (lambda x: np.multiply.reduce(x, axis=1), lambda x: np.prod(x, axis=1)),
        (lambda x: np.add.accumulate(x, axis=1), lambda x: np.cumsum(x, axis=1)),
        (lambda x: np.multiply.accumulate(x, axis=1), lambda x: np.cumprod(x, axis=1)),
        (lambda x: np.multiply.outer(x[0], x[0]), lambda x: np.multiply.outer(x[0], x[0])),
    ],
    ids=["add-reduce", "multiply-reduce", "add-accumulate", "multiply-accumulate", "outer"],
)
def test_supported_ufunc_methods_stage_and_serialize(
    operation: Any,
    expected: Any,
) -> None:
    program = ad.stage(operation, specs=(ad.ArraySpec((2, 3), "float64"),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    value = np.array([[0.7, 1.2, 1.8], [1.1, 0.8, 1.4]])

    for staged in (program, restored):
        np.testing.assert_allclose(staged(value), expected(value))


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
