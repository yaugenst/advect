"""Public contracts for NumPy's classic polynomial helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _assert_unary_jvp_matches_difference(
    function: Callable[[Any], Any],
    value: np.ndarray[Any, Any],
    direction: np.ndarray[Any, Any],
    *,
    rtol: float = 2e-5,
    atol: float = 2e-6,
) -> None:
    primal, tangent = ad.jvp(function)(value, tangents=direction)
    expected = function(value)
    step = 1e-6
    plus = function(value + step * direction)
    minus = function(value - step * direction)
    primal_leaves, primal_tree = ad.pytree.tree_flatten(primal)
    expected_leaves, expected_tree = ad.pytree.tree_flatten(expected)
    tangent_leaves, tangent_tree = ad.pytree.tree_flatten(tangent)
    plus_leaves, plus_tree = ad.pytree.tree_flatten(plus)
    minus_leaves, minus_tree = ad.pytree.tree_flatten(minus)
    assert primal_tree == expected_tree
    assert tangent_tree == primal_tree == plus_tree == minus_tree

    for actual, reference in zip(primal_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=rtol, atol=atol)
    for actual, upper, lower in zip(tangent_leaves, plus_leaves, minus_leaves, strict=True):
        np.testing.assert_allclose(
            actual,
            (np.asarray(upper) - np.asarray(lower)) / (2 * step),
            rtol=rtol,
            atol=atol,
        )


def test_poly_accepts_square_matrices_and_empty_root_vectors() -> None:
    matrix = np.array([[2.0, 0.3], [-0.2, -1.0]])
    direction = np.array([[0.1, -0.05], [0.2, 0.15]])
    _assert_unary_jvp_matches_difference(np.poly, matrix, direction)

    roots = np.empty(0)
    primal, tangent = ad.jvp(np.poly)(roots, tangents=roots)
    np.testing.assert_array_equal(primal, np.poly(roots))
    np.testing.assert_array_equal(tangent, np.array(0.0))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.ones((2, 3)), "non-empty and square"),
        (np.ones((1, 1, 1)), "one-dimensional or a square matrix"),
    ],
)
def test_poly_rejects_invalid_input_shapes(
    value: np.ndarray[Any, Any],
    message: str,
) -> None:
    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(np.poly)(value, tangents=np.ones_like(value))


def test_polyadd_pads_a_shorter_left_operand() -> None:
    left = np.array([2.0, -1.0])
    right = np.array([1.0, 3.0, 0.5])
    direction = np.array([0.2, -0.4])

    primal, tangent = ad.jvp(lambda value: np.polyadd(value, right))(
        left,
        tangents=direction,
    )

    np.testing.assert_array_equal(primal, np.polyadd(left, right))
    np.testing.assert_array_equal(tangent, np.array([0.0, *direction]))


@pytest.mark.parametrize("operation", [np.polyadd, np.polymul, np.polydiv])
def test_polynomial_arithmetic_requires_coefficient_vectors(
    operation: Callable[[Any, Any], Any],
) -> None:
    value = np.ones((2, 2))
    with pytest.raises(ad.TracingError, match="one-dimensional"):
        ad.jvp(lambda coefficients: operation(coefficients, np.ones(2)))(
            value,
            tangents=np.ones_like(value),
        )


def test_polyfit_full_preserves_numpy_output_contract() -> None:
    coordinates = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    observations = np.array([4.1, 0.8, 0.2, 1.2, 3.9])
    direction = np.array([0.1, -0.2, 0.05, 0.15, -0.1])

    _assert_unary_jvp_matches_difference(
        lambda values: np.polyfit(coordinates, values, 2, full=True),
        observations,
        direction,
        rtol=2e-4,
        atol=2e-5,
    )


@pytest.mark.parametrize(
    ("function", "values", "tangents", "message"),
    [
        pytest.param(
            lambda x, degree: np.polyfit(x, np.arange(4.0), degree),
            (np.arange(4.0), np.array(2.0)),
            (np.ones(4), np.array(0.0)),
            "degree must be a static",
            id="dynamic-degree",
        ),
        pytest.param(
            lambda x: np.polyfit(x, np.arange(4.0), -1),
            (np.arange(4.0),),
            (np.ones(4),),
            "degree must be non-negative",
            id="negative-degree",
        ),
        pytest.param(
            lambda x: np.polyfit(x, np.arange(4.0), 2),
            (np.arange(4.0).reshape(2, 2),),
            (np.ones((2, 2)),),
            "x must be a non-empty vector",
            id="matrix-x",
        ),
        pytest.param(
            lambda y: np.polyfit(np.arange(4.0), y, 2),
            (np.arange(3.0),),
            (np.ones(3),),
            "y must be one- or two-dimensional and match x",
            id="mismatched-y",
        ),
        pytest.param(
            lambda weights: np.polyfit(
                np.arange(4.0),
                np.arange(4.0),
                2,
                w=weights,
            ),
            (np.ones(3),),
            (np.ones(3),),
            "weights must be one-dimensional and match x",
            id="mismatched-weights",
        ),
        pytest.param(
            lambda x, rcond: np.polyfit(x, np.arange(4.0), 2, rcond=rcond),
            (np.arange(4.0), np.array(1e-12)),
            (np.ones(4), np.array(0.0)),
            "rcond must be static",
            id="dynamic-rcond",
        ),
        pytest.param(
            lambda x: np.polyfit(x, np.arange(4.0), 2),
            (np.ones(4),),
            (np.ones(4),),
            "rank-deficient design matrix",
            id="rank-deficient-design",
        ),
        pytest.param(
            lambda x: np.polyfit(x, np.arange(3.0), 2, cov=True),
            (np.arange(3.0),),
            (np.ones(3),),
            "covariance scaling requires more points",
            id="scaled-covariance-needs-residual-degrees-of-freedom",
        ),
    ],
)
def test_polyfit_rejects_dynamic_or_invalid_fit_controls(
    function: Callable[..., Any],
    values: tuple[np.ndarray[Any, Any], ...],
    tangents: tuple[np.ndarray[Any, Any], ...],
    message: str,
) -> None:
    argnums: int | tuple[int, ...] = 0 if len(values) == 1 else tuple(range(len(values)))
    tangent_input: object = tangents[0] if len(tangents) == 1 else tangents
    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(function, argnums=argnums)(*values, tangents=tangent_input)


def test_polyder_handles_excess_order_and_rejects_negative_order() -> None:
    coefficients = np.array([3.0])
    primal, tangent = ad.jvp(lambda values: np.polyder(values, m=2))(
        coefficients,
        tangents=np.ones_like(coefficients),
    )
    np.testing.assert_array_equal(primal, np.polyder(coefficients, m=2))
    np.testing.assert_array_equal(tangent, np.empty(0))

    with pytest.raises(ad.TracingError, match="order must be non-negative"):
        ad.jvp(lambda values: np.polyder(values, m=-1))(
            coefficients,
            tangents=np.ones_like(coefficients),
        )


def test_polyint_repeats_a_scalar_constant_for_each_integration() -> None:
    coefficients = np.array([2.0, -3.0])
    direction = np.array([0.4, -0.2])
    _assert_unary_jvp_matches_difference(
        lambda values: np.polyint(values, m=2, k=1.5),
        coefficients,
        direction,
    )


@pytest.mark.parametrize(
    ("order", "constants", "message"),
    [
        (-1, None, "order must be non-negative"),
        (3, [1.0, 2.0], "k must be scalar or contain at least m constants"),
    ],
)
def test_polyint_rejects_invalid_order_or_constants(
    order: int,
    constants: object,
    message: str,
) -> None:
    coefficients = np.array([2.0, -3.0])
    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(lambda values: np.polyint(values, m=order, k=constants))(
            coefficients,
            tangents=np.ones_like(coefficients),
        )


@pytest.mark.parametrize("coefficients", [np.zeros(3), np.array([0.0, 4.0])])
def test_roots_returns_empty_for_zero_and_constant_polynomials(
    coefficients: np.ndarray[Any, Any],
) -> None:
    primal, tangent = ad.jvp(np.roots)(
        coefficients,
        tangents=np.ones_like(coefficients),
    )
    np.testing.assert_array_equal(primal, np.roots(coefficients))
    np.testing.assert_array_equal(tangent, np.empty(0))


def test_roots_requires_a_vector_and_concrete_leading_coefficient() -> None:
    matrix = np.eye(2)
    with pytest.raises(ad.TracingError, match="coefficients must be one-dimensional"):
        ad.jvp(np.roots)(matrix, tangents=np.ones_like(matrix))

    with pytest.raises(ad.TracingError):
        ad.stage(np.roots, specs=ad.ArraySpec((3,), np.float64))
