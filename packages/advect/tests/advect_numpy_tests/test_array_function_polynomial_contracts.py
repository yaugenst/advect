"""Public contracts for NumPy's classic polynomial helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_tree_close,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_poly_accepts_square_matrices_and_empty_root_vectors() -> None:
    matrix = np.array([[2.0, 0.3], [-0.2, -1.0]])
    direction = np.array([[0.1, -0.05], [0.2, 0.15]])
    assert_jvp_matches_central_difference(np.poly, (matrix,), (direction,))

    roots = np.empty(0)
    primal, tangent = ad.jvp(np.poly)(roots, tangents=roots)
    np.testing.assert_array_equal(primal, np.poly(roots))
    np.testing.assert_array_equal(tangent, np.array(0.0))


def test_poly_of_a_matrix_differentiates_twice() -> None:
    # poly reads its concrete roots under every trace level of a Hessian.
    matrix = np.array([[2.0, 0.3], [-0.2, -1.0]])
    assert_jvp_matches_central_difference(
        ad.grad(lambda x: np.sum(np.poly(x) ** 2)),
        (matrix,),
        (np.array([[0.1, -0.05], [0.2, 0.15]]),),
    )


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

    assert_jvp_matches_central_difference(
        lambda values: np.polyfit(coordinates, values, 2, full=True),
        (observations,),
        (direction,),
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
    assert_jvp_matches_central_difference(
        lambda values: np.polyint(values, m=2, k=1.5), (coefficients,), (direction,)
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


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (lambda x: np.roots(np.astype(x, np.int64)), np.array([2.0, -3.0, 1.0])),
        (lambda x: np.roots(np.astype(x, np.float32)), np.array([3.2, 2.0, 1.0])),
        (np.roots, np.array([3.2, 2.0, 1.0])),
        (lambda x: np.roots(np.astype(x, np.complex128)), np.array([2.0, -3.0, 1.0])),
        # NumPy decides realness after stripping trailing zeros, so a repeated
        # root can make either choice differ from the full companion's.
        (np.roots, np.array([1.0, -7.0, 8.0, 16.0, 0.0, 0.0])),
        (np.roots, np.array([1.0, 1.0, -21.0, -9.0, 108.0, 0.0])),
        (lambda x: np.roots(np.astype(x, np.float32)), np.array([3.0, 0.0])),
        (lambda x: np.roots(np.astype(x, np.int64)), np.array([0.0, 4.0])),
        (lambda x: np.poly(np.astype(x, np.int64)), np.array([1.0, 2.0, 3.0])),
        (lambda x: np.poly(np.astype(x, np.float32)), np.array([1.0, 2.0, 3.0])),
        (lambda x: np.poly(x * np.array([1 + 1j, 1 - 1j])), np.array([1.5, 1.5])),
        (lambda x: np.poly(x * (1 + 2j)), np.array([1.0, 2.0])),
        (np.poly, np.array([[2.0, 1.0], [1.0, 3.0]])),
    ],
    ids=(
        "roots-integer",
        "roots-float32-complex",
        "roots-complex",
        "roots-complex-coefficients",
        "roots-trailing-zeros-complex",
        "roots-trailing-zero-real",
        "roots-float32-only-zero-roots",
        "roots-integer-constant",
        "poly-integer",
        "poly-float32",
        "poly-conjugate-roots",
        "poly-complex-roots",
        "poly-matrix",
    ),
)
def test_polynomial_helpers_keep_numpy_values_and_dtypes(
    function: Callable[[Any], Any],
    value: np.ndarray[Any, Any],
) -> None:
    primal, _tangent = ad.jvp(function)(value, tangents=np.ones_like(value))

    expected = function(value)
    assert primal.dtype == expected.dtype
    np.testing.assert_allclose(np.sort_complex(primal), np.sort_complex(expected), rtol=1e-6)


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


_QUARTERS = st.integers(min_value=-12, max_value=12).map(lambda value: value / 4)
# A coefficient change dp moves a simple root r by -dp(r) / p'(r): |r| to the
# degree over the product of r's distances to the other roots. Roots 0.25
# apart near 2.5 in magnitude, such as -2, -2.25 and -2.5, therefore curve
# within one central difference step by more than the tolerance. On
# [-1.5, 1.5] that truncation error stays within half the tolerance.
_ROOTS = st.lists(
    st.integers(min_value=-6, max_value=6).map(lambda value: value / 4),
    min_size=1,
    max_size=5,
    unique=True,
)
# Each helper reads a root vector (poly) or a coefficient vector (the rest), a
# second coefficient vector, and evaluation points.
_HELPERS: dict[str, Callable[[Any, np.ndarray[Any, Any], np.ndarray[Any, Any]], Any]] = {
    "poly": lambda roots, _second, _x: np.poly(roots),
    # Sorting fixes the order of distinct roots under a small perturbation.
    "roots": lambda coefficients, _second, _x: np.sort(np.roots(coefficients)),
    "polyval": lambda coefficients, _second, x: np.polyval(coefficients, x),
    "polyadd": lambda coefficients, second, _x: np.polyadd(coefficients, second),
    "polysub": lambda coefficients, second, _x: np.polysub(second, coefficients),
    "polymul": lambda coefficients, second, _x: np.polymul(coefficients, second),
    "polydiv": lambda coefficients, second, _x: np.polydiv(coefficients, second),
    "polyder": lambda coefficients, _second, _x: np.polyder(coefficients),
    "polyint": lambda coefficients, _second, _x: np.polyint(coefficients, k=0.5),
}


@given(
    name=st.sampled_from(sorted(_HELPERS)),
    roots=_ROOTS,
    leading=st.sampled_from((0.5, 1.0, 2.0, 3.0)),
    second=st.lists(_QUARTERS, min_size=1, max_size=3).filter(lambda values: abs(values[0]) >= 0.5),
    x=hnp.arrays(np.float64, 3, elements=_QUARTERS),
    dtype=st.sampled_from((np.int64, np.float32, np.float64, np.complex128)),
)
# polyadd pads the shorter, here the left, operand.
@example(
    name="polyadd",
    roots=[0.5],
    leading=2.0,
    second=[1.0, 3.0, 0.5],
    x=np.zeros(3),
    dtype=np.float64,
)
# An exact division trims the remainder, whose length a perturbation restores.
@example(
    name="polydiv",
    roots=[0.0, 0.5],
    leading=0.5,
    second=[-2.0, 1.0, 0.0],
    x=np.zeros(3),
    dtype=np.float64,
)
# The most clustered roots, whose central difference is the least accurate.
@example(
    name="roots",
    roots=[0.5, 0.75, 1.0, 1.25, 1.5],
    leading=0.5,
    second=[1.0],
    x=np.zeros(3),
    dtype=np.float64,
)
# Unscaled, int64 truncates 1.0 * poly([0, 0.25, -0.25]) to x**3, a triple root.
@example(
    name="roots",
    roots=[0.0, 0.25, -0.25],
    leading=1.0,
    second=[1.0],
    x=np.zeros(3),
    dtype=np.int64,
)
def test_polynomial_helpers_match_numpy_on_well_separated_roots(
    name: str,
    roots: list[float],
    leading: float,
    second: list[float],
    x: np.ndarray[Any, Any],
    dtype: type[np.generic],
) -> None:
    """Roots at least 0.25 apart in [-1.5, 1.5] keep every helper well conditioned.

    Every dtype holds the drawn input exactly: int64 draws scale the roots and
    the leading coefficient to integers, so the cast keeps the roots distinct.
    """
    if dtype is np.int64:
        roots, leading = [4 * root for root in roots], 2 * leading
    value = np.asarray(roots) if name == "poly" else leading * np.poly(roots)
    assert np.array_equal(np.astype(value, dtype), value)

    def function(current: Any) -> Any:
        return _HELPERS[name](np.astype(current, dtype), np.asarray(second), x)

    direction = np.cos(np.arange(value.size, dtype=np.float64))
    # NumPy trims a remainder's vanishing leading coefficients, so an exact
    # division has no central difference: its output shape moves with a step.
    smooth = name != "polydiv" or all(
        np.polydiv(value + step * direction, second)[1].shape == np.polydiv(value, second)[1].shape
        for step in (1e-6, -1e-6)
    )
    if dtype is np.float64 and smooth:
        assert_jvp_matches_central_difference(function, (value,), (direction,), rtol=1e-6)
        return
    primal, _ = ad.jvp(function)(value, tangents=np.zeros_like(value))
    tolerance = 1e-5 if dtype is np.float32 else 1e-9
    assert_tree_close(primal, function(value), rtol=tolerance, atol=tolerance)
