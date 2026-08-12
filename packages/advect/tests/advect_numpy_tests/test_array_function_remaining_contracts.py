"""Public contracts for remaining NumPy array-function variants."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("method", "quantile"),
    [
        ("averaged_inverted_cdf", 0.0),
        ("averaged_inverted_cdf", 1.0),
        ("hazen", 0.0),
        ("hazen", 1.0),
    ],
)
def test_quantile_boundary_methods_select_endpoint_values(
    method: str,
    quantile: float,
) -> None:
    value = np.array([1.0, 2.0, 4.0, 8.0])

    primal, tangent = ad.jvp(lambda x: np.quantile(x, quantile, method=method))(
        value,
        tangents=np.ones_like(value),
    )

    np.testing.assert_allclose(primal, np.quantile(value, quantile, method=method))
    np.testing.assert_allclose(tangent, 1.0)


def test_discrete_quantile_has_zero_coordinate_derivative() -> None:
    value = np.array([1.0, 2.0, 4.0, 8.0])
    quantile = np.array(0.6)

    primal, tangent = ad.jvp(
        lambda q: np.quantile(value, q, method="lower"),
    )(quantile, tangents=np.array(0.2))

    np.testing.assert_allclose(primal, np.quantile(value, quantile, method="lower"))
    np.testing.assert_array_equal(tangent, 0.0)


@pytest.mark.parametrize(
    ("operation", "value", "match"),
    [
        pytest.param(
            lambda x: np.quantile(x, 0.5),
            np.empty(0),
            "cannot reduce an empty axis",
            id="empty-axis",
        ),
        pytest.param(
            lambda x: np.quantile(x, 0.5, method="unsupported"),
            np.arange(4.0),
            "method=.*not supported",
            id="method",
        ),
        pytest.param(
            lambda x: np.quantile(x, 1.5),
            np.arange(4.0),
            "closed interval",
            id="coordinate",
        ),
        pytest.param(
            lambda x: np.nanquantile(x, 0.5),
            np.array([1.0 + 1.0j, 2.0 - 1.0j]),
            "does not support complex",
            id="nan-complex",
        ),
        pytest.param(
            lambda x: np.quantile(x, 0.5, overwrite_input=True),
            np.arange(4.0),
            "would mutate",
            id="overwrite",
        ),
        pytest.param(
            lambda x: np.quantile(x, 0.5, weights=np.ones(4)),
            np.arange(4.0),
            "weights=.*requires method",
            id="weights-method",
        ),
        pytest.param(
            lambda x: np.nanquantile(
                x,
                0.5,
                method="inverted_cdf",
                weights=np.ones(4),
            ),
            np.arange(4.0),
            "weighted NaN filtering",
            id="weighted-nan",
        ),
        pytest.param(
            lambda x: np.median(x, overwrite_input=True),
            np.arange(4.0),
            "would mutate",
            id="median-overwrite",
        ),
    ],
)
def test_order_statistics_reject_invalid_public_forms(
    operation: Callable[[Any], Any],
    value: np.ndarray[Any, Any],
    match: str,
) -> None:
    with pytest.raises(ad.TracingError, match=match):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


def test_nanquantile_preserves_all_nan_rows_and_keepdims() -> None:
    value = np.array([[np.nan, np.nan, np.nan], [1.0, 3.0, 5.0]])
    quantile = np.array(0.25)

    primal, tangent = ad.jvp(
        lambda x, q: np.nanquantile(x, q, axis=1, keepdims=True),
        argnums=(0, 1),
    )(
        value,
        quantile,
        tangents=(np.ones_like(value), np.array(0.1)),
    )

    with pytest.warns(RuntimeWarning, match="All-NaN"):
        expected = np.nanquantile(value, quantile, axis=1, keepdims=True)
    np.testing.assert_allclose(primal, expected, equal_nan=True)
    assert tangent.shape == expected.shape
    np.testing.assert_array_equal(tangent[0], 0.0)


@pytest.mark.parametrize(
    ("quantile", "weights", "match"),
    [
        pytest.param(0.5, np.ones(2), "must match one reduction axis", id="one-dimensional"),
        pytest.param(
            0.5,
            np.ones((2, 2)),
            "one-dimensional or match the input shape",
            id="shape",
        ),
        pytest.param(1.5, np.ones(3), "closed interval", id="coordinate"),
        pytest.param(0.5, np.array([1.0, -1.0, 1.0]), "non-negative", id="negative"),
        pytest.param(0.5, np.zeros(3), "positive finite sum", id="zero-sum"),
    ],
)
def test_weighted_quantile_validates_weights(
    quantile: float,
    weights: np.ndarray[Any, Any],
    match: str,
) -> None:
    value = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match=match):
        ad.jvp(
            lambda x: np.quantile(
                x,
                quantile,
                axis=1,
                method="inverted_cdf",
                weights=weights,
            )
        )(value, tangents=np.ones_like(value))


@pytest.mark.parametrize("axis", [0, np.int64(0)], ids=("python-int", "numpy-int"))
def test_unique_axis_preserves_classic_auxiliary_results(axis: int | np.integer[Any]) -> None:
    value = np.array([[2.0, 1.0], [2.0, 1.0], [3.0, 4.0]])
    direction = np.arange(value.size, dtype=float).reshape(value.shape)

    def unique(x: Any) -> Any:
        return np.unique(
            x,
            True,  # noqa: FBT003 - exercise NumPy's positional signature
            True,  # noqa: FBT003 - exercise NumPy's positional signature
            True,  # noqa: FBT003 - exercise NumPy's positional signature
            axis,
            equal_nan=False,
        )

    primal, tangent = ad.jvp(unique)(value, tangents=direction)
    expected = unique(value)

    for actual, reference in zip(primal, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(tangent[0], direction[[0, 2]])
    for discrete in tangent[1:]:
        np.testing.assert_array_equal(discrete, np.zeros_like(discrete))


def test_unique_rejects_a_non_integer_axis() -> None:
    value = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match="axis must be an integer or None"):
        ad.jvp(lambda x: np.unique(x, axis=0.5))(value, tangents=np.ones_like(value))


@pytest.mark.skipif(
    "sorted" not in inspect.signature(np.unique).parameters,
    reason="NumPy added unique(sorted=...) after 2.0",
)
def test_unique_accepts_the_versioned_sorted_option() -> None:
    value = np.array([2.0, 1.0, 2.0, 3.0])
    direction = np.array([10.0, 20.0, 30.0, 40.0])

    primal, tangent = ad.jvp(lambda x: np.unique(x, sorted=False))(
        value,
        tangents=direction,
    )

    expected, indices = np.unique(value, return_index=True, sorted=False)
    np.testing.assert_array_equal(primal, expected)
    np.testing.assert_array_equal(tangent, direction[indices])


@pytest.mark.parametrize("operation", [np.all, np.any], ids=("all", "any"))
def test_truth_reductions_honor_where_and_keepdims(operation: Callable[..., Any]) -> None:
    value = np.array([[0.0, 2.0, np.nan], [1.0, 0.0, np.nan]])
    where = np.array([[True, False, True], [False, True, True]])

    primal, tangent = ad.jvp(lambda x: operation(x, axis=1, keepdims=True, where=where))(
        value, tangents=np.ones_like(value)
    )

    np.testing.assert_array_equal(
        primal,
        operation(value, axis=1, keepdims=True, where=where),
    )
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (
            lambda x: np.isclose(
                x,
                [np.nan, 2.0, 3.0],
                1e-5,
                1e-8,
                True,  # noqa: FBT003 - exercise the positional signature
            ),
            [True, True, True],
        ),
        (lambda x: np.array_equal(x, np.ones((2, 2))), False),
        (
            lambda x: np.array_equal(
                x,
                [np.nan, 2.0, 3.0],
                True,  # noqa: FBT003 - exercise the positional signature
            ),
            True,
        ),
        (lambda x: np.array_equiv(x, np.ones(4)), False),
    ],
    ids=("isclose-equal-nan", "array-equal-shape", "array-equal-nan", "array-equiv-shape"),
)
def test_comparison_options_are_piecewise_constant(
    operation: Callable[[Any], Any],
    expected: object,
) -> None:
    value = np.array([np.nan, 2.0, 3.0])

    primal, tangent = ad.jvp(operation)(value, tangents=np.ones_like(value))

    np.testing.assert_array_equal(primal, expected)
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


def test_signal_operations_reject_an_invalid_mode() -> None:
    value = np.arange(4.0)

    with pytest.raises(ad.TracingError, match="mode must be full, same, or valid"):
        ad.jvp(lambda x: np.convolve(x, np.ones(2), mode="circular"))(
            value,
            tangents=np.ones_like(value),
        )


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda x: np.fft.fftn(x, s=None, axes=None), id="explicit-defaults"),
        pytest.param(lambda x: np.fft.fftshift(x, axes=-1), id="scalar-axis"),
    ],
)
def test_fft_metadata_variants_apply_to_primal_and_tangent(
    operation: Callable[[Any], Any],
) -> None:
    value = np.arange(8.0).reshape(2, 4)
    direction = np.linspace(-1.0, 1.0, value.size).reshape(value.shape)

    primal, tangent = ad.jvp(operation)(value, tangents=direction)

    np.testing.assert_allclose(primal, operation(value))
    np.testing.assert_allclose(tangent, operation(direction))


def test_irfftn_shape_defaults_to_the_trailing_axes() -> None:
    value = np.arange(24.0).reshape(2, 3, 4).astype(np.complex128)
    direction = np.linspace(-1.0, 1.0, value.size).reshape(value.shape).astype(np.complex128)

    primal, tangent = ad.jvp(lambda x: np.fft.irfftn(x, s=(3, 6)))(
        value,
        tangents=direction,
    )

    np.testing.assert_allclose(primal, np.fft.irfftn(value, s=(3, 6), axes=(-2, -1)))
    np.testing.assert_allclose(tangent, np.fft.irfftn(direction, s=(3, 6), axes=(-2, -1)))
