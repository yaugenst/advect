"""Public contracts for NumPy order statistics lowered to sorts and gathers."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect_numpy_tests._assertions import assert_jvp_matches_central_difference

if TYPE_CHECKING:
    from collections.abc import Callable


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
            lambda x: np.nanquantile(x, 1.5),
            np.full(3, np.nan),
            "closed interval",
            id="nan-coordinate",
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
        *(
            pytest.param(
                lambda x, quantile=quantile, weights=weights: np.quantile(
                    x, quantile, axis=1, method="inverted_cdf", weights=weights
                ),
                np.arange(6.0).reshape(2, 3),
                match,
                id=f"weights-{name}",
            )
            for name, quantile, weights, match in (
                ("one-dimensional", 0.5, np.ones(2), "must match one reduction axis"),
                ("shape", 0.5, np.ones((2, 2)), "one-dimensional or match the input shape"),
                ("coordinate", 1.5, np.ones(3), "closed interval"),
                ("negative", 0.5, np.array([1.0, -1.0, 1.0]), "non-negative"),
                ("zero-sum", 0.5, np.zeros(3), "positive finite sum"),
            )
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


_QUANTILE_METHODS = (
    "averaged_inverted_cdf",
    "closest_observation",
    "hazen",
    "higher",
    "interpolated_inverted_cdf",
    "inverted_cdf",
    "linear",
    "lower",
    "median_unbiased",
    "midpoint",
    "nearest",
    "normal_unbiased",
    "weibull",
)
_ORDER_STATISTIC_FORMS = (
    "median",
    "nanmedian",
    "nanpercentile",
    "nanquantile",
    "percentile",
    "quantile",
    "weighted",
)


@pytest.mark.parametrize("method", _QUANTILE_METHODS)
def test_quantile_methods_match_numpy_and_directional_differences(method: str) -> None:
    value = np.array([0.1, 1.0, 2.5, 4.0, 8.0])
    direction = np.array([0.3, -0.2, 0.4, 0.1, -0.5])

    primal, tangent = assert_jvp_matches_central_difference(
        lambda x: np.quantile(x, [0.2, 0.7], method=method),
        (value,),
        (direction,),
    )

    np.testing.assert_allclose(
        primal,
        np.quantile(value, [0.2, 0.7], method=method),
    )
    assert np.shape(tangent) == (2,)


def _cyclic_weights(shape: int | tuple[int, ...]) -> np.ndarray[Any, Any]:
    """Weights 2, 0, 1, ... in order, so a sorted slice can start with zero weight."""
    return ((np.arange(np.prod(shape)) + 2) % 3).reshape(shape).astype(np.float64)


def _order_statistic(
    form: str, method: str, quantile: Any, percent: Any, axis: int | None
) -> Callable[[Any], Any]:
    forms: dict[str, Callable[[Any], Any]] = {
        "median": lambda x: np.median(x, axis=axis),
        "nanmedian": lambda x: np.nanmedian(x, axis=axis),
        "nanpercentile": lambda x: np.nanpercentile(x, percent, axis=axis, method=method),
        "nanquantile": lambda x: np.nanquantile(x, quantile, axis=axis, method=method),
        "percentile": lambda x: np.percentile(x, percent, axis=axis, method=method),
        "quantile": lambda x: np.quantile(x, quantile, axis=axis, method=method),
        "weighted": lambda x: np.quantile(
            x,
            quantile,
            axis=axis,
            method="inverted_cdf",
            weights=_cyclic_weights(x.shape if axis is None else x.shape[axis]),
        ),
    }
    return forms[form]


_ORDER_STATISTIC_SHAPES = hnp.array_shapes(min_dims=2, max_dims=2, max_side=4)


@given(
    value=hnp.arrays(
        st.sampled_from([np.float32, np.float64]),
        _ORDER_STATISTIC_SHAPES,
        elements=st.floats(min_value=-1e6, max_value=1e6, width=32)
        | st.sampled_from([np.nan, np.inf, -np.inf]),
    )
    | hnp.arrays(
        st.sampled_from([np.int32, np.int64]),
        _ORDER_STATISTIC_SHAPES,
        elements=st.integers(-1000, 1000),
    )
    # Small integers span their whole range, so a mean of two can overflow them.
    | hnp.arrays(st.sampled_from([np.int8, np.uint8]), _ORDER_STATISTIC_SHAPES)
    # Half precision sums the middle pair within range, below the deliberate
    # nanmedian divergence at the float16 limit.
    | hnp.arrays(
        np.float16,
        _ORDER_STATISTIC_SHAPES,
        elements=st.floats(min_value=-3e4, max_value=3e4, width=16)
        | st.sampled_from([np.nan, np.inf, -np.inf]),
    ),
    form=st.sampled_from(_ORDER_STATISTIC_FORMS),
    method=st.sampled_from(_QUANTILE_METHODS),
    quantile=st.floats(min_value=0.0, max_value=1.0)
    | st.sampled_from([0, 1])
    | hnp.arrays(np.float64, st.integers(1, 2), elements=st.floats(0.0, 1.0))
    | hnp.arrays(np.float32, st.integers(1, 2), elements=st.floats(0.0, 1.0, width=32))
    | hnp.arrays(
        st.sampled_from([np.bool_, np.int8, np.int64]),
        st.integers(1, 2),
        elements=st.integers(0, 1),
    ),
    percent=st.floats(min_value=0.0, max_value=100.0)
    | st.integers(0, 100)
    | st.integers(0, 100).map(np.int64)
    | st.lists(st.integers(0, 100), min_size=1, max_size=3)
    | hnp.arrays(
        st.sampled_from([np.int32, np.int64, np.float32]),
        st.integers(1, 2),
        elements=st.integers(0, 100),
    ),
    axis=st.sampled_from([None, 1]),
)
@example(
    value=np.array([[1.0, np.nan, 3.0, 2.0, 5.0]]),
    form="median",
    method="linear",
    quantile=0.5,
    percent=50,
    axis=None,
)
@example(
    value=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    form="percentile",
    method="linear",
    quantile=0.3,
    percent=30.0,
    axis=None,
)
@example(
    value=np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float32),
    form="nanmedian",
    method="linear",
    quantile=0.5,
    percent=50,
    axis=1,
)
# A 0-d result for a NaN slice is NumPy's NaN element in the data dtype.
@example(
    value=np.array([[np.nan]], dtype=np.float32),
    form="quantile",
    method="linear",
    quantile=np.float64(0.5),
    percent=50,
    axis=None,
)
# Percentile divides q by 100 before NumPy picks its path, so integer
# percentiles interpolate integer data in float64.
@example(
    value=np.array([[1, 2, 3, 4]]),
    form="percentile",
    method="linear",
    quantile=0.5,
    percent=[25, 50, 75],
    axis=None,
)
@example(
    value=np.array([[1, 2, 3, 4]]),
    form="nanpercentile",
    method="linear",
    quantile=0.5,
    percent=50,
    axis=None,
)
@example(
    value=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    form="percentile",
    method="linear",
    quantile=0.5,
    percent=np.array([25, 75]),
    axis=None,
)
# NumPy's weibull virtual index keeps an integer quantile's dtype.
@example(
    value=np.array([[1, 2, 3, 4]], dtype=np.int32),
    form="quantile",
    method="weibull",
    quantile=np.array([0, 1], dtype=np.int8),
    percent=50,
    axis=None,
)
@example(
    value=np.array([[1, 2, 3, 4]], dtype=np.int32),
    form="quantile",
    method="interpolated_inverted_cdf",
    quantile=np.array([True, False]),
    percent=50,
    axis=None,
)
# NumPy computes a float32 quantile's index as (n - 1) * q in float32.
@example(
    value=np.array([[0.0, 1.0]]),
    form="percentile",
    method="linear",
    quantile=0.0,
    percent=np.array([1.0], dtype=np.float32),
    axis=None,
)
# NumPy interpolates from the nearer observation, so an infinite neighbour
# gives NaN or keeps the infinity.
@example(
    value=np.array([[1.0, np.inf], [-np.inf, 1.0]]),
    form="quantile",
    method="linear",
    quantile=np.array([0.5, 0.75]),
    percent=50,
    axis=1,
)
@example(
    value=np.array([[1.0, np.inf]]),
    form="quantile",
    method="averaged_inverted_cdf",
    quantile=0.75,
    percent=50,
    axis=None,
)
@example(
    value=np.array([[1.0, np.inf]]),
    form="median",
    method="linear",
    quantile=0.5,
    percent=50,
    axis=None,
)
# NumPy's nan forms cast every slice to the first slice's dtype, here the
# all-NaN slice's data dtype.
@example(
    value=np.array([[np.nan, np.nan], [6.6, 1.5]], dtype=np.float32),
    form="nanquantile",
    method="linear",
    quantile=np.array([0.3]),
    percent=50,
    axis=1,
)
# np.median's mean accumulates integers in float64 and float16 in float32.
@example(
    value=np.array([[100, 120]], dtype=np.int8),
    form="median",
    method="linear",
    quantile=0.5,
    percent=50,
    axis=None,
)
@example(
    value=np.array([[65504.0, 65504.0]], dtype=np.float16),
    form="median",
    method="linear",
    quantile=0.5,
    percent=50,
    axis=1,
)
# Only a Python float quantile is weak: a NumPy float64 quantile keeps the fixed
# midpoint and averaged_inverted_cdf weights in float64.
@example(
    value=np.array([[48577.0, 999999.8]], dtype=np.float32),
    form="percentile",
    method="midpoint",
    quantile=0.5,
    percent=np.int64(1),
    axis=None,
)
@example(
    value=np.array([[48577.0, 999999.8]], dtype=np.float32),
    form="quantile",
    method="averaged_inverted_cdf",
    quantile=np.array([0.5]),
    percent=50,
    axis=None,
)
# NumPy's inverted CDF lowers zero cumulative weights to -1, so q=0 skips a
# leading zero-weight observation.
@example(
    value=np.array([[3.0, 2.0]]),
    form="weighted",
    method="inverted_cdf",
    quantile=0,
    percent=50,
    axis=None,
)
def test_order_statistics_match_numpy_values_nans_and_dtypes(
    value: np.ndarray[Any, Any],
    form: str,
    method: str,
    quantile: Any,
    percent: Any,
    axis: int | None,
) -> None:
    function = _order_statistic(form, method, quantile, percent, axis)
    # Integer data is traced through a float64 input cast back to its dtype.
    traced_value = value.astype(np.float64) if value.dtype.kind in "iu" else value
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.asarray(function(value))
        actual = np.asarray(
            ad.jvp(lambda x: function(np.astype(x, value.dtype)))(
                traced_value, tangents=np.ones_like(traced_value)
            )[0]
        )

    assert actual.dtype == expected.dtype
    # The lowering evaluates NumPy's own index and interpolation formulas.
    np.testing.assert_array_equal(actual, expected)


@st.composite
def _order_statistic_case(draw: st.DrawFn) -> tuple[Callable[[Any], Any], np.ndarray[Any, Any]]:
    shape = draw(hnp.array_shapes(min_dims=1, max_dims=3, max_side=4))
    size = math.prod(shape)
    dtype = draw(st.sampled_from((np.float32, np.float64)))
    # Distinct values keep the sort, and so every quantile's derivative, unique.
    order = np.asarray(draw(st.permutations(range(size))))
    value = (order * 0.37 - 1.0).reshape(shape).astype(dtype)
    ndim = len(shape)
    axis = draw(
        st.none()
        | st.integers(-ndim, ndim - 1)
        | st.lists(st.integers(0, ndim - 1), min_size=1, max_size=ndim, unique=True).map(tuple)
    )
    keepdims = draw(st.booleans())
    form = draw(st.sampled_from(_ORDER_STATISTIC_FORMS))
    if form.startswith("nan"):
        # Every reduced slice keeps its minimum, so it has a finite value.
        reduced = None if axis is None else tuple(np.atleast_1d(axis) % ndim)
        minimum = np.min(value, axis=reduced, keepdims=True)
        value[draw(hnp.arrays(np.bool_, shape)) & (value != minimum)] = np.nan
    method = draw(st.sampled_from(_QUANTILE_METHODS))
    quantile = draw(
        st.floats(0.0, 1.0) | st.lists(st.floats(0.0, 1.0), min_size=1, max_size=3).map(np.array)
    )
    options: dict[str, Any] = {"axis": axis, "keepdims": keepdims}
    functions: dict[str, Callable[[Any], Any]] = {
        "median": lambda x: np.median(x, **options),
        "nanmedian": lambda x: np.nanmedian(x, **options),
        "quantile": lambda x: np.quantile(x, quantile, method=method, **options),
        "nanquantile": lambda x: np.nanquantile(x, quantile, method=method, **options),
        "percentile": lambda x: np.percentile(
            x, np.multiply(quantile, 100), method=method, **options
        ),
        "nanpercentile": lambda x: np.nanpercentile(
            x, np.multiply(quantile, 100), method=method, **options
        ),
        "weighted": lambda x: np.quantile(
            x, quantile, method="inverted_cdf", weights=np.ones(x.shape), **options
        ),
    }
    return functions[form], value


@given(case=_order_statistic_case())
@example(case=(lambda x: np.quantile(x, 0.0, method="hazen"), np.array([1.0, 2.0, 4.0, 8.0])))
@example(case=(lambda x: np.quantile(x, 1.0, method="hazen"), np.array([1.0, 2.0, 4.0, 8.0])))
@example(
    case=(
        lambda x: np.quantile(x, 0.0, method="averaged_inverted_cdf"),
        np.array([1.0, 2.0, 4.0, 8.0]),
    )
)
@example(
    case=(
        lambda x: np.quantile(x, 1.0, method="averaged_inverted_cdf"),
        np.array([1.0, 2.0, 4.0, 8.0]),
    )
)
def test_order_statistic_tangents_are_homogeneous_and_translation_equivariant(
    case: tuple[Callable[[Any], Any], np.ndarray[Any, Any]],
) -> None:
    function, value = case
    tolerance = 2e-6 if value.dtype == np.float32 else 1e-12
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        # f((1 + t) x) = (1 + t) f(x): sort order, NaN positions and weights stay fixed.
        primal, scaled = ad.jvp(function)(value, tangents=np.nan_to_num(value))
        # f(x + t) = f(x) + t: every method's interpolation weights sum to one.
        _, shifted = ad.jvp(function)(value, tangents=np.ones_like(value))

    np.testing.assert_allclose(scaled, primal, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(shifted, np.ones_like(primal), rtol=tolerance, atol=tolerance)


@given(
    value=hnp.arrays(
        np.float64, hnp.array_shapes(max_dims=2, max_side=4), elements=st.floats(-1e3, 1e3)
    ),
    quantile=hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=0, max_dims=1, min_side=0, max_side=3),
        elements=st.floats(0.0, 1.0),
    ),
    method=st.sampled_from(("closest_observation", "higher", "inverted_cdf", "lower", "nearest")),
)
@example(value=np.array([1.0, 2.0, 4.0, 8.0]), quantile=np.array(0.6), method="lower")
@example(value=np.array([[0.0, 1.0], [2.0, 3.0]]), quantile=np.array([]), method="nearest")
def test_discrete_quantile_methods_have_zero_coordinate_derivatives(
    value: np.ndarray[Any, Any], quantile: np.ndarray[Any, Any], method: str
) -> None:
    def order_statistic(q: Any) -> Any:
        return np.quantile(value, q, axis=-1, method=method)

    primal, tangent = ad.jvp(order_statistic)(quantile, tangents=np.ones_like(quantile))

    np.testing.assert_array_equal(primal, order_statistic(quantile))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


# Each quantile, 0.35 or 0.5 * 0.35 + 0.2, lies strictly between two knots of
# every method on six observations, so a central difference measures the derivative.
_QUANTILE_PROMOTIONS: dict[str, Callable[[Any], Any]] = {
    "weak": lambda s: s,
    "weak-derived": lambda s: s * 0.5 + 0.2,
    "strong": lambda s: np.multiply(s, np.ones(1)),
}


@pytest.mark.parametrize("promotion", sorted(_QUANTILE_PROMOTIONS))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["hazen", "linear", "midpoint", "nearest", "weibull"])
@pytest.mark.parametrize("form", ["nanpercentile", "nanquantile", "percentile", "quantile"])
def test_a_selected_python_scalar_quantile_promotes_as_eager_numpy(
    form: str, method: str, dtype: type[np.floating[Any]], promotion: str
) -> None:
    # A weak quantile interpolates float32 data in float32, as NumPy's
    # float(gamma) does, and a strong one promotes it to float64. The selected
    # scalar used to become a strong array, which returned float64.
    value = np.array([0.5, -1.0, 2.25, 1.0, 4.0, 3.0], dtype)
    if form.startswith("nan"):
        value = np.insert(value, 2, np.nan)
    function = getattr(np, form)
    scale = 100 if form.endswith("percentile") else 1
    lift = _QUANTILE_PROMOTIONS[promotion]

    def order_statistic(s: Any, data: Any = value) -> Any:
        return function(data, lift(s) * scale, method=method)

    expected = order_statistic(0.35)
    primal, tangent = ad.jvp(order_statistic)(0.35, tangents=1.0)
    gradient = ad.grad(lambda s: np.sum(order_statistic(s)))(0.35)
    step = 1e-6
    wide = value.astype(np.float64)
    slope = (order_statistic(0.35 + step, wide) - order_statistic(0.35 - step, wide)) / (2 * step)

    assert np.asarray(primal).dtype == np.asarray(tangent).dtype == np.asarray(expected).dtype
    np.testing.assert_array_equal(primal, expected)
    np.testing.assert_allclose(tangent, slope, rtol=1e-6, atol=1e-6)
    assert type(gradient) is float
    np.testing.assert_allclose(gradient, np.sum(tangent), rtol=1e-6)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda v: np.median(v**3), id="median"),
        pytest.param(lambda v: np.quantile(v**3, 0.3), id="quantile"),
        pytest.param(lambda v: np.percentile(v**3, 30.0), id="percentile"),
        pytest.param(lambda v: np.nanmedian(v**3), id="nanmedian"),
        pytest.param(lambda v: np.nanquantile(v**3, 0.3), id="nanquantile"),
        pytest.param(
            lambda v: np.quantile(v**3, 0.3, method="inverted_cdf", weights=np.ones(4)),
            id="weighted",
        ),
        pytest.param(lambda v: np.quantile(v[1:] ** 3, v[0] ** 2), id="traced-quantile"),
    ],
)
def test_order_statistics_differentiate_twice(operation: Callable[[Any], Any]) -> None:
    value = np.array([0.5, 5.0, 3.0, 2.0])
    step = 1e-6
    gradient = ad.grad(operation)
    expected = np.stack(
        [
            (gradient(value + step * direction) - gradient(value - step * direction)) / (2 * step)
            for direction in np.eye(value.size)
        ]
    )

    np.testing.assert_allclose(ad.hessian(operation)(value), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda v: np.median(v, axis=1), id="median"),
        pytest.param(lambda v: np.quantile(v, 0.3, axis=0), id="quantile"),
        pytest.param(lambda v: np.percentile(v, [10, 75], axis=1, keepdims=True), id="percentile"),
    ],
)
def test_staged_derivatives_propagate_order_statistic_nans(
    operation: Callable[[Any], Any],
) -> None:
    value = np.array([[0.3, -1.2, 2.0, 0.7], [1.1, 0.4, -0.5, 0.9], [-0.2, 1.5, 0.8, -1.0]])
    with_nan = value.copy()
    with_nan[1, 2] = np.nan

    def value_and_tangent(x: Any) -> Any:
        return ad.jvp(operation)(x, tangents=np.ones_like(x))

    # The dynamic derivative runs over abstract payloads while it is staged.
    program = ad.stage(value_and_tangent, value)
    for data in (value, with_nan):
        primal, tangent = program(data)
        np.testing.assert_array_equal(primal, operation(data))
        np.testing.assert_array_equal(tangent, value_and_tangent(data)[1])


@pytest.mark.parametrize("size", [6, 7])
@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(np.median, id="median"),
        pytest.param(lambda v: np.median(v.astype(np.int64)), id="integer-median"),
        pytest.param(lambda v: np.quantile(v, 0.3), id="quantile"),
        pytest.param(lambda v: np.percentile(v, 30.0, axis=0), id="percentile"),
        pytest.param(np.nanmedian, id="nanmedian"),
        pytest.param(lambda v: np.nanquantile(v, np.array(0.3)), id="nanquantile"),
    ],
)
def test_scalar_order_statistics_return_numpy_scalars(
    operation: Callable[[Any], Any], size: int
) -> None:
    value = np.linspace(-1.0, 2.0, size)

    primal, tangent = ad.jvp(operation)(value, tangents=np.ones_like(value))

    assert type(primal) is type(tangent) is type(operation(value))


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
    ("quantile", "weights"),
    [
        # NumPy compares the cumulative weights in a float quantile's dtype, where
        # float32(0.4) reaches the second of five equal weights.
        pytest.param(np.float32(0.4), np.ones(5), id="float32-quantile"),
        pytest.param(np.array([0.4, 0.8], np.float32), np.ones(5), id="float32-quantiles"),
        # It accumulates in float64, where 7 / 10 of float32 weights reaches 0.7.
        pytest.param(0.7, np.array([7.0, 3.0], np.float32), id="float32-weights"),
    ],
)
def test_weighted_inverted_cdf_builds_numpys_cumulative_weights(
    quantile: Any,
    weights: np.ndarray[Any, Any],
) -> None:
    value = np.arange(float(weights.size))

    def weighted(x: Any) -> Any:
        return np.quantile(x, quantile, method="inverted_cdf", weights=weights)

    expected = weighted(value)
    jacobian = ad.jacobian(weighted)(value)

    np.testing.assert_array_equal(ad.jvp(weighted)(value, tangents=value)[0], expected)
    # The derivative selects the same observation as NumPy.
    np.testing.assert_array_equal(jacobian, np.equal.outer(expected, value))
