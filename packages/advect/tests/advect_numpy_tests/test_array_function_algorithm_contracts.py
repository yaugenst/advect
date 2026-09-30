"""Public contracts for NumPy algorithms with composite trace lowerings."""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect.core._pytree import tree_flatten
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_tree_close,
    seeded_like,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_apply_along_axis_supports_array_results_and_callback_arguments() -> None:
    values = np.arange(24.0).reshape(2, 3, 4)
    direction = np.linspace(-0.5, 0.5, values.size).reshape(values.shape)

    def summarize(row: Any, scale: float, *, offset: float) -> Any:
        return np.stack((np.sum(row) + offset, scale * np.mean(row)))

    def apply(array: Any) -> Any:
        return np.apply_along_axis(summarize, -2, array, 2.0, offset=1.5)

    primal, tangent = ad.jvp(apply)(values, tangents=direction)

    np.testing.assert_allclose(primal, apply(values))
    np.testing.assert_allclose(
        tangent,
        np.apply_along_axis(summarize, -2, direction, 2.0, offset=0.0),
    )


def test_apply_along_axis_lifts_constant_callback_results() -> None:
    values = np.arange(6.0).reshape(2, 3)

    primal, tangent = ad.jvp(
        lambda array: np.apply_along_axis(lambda _row: np.array([1.0, 2.0]), 1, array)
    )(values, tangents=np.ones_like(values))

    np.testing.assert_array_equal(primal, np.array([[1.0, 2.0], [1.0, 2.0]]))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


def test_sliding_window_view_supports_multiple_and_negative_axes() -> None:
    values = np.arange(40.0).reshape(2, 4, 5)
    direction = np.linspace(-1.0, 1.0, values.size).reshape(values.shape)

    def windows(array: Any) -> Any:
        return np.lib.stride_tricks.sliding_window_view(
            array,
            (2, 3),
            axis=(0, -1),
        )

    primal, tangent = ad.jvp(windows)(values, tangents=direction)

    np.testing.assert_array_equal(primal, windows(values))
    np.testing.assert_array_equal(tangent, windows(direction))


def test_bincount_accepts_traced_indices_without_weights() -> None:
    indices = np.array([0, 2, 2, 4], dtype=np.int64)

    primal, tangent = ad.jvp(lambda current: np.bincount(current, minlength=6))(
        indices,
        tangents=np.zeros_like(indices),
    )

    np.testing.assert_array_equal(primal, np.bincount(indices, minlength=6))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


@pytest.mark.parametrize(
    "operation",
    [
        lambda weights: np.bincount(np.array([0, 2, 2, 1]), weights=weights, minlength=4),
        lambda weights: np.histogram(np.array([0.1, 0.4, 0.45, 0.9]), bins=3, weights=weights)[0],
        lambda weights: np.histogram2d(
            np.array([0.1, 0.4, 0.45, 0.9]),
            np.array([0.9, 0.2, 0.3, 0.1]),
            bins=2,
            weights=weights,
        )[0],
    ],
    ids=["bincount", "histogram", "histogram2d"],
)
def test_weighted_counts_have_second_derivatives(operation: Any) -> None:
    weights = np.array([1.0, 2.0, 0.5, 1.5])

    def loss(current: Any) -> Any:
        return np.sum(operation(current) ** 2)

    # Each weight lands in one bin, so d2/dw_i dw_j is 2 when i and j share it.
    bins = np.array([np.flatnonzero(operation(unit))[0] for unit in np.eye(weights.size)])
    expected = 2.0 * (bins[:, None] == bins[None, :])

    np.testing.assert_allclose(ad.hessian(loss)(weights), expected)


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        (np.array([[0, 1]], dtype=np.int64), "one-dimensional integer array"),
        (np.array([0.0, 1.0]), "one-dimensional integer array"),
        (np.array([0, -1], dtype=np.int64), "must be non-negative"),
    ],
)
def test_bincount_validates_indices(indices: np.ndarray[Any, Any], match: str) -> None:
    with pytest.raises(ad.TracingError, match=match):
        ad.jvp(np.bincount)(indices, tangents=np.zeros_like(indices))


def test_bincount_validates_minlength_and_weight_shape() -> None:
    indices = np.array([0, 1, 1], dtype=np.int64)

    with pytest.raises(ad.TracingError, match="minlength must be a static integer"):
        ad.jvp(lambda weights: np.bincount(indices, weights=weights, minlength=2.5))(
            np.ones(3),
            tangents=np.ones(3),
        )

    with pytest.raises(ValueError, match="must not be negative"):
        ad.jvp(lambda weights: np.bincount(indices, weights=weights, minlength=-1))(
            np.ones(3),
            tangents=np.ones(3),
        )

    with pytest.raises((ValueError, ad.TracingError), match="weights"):
        ad.jvp(lambda weights: np.bincount(indices, weights=weights))(
            np.ones(2),
            tangents=np.ones(2),
        )


def test_insert_requires_static_indices() -> None:
    source = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match="obj= must be static"):
        ad.jvp(lambda obj: np.insert(source, obj, 1.0))(
            np.array(1, dtype=np.int64),
            tangents=np.array(0, dtype=np.int64),
        )


_BOUNDS = st.floats(min_value=-8.0, max_value=8.0, width=32)


@given(
    samples=hnp.arrays(
        st.sampled_from([np.float32, np.float64]),
        st.integers(min_value=0, max_value=6),
        elements=_BOUNDS,
    ),
    bins=st.integers(min_value=1, max_value=4),
    histogram_range=st.none()
    | st.tuples(_BOUNDS, st.sampled_from([0.0, 0.5, 4.0])).map(
        lambda bounds: (bounds[0], bounds[0] + bounds[1])
    ),
    density=st.booleans(),
    weighted=st.booleans(),
    traced_range=st.booleans(),
)
@example(
    samples=np.array([0.2, 1.0, 1.4]),
    bins=3,
    histogram_range=(1.0, 1.0),
    density=True,
    weighted=False,
    traced_range=False,
)
@example(
    samples=np.array([0.2, 1.0, 1.4], dtype=np.float32),
    bins=2,
    histogram_range=(0.0, 2.0),
    density=True,
    weighted=True,
    traced_range=False,
)
# NumPy rejects a subnormal range as too narrow for two finite bins.
@example(
    samples=np.array([1e-45, 0.0], dtype=np.float32),
    bins=2,
    histogram_range=None,
    density=False,
    weighted=False,
    traced_range=False,
)
# Traced equal bounds are widened by 0.5 like NumPy's concrete ones.
@example(
    samples=np.array([0.2, 1.0, 1.4], dtype=np.float32),
    bins=3,
    histogram_range=(1.0, 1.0),
    density=False,
    weighted=False,
    traced_range=True,
)
def test_histogram_edges_and_counts_match_numpy(
    samples: np.ndarray[Any, Any],
    bins: int,
    histogram_range: tuple[float, float] | None,
    *,
    density: bool,
    weighted: bool,
    traced_range: bool,
) -> None:
    weights = np.linspace(0.5, 1.5, samples.size, dtype=samples.dtype) if weighted else None

    def histogram(values: Any, *traced: Any) -> Any:
        weight, *bounds = traced if weighted else (None, *traced)
        edge_range = tuple(bounds) or histogram_range
        return (
            np.histogram(values, bins, range=edge_range, density=density, weights=weight),
            np.histogram_bin_edges(values, bins, range=edge_range),
        )

    # Traced bounds are 0-d float64 arrays, which NumPy promotes strongly.
    bounds = (
        tuple(np.asarray(bound) for bound in histogram_range)
        if histogram_range is not None and traced_range
        else ()
    )
    arguments = (samples, *((weights,) if weighted else ()), *bounds)
    directions = tuple(
        seeded_like(argument, str(index)) for index, argument in enumerate(arguments)
    )

    def traced() -> Any:
        return ad.jvp(histogram, argnums=tuple(range(len(arguments))))(
            *arguments, tangents=directions
        )

    with np.errstate(all="ignore"):
        try:
            expected = histogram(*arguments)
        except ValueError as error:
            # NumPy rejects a range too narrow for finite bins; tracing must agree.
            with pytest.raises(ValueError, match=re.escape(str(error))):
                traced()
            return
        actual, tangent = traced()
        if weighted and not density:
            # Counts are locally constant in the samples and bounds and linear in the weights.
            linear = histogram(samples, directions[1], *bounds)[0][0]
            assert_tree_close(tangent[0][0], linear, rtol=1e-6, atol=1e-6)
    for actual_leaf, expected_leaf in zip(
        tree_flatten(actual)[0], tree_flatten(expected)[0], strict=True
    ):
        assert actual_leaf.dtype == expected_leaf.dtype
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, equal_nan=True)


def test_i0_accepts_integer_arrays() -> None:
    integers = np.array([0, 1, 2], dtype=np.int64)
    primal, tangent = ad.jvp(np.i0)(integers, tangents=np.zeros_like(integers))
    np.testing.assert_allclose(primal, np.i0(integers))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


def test_arange_differentiates_a_traced_start_and_step() -> None:
    def sequence(start: Any, step: Any) -> Any:
        return np.arange(start, 5.0, step, like=start)

    primal, tangent = ad.jvp(sequence, argnums=(0, 1))(
        np.array(1.0),
        np.array(1.0),
        tangents=(np.array(0.2), np.array(-0.1)),
    )

    np.testing.assert_array_equal(primal, np.arange(1.0, 5.0, 1.0))
    np.testing.assert_allclose(tangent, 0.2 - 0.1 * np.arange(4))


def test_block_supports_nested_mixed_arrays() -> None:
    values = np.array([[1.0, 2.0]])

    def assemble(array: Any) -> Any:
        return np.block(
            [
                [array, np.zeros_like(values)],
                [np.ones_like(values), 2.0 * array],
            ]
        )

    primal, tangent = ad.jvp(assemble)(values, tangents=np.ones_like(values))
    np.testing.assert_array_equal(primal, assemble(values))
    np.testing.assert_array_equal(
        tangent,
        np.block(
            [
                [np.ones_like(values), np.zeros_like(values)],
                [np.zeros_like(values), 2.0 * np.ones_like(values)],
            ]
        ),
    )


def test_logspace_and_geomspace_accept_axis_and_dtype_controls() -> None:
    start = np.array([0.0, 1.0])
    stop = np.array([1.0, 2.0])
    direction = np.array([0.1, -0.2])

    def assert_directional(function: Any, value: np.ndarray[Any, Any]) -> None:
        assert function(value).dtype == np.dtype(np.float32)
        assert_jvp_matches_central_difference(
            function, (value,), (direction,), rtol=1e-3, atol=1e-4, step=1e-3
        )

    assert_directional(
        lambda value: np.logspace(value, stop, num=4, axis=-1, dtype=np.float32),
        start,
    )

    positive_start = start + 1.0
    positive_stop = stop + 2.0
    assert_directional(
        lambda value: np.geomspace(
            value,
            positive_stop,
            num=4,
            axis=-1,
            dtype=np.float32,
        ),
        positive_start,
    )


@pytest.mark.parametrize("num", [2, 3])
@pytest.mark.parametrize("axis", [0, -1])
def test_logspace_places_an_array_base_on_the_sample_axis(num: int, axis: int) -> None:
    start = np.array([0.0, 1.0])
    stop = np.array([1.0, 2.0])
    base = np.array([2.0, 10.0])

    def logspace(value: Any, bases: Any) -> Any:
        return np.logspace(value, stop, num=num, base=bases, axis=axis)

    assert_jvp_matches_central_difference(
        logspace,
        (start, base),
        (np.array([0.1, -0.2]), np.array([0.3, 0.05])),
        rtol=1e-5,
        atol=1e-6,
    )


def test_union1d_selects_nan_and_finite_tangents_from_the_inputs() -> None:
    left = np.array([np.nan, 2.0])
    right = np.array([1.0, np.nan])
    direction = np.array([0.3, -0.2])

    primal, tangent = ad.jvp(lambda values: np.union1d(values, right))(
        left,
        tangents=direction,
    )

    np.testing.assert_array_equal(primal, np.union1d(left, right))
    np.testing.assert_array_equal(tangent, np.array([0.0, -0.2, 0.3]))


def test_unique_array_api_results_preserve_named_fields_and_selected_tangents() -> None:
    value = np.array([2.0, 1.0, 2.0, 3.0])
    direction = np.array([10.0, 20.0, 30.0, 40.0])

    primal, tangent = ad.jvp(np.unique_all)(value, tangents=direction)

    assert type(primal) is type(np.unique_all(value))
    assert type(tangent) is type(primal)
    np.testing.assert_array_equal(primal.values, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(tangent.values, np.array([20.0, 10.0, 40.0]))
    np.testing.assert_array_equal(tangent.indices, np.zeros(3, dtype=np.int64))
    np.testing.assert_array_equal(tangent.inverse_indices, np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(tangent.counts, np.zeros(3, dtype=np.int64))


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

    assert_tree_close(primal, expected, rtol=0.0)
    np.testing.assert_array_equal(tangent[0], direction[[0, 2]])
    for discrete in tangent[1:]:
        np.testing.assert_array_equal(discrete, np.zeros_like(discrete))


@pytest.mark.parametrize(
    "operation",
    [np.unique, np.unique_values, lambda x: np.unique_counts(x).values],
    ids=("unique", "unique_values", "unique_counts"),
)
def test_unique_values_have_second_derivatives(operation: Callable[..., Any]) -> None:
    value = np.array([1.0, 2.0, 0.5, 2.0, 0.3])

    hessian = ad.hessian(lambda x: np.sum(operation(x) ** 3))(value)

    # Each distinct value is gathered from its first occurrence.
    first = np.array([True, True, True, False, True])
    np.testing.assert_allclose(hessian, np.diag(6.0 * value * first))


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


def test_trim_zeros_accepts_a_negative_axis() -> None:
    values = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    direction = np.arange(values.size, dtype=float).reshape(values.shape)

    primal, tangent = ad.jvp(lambda array: np.trim_zeros(array, axis=-1))(
        values,
        tangents=direction,
    )

    np.testing.assert_array_equal(primal, np.trim_zeros(values, axis=-1))
    np.testing.assert_array_equal(tangent, direction[:, 1:2])


def test_trim_zeros_removes_an_all_zero_array() -> None:
    values = np.zeros(4)

    primal, tangent = ad.jvp(np.trim_zeros)(values, tangents=np.ones_like(values))

    assert primal.size == tangent.size == 0


_TRIM_MATRIX = np.array(
    [[0.0, 0.0, 2.0, 3.0, 0.0], [0.0, 1.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
)


@pytest.mark.parametrize(
    ("value", "kwargs"),
    [
        (np.zeros(3), {"trim": "f"}),
        (np.zeros(3), {"trim": "b"}),
        (np.zeros((2, 3)), {"trim": "b", "axis": 1}),
        (_TRIM_MATRIX, {}),
        (_TRIM_MATRIX, {"axis": (0, 1)}),
        (_TRIM_MATRIX, {"trim": "f", "axis": 0}),
    ],
    ids=("zeros-front", "zeros-back", "zero-matrix-axis", "matrix", "axis-tuple", "matrix-front"),
)
def test_trim_zeros_trims_numpy_bounding_boxes(
    value: np.ndarray[Any, Any],
    kwargs: dict[str, Any],
) -> None:
    direction = np.arange(value.size, dtype=float).reshape(value.shape)

    primal, tangent = ad.jvp(lambda array: np.trim_zeros(array, **kwargs))(
        value,
        tangents=direction,
    )

    expected = np.trim_zeros(value, **kwargs)
    np.testing.assert_array_equal(primal, expected)
    assert tangent.shape == expected.shape


def test_arange_accepts_the_cpu_device_keyword() -> None:
    stop = np.array(5.0)

    primal, tangent = ad.jvp(
        lambda endpoint: np.arange(
            1.0,
            endpoint,
            dtype=np.float32,
            device="cpu",
            like=endpoint,
        )
    )(stop, tangents=np.array(0.0))

    np.testing.assert_array_equal(primal, np.arange(1.0, stop, dtype=np.float32, device="cpu"))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


def test_apply_over_axes_lifts_a_constant_rank_reducing_result() -> None:
    values = np.arange(6.0).reshape(2, 3)

    def reduce_to_constant(array: Any) -> Any:
        return np.apply_over_axes(
            lambda item, _axis: np.ones(item.shape[1:]),
            array,
            (0,),
        )

    primal, tangent = ad.jvp(reduce_to_constant)(values, tangents=np.ones_like(values))

    np.testing.assert_array_equal(primal, reduce_to_constant(values))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


_X = np.array([0.1, 0.4, 0.8, 0.2])
_Y = np.array([0.2, 0.7, 0.9, 0.6])
_UNIT_SQUARE = ((0.0, 1.0), (0.0, 1.0))
_EDGE_ROWS = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 1.0]])
_SHIFT = np.array([0.2, -0.1, 0.3, 0.4])


# Every sample stays well inside its bin, so counts are locally constant and the
# edges are affine in the range and in the explicit edges.
@pytest.mark.parametrize(
    ("function", "primals", "directions"),
    [
        pytest.param(
            lambda x: np.histogram(x, bins=4, range=(0.0, 1.0), density=True),
            (_X,),
            (_SHIFT,),
            id="density-range",
        ),
        # Equal samples widen to +-0.5 around them, so only a uniform shift is smooth.
        pytest.param(
            lambda x: np.histogram(x, bins=2),
            (np.full(3, 2.0),),
            (np.full(3, 0.25),),
            id="equal-samples",
        ),
        pytest.param(
            lambda x, low, high: np.histogram(x, bins=3, range=(low, high)),
            (_X[:3], np.array(0.0), np.array(1.0)),
            (_SHIFT[:3], np.array(0.2), np.array(-0.1)),
            id="traced-range",
        ),
        pytest.param(
            lambda edges: np.histogram(_X[:3], bins=edges),
            (np.array([0.0, 0.25, 0.75, 1.0]),),
            (np.array([0.0, 0.05, -0.05, 0.0]),),
            id="traced-edges",
        ),
        pytest.param(
            lambda x: np.histogram(x, bins=2), (np.empty(0),), (np.empty(0),), id="empty-samples"
        ),
        *(
            pytest.param(
                lambda x, y, bins=bins: np.histogram2d(
                    x, y, bins=bins, range=_UNIT_SQUARE, density=True
                ),
                (_X, _Y),
                (_SHIFT, -_SHIFT),
                id=f"histogram2d-{name}",
            )
            for name, bins in (("scalar-bins", 3), ("shared-edges", np.array([0.0, 0.5, 1.0])))
        ),
        pytest.param(
            lambda x: np.histogram2d(x, _Y, bins=_EDGE_ROWS),
            (_X,),
            (_SHIFT,),
            id="histogram2d-static-edge-rows",
        ),
        pytest.param(
            lambda edges: np.histogram2d(_X, _Y, bins=edges),
            (_EDGE_ROWS,),
            (np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]]),),
            id="histogram2d-traced-edge-rows",
        ),
        pytest.param(
            lambda x, y: np.histogramdd((x, y), bins=2, range=_UNIT_SQUARE, density=True),
            (_X, _Y),
            (_SHIFT, -_SHIFT),
            id="histogramdd-coordinates",
        ),
        pytest.param(
            lambda edges: np.histogramdd(_X[:3, None], bins=(edges,)),
            (np.array([0.0, 0.5, 1.0]),),
            (np.array([0.0, 0.1, 0.0]),),
            id="histogramdd-traced-edge-sequence",
        ),
    ],
)
def test_histogram_family_matches_directional_differences(
    function: Callable[..., Any],
    primals: tuple[np.ndarray[Any, Any], ...],
    directions: tuple[np.ndarray[Any, Any], ...],
) -> None:
    assert_jvp_matches_central_difference(function, primals, directions)


def _inconsistent_rows(array: Any) -> Any:
    lengths = iter((1, 2))
    return np.apply_along_axis(lambda row: row[: next(lengths)], 1, array)


_MATRIX = np.arange(6.0).reshape(2, 3)


@pytest.mark.parametrize(
    ("operation", "value", "error", "match"),
    [
        pytest.param(
            lambda x: np.apply_along_axis(np.sum, 1, x),
            np.empty((0, 3)),
            ad.TracingError,
            "cannot iterate an empty batch",
            id="apply-along-axis-empty",
        ),
        pytest.param(
            lambda x: np.apply_along_axis(None, 1, x),
            _MATRIX,
            ad.TracingError,
            "func1d must be callable",
            id="apply-along-axis-callable",
        ),
        pytest.param(
            _inconsistent_rows,
            _MATRIX,
            ad.TracingError,
            "returned inconsistent shapes",
            id="apply-along-axis-shapes",
        ),
        pytest.param(
            lambda x: np.apply_over_axes(lambda item, _axis: np.ravel(item), x, (0,)),
            np.arange(24.0).reshape(2, 3, 4),
            ad.TracingError,
            "preserve rank or remove only its axis",
            id="apply-over-axes-rank",
        ),
        pytest.param(
            lambda x: np.apply_over_axes(None, x, (0,)),
            _MATRIX,
            ad.TracingError,
            "func must be callable",
            id="apply-over-axes-callable",
        ),
        *(
            pytest.param(
                lambda x, window=window, axis=axis: np.lib.stride_tricks.sliding_window_view(
                    x, window, axis=axis
                ),
                np.arange(8.0).reshape(2, 4),
                ad.TracingError,
                match,
                id=f"sliding-window-{name}",
            )
            for name, window, axis, match in (
                ("lengths", (2, 2), 1, "matching lengths"),
                ("zero", 0, 1, "must be positive"),
                ("oversized", 5, 1, "exceeds an input dimension"),
            )
        ),
        pytest.param(
            lambda x: np.lib.stride_tricks.sliding_window_view(x, 2, writeable=True),
            np.arange(6.0),
            ad.TracingError,
            "non-writeable base-independent",
            id="sliding-window-writeable",
        ),
        pytest.param(
            lambda x: np.insert(x, 0, 1.0, axis=3),
            _MATRIX,
            ad.TracingError,
            "axis 3 is out of bounds",
            id="insert-axis",
        ),
        pytest.param(
            lambda x: np.insert(x, 10, 1.0),
            np.arange(3.0),
            IndexError,
            "index 10 is out of bounds for axis 0 with size 3",
            id="insert-index",
        ),
        pytest.param(
            lambda x: np.trim_zeros(x, trim="fx"),
            np.array([0.0, 1.0, 0.0]),
            ad.TracingError,
            "must contain only",
            id="trim-zeros-selector",
        ),
        pytest.param(
            lambda x: np.arange(x, None, like=x),
            np.array(1.0),
            ad.TracingError,
            "requires a stop value",
            id="arange-none-stop",
        ),
        *(
            pytest.param(
                lambda x, operation=operation: operation(x, bins="auto"),
                _X,
                ad.TracingError,
                r"string .*estimators are data-dependent",
                id=f"{operation.__name__}-string-bins",
            )
            for operation in (np.histogram, np.histogram_bin_edges)
        ),
        pytest.param(np.histogramdd, _X, ad.TracingError, r"shape \(N, D\)", id="histogramdd-1d"),
        pytest.param(
            lambda x: np.histogramdd((x, np.array([0.1, 0.2]))),
            _X,
            ad.TracingError,
            "columns must have equal lengths",
            id="histogramdd-ragged",
        ),
        pytest.param(
            lambda weights: np.histogramdd((), weights=weights),
            np.empty(0),
            ad.TracingError,
            "at least one sample dimension",
            id="histogramdd-empty",
        ),
        pytest.param(
            np.i0,
            np.array([1.0 + 0.5j]),
            TypeError,
            "does not support complex",
            id="i0-complex",
        ),
        pytest.param(
            lambda x: np.block([x, []]),
            np.array([[1.0, 2.0]]),
            ad.TracingError,
            "does not accept empty lists",
            id="block-empty-list",
        ),
        pytest.param(
            lambda x: np.block([x, [x]]),
            np.array([[1.0, 2.0]]),
            ad.TracingError,
            "list depths must match",
            id="block-depths",
        ),
        pytest.param(
            lambda x: np.geomspace(x * 0, 1.0, 3),
            np.array(2.0),
            ValueError,
            "Geometric sequence cannot include zero",
            id="geomspace-zero",
        ),
        pytest.param(
            lambda x: np.unique(x, axis=0.5),
            _MATRIX,
            ad.TracingError,
            "axis must be an integer or None",
            id="unique-axis",
        ),
        pytest.param(
            lambda x: np.convolve(x, np.ones(2), mode="circular"),
            np.arange(4.0),
            ad.TracingError,
            "mode must be full, same, or valid",
            id="convolve-mode",
        ),
    ],
)
def test_algorithm_forms_reject_invalid_public_calls(
    operation: Callable[[Any], Any],
    value: np.ndarray[Any, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.jvp(operation)(value, tangents=np.ones_like(value))
