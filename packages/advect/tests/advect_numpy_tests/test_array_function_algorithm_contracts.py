"""Public contracts for NumPy algorithms with composite trace lowerings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad


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


def test_apply_along_axis_rejects_an_empty_iteration_batch() -> None:
    values = np.empty((0, 3))

    with pytest.raises(ad.TracingError, match="cannot iterate an empty batch"):
        ad.jvp(lambda array: np.apply_along_axis(np.sum, 1, array))(
            values,
            tangents=np.empty_like(values),
        )


def test_apply_along_axis_requires_a_callable() -> None:
    values = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match="func1d must be callable"):
        ad.jvp(lambda array: np.apply_along_axis(None, 1, array))(
            values,
            tangents=np.ones_like(values),
        )


def test_apply_over_axes_rejects_a_callback_with_an_invalid_rank() -> None:
    values = np.arange(24.0).reshape(2, 3, 4)

    with pytest.raises(ad.TracingError, match="preserve rank or remove only its axis"):
        ad.jvp(lambda array: np.apply_over_axes(lambda item, _axis: np.ravel(item), array, (0,)))(
            values,
            tangents=np.ones_like(values),
        )


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


@pytest.mark.parametrize(
    ("window_shape", "axis", "match"),
    [
        ((2, 2), 1, "matching lengths"),
        (0, 1, "must be positive"),
        (5, 1, "exceeds an input dimension"),
    ],
)
def test_sliding_window_view_validates_window_shape(
    window_shape: object,
    axis: object,
    match: str,
) -> None:
    values = np.arange(8.0).reshape(2, 4)

    with pytest.raises(ad.TracingError, match=match):
        ad.jvp(
            lambda array: np.lib.stride_tricks.sliding_window_view(
                array,
                window_shape,
                axis=axis,
            )
        )(values, tangents=np.ones_like(values))


def test_sliding_window_view_rejects_a_writeable_traced_result() -> None:
    values = np.arange(6.0)

    with pytest.raises(ad.TracingError, match="non-writeable base-independent"):
        ad.jvp(
            lambda array: np.lib.stride_tricks.sliding_window_view(
                array,
                2,
                writeable=True,
            )
        )(values, tangents=np.ones_like(values))


def test_bincount_differentiates_weights_and_honors_minlength() -> None:
    indices = np.array([0, 2, 2, 4], dtype=np.int64)
    weights = np.array([1.0, 2.0, 3.0, 0.5])
    direction = np.array([0.2, -0.1, 0.3, 0.4])

    primal, tangent = ad.jvp(lambda current: np.bincount(indices, weights=current, minlength=7))(
        weights, tangents=direction
    )

    np.testing.assert_array_equal(primal, np.bincount(indices, weights=weights, minlength=7))
    np.testing.assert_array_equal(
        tangent,
        np.bincount(indices, weights=direction, minlength=7),
    )


def test_bincount_accepts_traced_indices_without_weights() -> None:
    indices = np.array([0, 2, 2, 4], dtype=np.int64)

    primal, tangent = ad.jvp(lambda current: np.bincount(current, minlength=6))(
        indices,
        tangents=np.zeros_like(indices),
    )

    np.testing.assert_array_equal(primal, np.bincount(indices, minlength=6))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


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


def test_insert_supports_an_axis_and_traced_values() -> None:
    source = np.arange(6.0).reshape(2, 3)
    inserted = np.array([10.0, 20.0])
    source_direction = np.linspace(-0.3, 0.2, source.size).reshape(source.shape)
    inserted_direction = np.array([0.4, -0.2])

    primal, tangent = ad.jvp(
        lambda array, values: np.insert(array, 1, values, axis=1),
        argnums=(0, 1),
    )(
        source,
        inserted,
        tangents=(source_direction, inserted_direction),
    )

    np.testing.assert_array_equal(primal, np.insert(source, 1, inserted, axis=1))
    np.testing.assert_array_equal(
        tangent,
        np.insert(source_direction, 1, inserted_direction, axis=1),
    )


def test_insert_handles_empty_sources_and_empty_insertions() -> None:
    inserted = np.array([1.0, 2.0])
    primal, tangent = ad.jvp(lambda values: np.insert(np.empty(0), 0, values))(
        inserted,
        tangents=np.array([0.2, -0.1]),
    )
    np.testing.assert_array_equal(primal, inserted)
    np.testing.assert_array_equal(tangent, np.array([0.2, -0.1]))

    source = np.array([1.0, 2.0, 3.0])
    primal, tangent = ad.jvp(lambda array: np.insert(array, [], []))(
        source,
        tangents=np.array([0.3, -0.2, 0.1]),
    )
    np.testing.assert_array_equal(primal, source)
    np.testing.assert_array_equal(tangent, np.array([0.3, -0.2, 0.1]))


def test_insert_requires_static_indices_and_a_valid_axis() -> None:
    source = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match="obj= must be static"):
        ad.jvp(lambda obj: np.insert(source, obj, 1.0))(
            np.array(1, dtype=np.int64),
            tangents=np.array(0, dtype=np.int64),
        )

    with pytest.raises(ad.TracingError, match="axis 3 is out of bounds"):
        ad.jvp(lambda array: np.insert(array, 0, 1.0, axis=3))(
            source,
            tangents=np.ones_like(source),
        )


def test_histogram_supports_density_with_an_explicit_range() -> None:
    samples = np.array([0.1, 0.4, 0.8, 0.2])
    direction = np.array([0.2, -0.1, 0.3, 0.4])

    def histogram(values: Any) -> Any:
        return np.histogram(values, bins=4, range=(0.0, 1.0), density=True)

    primal, tangent = ad.jvp(histogram)(samples, tangents=direction)
    expected = histogram(samples)

    np.testing.assert_allclose(primal[0], expected[0])
    np.testing.assert_allclose(primal[1], expected[1])
    np.testing.assert_array_equal(tangent[0], np.zeros_like(primal[0]))
    np.testing.assert_array_equal(tangent[1], np.zeros_like(primal[1]))


def test_histogram_edges_follow_equal_samples_and_explicit_bounds() -> None:
    samples = np.full(3, 2.0)
    primal, tangent = ad.jvp(lambda values: np.histogram(values, bins=2))(
        samples,
        tangents=np.full(3, 0.25),
    )
    expected = np.histogram(samples, bins=2)

    np.testing.assert_array_equal(primal[0], expected[0])
    np.testing.assert_allclose(primal[1], expected[1])
    np.testing.assert_array_equal(tangent[0], np.zeros_like(primal[0]))
    np.testing.assert_allclose(tangent[1], np.full(3, 0.25))

    fixed_samples = np.array([0.1, 0.4, 0.8])
    primal, tangent = ad.jvp(
        lambda values, low, high: np.histogram(values, bins=3, range=(low, high)),
        argnums=(0, 1, 2),
    )(
        fixed_samples,
        np.array(0.0),
        np.array(1.0),
        tangents=(np.zeros_like(fixed_samples), np.array(0.2), np.array(-0.1)),
    )
    np.testing.assert_array_equal(primal[0], np.histogram(fixed_samples, bins=3)[0])
    np.testing.assert_allclose(primal[1], np.linspace(0.0, 1.0, 4))
    np.testing.assert_array_equal(tangent[0], np.zeros_like(primal[0]))
    np.testing.assert_allclose(tangent[1], np.linspace(0.2, -0.1, 4), atol=1e-15)


def test_histogram_accepts_traced_explicit_edges_and_empty_samples() -> None:
    samples = np.array([0.1, 0.4, 0.8])
    edges = np.array([0.0, 0.25, 0.75, 1.0])
    edge_direction = np.array([0.0, 0.05, -0.05, 0.0])

    primal, tangent = ad.jvp(lambda current: np.histogram(samples, bins=current))(
        edges,
        tangents=edge_direction,
    )
    expected = np.histogram(samples, bins=edges)
    np.testing.assert_array_equal(primal[0], expected[0])
    np.testing.assert_array_equal(primal[1], edges)
    np.testing.assert_array_equal(tangent[0], np.zeros_like(primal[0]))
    np.testing.assert_array_equal(tangent[1], edge_direction)

    primal, tangent = ad.jvp(lambda values: np.histogram(values, bins=2))(
        np.empty(0),
        tangents=np.empty(0),
    )
    expected = np.histogram(np.empty(0), bins=2)
    np.testing.assert_array_equal(primal[0], expected[0])
    np.testing.assert_array_equal(primal[1], expected[1])
    np.testing.assert_array_equal(tangent[0], np.zeros_like(primal[0]))
    np.testing.assert_array_equal(tangent[1], np.zeros_like(primal[1]))


@pytest.mark.parametrize("operation", [np.histogram, np.histogram_bin_edges])
def test_histogram_string_estimators_are_rejected_during_tracing(operation: Any) -> None:
    samples = np.array([0.1, 0.4, 0.8])

    with pytest.raises(ad.TracingError, match=r"string .*estimators are data-dependent"):
        ad.jvp(lambda values: operation(values, bins="auto"))(
            samples,
            tangents=np.ones_like(samples),
        )


@pytest.mark.parametrize("bins", [3, np.array([0.0, 0.5, 1.0])])
def test_histogram2d_supports_scalar_and_shared_edge_bins(bins: object) -> None:
    x = np.array([0.1, 0.4, 0.8, 0.2])
    y = np.array([0.2, 0.7, 0.9, 0.6])

    def histogram(left: Any, right: Any) -> Any:
        return np.histogram2d(
            left,
            right,
            bins=bins,
            range=((0.0, 1.0), (0.0, 1.0)),
            density=True,
        )

    primal, tangent = ad.jvp(histogram, argnums=(0, 1))(
        x,
        y,
        tangents=(np.ones_like(x), np.ones_like(y)),
    )
    expected = histogram(x, y)

    for actual, reference in zip(primal, expected, strict=True):
        np.testing.assert_allclose(actual, reference)
    for value in tangent:
        np.testing.assert_array_equal(value, np.zeros_like(value))


def test_histogramdd_supports_coordinate_tuples_scalar_bins_and_density() -> None:
    x = np.array([0.1, 0.4, 0.8, 0.2])
    y = np.array([0.2, 0.7, 0.9, 0.6])

    def histogram(left: Any, right: Any) -> Any:
        return np.histogramdd(
            (left, right),
            bins=2,
            range=((0.0, 1.0), (0.0, 1.0)),
            density=True,
        )

    primal, tangent = ad.jvp(histogram, argnums=(0, 1))(
        x,
        y,
        tangents=(np.ones_like(x), np.ones_like(y)),
    )
    expected = histogram(x, y)

    np.testing.assert_allclose(primal[0], expected[0])
    for actual, reference in zip(primal[1], expected[1], strict=True):
        np.testing.assert_allclose(actual, reference)
    np.testing.assert_array_equal(tangent[0], np.zeros_like(tangent[0]))
    for edge_tangent in tangent[1]:
        np.testing.assert_array_equal(edge_tangent, np.zeros_like(edge_tangent))


def test_histogramdd_validates_public_sample_shapes() -> None:
    samples = np.array([0.1, 0.4, 0.8])

    with pytest.raises(ad.TracingError, match=r"shape \(N, D\)"):
        ad.jvp(np.histogramdd)(samples, tangents=np.ones_like(samples))

    with pytest.raises(ad.TracingError, match="columns must have equal lengths"):
        ad.jvp(lambda values: np.histogramdd((values, np.array([0.1, 0.2]))))(
            samples,
            tangents=np.ones_like(samples),
        )


def test_i0_accepts_integer_arrays_and_rejects_complex_values() -> None:
    integers = np.array([0, 1, 2], dtype=np.int64)
    primal, tangent = ad.jvp(np.i0)(integers, tangents=np.zeros_like(integers))
    np.testing.assert_allclose(primal, np.i0(integers))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))

    complex_values = np.array([1.0 + 0.5j])
    with pytest.raises(TypeError, match="does not support complex"):
        ad.jvp(np.i0)(complex_values, tangents=np.ones_like(complex_values))


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


def test_block_supports_nested_mixed_arrays_and_validates_nesting() -> None:
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

    with pytest.raises(ad.TracingError, match="does not accept empty lists"):
        ad.jvp(lambda array: np.block([array, []]))(values, tangents=np.ones_like(values))

    with pytest.raises(ad.TracingError, match="list depths must match"):
        ad.jvp(lambda array: np.block([array, [array]]))(
            values,
            tangents=np.ones_like(values),
        )


def test_logspace_and_geomspace_accept_axis_and_dtype_controls() -> None:
    start = np.array([0.0, 1.0])
    stop = np.array([1.0, 2.0])
    direction = np.array([0.1, -0.2])

    def assert_directional(function: Any, value: np.ndarray[Any, Any]) -> None:
        primal, tangent = ad.jvp(function)(value, tangents=direction)
        reference = function(value)
        step = 1e-3
        numerical = (
            function(value + step * direction).astype(np.float64)
            - function(value - step * direction).astype(np.float64)
        ) / (2 * step)

        assert primal.shape == tangent.shape == reference.shape == (2, 4)
        assert primal.dtype == tangent.dtype == reference.dtype == np.dtype(np.float32)
        np.testing.assert_allclose(primal, reference, rtol=1e-6)
        np.testing.assert_allclose(tangent, numerical, rtol=1e-3, atol=1e-4)

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
