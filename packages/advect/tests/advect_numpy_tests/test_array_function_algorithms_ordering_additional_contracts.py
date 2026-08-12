"""Additional public contracts for NumPy algorithms and ordering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad


def test_histogram2d_accepts_static_ndarray_edge_rows() -> None:
    x = np.array([0.1, 0.4, 0.8, 0.2])
    y = np.array([0.2, 0.7, 0.9, 0.6])
    bins = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 1.0]])

    primal, tangent = ad.jvp(lambda values: np.histogram2d(values, y, bins=bins))(
        x,
        tangents=np.ones_like(x),
    )

    expected = np.histogram2d(x, y, bins=bins)
    for actual, reference in zip(primal, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    for value in tangent:
        np.testing.assert_array_equal(value, np.zeros_like(value))


def test_histogram2d_preserves_traced_ndarray_edge_rows() -> None:
    x = np.array([0.1, 0.4, 0.8, 0.2])
    y = np.array([0.2, 0.7, 0.9, 0.6])
    bins = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 1.0]])
    direction = np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]])

    primal, tangent = ad.jvp(lambda edges: np.histogram2d(x, y, bins=edges))(
        bins,
        tangents=direction,
    )

    expected = np.histogram2d(x, y, bins=bins)
    for actual, reference in zip(primal, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(tangent[0], np.zeros_like(primal[0]))
    np.testing.assert_array_equal(tangent[1], direction[0])
    np.testing.assert_array_equal(tangent[2], direction[1])


def test_histogramdd_preserves_a_traced_edge_sequence() -> None:
    samples = np.array([[0.1], [0.4], [0.8]])
    edges = np.array([0.0, 0.5, 1.0])
    direction = np.array([0.0, 0.1, 0.0])

    primal, tangent = ad.jvp(lambda values: np.histogramdd(samples, bins=(values,)))(
        edges,
        tangents=direction,
    )

    expected = np.histogramdd(samples, bins=(edges,))
    np.testing.assert_array_equal(primal[0], expected[0])
    np.testing.assert_array_equal(primal[1][0], expected[1][0])
    np.testing.assert_array_equal(tangent[0], np.zeros_like(primal[0]))
    np.testing.assert_array_equal(tangent[1][0], direction)


def test_histogramdd_rejects_an_empty_coordinate_sequence() -> None:
    weights = np.empty(0)

    with pytest.raises(ad.TracingError, match="at least one sample dimension"):
        ad.jvp(lambda values: np.histogramdd((), weights=values))(
            weights,
            tangents=weights,
        )


def test_lexsort_accepts_a_traced_key_matrix() -> None:
    keys = np.array([[2.0, 1.0, 2.0], [1.0, 3.0, 0.0]])

    primal, tangent = ad.jvp(np.lexsort)(keys, tangents=np.ones_like(keys))

    np.testing.assert_array_equal(primal, np.lexsort(keys))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


def test_isin_honors_kind_and_boolean_controls() -> None:
    values = np.array([1.0, 2.0, 4.0, 7.0])
    candidates = np.array([2.0, 3.0, 7.0])

    def membership(array: Any) -> Any:
        return np.isin(
            array,
            candidates,
            assume_unique=True,
            invert=True,
            kind="sort",
        )

    primal, tangent = ad.jvp(membership)(values, tangents=np.ones_like(values))

    np.testing.assert_array_equal(primal, membership(values))
    np.testing.assert_array_equal(tangent, np.zeros_like(primal))


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


def test_trim_zeros_rejects_an_invalid_trim_selector() -> None:
    values = np.array([0.0, 1.0, 0.0])

    with pytest.raises(ad.TracingError, match="must contain only"):
        ad.jvp(lambda array: np.trim_zeros(array, trim="fx"))(
            values,
            tangents=np.ones_like(values),
        )


def test_trim_zeros_requires_an_axis_for_a_matrix() -> None:
    values = np.zeros((2, 3))

    with pytest.raises(ad.TracingError, match="without axis requires a one-dimensional array"):
        ad.jvp(np.trim_zeros)(values, tangents=np.ones_like(values))


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


def test_arange_rejects_an_explicit_none_stop() -> None:
    start = np.array(1.0)

    with pytest.raises(ad.TracingError, match="requires a stop value"):
        ad.jvp(lambda value: np.arange(value, None, like=value))(
            start,
            tangents=np.array(0.0),
        )


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


def test_apply_over_axes_requires_a_callable() -> None:
    values = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match="func must be callable"):
        ad.jvp(lambda array: np.apply_over_axes(None, array, (0,)))(
            values,
            tangents=np.ones_like(values),
        )


def test_apply_along_axis_rejects_inconsistent_callback_shapes() -> None:
    values = np.arange(6.0).reshape(2, 3)

    def inconsistent(array: Any) -> Any:
        lengths = iter((1, 2))
        return np.apply_along_axis(lambda row: row[: next(lengths)], 1, array)

    with pytest.raises(ad.TracingError, match="returned inconsistent shapes"):
        ad.jvp(inconsistent)(values, tangents=np.ones_like(values))


def test_insert_accepts_a_negative_axis() -> None:
    values = np.arange(6.0).reshape(2, 3)
    direction = np.linspace(-0.3, 0.3, values.size).reshape(values.shape)

    primal, tangent = ad.jvp(lambda array: np.insert(array, 1, [10.0, 20.0], axis=-1))(
        values,
        tangents=direction,
    )

    np.testing.assert_array_equal(primal, np.insert(values, 1, [10.0, 20.0], axis=-1))
    np.testing.assert_array_equal(tangent, np.insert(direction, 1, [0.0, 0.0], axis=-1))


def test_insert_preserves_numpy_index_errors() -> None:
    values = np.arange(3.0)

    with pytest.raises(IndexError):
        ad.jvp(lambda array: np.insert(array, 10, 1.0))(
            values,
            tangents=np.ones_like(values),
        )
