"""User-facing qualification for NumPy functions lowered compositionally."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad
from advect_numpy_tests._assertions import assert_jvp_matches_central_difference

if TYPE_CHECKING:
    from collections.abc import Callable


def test_heaviside_differentiates_its_value_at_zero_argument() -> None:
    value = np.array([0.25, 0.75])
    primal, tangent = ad.jvp(lambda x: np.heaviside(np.zeros_like(x), x))(
        value,
        tangents=np.ones_like(value),
    )

    np.testing.assert_array_equal(primal, value)
    np.testing.assert_array_equal(tangent, np.ones_like(value))


def test_signbit_is_traceable_as_a_piecewise_constant_mask() -> None:
    value = np.array([-2.0, 0.5, -0.25])
    direction = np.array([0.4, -0.2, 0.1])

    def absolute_from_mask(x: Any) -> Any:
        return np.where(np.signbit(x), -x, x)

    primal, tangent = ad.jvp(absolute_from_mask)(value, tangents=direction)
    np.testing.assert_array_equal(primal, np.abs(value))
    np.testing.assert_array_equal(tangent, np.array([-0.4, -0.2, -0.1]))

    program = ad.stage(
        absolute_from_mask,
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    np.testing.assert_array_equal(program(value), np.abs(value))


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("constant", {"constant_values": np.array([1.5, -0.5])}),
        ("linear_ramp", {"end_values": np.array([1.5, -0.5])}),
        ("reflect", {"reflect_type": "odd"}),
        ("symmetric", {"reflect_type": "even"}),
        ("wrap", {}),
        ("mean", {"stat_length": 2}),
        ("median", {"stat_length": 2}),
        ("maximum", {"stat_length": 2}),
        ("minimum", {"stat_length": 2}),
    ],
)
def test_pad_numeric_modes_match_directional_differences(
    mode: str,
    kwargs: dict[str, object],
) -> None:
    value = np.array([0.2, 1.0, 2.5, 4.0])
    direction = np.array([0.3, -0.2, 0.4, 0.1])

    assert_jvp_matches_central_difference(
        lambda x: np.pad(x, (2, 1), mode=mode, **kwargs),
        (value,),
        (direction,),
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: np.pad(np.astype(x, np.int64), (1, 2), mode="mean"),
        lambda x: np.pad(np.astype(x, np.int64), 1, mode="median", stat_length=2),
        lambda x: np.pad(x, (2, 1), mode="mean", stat_length=1.6),
        lambda x: np.pad(np.reshape(x, (2, 2)), ((1, 2), (2, 1)), mode="minimum"),
        # NumPy returns an empty array before it computes or validates statistics.
        lambda x: np.pad(np.reshape(x, (4, 1))[:0], ((0, 0), (1, 2)), mode="maximum"),
        lambda x: np.pad(
            np.reshape(x, (4, 1))[:0], ((0, 0), (1, 1)), mode="minimum", stat_length=0
        ),
        lambda x: np.pad(
            np.reshape(x, (4, 1))[:0], ((0, 0), (2, 1)), mode="median", stat_length=-1
        ),
    ],
    ids=(
        "integer-mean",
        "integer-median",
        "rounded-stat-length",
        "matrix-minimum",
        "empty-maximum",
        "empty-zero-stat-length",
        "empty-negative-stat-length",
    ),
)
def test_statistical_pad_matches_numpy_rounding_and_lengths(
    operation: Callable[[Any], Any],
) -> None:
    value = np.array([1.0, 2.0, 4.0, 7.0])
    primal, _tangent = ad.jvp(operation)(value, tangents=np.ones_like(value))

    expected = operation(value)
    assert primal.dtype == expected.dtype
    np.testing.assert_array_equal(primal, expected)


def test_interp_differentiates_fill_values_with_static_period() -> None:
    coordinates = np.array([-0.3, 0.2, 1.7, 2.4])
    samples = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, -0.5, 2.0])
    fills = np.array([3.0, -2.0])
    direction = np.array([0.4, -0.1])

    assert_jvp_matches_central_difference(
        lambda y, fill: np.interp(
            coordinates,
            samples,
            y,
            left=fill[0],
            right=fill[1],
        ),
        (values, fills),
        (np.array([0.2, -0.3, 0.1]), direction),
    )
    assert_jvp_matches_central_difference(
        lambda y: np.interp(coordinates, samples, y, period=3.0),
        (values,),
        (np.array([0.2, -0.3, 0.1]),),
    )


def test_static_shape_queries_are_available_inside_a_trace() -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.ones_like(value)

    primal, tangent = ad.jvp(
        lambda x: x * (np.ndim(x) + np.size(x) + len(np.shape(x))),
    )(value, tangents=direction)

    np.testing.assert_array_equal(primal, value * 10)
    np.testing.assert_array_equal(tangent, direction * 10)


@pytest.mark.parametrize(
    "query",
    [
        lambda x: np.ndim(a=x),
        lambda x: np.shape(a=x),
        lambda x: np.size(a=x),
        lambda x: np.isrealobj(x=x),
        lambda x: np.iscomplexobj(x=x),
        lambda x: np.can_cast(from_=x, to=np.float32),
    ],
    ids=["ndim", "shape", "size", "isrealobj", "iscomplexobj", "can_cast"],
)
def test_static_queries_accept_keyword_operands_inside_a_trace(query: Any) -> None:
    value = np.arange(6.0).reshape(2, 3)
    answers: list[object] = []

    def record(x: Any) -> Any:
        answers.append(query(x))
        return x

    ad.jvp(record)(value, tangents=np.ones_like(value))

    assert answers == [query(value)]


@pytest.mark.parametrize("operation", [np.mean, np.var, np.std])
def test_controlled_reductions_preserve_float32_dtype(operation: Any) -> None:
    value = np.array([0.5, 1.5, 2.5, 4.0], dtype=np.float32)
    direction = np.array([0.2, -0.1, 0.3, 0.4], dtype=np.float32)
    mask = np.array([True, False, True, True])

    kwargs = {"correction": 1} if operation in {np.var, np.std} else {}

    def reduce(x: Any) -> Any:
        return operation(x, where=mask, **kwargs)

    primal, tangent = ad.jvp(reduce)(value, tangents=direction)
    gradient = ad.grad(reduce)(value)

    assert np.asarray(primal).dtype == np.dtype(np.float32)
    assert np.asarray(tangent).dtype == np.dtype(np.float32)
    assert np.asarray(gradient).dtype == np.dtype(np.float32)


def test_piecewise_callables_receive_only_their_selected_subset() -> None:
    value = np.array([-3.0, -1.0, 0.5, 2.0])
    direction = np.array([0.2, -0.3, 0.4, -0.1])

    def centered_negative(x: Any) -> Any:
        return np.piecewise(
            x,
            [x < 0],
            [lambda selected: selected - np.mean(selected), 2.0],
        )

    primal, _tangent = assert_jvp_matches_central_difference(
        centered_negative,
        (value,),
        (direction,),
    )
    np.testing.assert_allclose(primal, centered_negative(value))


def test_choose_mode_raise_validates_concrete_indices() -> None:
    indices = np.array([0, 2])
    with pytest.raises(ValueError, match="invalid entry"):
        ad.jvp(lambda x: np.choose(indices, (x, x + 1.0), mode="raise"))(
            np.array([1.0, 2.0]),
            tangents=np.ones(2),
        )


@pytest.mark.parametrize("selector", [slice(1, 4, 2), [1, 3]])
def test_insert_differentiates_array_and_inserted_values(selector: Any) -> None:
    source = np.array([0.2, 1.0, 2.5, 4.0])
    inserted = np.array([-1.0, 3.0])
    source_direction = np.array([0.3, -0.2, 0.4, 0.1])
    inserted_direction = np.array([-0.5, 0.25])

    primal, _tangent = assert_jvp_matches_central_difference(
        lambda x, values: np.insert(x, selector, values),
        (source, inserted),
        (source_direction, inserted_direction),
    )
    np.testing.assert_array_equal(primal, np.insert(source, selector, inserted))


def test_geomspace_accepts_one_traced_and_one_static_endpoint() -> None:
    start = np.array(-1.0)
    direction = np.array(0.2)

    primal, _tangent = assert_jvp_matches_central_difference(
        lambda value: np.geomspace(value, -100.0, num=7),
        (start,),
        (direction,),
    )
    np.testing.assert_allclose(primal, np.geomspace(start, -100.0, num=7))


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"fweights": np.array([1.0, 1.5, 2.0])}, TypeError),
        ({"aweights": np.array([1.0, -1.0, 2.0])}, ValueError),
        ({"fweights": np.ones((1, 3))}, RuntimeError),
    ],
)
def test_cov_validates_weight_contracts(
    kwargs: dict[str, np.ndarray[Any, Any]],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ad.jvp(lambda x: np.cov(x, **kwargs))(
            np.array([0.2, 1.0, 2.5]),
            tangents=np.ones(3),
        )


def test_average_supports_tuple_axes_and_validates_weight_shape() -> None:
    value = np.arange(24.0).reshape(2, 3, 4)
    weights = np.arange(8.0).reshape(2, 4) + 1.0
    direction = np.linspace(-0.5, 0.5, value.size).reshape(value.shape)

    primal, _tangent = assert_jvp_matches_central_difference(
        lambda x: np.average(x, axis=(0, 2), weights=weights),
        (value,),
        (direction,),
    )
    np.testing.assert_allclose(
        primal,
        np.average(value, axis=(0, 2), weights=weights),
    )

    with pytest.raises(ValueError, match="Shape of weights"):
        ad.jvp(lambda x: np.average(x, axis=1, weights=np.ones(2)))(
            value,
            tangents=direction,
        )


def test_resize_of_an_empty_array_has_a_zero_traceable_result() -> None:
    value = np.empty(0, dtype=np.float64)
    primal, tangent = ad.jvp(lambda x: np.resize(x, (2, 3)))(
        value,
        tangents=np.empty_like(value),
    )

    np.testing.assert_array_equal(primal, np.zeros((2, 3)))
    np.testing.assert_array_equal(tangent, np.zeros((2, 3)))
