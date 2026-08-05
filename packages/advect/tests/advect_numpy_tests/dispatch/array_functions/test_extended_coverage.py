"""User-facing qualification for NumPy functions lowered compositionally."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, strategies as st

import advect as ad
from advect.core._pytree import tree_leaves


def _assert_directional_difference(
    function: Any,
    primals: tuple[np.ndarray[Any, Any], ...],
    tangents: tuple[np.ndarray[Any, Any], ...],
    *,
    rtol: float = 2e-5,
    atol: float = 2e-6,
) -> tuple[Any, Any]:
    argnums = tuple(range(len(primals)))
    primal, tangent = ad.jvp(function, argnums=argnums)(*primals, tangents=tangents)
    reference = function(*primals)
    primal_leaves = tree_leaves(primal)
    reference_leaves = tree_leaves(reference)
    assert len(primal_leaves) == len(reference_leaves)
    for actual, expected in zip(primal_leaves, reference_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    step = 1e-6
    plus = function(
        *(
            np.asarray(value + step * direction)
            for value, direction in zip(primals, tangents, strict=True)
        )
    )
    minus = function(
        *(
            np.asarray(value - step * direction)
            for value, direction in zip(primals, tangents, strict=True)
        )
    )
    numerical_leaves = [
        (np.asarray(right) - np.asarray(left)) / (2 * step)
        for right, left in zip(tree_leaves(plus), tree_leaves(minus), strict=True)
    ]
    tangent_leaves = tree_leaves(tangent)
    assert len(tangent_leaves) == len(numerical_leaves)
    for actual, expected in zip(tangent_leaves, numerical_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return primal, tangent


def test_heaviside_differentiates_its_value_at_zero_argument() -> None:
    value = np.array([0.25, 0.75])
    primal, tangent = ad.jvp(lambda x: np.heaviside(np.zeros_like(x), x))(
        value,
        tangents=np.ones_like(value),
    )

    np.testing.assert_array_equal(primal, value)
    np.testing.assert_array_equal(tangent, np.ones_like(value))


def test_i0_matches_a_directional_difference() -> None:
    _assert_directional_difference(
        np.i0,
        (np.array([-2.0, -0.3, 0.5, 2.5]),),
        (np.array([0.5, -1.0, 2.0, -0.25]),),
    )


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


def test_lstsq_differentiates_solution_residuals_and_singular_values() -> None:
    matrix = np.array([[1.0, 0.2], [0.3, 1.4], [1.2, -0.4]])
    rhs = np.array([0.5, -1.0, 2.0])
    matrix_tangent = np.array([[0.1, -0.2], [0.05, 0.3], [-0.15, 0.2]])
    rhs_tangent = np.array([0.2, -0.1, 0.3])

    primal, tangent = _assert_directional_difference(
        lambda a, b: np.linalg.lstsq(a, b, rcond=None),
        (matrix, rhs),
        (matrix_tangent, rhs_tangent),
        rtol=2e-4,
        atol=2e-5,
    )

    assert primal[2] == 2
    assert tangent[2] == 0


def test_lstsq_differentiates_the_right_hand_side_independently() -> None:
    matrix = np.array([[1.0, 0.2], [0.3, 1.4], [1.2, -0.4]])
    rhs = np.array([0.5, -1.0, 2.0])
    rhs_tangent = np.array([0.2, -0.1, 0.3])

    primal, tangent = _assert_directional_difference(
        lambda b: np.linalg.lstsq(matrix, b, rcond=None),
        (rhs,),
        (rhs_tangent,),
        rtol=2e-4,
        atol=2e-5,
    )

    assert primal[2] == 2
    assert tangent[2] == 0
    np.testing.assert_array_equal(tangent[3], np.zeros_like(primal[3]))


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


def test_histogramdd_differentiates_weights_and_preserves_nested_edges() -> None:
    sample = np.array([[0.1, 0.2], [0.4, 0.7], [0.8, 0.9], [0.2, 0.6]])
    weights = np.array([1.0, 2.0, 3.0, 0.5])
    direction = np.array([0.2, -0.1, 0.3, 0.4])
    bins = ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])

    primal, tangent = _assert_directional_difference(
        lambda w: np.histogramdd(sample, bins=bins, weights=w, density=True),
        (weights,),
        (direction,),
    )

    assert isinstance(primal[1], list)
    assert isinstance(tangent[1], list)
    for edge_tangent in tangent[1]:
        np.testing.assert_array_equal(edge_tangent, np.zeros(3))


@pytest.mark.parametrize(
    "method",
    [
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ],
)
def test_quantile_methods_match_numpy_and_directional_differences(method: str) -> None:
    value = np.array([0.1, 1.0, 2.5, 4.0, 8.0])
    direction = np.array([0.3, -0.2, 0.4, 0.1, -0.5])

    primal, tangent = _assert_directional_difference(
        lambda x: np.quantile(x, [0.2, 0.7], method=method),
        (value,),
        (direction,),
    )

    np.testing.assert_allclose(
        primal,
        np.quantile(value, [0.2, 0.7], method=method),
    )
    assert np.shape(tangent) == (2,)


def test_quantile_differentiates_the_quantile_coordinate_for_linear_method() -> None:
    value = np.array([0.0, 1.0, 3.0, 7.0])
    quantile = np.array([0.25, 0.6])
    direction = np.array([0.2, -0.1])

    _assert_directional_difference(
        lambda q: np.quantile(value, q, method="linear"),
        (quantile,),
        (direction,),
    )


def test_gradient_differentiates_coordinate_spacing_with_second_order_edges() -> None:
    values = np.array([0.2, 0.8, 1.9, 3.5, 5.0])
    coordinates = np.array([0.0, 0.4, 1.1, 2.0, 3.2])
    value_tangent = np.array([0.1, -0.2, 0.3, 0.05, -0.1])
    coordinate_tangent = np.array([0.0, 0.1, -0.05, 0.2, 0.1])

    _assert_directional_difference(
        lambda x, spacing: np.gradient(x, spacing, edge_order=2),
        (values, coordinates),
        (value_tangent, coordinate_tangent),
        rtol=2e-4,
        atol=2e-5,
    )


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

    _assert_directional_difference(
        lambda x: np.pad(x, (2, 1), mode=mode, **kwargs),
        (value,),
        (direction,),
    )


@given(
    size=st.integers(min_value=1, max_value=8),
    before=st.integers(min_value=0, max_value=18),
    after=st.integers(min_value=0, max_value=18),
    mode_and_reflection=st.sampled_from(
        (
            ("edge", None),
            ("linear_ramp", None),
            ("reflect", "even"),
            ("reflect", "odd"),
            ("symmetric", "even"),
            ("symmetric", "odd"),
            ("wrap", None),
        )
    ),
)
def test_pad_linear_lowerings_match_numpy_beyond_one_input_period(
    size: int,
    before: int,
    after: int,
    mode_and_reflection: tuple[str, str | None],
) -> None:
    mode, reflect_type = mode_and_reflection
    value = np.linspace(-0.7, 1.3, size)
    direction = np.linspace(0.4, -0.2, size)
    kwargs: dict[str, object] = {}
    tangent_kwargs: dict[str, object] = {}
    if mode == "linear_ramp":
        kwargs["end_values"] = (1.2, -0.8)
        tangent_kwargs["end_values"] = 0
    elif reflect_type is not None:
        kwargs["reflect_type"] = reflect_type
        tangent_kwargs["reflect_type"] = reflect_type

    primal, tangent = ad.jvp(lambda x: np.pad(x, (before, after), mode=mode, **kwargs))(
        value, tangents=direction
    )

    np.testing.assert_allclose(
        primal,
        np.pad(value, (before, after), mode=mode, **kwargs),
    )
    np.testing.assert_allclose(
        tangent,
        np.pad(direction, (before, after), mode=mode, **tangent_kwargs),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "mode",
    ["constant", "linear_ramp"],
)
def test_pad_differentiates_boundary_parameters(mode: str) -> None:
    value = np.array([0.2, 1.0, 2.5])
    parameters = np.array([1.5, -0.5])
    value_direction = np.array([0.3, -0.2, 0.4])
    parameter_direction = np.array([-0.1, 0.25])
    keyword = "constant_values" if mode == "constant" else "end_values"

    _assert_directional_difference(
        lambda x, boundary: np.pad(
            x,
            (7, 5),
            mode=mode,
            **{keyword: boundary},
        ),
        (value, parameters),
        (value_direction, parameter_direction),
    )


def test_linspace_retstep_differentiates_both_outputs() -> None:
    start = np.array(-1.0)
    stop = np.array(2.0)
    start_tangent = np.array(0.2)
    stop_tangent = np.array(-0.3)

    _assert_directional_difference(
        lambda left, right: np.linspace(left, right, 6, endpoint=False, retstep=True),
        (start, stop),
        (start_tangent, stop_tangent),
    )


def test_interp_differentiates_fill_values_with_static_period() -> None:
    coordinates = np.array([-0.3, 0.2, 1.7, 2.4])
    samples = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, -0.5, 2.0])
    fills = np.array([3.0, -2.0])
    direction = np.array([0.4, -0.1])

    _assert_directional_difference(
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
    _assert_directional_difference(
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

    primal, _tangent = _assert_directional_difference(
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

    primal, _tangent = _assert_directional_difference(
        lambda x, values: np.insert(x, selector, values),
        (source, inserted),
        (source_direction, inserted_direction),
    )
    np.testing.assert_array_equal(primal, np.insert(source, selector, inserted))


def test_geomspace_accepts_one_traced_and_one_static_endpoint() -> None:
    start = np.array(-1.0)
    direction = np.array(0.2)

    primal, _tangent = _assert_directional_difference(
        lambda value: np.geomspace(value, -100.0, num=7),
        (start,),
        (direction,),
    )
    np.testing.assert_allclose(primal, np.geomspace(start, -100.0, num=7))


def test_polyfit_is_differentiable_when_an_abscissa_is_zero() -> None:
    coordinates = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.2, 1.1, 3.8, 9.2])
    direction = np.array([0.1, -0.2, 0.15, 0.05])

    _primal, tangent = _assert_directional_difference(
        lambda x: np.polyfit(x, values, 2),
        (coordinates,),
        (direction,),
        rtol=5e-4,
        atol=5e-5,
    )
    assert np.all(np.isfinite(tangent))


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

    primal, _tangent = _assert_directional_difference(
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


def test_histogram_variants_use_sparse_weight_reductions() -> None:
    weights = np.array([1.0, 2.0, 3.0, 0.5])
    direction = np.array([0.2, -0.1, 0.3, 0.4])
    x = np.array([0.1, 0.4, 0.8, 0.2])
    y = np.array([0.2, 0.7, 0.9, 0.6])
    bins = np.array([0.0, 0.5, 1.0])

    _assert_directional_difference(
        lambda w: np.histogram(x, bins=bins, weights=w, density=True)[0],
        (weights,),
        (direction,),
    )
    _assert_directional_difference(
        lambda w: np.histogram2d(
            x,
            y,
            bins=(bins, bins),
            weights=w,
            density=True,
        )[0],
        (weights,),
        (direction,),
    )
