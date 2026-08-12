"""Public contracts for NumPy conveniences lowered compositionally."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _assert_unary_jvp_matches_difference(
    function: Callable[[np.ndarray[Any, Any]], Any],
    value: np.ndarray[Any, Any],
    direction: np.ndarray[Any, Any],
    *,
    rtol: float = 2e-5,
    atol: float = 2e-6,
) -> None:
    primal, tangent = ad.jvp(function)(value, tangents=direction)
    np.testing.assert_allclose(primal, function(value), rtol=rtol, atol=atol)
    step = 1e-6
    difference = (function(value + step * direction) - function(value - step * direction)) / (
        2 * step
    )
    np.testing.assert_allclose(tangent, difference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("operation", [np.hstack, np.vstack, np.dstack, np.column_stack])
def test_stack_families_require_an_explicit_array_sequence(
    operation: Callable[..., Any],
) -> None:
    with pytest.raises(ad.TracingError, match="non-empty tuple or list"):
        ad.jvp(operation)(
            np.arange(3.0),
            tangents=np.ones(3),
        )


def test_hstack_honors_dtype_and_casting() -> None:
    value = np.array([1.0, 2.0], dtype=np.float32)
    direction = np.array([0.25, -0.5], dtype=np.float32)

    primal, tangent = ad.jvp(
        lambda array: np.hstack(
            (array, array + 1),
            dtype=np.float64,
            casting="safe",
        )
    )(value, tangents=direction)

    assert primal.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(primal, np.hstack((value, value + 1), dtype=np.float64))
    np.testing.assert_array_equal(tangent, np.hstack((direction, direction)))


def test_append_differentiates_both_arrays_along_an_explicit_axis() -> None:
    left = np.arange(6.0).reshape(2, 3)
    right = np.array([[6.0], [7.0]])
    left_direction = np.linspace(-0.3, 0.2, left.size).reshape(left.shape)
    right_direction = np.array([[0.4], [-0.2]])

    primal, tangent = ad.jvp(
        lambda first, second: np.append(first, second, axis=1),
        argnums=(0, 1),
    )(
        left,
        right,
        tangents=(left_direction, right_direction),
    )

    np.testing.assert_array_equal(primal, np.append(left, right, axis=1))
    np.testing.assert_array_equal(
        tangent,
        np.append(left_direction, right_direction, axis=1),
    )


def test_ediff1d_differentiates_traced_boundaries() -> None:
    value = np.array([0.2, 1.0, 2.5, 4.0])
    direction = np.array([0.3, -0.2, 0.4, 0.1])

    primal, tangent = ad.jvp(
        lambda array: np.ediff1d(
            array,
            to_begin=-array[:1],
            to_end=array[-1:],
        )
    )(value, tangents=direction)

    np.testing.assert_array_equal(
        primal,
        np.ediff1d(value, to_begin=-value[:1], to_end=value[-1:]),
    )
    np.testing.assert_array_equal(
        tangent,
        np.ediff1d(direction, to_begin=-direction[:1], to_end=direction[-1:]),
    )


def test_delete_supports_an_explicit_axis_and_requires_a_static_selector() -> None:
    value = np.arange(12.0).reshape(3, 4)
    direction = np.linspace(-0.5, 0.5, value.size).reshape(value.shape)

    primal, tangent = ad.jvp(lambda array: np.delete(array, [0, 2], axis=1))(
        value,
        tangents=direction,
    )
    np.testing.assert_array_equal(primal, np.delete(value, [0, 2], axis=1))
    np.testing.assert_array_equal(tangent, np.delete(direction, [0, 2], axis=1))

    with pytest.raises(ad.TracingError, match="obj= must be static"):
        ad.jvp(
            lambda array, selector: np.delete(array, selector, axis=0),
            argnums=(0, 1),
        )(
            value,
            np.array(1.0),
            tangents=(direction, np.array(0.0)),
        )


def test_resize_supports_zero_sized_outputs_and_requires_a_static_shape() -> None:
    value = np.arange(3.0)
    direction = np.array([0.2, -0.1, 0.3])

    primal, tangent = ad.jvp(lambda array: np.resize(array, (2, 0)))(
        value,
        tangents=direction,
    )
    assert primal.shape == tangent.shape == (2, 0)

    with pytest.raises(ad.TracingError, match="new_shape must be static"):
        ad.jvp(np.resize, argnums=(0, 1))(
            value,
            np.array(2.0),
            tangents=(direction, np.array(0.0)),
        )


def test_meshgrid_supports_sparse_mixed_inputs() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-1.0, 3.0])
    direction = np.array([0.2, -0.1, 0.3])

    primal, tangent = ad.jvp(lambda value: np.meshgrid(value, y, sparse=True, indexing="ij"))(
        x, tangents=direction
    )
    expected = np.meshgrid(x, y, sparse=True, indexing="ij")
    expected_tangent = np.meshgrid(
        direction,
        np.zeros_like(y),
        sparse=True,
        indexing="ij",
    )

    for actual, reference in zip(primal, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    for actual, reference in zip(tangent, expected_tangent, strict=True):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"copy": False}, "aliasing views"),
        ({"indexing": "yx"}, "indexing="),
    ],
)
def test_meshgrid_rejects_unsafe_or_invalid_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(lambda value: np.meshgrid(value, **kwargs))(
            np.arange(3.0),
            tangents=np.ones(3),
        )


def test_average_returns_the_normalization_for_unweighted_and_weighted_inputs() -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.linspace(-0.2, 0.3, value.size).reshape(value.shape)

    primal, tangent = ad.jvp(
        lambda array: np.average(
            array,
            axis=(0, 1),
            keepdims=True,
            returned=True,
        )
    )(value, tangents=direction)
    expected = np.average(value, axis=(0, 1), keepdims=True, returned=True)
    np.testing.assert_allclose(primal[0], expected[0])
    np.testing.assert_allclose(primal[1], expected[1])
    np.testing.assert_allclose(tangent[0], np.average(direction, axis=(0, 1), keepdims=True))
    np.testing.assert_array_equal(tangent[1], np.zeros_like(expected[1]))

    weights = np.array([1.0, 2.0, 3.0])
    weight_direction = np.array([0.1, -0.2, 0.3])

    def weighted(array: Any, weight: Any) -> Any:
        return np.average(array, axis=1, weights=weight, returned=True)

    weighted_primal, weighted_tangent = ad.jvp(weighted, argnums=(0, 1))(
        value,
        weights,
        tangents=(direction, weight_direction),
    )
    weighted_expected = weighted(value, weights)
    step = 1e-6
    upper = weighted(value + step * direction, weights + step * weight_direction)
    lower = weighted(value - step * direction, weights - step * weight_direction)
    for actual, reference in zip(weighted_primal, weighted_expected, strict=True):
        np.testing.assert_allclose(actual, reference)
    for actual, positive, negative in zip(weighted_tangent, upper, lower, strict=True):
        np.testing.assert_allclose(actual, (positive - negative) / (2 * step))


@pytest.mark.parametrize(
    ("axis", "weights", "exception", "message"),
    [
        (None, np.ones(2), TypeError, "Axis must be specified"),
        (1.5, np.ones(2), TypeError, "not iterable"),
        ((2,), np.ones(2), ad.TracingError, "out of bounds"),
        ((0, 0), np.ones((2, 2)), ad.TracingError, "duplicates"),
    ],
)
def test_average_validates_weight_axes(
    axis: object,
    weights: np.ndarray[Any, Any],
    exception: type[Exception],
    message: str,
) -> None:
    value = np.arange(6.0).reshape(2, 3)
    with pytest.raises(exception, match=message):
        ad.jvp(lambda array: np.average(array, axis=axis, weights=weights))(
            value,
            tangents=np.ones_like(value),
        )


def test_average_rejects_zero_weight_sums() -> None:
    with pytest.raises(ZeroDivisionError, match="sum to zero"):
        ad.jvp(lambda value: np.average(value, weights=np.zeros(3)))(
            np.arange(3.0),
            tangents=np.ones(3),
        )


def test_trapezoid_supports_scalar_spacing_and_broadcast_coordinates() -> None:
    value = np.arange(6.0).reshape(3, 2)
    direction = np.linspace(-0.2, 0.3, value.size).reshape(value.shape)
    _assert_unary_jvp_matches_difference(
        lambda array: np.trapezoid(array, dx=2.0, axis=0),
        value,
        direction,
    )

    row_value = value.T
    row_direction = direction.T
    coordinates = np.array([0.0, 1.0, 3.0])
    _assert_unary_jvp_matches_difference(
        lambda array: np.trapezoid(array, x=coordinates, axis=1),
        row_value,
        row_direction,
    )


def test_matrix_power_zero_returns_a_zero_tangent_identity() -> None:
    value = np.array([[2.0, 0.5], [0.3, 1.5]])
    direction = np.array([[0.2, -0.1], [0.3, 0.1]])

    primal, tangent = ad.jvp(lambda matrix: np.linalg.matrix_power(matrix, 0))(
        value,
        tangents=direction,
    )

    np.testing.assert_array_equal(primal, np.eye(2))
    np.testing.assert_array_equal(tangent, np.zeros((2, 2)))


def test_matrix_power_requires_a_static_integer_and_square_matrix() -> None:
    matrix = np.eye(2)
    matrix_direction = np.ones_like(matrix)

    with pytest.raises(ad.TracingError, match="exponent must be a static integer"):
        ad.jvp(np.linalg.matrix_power, argnums=(0, 1))(
            matrix,
            np.array(2.0),
            tangents=(matrix_direction, np.array(0.0)),
        )

    with pytest.raises(ad.TracingError, match="requires square matrices"):
        ad.jvp(lambda value: np.linalg.matrix_power(value, 2))(
            np.arange(6.0).reshape(2, 3),
            tangents=np.ones((2, 3)),
        )


def test_multi_dot_requires_at_least_two_arrays() -> None:
    value = np.eye(2)
    with pytest.raises(ad.TracingError, match="requires at least two arrays"):
        ad.jvp(lambda matrix: np.linalg.multi_dot([matrix]))(
            value,
            tangents=np.ones_like(value),
        )


def test_tensorinv_validates_its_static_dimension_split() -> None:
    tensor = np.eye(4).reshape(2, 2, 2, 2)
    direction = np.ones_like(tensor)

    with pytest.raises(ad.TracingError, match="ind must be a static integer"):
        ad.jvp(np.linalg.tensorinv, argnums=(0, 1))(
            tensor,
            np.array(2.0),
            tangents=(direction, np.array(0.0)),
        )

    with pytest.raises(ad.TracingError, match="ind must split the tensor dimensions"):
        ad.jvp(lambda value: np.linalg.tensorinv(value, ind=0))(
            tensor,
            tangents=direction,
        )

    with pytest.raises(ad.TracingError, match="requires equal products"):
        ad.jvp(lambda value: np.linalg.tensorinv(value, ind=1))(
            np.arange(8.0).reshape(2, 2, 2),
            tangents=np.ones((2, 2, 2)),
        )


def test_tensorsolve_requires_a_square_flattened_operator() -> None:
    with pytest.raises(ad.TracingError, match="requires a square flattened operator"):
        ad.jvp(lambda operator: np.linalg.tensorsolve(operator, np.ones(2)))(
            np.ones((2, 3)),
            tangents=np.ones((2, 3)),
        )


def test_tensorsolve_supports_moving_operator_axes() -> None:
    operator = np.eye(4).reshape(2, 2, 2, 2) + 0.1
    direction = np.linspace(-0.2, 0.3, operator.size).reshape(operator.shape)
    right = np.array([[1.0, 2.0], [3.0, 4.0]])
    _assert_unary_jvp_matches_difference(
        lambda value: np.linalg.tensorsolve(value, right, axes=(0, 1)),
        operator,
        direction,
    )


def test_cond_supports_nonspectral_norm_orders() -> None:
    value = np.array([[2.0, 0.5], [0.3, 1.5]])
    direction = np.array([[0.2, -0.1], [0.3, 0.1]])
    _assert_unary_jvp_matches_difference(
        lambda matrix: np.linalg.cond(matrix, p=1),
        value,
        direction,
    )


def test_broadcast_arrays_rejects_subclass_preservation() -> None:
    with pytest.raises(ad.TracingError, match="subok=True"):
        ad.jvp(lambda value: np.broadcast_arrays(value, np.ones(3), subok=True))(
            np.arange(6.0).reshape(2, 3),
            tangents=np.ones((2, 3)),
        )


@pytest.mark.parametrize("case", ["non-sequences", "unequal-lengths"])
def test_select_validates_condition_and_choice_sequences(case: str) -> None:
    def function(value: Any) -> Any:
        if case == "non-sequences":
            return np.select(value > 0, [value])
        return np.select([value > 0], [value, value + 1])

    message = "must be sequences" if case == "non-sequences" else "equally sized non-empty"

    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(function)(np.arange(3.0), tangents=np.ones(3))


def test_select_differentiates_mixed_choices_and_a_traced_default() -> None:
    first = np.array([True, False, False, False])
    second = np.array([False, True, False, False])
    choice = np.arange(4.0)
    default = np.array([10.0, 20.0, 30.0, 40.0])

    primal, tangent = ad.jvp(
        lambda selected, fallback: np.select(
            [first, second],
            [selected, 2.0],
            default=fallback,
        ),
        argnums=(0, 1),
    )(
        choice,
        default,
        tangents=(np.ones(4), np.full(4, 3.0)),
    )

    np.testing.assert_array_equal(primal, np.array([0.0, 2.0, 30.0, 40.0]))
    np.testing.assert_array_equal(tangent, np.array([1.0, 0.0, 3.0, 3.0]))


def test_piecewise_validates_branch_count_and_callable_output_size() -> None:
    value = np.array([-2.0, -1.0, 1.0, 2.0])
    direction = np.ones_like(value)

    with pytest.raises(ad.TracingError, match="funclist must match condlist"):
        ad.jvp(lambda array: np.piecewise(array, [array > 0], [array, array, array]))(
            value,
            tangents=direction,
        )

    with pytest.raises(ad.TracingError, match="output must be scalar or match"):
        ad.jvp(
            lambda array: np.piecewise(
                array,
                [array > 0],
                [lambda selected: np.concatenate((selected, selected))],
            )
        )(value, tangents=direction)


def test_choose_validates_its_choice_contract() -> None:
    with pytest.raises(ad.TracingError, match="non-empty choice sequence"):
        ad.jvp(lambda traced_indices: np.choose(traced_indices, []))(
            np.array([0.0]),
            tangents=np.zeros(1),
        )
    with pytest.raises(ad.TracingError, match="mode must be raise, wrap, or clip"):
        ad.jvp(lambda array: np.choose(np.array([0]), [array], mode="invalid"))(
            np.array([1.0]),
            tangents=np.ones(1),
        )


def test_vander_supports_empty_outputs_and_validates_static_shape_controls() -> None:
    value = np.arange(4.0)
    direction = np.array([0.2, -0.1, 0.3, 0.4])

    primal, tangent = ad.jvp(lambda array: np.vander(array, N=0))(
        value,
        tangents=direction,
    )
    assert primal.shape == tangent.shape == (4, 0)

    with pytest.raises(ad.TracingError, match="N must be non-negative"):
        ad.jvp(lambda array: np.vander(array, N=-1))(
            value,
            tangents=direction,
        )

    with pytest.raises(ad.TracingError, match="input must be one-dimensional"):
        ad.jvp(lambda array: np.vander(array, N=2))(
            value.reshape(2, 2),
            tangents=direction.reshape(2, 2),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fweights": np.array([1, 2, 1, 3])},
        {"aweights": np.array([0.5, 1.0, 2.0, 1.5]), "bias": True},
        {
            "fweights": np.array([1, 2, 1, 3]),
            "aweights": np.array([0.5, 1.0, 2.0, 1.5]),
        },
    ],
    ids=["frequency", "analytic-bias", "combined"],
)
def test_cov_weighted_variants_match_numpy_and_directional_differences(
    kwargs: dict[str, object],
) -> None:
    value = np.array([[0.2, 1.0, 2.5, 4.0], [1.2, -0.5, 3.0, 2.0]])
    direction = np.array([[0.1, -0.2, 0.3, 0.05], [-0.1, 0.2, 0.1, -0.3]])
    _assert_unary_jvp_matches_difference(
        lambda array: np.cov(array, **kwargs),
        value,
        direction,
    )


def test_cov_supports_an_additional_rowvar_false_dataset() -> None:
    left = np.array([[0.2, 1.2], [1.0, -0.5], [2.5, 3.0], [4.0, 2.0]])
    right = np.array([[2.0, 0.5], [1.0, 1.5], [0.0, 2.5], [-1.0, 3.5]])
    left_direction = np.linspace(-0.3, 0.2, left.size).reshape(left.shape)
    right_direction = np.linspace(0.2, -0.1, right.size).reshape(right.shape)

    def covariance(first: Any, second: Any) -> Any:
        return np.cov(first, second, rowvar=False)

    primal, tangent = ad.jvp(covariance, argnums=(0, 1))(
        left,
        right,
        tangents=(left_direction, right_direction),
    )
    step = 1e-6
    difference = (
        covariance(left + step * left_direction, right + step * right_direction)
        - covariance(left - step * left_direction, right - step * right_direction)
    ) / (2 * step)
    np.testing.assert_allclose(primal, covariance(left, right))
    np.testing.assert_allclose(tangent, difference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"ddof": 1.5}, ValueError, "ddof must be integer"),
        ({"fweights": np.ones(2)}, RuntimeError, "incompatible numbers of samples"),
        ({"fweights": np.zeros(4)}, ZeroDivisionError, "sum to zero"),
    ],
)
def test_cov_validates_normalization_inputs(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    value = np.arange(8.0).reshape(2, 4)
    with pytest.raises(exception, match=message):
        ad.jvp(lambda array: np.cov(array, **kwargs))(
            value,
            tangents=np.ones_like(value),
        )


def test_corrcoef_supports_scalar_results() -> None:
    value = np.array([0.2, 1.0, 2.5, 4.0])
    direction = np.array([0.1, -0.2, 0.3, 0.05])
    primal, tangent = ad.jvp(np.corrcoef)(value, tangents=direction)
    np.testing.assert_allclose(primal, np.corrcoef(value))
    np.testing.assert_allclose(tangent, 0.0, atol=1e-14)
