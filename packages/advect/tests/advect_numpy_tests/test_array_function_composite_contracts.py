"""Public contracts for NumPy conveniences lowered compositionally."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp
from numpy.lib import scimath

import advect as ad
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_staged_round_trip,
    assert_tree_close,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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


def _ediff1d_boundary_loss(boundary_dtype: type[np.floating[Any]]) -> Callable[[Any], Any]:
    def loss(x: Any) -> Any:
        begin = np.astype(x[:1] ** 3, boundary_dtype)
        return np.sum(np.ediff1d(x * x, to_begin=begin, to_end=x[-1:]) ** 2)

    return loss


@pytest.mark.parametrize("boundary_dtype", [np.float64, np.float32])
def test_ediff1d_differentiates_traced_boundaries_twice(
    boundary_dtype: type[np.floating[Any]],
) -> None:
    value = np.array([0.2, 1.0, 2.5, 4.0])
    # A float32 boundary is cast back to float64, which only rounds its value,
    # so float64 finite differences are the reference for both dtypes.
    step = 1e-6
    gradient = ad.grad(_ediff1d_boundary_loss(np.float64))
    expected = np.stack(
        [
            (gradient(value + step * direction) - gradient(value - step * direction)) / (2 * step)
            for direction in np.eye(value.size)
        ]
    )

    hessian = ad.hessian(_ediff1d_boundary_loss(boundary_dtype))(value)

    np.testing.assert_allclose(hessian, expected, rtol=1e-5, atol=1e-5)


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

    assert_tree_close(primal, expected, rtol=0.0)
    assert_tree_close(tangent, expected_tangent, rtol=0.0)


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


@pytest.mark.parametrize(
    "lifetime",
    [
        lambda function, value: ad.jvp(function)(value, tangents=np.ones_like(value)),
        lambda function, value: ad.stage(function, specs=(ad.ArraySpec(value.shape, value.dtype),)),
    ],
    ids=("dynamic", "staged"),
)
@pytest.mark.parametrize(
    ("axis", "weights", "exception", "message"),
    [
        (None, np.ones(2), TypeError, "Axis must be specified"),
        (1, np.ones(2), ValueError, "Shape of weights must be consistent"),
        (1.5, np.ones(2), TypeError, "not iterable"),
        ((2,), np.ones(2), np.exceptions.AxisError, "out of bounds"),
        ((0, 0), np.ones((2, 2)), ValueError, "repeated axis"),
    ],
)
def test_average_validates_weight_axes_like_numpy(
    lifetime: Callable[[Callable[[Any], Any], np.ndarray[Any, Any]], object],
    axis: object,
    weights: np.ndarray[Any, Any],
    exception: type[Exception],
    message: str,
) -> None:
    value = np.arange(6.0).reshape(2, 3)

    def average(array: Any) -> Any:
        return np.average(array, axis=axis, weights=weights)

    with pytest.raises(exception, match=message):
        average(value)
    with pytest.raises(exception, match=message):
        lifetime(average, value)


def test_average_rejects_zero_weight_sums() -> None:
    with pytest.raises(ZeroDivisionError, match="sum to zero"):
        ad.jvp(lambda value: np.average(value, weights=np.zeros(3)))(
            np.arange(3.0),
            tangents=np.ones(3),
        )


def test_trapezoid_supports_scalar_spacing_and_broadcast_coordinates() -> None:
    value = np.arange(6.0).reshape(3, 2)
    direction = np.linspace(-0.2, 0.3, value.size).reshape(value.shape)
    assert_jvp_matches_central_difference(
        lambda array: np.trapezoid(array, dx=2.0, axis=0), (value,), (direction,)
    )

    row_value = value.T
    row_direction = direction.T
    coordinates = np.array([0.0, 1.0, 3.0])
    assert_jvp_matches_central_difference(
        lambda array: np.trapezoid(array, x=coordinates, axis=1), (row_value,), (row_direction,)
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
    assert_jvp_matches_central_difference(
        lambda value: np.linalg.tensorsolve(value, right, axes=(0, 1)), (operator,), (direction,)
    )


def test_cond_supports_nonspectral_norm_orders() -> None:
    value = np.array([[2.0, 0.5], [0.3, 1.5]])
    direction = np.array([[0.2, -0.1], [0.3, 0.1]])
    assert_jvp_matches_central_difference(
        lambda matrix: np.linalg.cond(matrix, p=1), (value,), (direction,)
    )


def test_broadcast_arrays_rejects_subclass_preservation() -> None:
    with pytest.raises(ad.TracingError, match="subok=True"):
        ad.jvp(lambda value: np.broadcast_arrays(value, np.ones(3), subok=True))(
            np.arange(6.0).reshape(2, 3),
            tangents=np.ones((2, 3)),
        )


_CONSTANT_LIFTING_FORMS: dict[str, Callable[[Any], Any]] = {
    "block": lambda x: np.block([x, np.array([1, 2])]),
    "broadcast_arrays": lambda x: np.broadcast_arrays(x, np.array([[1], [2]])),
    # NumPy reads a Python scalar or a nested list as its array; the handler
    # read their missing shape attribute.
    "broadcast_arrays[python]": lambda x: np.broadcast_arrays(x, 2.5, [[1], [2]]),
    "choose": lambda x: np.choose(np.arange(x.size) % 2, [x, np.arange(x.size)]),
    "insert": lambda x: np.insert(x, 1, 5.0),
    "logn": lambda x: scimath.logn(2.0, x),
    "meshgrid": lambda x: np.meshgrid(x, np.array([1.5, 2.5])),
    "piecewise": lambda x: np.piecewise(x, [x < 0], [lambda v: 2 * v, 3.0]),
    "power": lambda x: scimath.power(x, 2.0),
    "select": lambda x: np.select([x > 0], [x], 0.5),
}


@given(
    form=st.sampled_from(sorted(_CONSTANT_LIFTING_FORMS)),
    value=hnp.arrays(
        st.sampled_from([np.float32, np.float64]),
        st.integers(min_value=1, max_value=5),
        elements=st.floats(width=32),
    ),
)
@example(form="broadcast_arrays[python]", value=np.array([1.0, 2.0]))
@example(form="choose", value=np.array([np.inf, -1.0, 2.0]))
@example(form="piecewise", value=np.array([3.0, np.nan, 2.0], dtype=np.float32))
@example(form="select", value=np.array([np.nan, -1.0], dtype=np.float32))
def test_lifted_constants_keep_numpy_values_and_dtypes(
    form: str,
    value: np.ndarray[Any, Any],
) -> None:
    _assert_traced_primal_is_numpy_exactly(_CONSTANT_LIFTING_FORMS[form], value)


def _assert_traced_primal_is_numpy_exactly(
    function: Callable[[Any], Any],
    value: np.ndarray[Any, Any],
) -> None:
    with np.errstate(all="ignore"):
        expected = function(value)
        actual, pullback = ad.vjp(function)(value)
    pullback.close()

    assert_tree_close(actual, expected, rtol=0.0)


@st.composite
def _round_inputs(draw: st.DrawFn) -> np.ndarray[Any, Any]:
    dtype = np.dtype(draw(st.sampled_from(["float16", "float32", "float64", "int32", "int64"])))
    if draw(st.booleans()):
        # Advect recombines the rounded parts as real + imag * 1j, where a
        # non-finite rounded imaginary part also turns the real part into NaN.
        # The complex domain therefore stays finite after scaling by 10**4.
        part = st.floats(min_value=-(2.0**100), max_value=2.0**100, width=32)
        elements = st.builds(complex, part, part)
        dtype = np.dtype(np.complex64 if dtype.itemsize < 8 else np.complex128)
        return draw(hnp.arrays(dtype, st.integers(1, 4), elements=elements))
    return draw(hnp.arrays(dtype, st.integers(1, 4)))


@given(value=_round_inputs(), decimals=st.integers(min_value=-4, max_value=4))
@example(value=np.array([15, 25, -35, 14]), decimals=-1)
@example(value=np.array([16777215.0], dtype=np.float32), decimals=-3)
@example(value=np.array([0.3 + 0.7j, 0.29 + 0.31j]), decimals=1)
# NumPy's power of ten saturates to inf, so extreme decimals give NaN.
@example(value=np.array([1.5, -2.25]), decimals=-400)
@example(value=np.array([1.5, -2.25]), decimals=400)
# Staging once rounded integers with rint, which returns float64.
@example(value=np.array([15, -35], dtype=np.int32), decimals=0)
def test_round_is_numpy_round_exactly(value: np.ndarray[Any, Any], decimals: int) -> None:
    _assert_traced_primal_is_numpy_exactly(lambda x: np.round(x, decimals), value)
    _assert_traced_primal_is_numpy_exactly(lambda x: np.around(x, decimals), value)
    if decimals == 0:
        # Staging supports only the default decimals.
        expected = np.round(value)
        staged = ad.stage(np.round, specs=(ad.ArraySpec(value.shape, value.dtype),))(value)
        assert staged.dtype == expected.dtype
        np.testing.assert_array_equal(staged, expected)


def test_round_rints_bool_input_like_numpy() -> None:
    _assert_traced_primal_is_numpy_exactly(np.round, np.array([True, False]))


@pytest.mark.parametrize("decimals", [1, -1])
def test_round_rejects_scaling_bool_input_like_numpy(decimals: int) -> None:
    value = np.array([True, False])
    with pytest.raises(TypeError):
        np.round(value, decimals)

    with pytest.raises(TypeError, match="scaled bool array"):
        ad.vjp(lambda x: np.round(x, decimals))(value)


@pytest.mark.parametrize(
    "function",
    [
        lambda x: np.ediff1d(x, to_begin=[[1.0, 2.0]], to_end=np.array([[5.0], [6.0]])),
        lambda x: np.ediff1d(np.astype(x, np.float32), to_begin=[[1.0, 2.0]], to_end=7.5),
        lambda x: np.ediff1d(np.astype(x * 4, np.int64), to_begin=[5], to_end=np.int8(3)),
    ],
    ids=("matrix-boundaries", "float32-input", "integer-input"),
)
def test_ediff1d_ravels_boundaries_into_the_input_dtype(function: Callable[[Any], Any]) -> None:
    _assert_traced_primal_is_numpy_exactly(function, np.array([0.5, 1.0, 2.25]))


def test_ediff1d_rejects_boundaries_that_cannot_cast_to_the_input_dtype() -> None:
    with pytest.raises(TypeError, match="`to_begin` must be compatible"):
        ad.jvp(lambda x: np.ediff1d(np.astype(x, np.int64), to_begin=[0.5]))(
            np.array([1.0, 2.0]),
            tangents=np.ones(2),
        )


def test_compress_and_extract_select_numpy_positions() -> None:
    value = np.array([0.5, 1.0, 2.0, 3.25])

    _assert_traced_primal_is_numpy_exactly(
        lambda x: np.compress([True, False, True, False, False], x),
        value,
    )
    _assert_traced_primal_is_numpy_exactly(
        lambda x: np.extract(np.reshape(x, (2, 2)) > 0.75, np.reshape(x, (2, 2))),
        value,
    )
    with pytest.raises(ValueError, match="condition must be a 1-d array"):
        ad.jvp(lambda x: np.compress([[True, False]], x))(value, tangents=np.ones_like(value))


@pytest.mark.parametrize("function", [np.compress, np.extract])
def test_compress_and_extract_select_from_a_static_sequence(
    function: Callable[..., Any],
) -> None:
    # Only the condition is traced, so the selection is a constant of its trace.
    _assert_traced_primal_is_numpy_exactly(
        lambda x: function(x > 0.75, [10.0, 20.0, 30.0, 40.0]),
        np.array([0.5, 1.0, 2.0, 3.25]),
    )


@pytest.mark.parametrize(
    "lifetime",
    [
        lambda function, value: ad.jvp(function)(value, tangents=np.ones_like(value)),
        lambda function, value: ad.stage(function, specs=(ad.ArraySpec(value.shape, value.dtype),)),
    ],
    ids=("dynamic", "staged"),
)
@pytest.mark.parametrize(
    ("shape", "axis", "error", "match"),
    [
        ((4,), None, IndexError, "index 4 is out of bounds for axis 0 with size 4"),
        ((3, 2), -1, IndexError, "index 2 is out of bounds for axis 1 with size 2"),
        ((3, 2), 1.0, TypeError, "'float' object cannot be interpreted as an integer"),
    ],
    ids=("flattened", "negative-axis", "float-axis"),
)
def test_compress_rejects_numpy_invalid_positions_and_axes(
    lifetime: Callable[[Callable[[Any], Any], np.ndarray[Any, Any]], object],
    shape: tuple[int, ...],
    axis: object,
    error: type[Exception],
    match: str,
) -> None:
    def over_long(x: Any) -> Any:
        return np.compress([True, False, True, False, True], x, axis=axis)

    with pytest.raises(error, match=match):
        over_long(np.ones(shape))
    with pytest.raises(error, match=match):
        lifetime(over_long, np.ones(shape))


@pytest.mark.parametrize("axis", [np.int64(-1), 1], ids=("numpy-integer", "python-integer"))
def test_compress_accepts_integer_axes_in_every_lifetime(axis: object) -> None:
    value = np.arange(6.0).reshape(3, 2)

    def function(x: Any) -> Any:
        return np.compress([False, True], x, axis=axis)

    np.testing.assert_array_equal(ad.jvp(function)(value, tangents=value)[0], function(value))
    assert_staged_round_trip(function, value, rtol=0.0)


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


def test_select_promotes_from_dtypes_inside_a_staged_derivative() -> None:
    value = np.array([0.3, -1.2, 2.0, 0.7, -0.5, 1.5], dtype=np.float32)

    def loss(x: Any) -> Any:
        return np.sum(np.select([x > 1, x < -1], [x * 2, x**2], default=x))

    staged = ad.stage(ad.grad(loss), value)(value)

    assert staged.dtype == np.float32
    np.testing.assert_array_equal(staged, ad.grad(loss)(value))


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
    assert_jvp_matches_central_difference(
        lambda array: np.cov(array, **kwargs), (value,), (direction,)
    )


def test_cov_supports_an_additional_rowvar_false_dataset() -> None:
    left = np.array([[0.2, 1.2], [1.0, -0.5], [2.5, 3.0], [4.0, 2.0]])
    right = np.array([[2.0, 0.5], [1.0, 1.5], [0.0, 2.5], [-1.0, 3.5]])
    left_direction = np.linspace(-0.3, 0.2, left.size).reshape(left.shape)
    right_direction = np.linspace(0.2, -0.1, right.size).reshape(right.shape)

    def covariance(first: Any, second: Any) -> Any:
        return np.cov(first, second, rowvar=False)

    assert_jvp_matches_central_difference(
        covariance, (left, right), (left_direction, right_direction)
    )


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
