"""Public contracts for miscellaneous NumPy array-function forms."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_staged_round_trip,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_nan_to_num_static_replacements_zero_replaced_tangents() -> None:
    value = np.array([np.nan, np.inf, -np.inf, 2.0])
    direction = np.array([1.0, 2.0, 3.0, 4.0])

    primal, tangent = ad.jvp(lambda x: np.nan_to_num(x, nan=-2.0, posinf=5.0, neginf=-7.0))(
        value, tangents=direction
    )

    np.testing.assert_array_equal(primal, [-2.0, 5.0, -7.0, 2.0])
    np.testing.assert_array_equal(tangent, [0.0, 0.0, 0.0, 4.0])


@pytest.mark.parametrize(
    ("value", "direction", "expected", "expected_tangent"),
    [
        (
            np.array([np.nan, np.inf, -np.inf, 2.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([-2.0, 5.0, -7.0, 2.0]),
            np.array([0.1, 0.2, 0.3, 4.0]),
        ),
        (
            np.array([complex(np.nan, 1), complex(2, np.inf), complex(-np.inf, np.nan)]),
            np.array([1 + 2j, 2 + 3j, 3 + 4j]),
            np.array([-2 + 1j, 2 + 5j, -7 - 2j]),
            np.array([0.1 + 2j, 2 + 0.2j, 0.3 + 0.1j]),
        ),
    ],
    ids=("real", "complex"),
)
def test_nan_to_num_replacements_are_differentiable(
    value: np.ndarray[Any, Any],
    direction: np.ndarray[Any, Any],
    expected: np.ndarray[Any, Any],
    expected_tangent: np.ndarray[Any, Any],
) -> None:
    primal, tangent = ad.jvp(
        lambda x, nan, posinf, neginf: np.nan_to_num(
            x,
            nan=nan,
            posinf=posinf,
            neginf=neginf,
        ),
        argnums=(0, 1, 2, 3),
    )(
        value,
        np.array(-2.0),
        np.array(5.0),
        np.array(-7.0),
        tangents=(direction, np.array(0.1), np.array(0.2), np.array(0.3)),
    )

    np.testing.assert_array_equal(primal, expected)
    np.testing.assert_array_equal(tangent, expected_tangent)


@pytest.mark.parametrize(
    ("keyword", "replacement"),
    [("nan", 5.0), ("posinf", np.inf), ("neginf", np.nan)],
)
def test_nan_to_num_preserves_an_integer_input_with_traced_replacements(
    keyword: str,
    replacement: float,
) -> None:
    value = np.array([1, 2, 3], dtype=np.int64)

    primal, tangent = ad.jvp(
        lambda x, traced_replacement: np.nan_to_num(x, **{keyword: traced_replacement}),
        argnums=(0, 1),
    )(
        value,
        np.array(replacement),
        tangents=(np.ones_like(value), np.array(0.25)),
    )

    np.testing.assert_array_equal(primal, value)
    assert primal.dtype == value.dtype
    np.testing.assert_array_equal(tangent, np.zeros_like(value))


def test_angle_accepts_positional_degree_metadata() -> None:
    value = np.array([1.0 + 2.0j, -2.0 + 1.0j])
    direction = np.array([0.2 - 0.1j, 0.1 + 0.3j])

    def angle(x: Any) -> Any:
        return np.angle(x, True)  # noqa: FBT003 - exercise NumPy's positional form

    assert_jvp_matches_central_difference(angle, (value,), (direction,), rtol=1e-7, atol=0.0)
    assert_staged_round_trip(angle, value)


def test_gradient_of_a_selected_python_scalar_has_no_axes() -> None:
    # NumPy's gradient of a rank-zero value is an empty tuple. The handler read
    # the rank of the scalar's Python float: "'float' object has no attribute 'ndim'".
    primal, tangent = ad.jvp(np.gradient)(0.5, tangents=1.0)

    assert primal == tangent == np.gradient(0.5) == ()


def test_linspace_positional_options_and_axis_match_numpy() -> None:
    start = np.array([0.0, 1.0])
    stop = np.array([2.0, 3.0])
    start_direction = np.array([0.1, 0.2])
    stop_direction = np.array([0.3, 0.4])

    def spaced(left: Any, right: Any) -> Any:
        return np.linspace(
            left,
            right,
            3,
            False,  # noqa: FBT003 - exercise NumPy's positional contract
            False,  # noqa: FBT003 - exercise NumPy's positional contract
            np.float32,
            1,
            device="cpu",
        )

    primal, tangent = ad.jvp(spaced, argnums=(0, 1))(
        start,
        stop,
        tangents=(start_direction, stop_direction),
    )

    np.testing.assert_allclose(primal, spaced(start, stop))
    np.testing.assert_allclose(tangent, spaced(start_direction, stop_direction))


def test_linspace_empty_retstep_has_a_constant_nan_step() -> None:
    primal, tangent = ad.jvp(
        lambda start, stop: np.linspace(start, stop, num=0, retstep=True),
        argnums=(0, 1),
    )(
        np.array(0.0),
        np.array(2.0),
        tangents=(np.array(0.1), np.array(0.2)),
    )

    assert primal[0].size == tangent[0].size == 0
    assert np.isnan(primal[1])
    assert tangent[1] == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: np.sort(x, axis=0, stable=True),
        lambda x: np.partition(x, (0, 2), axis=1, kind="introselect"),
        lambda x: np.sort(x, axis=None),
        lambda x: np.partition(x, 4, axis=None),
    ],
    ids=("sort", "partition", "sort-flattened", "partition-flattened"),
)
def test_ordering_operations_apply_the_primal_permutation_to_tangents(
    operation: Callable[[Any], Any],
) -> None:
    value = np.array([[2.0, 7.0, 1.0], [5.0, 4.0, 9.0]])
    direction = np.array([[0.2, -0.3, 0.5], [0.7, 0.1, -0.2]])

    assert_jvp_matches_central_difference(operation, (value,), (direction,), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    "operation",
    [lambda x: np.sort(x, axis=None), lambda x: np.sort(x, None, "stable")],
    ids=("keyword", "positional"),
)
def test_sort_with_axis_none_sorts_the_flattened_array_in_every_lifetime(
    operation: Callable[[Any], Any],
) -> None:
    value = np.array([[2.0, 7.0, 1.0], [5.0, 4.0, 9.0]])
    weights = np.arange(1.0, 7.0)
    expected = operation(value)

    np.testing.assert_array_equal(ad.jvp(operation)(value, tangents=value)[0], expected)
    gradient_fn = ad.grad(lambda x: np.sum(operation(x) * weights))
    gradient = gradient_fn(value)
    np.testing.assert_array_equal(
        gradient.ravel()[np.argsort(value, axis=None, kind="stable")],
        weights,
    )
    for function in (operation, gradient_fn):
        assert_staged_round_trip(function, value, rtol=0.0)


@pytest.mark.parametrize(
    ("operation", "staged"),
    [
        (lambda x: np.argsort(x, axis=None), True),
        (lambda x: np.argsort(x, None, "stable"), True),
        (lambda x: np.argsort(x, 0), True),
        (lambda x: np.argpartition(x, 4, axis=None), False),
        (lambda x: np.argpartition(x, 1, None), False),
    ],
    ids=(
        "argsort-keyword",
        "argsort-positional",
        "argsort-positional-axis",
        "argpartition-keyword",
        "argpartition-positional",
    ),
)
def test_argsort_and_argpartition_follow_numpy_axes_in_every_lifetime(
    operation: Callable[[Any], Any],
    *,
    staged: bool,
) -> None:
    value = np.array([[2.0, 7.0, 1.0], [5.0, 4.0, 9.0]])
    expected = operation(value)

    np.testing.assert_array_equal(ad.jvp(operation)(value, tangents=value)[0], expected)
    if staged:
        assert_staged_round_trip(operation, value, rtol=0.0)


_DESCENDING = pytest.mark.skipif(
    "descending" not in inspect.signature(np.sort).parameters,
    reason="NumPy added sort(descending=...) in 2.5",
)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        pytest.param(lambda x: np.sort(x, descending=True), "descending", marks=_DESCENDING),
        pytest.param(lambda x: np.argsort(x, descending=True), "descending", marks=_DESCENDING),
        (lambda x: np.take(x, [0], mode="invalid"), "mode must be raise, wrap, or clip"),
    ],
    ids=("sort-descending", "argsort-descending", "take-mode"),
)
def test_unsupported_sort_and_take_options_fail_while_tracing_or_staging(
    operation: Callable[[Any], Any],
    message: str,
) -> None:
    value = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(operation)(value, tangents=np.ones_like(value))
    with pytest.raises(ad.TracingError, match=message):
        ad.stage(operation, specs=(ad.ArraySpec(value.shape, value.dtype),))


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: np.sort(x, descending=False),
        lambda x: np.sort(x, None, descending=None),
        lambda x: np.take_along_axis(x, np.argsort(x, 0, descending=0), axis=0),
    ],
    ids=("sort", "sort-flattened", "argsort"),
)
@_DESCENDING
def test_ascending_descending_option_traces_stages_and_differentiates(
    operation: Callable[[Any], Any],
) -> None:
    value = np.array([[3.0, 1.0, 2.0], [0.5, -1.0, 4.0]])
    spec = ad.ArraySpec(value.shape, value.dtype)

    def loss(x: Any) -> Any:
        return np.sum(operation(x) ** 3)

    primal, _ = ad.jvp(operation)(value, tangents=np.ones_like(value))
    np.testing.assert_array_equal(primal, operation(value))
    assert_staged_round_trip(operation, value)
    np.testing.assert_allclose(ad.grad(ad.stage(loss, specs=(spec,)))(value), ad.grad(loss)(value))


@st.composite
def _truth_reduction(draw: st.DrawFn) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    shape = draw(hnp.array_shapes(max_dims=3, min_side=0, max_side=3))
    value = draw(hnp.arrays(np.float64, shape, elements=st.sampled_from((0.0, 2.0, np.nan))))
    axes = st.integers(-len(shape), len(shape) - 1) | st.lists(
        st.integers(0, len(shape) - 1), unique=True
    ).map(tuple)
    options = draw(
        st.fixed_dictionaries(
            {},
            optional={
                "axis": st.none() | axes,
                "keepdims": st.booleans(),
                "where": hnp.arrays(np.bool_, shape),
            },
        )
    )
    return value, options


_MASKED_ROWS = (
    np.array([[0.0, 2.0, np.nan], [1.0, 0.0, np.nan]]),
    {"axis": 1, "keepdims": True, "where": np.array([[True, False, True], [False, True, True]])},
)


@given(case=_truth_reduction(), reduce=st.sampled_from((np.all, np.any)))
@example(case=_MASKED_ROWS, reduce=np.all)
@example(case=_MASKED_ROWS, reduce=np.any)
def test_truth_reductions_are_numpys_piecewise_constants(
    case: tuple[np.ndarray[Any, Any], dict[str, Any]],
    reduce: Callable[..., Any],
) -> None:
    value, options = case

    primal, tangent = ad.jvp(lambda x: reduce(x, **options))(value, tangents=np.ones_like(value))

    expected = reduce(value, **options)
    assert np.shape(primal) == np.shape(expected)
    np.testing.assert_array_equal(primal, expected)
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


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda x: np.take_along_axis(x, np.zeros_like(x, dtype=int)), "requires axis"),
        (lambda x: np.take_along_axis(x, np.zeros_like(x, dtype=int), axis=None), "axis=None"),
        (lambda x: np.repeat(x, [1, 2, 1], axis=1), "only scalar repeats"),
        (lambda x: np.tile(x, 2.5), "int or tuple of ints"),
    ],
    ids=("take-axis-missing", "take-axis-none", "repeat", "tile"),
)
def test_shape_and_selection_controls_report_unsupported_public_forms(
    operation: Callable[[Any], Any],
    message: str,
) -> None:
    value = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


@pytest.mark.parametrize(
    ("update", "message", "exception"),
    [
        (
            lambda out: np.put(out, [0], [1.0], mode="invalid"),
            "clipmode must be",
            ad.TracingError,
        ),
        (
            lambda out: np.put(out, [20], [1.0]),
            "index is out of bounds",
            ad.TracingError,
        ),
        (
            lambda out: np.put(out, [0], np.array([])),
            "empty replacements",
            ad.TracingError,
        ),
        (
            lambda out: np.put_along_axis(out, np.array([0, 1]), [2.0, 3.0], axis=1),
            "same number of dimensions",
            ValueError,
        ),
        (
            lambda out: np.put_along_axis(
                out,
                np.zeros((3, 1), dtype=int),
                np.ones((3, 1)),
                axis=1,
            ),
            "broadcast against",
            ValueError,
        ),
        (
            lambda out: np.put_along_axis(
                out,
                np.zeros((2, 2), dtype=int),
                np.ones((3, 2)),
                axis=1,
            ),
            "shape mismatch",
            ValueError,
        ),
        (
            lambda out: np.put_along_axis(
                out,
                np.array([[0, 3], [0, 1]]),
                1.0,
                axis=1,
            ),
            "index is out of bounds",
            ad.TracingError,
        ),
    ],
    ids=(
        "put-mode",
        "put-bounds",
        "put-empty",
        "axis-rank",
        "axis-broadcast",
        "values",
        "axis-bounds",
    ),
)
def test_mutation_controls_report_invalid_public_forms(
    update: Callable[[Any], Any],
    message: str,
    exception: type[Exception],
) -> None:
    def apply(value: Any) -> Any:
        result = value.copy()
        update(result)
        return result

    value = np.arange(6.0).reshape(2, 3)
    with pytest.raises(exception, match=message):
        ad.jvp(apply)(value, tangents=np.ones_like(value))


@pytest.mark.parametrize(
    ("mode", "positions"),
    [(None, [2, 0]), (2, [2, 0]), (b"wrap", [5, -6]), (np.int64(1), [5, -6]), (0, [5, -6])],
    ids=("none-raise", "integer-raise", "bytes-wrap", "numpy-integer-wrap", "integer-clip"),
)
def test_take_binds_numpy_mode_spellings_in_every_lifetime(
    mode: object,
    positions: list[int],
) -> None:
    values = np.arange(12.0).reshape(3, 4)
    indices = np.array(positions)

    def take(array: Any) -> Any:
        return np.take(array, indices, axis=1, mode=cast("Any", mode))

    def loss(array: Any) -> Any:
        return np.sum(take(array) ** 2)

    primal, _ = ad.jvp(take)(values, tangents=np.ones_like(values))
    np.testing.assert_array_equal(primal, take(values))
    assert_staged_round_trip(take, values)
    staged_loss = ad.stage(loss, specs=(ad.ArraySpec(values.shape, values.dtype),))
    np.testing.assert_allclose(ad.grad(staged_loss)(values), ad.grad(loss)(values))
