"""Public contracts for miscellaneous NumPy array-function forms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

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


def test_nan_to_num_preserves_an_integer_input_with_traced_replacements() -> None:
    value = np.array([1, 2, 3], dtype=np.int64)

    primal, tangent = ad.jvp(
        lambda x, replacement: np.nan_to_num(x, nan=replacement),
        argnums=(0, 1),
    )(
        value,
        np.array(5.0),
        tangents=(np.ones_like(value), np.array(0.25)),
    )

    np.testing.assert_array_equal(primal, value)
    assert primal.dtype == value.dtype
    np.testing.assert_array_equal(tangent, np.zeros_like(value))


@pytest.mark.parametrize("mode", ["wrap", "clip"])
def test_take_modes_select_primal_and_tangent(mode: str) -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.arange(10.0, 16.0).reshape(2, 3)
    indices = np.array([-1, 4])

    primal, tangent = ad.jvp(lambda x: np.take(x, indices, axis=1, mode=mode))(
        value, tangents=direction
    )

    np.testing.assert_array_equal(primal, np.take(value, indices, axis=1, mode=mode))
    np.testing.assert_array_equal(tangent, np.take(direction, indices, axis=1, mode=mode))


def test_take_along_axis_accepts_a_positional_negative_axis() -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.arange(10.0, 16.0).reshape(2, 3)
    indices = np.array([[2, 0], [1, 1]])

    primal, tangent = ad.jvp(lambda x: np.take_along_axis(x, indices, -1))(
        value,
        tangents=direction,
    )

    np.testing.assert_array_equal(primal, np.take_along_axis(value, indices, axis=-1))
    np.testing.assert_array_equal(tangent, np.take_along_axis(direction, indices, axis=-1))


def test_diff_differentiates_prepend_and_append_operands() -> None:
    value = np.arange(6.0).reshape(2, 3)
    prepend = np.array([[10.0], [20.0]])
    append = np.array([[30.0], [40.0]])
    directions = (
        np.arange(10.0, 16.0).reshape(2, 3),
        np.array([[1.0], [2.0]]),
        np.array([[3.0], [4.0]]),
    )

    def difference(x: Any, before: Any, after: Any) -> Any:
        return np.diff(x, n=2, axis=1, prepend=before, append=after)

    primal, tangent = ad.jvp(difference, argnums=(0, 1, 2))(
        value,
        prepend,
        append,
        tangents=directions,
    )

    np.testing.assert_array_equal(primal, difference(value, prepend, append))
    np.testing.assert_array_equal(tangent, difference(*directions))


def test_static_diff_boundaries_survive_staging() -> None:
    prepend = np.array([[10.0, 20.0, 30.0]])
    append = np.array([[40.0, 50.0, 60.0]])

    def difference(x: Any) -> Any:
        return np.diff(x, axis=0, prepend=prepend, append=append)

    value = np.arange(6.0).reshape(2, 3)
    program = ad.stage(difference, value)

    np.testing.assert_array_equal(program(value), difference(value))
    np.testing.assert_array_equal(
        ad.StagedProgram.from_dict(program.to_dict())(value), difference(value)
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: np.repeat(x, repeats=2, axis=0),
        lambda x: np.tile(x, reps=np.array([2, 1], dtype=np.int64)),
    ],
    ids=("repeat", "tile"),
)
def test_repeat_and_tile_apply_the_same_layout_to_tangents(
    operation: Callable[[Any], Any],
) -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.arange(10.0, 16.0).reshape(2, 3)

    primal, tangent = ad.jvp(operation)(value, tangents=direction)

    np.testing.assert_array_equal(primal, operation(value))
    np.testing.assert_array_equal(tangent, operation(direction))


def test_full_preserves_options_and_the_fill_derivative() -> None:
    fill = np.array(2.0)
    direction = np.array(0.25)

    def create(value: Any) -> Any:
        return np.full(
            (2, 3),
            value,
            dtype=np.float32,
            order="F",
            device="cpu",
            like=value,
        )

    primal, tangent = ad.jvp(create)(fill, tangents=direction)

    np.testing.assert_array_equal(primal, create(fill))
    np.testing.assert_array_equal(tangent, np.full((2, 3), direction, dtype=np.float32))
    assert primal.flags.f_contiguous


def test_full_like_uses_shape_options_and_only_the_fill_derivative() -> None:
    anchor = np.arange(6.0).reshape(2, 3)
    fill = np.array(2.0)

    def create(array: Any, value: Any) -> Any:
        return np.full_like(
            array,
            value,
            dtype=np.float32,
            order="F",
            subok=False,
            shape=(3, 2),
            device="cpu",
        )

    primal, tangent = ad.jvp(create, argnums=(0, 1))(
        anchor,
        fill,
        tangents=(np.ones_like(anchor), np.array(0.25)),
    )

    np.testing.assert_array_equal(primal, create(anchor, fill))
    np.testing.assert_array_equal(tangent, np.full((3, 2), 0.25, dtype=np.float32))
    assert primal.flags.f_contiguous


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
    ],
    ids=("sort", "partition"),
)
def test_ordering_operations_apply_the_primal_permutation_to_tangents(
    operation: Callable[[Any], Any],
) -> None:
    value = np.array([[2.0, 7.0, 1.0], [5.0, 4.0, 9.0]])
    direction = np.array([[0.2, -0.3, 0.5], [0.7, 0.1, -0.2]])

    primal, tangent = ad.jvp(operation)(value, tangents=direction)
    step = 1e-6
    numerical = (operation(value + step * direction) - operation(value - step * direction)) / (
        2 * step
    )

    np.testing.assert_array_equal(primal, operation(value))
    np.testing.assert_allclose(tangent, numerical, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("mode", ["wrap", "clip"])
def test_put_modes_preserve_last_write_derivatives(mode: str) -> None:
    indices = np.array([-1, 6, 1])

    def update(array: Any, replacements: Any) -> Any:
        result = array.copy()
        np.put(result, indices, replacements, mode=mode)
        return result

    value = np.arange(4.0)
    replacements = np.array([10.0, 20.0])
    directions = (np.arange(0.1, 4.1), np.array([0.2, 0.3]))
    primal, tangent = ad.jvp(update, argnums=(0, 1))(
        value,
        replacements,
        tangents=directions,
    )

    np.testing.assert_array_equal(primal, update(value, replacements))
    np.testing.assert_array_equal(tangent, update(*directions))


def test_put_along_axis_accepts_axis_none() -> None:
    indices = np.array([0, 3])

    def update(array: Any, replacements: Any) -> Any:
        result = array.copy()
        np.put_along_axis(result, indices, replacements, axis=None)
        return result

    value = np.arange(6.0).reshape(2, 3)
    replacements = np.array([10.0, 20.0])
    directions = (np.arange(10.0, 16.0).reshape(2, 3), np.array([0.2, 0.3]))
    primal, tangent = ad.jvp(update, argnums=(0, 1))(
        value,
        replacements,
        tangents=directions,
    )

    np.testing.assert_array_equal(primal, update(value, replacements))
    np.testing.assert_array_equal(tangent, update(*directions))


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda x: np.take(x, [0], mode="invalid"), "mode must be"),
        (lambda x: np.take_along_axis(x, np.zeros_like(x, dtype=int)), "requires axis"),
        (lambda x: np.take_along_axis(x, np.zeros_like(x, dtype=int), axis=None), "axis=None"),
        (lambda x: np.diff(x, n=-1), "requires n >= 0"),
        (lambda x: np.repeat(x, [1, 2, 1], axis=1), "only scalar repeats"),
        (lambda x: np.tile(x, 2.5), "int or tuple of ints"),
    ],
    ids=("take-mode", "take-axis-missing", "take-axis-none", "diff-n", "repeat", "tile"),
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
