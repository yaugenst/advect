"""Additional public contracts for NumPy shape and creation metadata."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _assert_staged_matches(
    operation: Callable[..., Any],
    *values: np.ndarray[Any, Any],
) -> None:
    program = ad.stage(
        operation,
        specs=tuple(ad.ArraySpec(value.shape, value.dtype) for value in values),
    )
    expected = operation(*values)
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = staged(*values)
        actual_items = actual if isinstance(actual, tuple) else (actual,)
        expected_items = expected if isinstance(expected, tuple) else (expected,)
        for item, reference in zip(actual_items, expected_items, strict=True):
            np.testing.assert_allclose(item, reference)


def test_angle_accepts_positional_degree_metadata() -> None:
    value = np.array([1.0 + 2.0j, -2.0 + 1.0j])
    direction = np.array([0.2 - 0.1j, 0.1 + 0.3j])

    def angle(x: Any) -> Any:
        return np.angle(x, True)  # noqa: FBT003 - exercise NumPy's positional form

    primal, tangent = ad.jvp(angle)(value, tangents=direction)
    epsilon = 1e-6

    np.testing.assert_allclose(primal, angle(value))
    np.testing.assert_allclose(
        tangent,
        (angle(value + epsilon * direction) - angle(value - epsilon * direction)) / (2 * epsilon),
    )
    _assert_staged_matches(angle, value)


def test_copy_take_and_sort_accept_positional_metadata() -> None:
    value = np.array([[4.0, 1.0, 7.0], [2.0, 6.0, 3.0]])
    direction = np.arange(0.1, 0.7, 0.1).reshape(2, 3)

    def arrange(x: Any) -> tuple[Any, ...]:
        return (
            np.copy(x, "F", False),  # noqa: FBT003 - exercise NumPy's positional form
            np.take(x, [-1, 3], 1, None, "wrap"),
            np.sort(x, 0, "mergesort"),
        )

    primal, tangent = ad.jvp(arrange)(value, tangents=direction)
    epsilon = 1e-6
    expected = arrange(value)
    numerical = tuple(
        (upper - lower) / (2 * epsilon)
        for upper, lower in zip(
            arrange(value + epsilon * direction),
            arrange(value - epsilon * direction),
            strict=True,
        )
    )

    for actual, reference in zip(primal, expected, strict=True):
        np.testing.assert_allclose(actual, reference)
    for actual, reference in zip(tangent, numerical, strict=True):
        np.testing.assert_allclose(actual, reference)


def test_shape_operations_preserve_positional_static_metadata() -> None:
    value = np.arange(1.0, 10.0).reshape(3, 3)
    direction = np.arange(0.1, 1.0, 0.1).reshape(3, 3)

    def reshape(x: Any) -> tuple[Any, ...]:
        return (
            np.reshape(x, (1, 9), "F"),
            np.diag(x, -1),
            np.trace(x, 1, 0, 1, np.float64),
            np.diff(x, 2, 1, 0.0, 10.0),
        )

    primal, tangent = ad.jvp(reshape)(value, tangents=direction)
    expected = reshape(value)
    expected_tangent = (
        np.reshape(direction, (1, 9), "F"),
        np.diag(direction, -1),
        np.trace(direction, 1, 0, 1, np.float64),
        np.diff(direction, 2, 1, 0.0, 0.0),
    )

    for actual, reference in zip(primal, expected, strict=True):
        np.testing.assert_allclose(actual, reference)
    for actual, reference in zip(tangent, expected_tangent, strict=True):
        np.testing.assert_allclose(actual, reference)


def test_axis_metadata_accepts_numpy_integer_and_scalar_array_forms() -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.arange(0.1, 0.7, 0.1).reshape(2, 3)

    def arrange(x: Any) -> tuple[Any, ...]:
        return (
            np.moveaxis(x, np.int64(0), 1),
            np.tile(x, np.asarray(2)),
        )

    primal, tangent = ad.jvp(arrange)(value, tangents=direction)
    for actual, reference in zip(primal, arrange(value), strict=True):
        np.testing.assert_array_equal(actual, reference)
    for actual, reference in zip(tangent, arrange(direction), strict=True):
        np.testing.assert_array_equal(actual, reference)


def test_full_constructors_accept_positional_options() -> None:
    anchor = np.arange(6.0).reshape(2, 3)
    fill = np.asarray(2.0)

    def create(array: Any, value: Any) -> tuple[Any, ...]:
        return (
            np.full((2, 3), value, np.float32, "F", like=array),
            np.full_like(
                array,
                value,
                np.float32,
                "F",
                False,  # noqa: FBT003 - exercise NumPy's positional form
                (3, 2),
                device="cpu",
            ),
        )

    primal, tangent = ad.jvp(create, argnums=(0, 1))(
        anchor,
        fill,
        tangents=(np.ones_like(anchor), np.asarray(0.25)),
    )

    for actual, reference in zip(primal, create(anchor, fill), strict=True):
        np.testing.assert_array_equal(actual, reference)
        assert actual.flags.f_contiguous
    for actual in tangent:
        np.testing.assert_array_equal(actual, np.full(actual.shape, 0.25, dtype=np.float32))
    _assert_staged_matches(create, anchor, fill)


def test_gradient_broadcasts_one_scalar_spacing_across_selected_axes() -> None:
    value = np.arange(9.0).reshape(3, 3) ** 2
    direction = np.arange(0.1, 1.0, 0.1).reshape(3, 3)

    def gradient(x: Any) -> tuple[Any, ...]:
        return tuple(np.gradient(x, 2.0, axis=(0, 1), edge_order=2))

    primal, tangent = ad.jvp(gradient)(value, tangents=direction)
    for actual, reference in zip(primal, gradient(value), strict=True):
        np.testing.assert_allclose(actual, reference)
    for actual, reference in zip(tangent, gradient(direction), strict=True):
        np.testing.assert_allclose(actual, reference)
    _assert_staged_matches(gradient, value)


@pytest.mark.parametrize("operation", [np.split, np.array_split], ids=("split", "array-split"))
def test_split_accepts_a_positional_axis(operation: Callable[..., Any]) -> None:
    value = np.arange(12.0).reshape(3, 4)
    direction = np.arange(0.1, 1.3, 0.1).reshape(3, 4)

    def split(x: Any) -> list[Any]:
        return operation(x, 2, 1)

    primal, tangent = ad.jvp(split)(value, tangents=direction)
    for actual, reference in zip(primal, split(value), strict=True):
        np.testing.assert_array_equal(actual, reference)
    for actual, reference in zip(tangent, split(direction), strict=True):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize(
    ("operation", "value"),
    [
        (np.split, np.arange(8.0)),
        (np.array_split, np.arange(7.0)),
        (np.hsplit, np.arange(8.0).reshape(2, 4)),
        (np.vsplit, np.arange(8.0).reshape(4, 2)),
        (np.dsplit, np.arange(8.0).reshape(1, 2, 4)),
    ],
    ids=("split", "array-split", "hsplit", "vsplit", "dsplit"),
)
def test_split_requires_the_data_array_to_own_the_trace(
    operation: Callable[..., Any],
    value: np.ndarray[Any, Any],
) -> None:
    with pytest.raises(ad.TracingError, match="requires a traced first argument"):
        ad.jvp(lambda sections: operation(value, sections))(
            np.asarray(2),
            tangents=np.asarray(0),
        )


@pytest.mark.skipif(
    "descending" not in inspect.signature(np.sort).parameters,
    reason="NumPy added sort(descending=...) in 2.5",
)
def test_sort_rejects_the_unimplemented_descending_option() -> None:
    value = np.arange(6.0).reshape(2, 3)

    with pytest.raises(ad.TracingError, match="descending"):
        ad.jvp(lambda x: np.sort(x, descending=True))(
            value,
            tangents=np.ones_like(value),
        )


@pytest.mark.parametrize(
    ("boundary", "widths"),
    [
        (np.asarray(2.0), ((1, 0), (0, 2))),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), ((1, 2), (2, 1))),
        (np.asarray(2.0), ((1, 2),)),
    ],
    ids=("scalar", "per-axis", "broadcast-width"),
)
def test_pad_differentiates_boundary_parameters(
    boundary: np.ndarray[Any, Any],
    widths: tuple[tuple[int, int], ...],
) -> None:
    value = np.arange(6.0).reshape(2, 3)

    def pad(x: Any, edges: Any) -> Any:
        return np.pad(x, widths, mode="constant", constant_values=edges)

    directions = (np.full_like(value, 0.1), np.full_like(boundary, 0.25))
    primal, tangent = ad.jvp(pad, argnums=(0, 1))(
        value,
        boundary,
        tangents=directions,
    )

    np.testing.assert_array_equal(primal, pad(value, boundary))
    np.testing.assert_array_equal(
        tangent,
        np.pad(directions[0], widths, mode="constant", constant_values=directions[1]),
    )
    np.testing.assert_array_equal(
        ad.grad(lambda x: np.sum(pad(x, boundary)))(value),
        np.ones_like(value),
    )


def test_statistical_pad_defaults_to_the_complete_edge_region() -> None:
    value = np.arange(1.0, 7.0).reshape(2, 3)
    direction = np.arange(0.1, 0.7, 0.1).reshape(2, 3)

    def pad(x: Any) -> Any:
        return np.pad(x, ((1, 2), (2, 1)), mode="mean")

    primal, tangent = ad.jvp(pad)(value, tangents=direction)

    np.testing.assert_allclose(primal, pad(value))
    np.testing.assert_allclose(tangent, pad(direction))


@pytest.mark.parametrize(
    ("operation", "value", "error", "match"),
    [
        pytest.param(
            lambda x: np.pad(x, 1, mode="empty"),
            np.arange(3.0),
            ad.TracingError,
            "not differentiable",
            id="unsupported-mode",
        ),
        pytest.param(
            lambda x: np.pad(x, (-1, 1), mode="edge"),
            np.arange(3.0),
            ad.TracingError,
            "cannot contain negative",
            id="negative-width",
        ),
        pytest.param(
            lambda x: np.pad(x, 1, mode="reflect", reflect_type="neither"),
            np.arange(3.0),
            ad.TracingError,
            "reflect_type",
            id="reflect-type",
        ),
        pytest.param(
            lambda x: np.pad(x, 1, mode="edge"),
            np.empty(0),
            ValueError,
            "can't extend empty axis",
            id="empty-edge",
        ),
        pytest.param(
            lambda x: np.pad(
                x,
                ((1, 1), (1, 1)),
                mode="linear_ramp",
                end_values=(1.0, 2.0, 3.0),
            ),
            np.arange(6.0).reshape(2, 3),
            ad.TracingError,
            "end_values shape",
            id="static-boundary-shape",
        ),
        pytest.param(
            lambda x: np.pad(
                x,
                ((1, 1), (1, 1)),
                mode="mean",
                stat_length=(1, 2, 3),
            ),
            np.arange(6.0).reshape(2, 3),
            ad.TracingError,
            "stat_length shape",
            id="stat-length-shape",
        ),
        pytest.param(
            lambda x: np.gradient(x, edge_order=3),
            np.arange(9.0).reshape(3, 3),
            ad.TracingError,
            "edge_order must be 1 or 2",
            id="gradient-edge-order",
        ),
        pytest.param(
            lambda x: np.gradient(x, 1.0, 2.0, 3.0, axis=(0, 1)),
            np.arange(9.0).reshape(3, 3),
            ad.TracingError,
            "one spacing per gradient axis",
            id="gradient-spacing-count",
        ),
    ],
)
def test_shape_and_creation_metadata_rejects_invalid_public_forms(
    operation: Callable[[Any], Any],
    value: np.ndarray[Any, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


def test_pad_rejects_a_live_boundary_with_the_wrong_shape() -> None:
    value = np.arange(6.0).reshape(2, 3)
    boundary = np.arange(3.0)

    with pytest.raises(ad.TracingError, match="constant_values shape"):
        ad.jvp(
            lambda x, edge: np.pad(x, 1, mode="constant", constant_values=edge),
            argnums=(0, 1),
        )(
            value,
            boundary,
            tangents=(np.ones_like(value), np.ones_like(boundary)),
        )
