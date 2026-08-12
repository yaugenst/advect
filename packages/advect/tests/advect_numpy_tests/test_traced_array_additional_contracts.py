"""Additional public contracts for NumPy's concrete traced-array boundary."""

from __future__ import annotations

import copy
import operator
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


def _assert_tuple_allclose(actual: tuple[Any, ...], expected: tuple[Any, ...]) -> None:
    for item, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(item, reference)


def test_properties_copy_protocols_and_transpose_forms_preserve_tangents() -> None:
    value = np.asarray([[1 + 2j, 3 - 1j], [-2 + 0.5j, 4 + 3j]])
    direction = np.asarray([[0.2 - 0.1j, -0.3 + 0.4j], [0.5 + 0.2j, -0.1 - 0.6j]])

    def operation(array: Any) -> tuple[Any, ...]:
        return (
            array.real,
            array.imag,
            copy.copy(array),
            copy.deepcopy(array),
            array.transpose((1, 0)),
            array.transpose(1, 0),
        )

    primal, tangent = ad.jvp(operation)(value, tangents=direction)

    _assert_tuple_allclose(primal, operation(value))
    _assert_tuple_allclose(tangent, operation(direction))


def test_ufunc_out_rejects_a_destination_view_before_mutation() -> None:
    value = np.asarray([1 + 2j, 3 - 1j])

    def operation(array: Any) -> Any:
        owned = array.copy()
        destination = owned.real
        np.add(destination, 1.0, out=destination)
        return destination

    with pytest.raises(ad.MutationError, match=r"ufunc out=.*traced view"):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


@pytest.mark.parametrize(
    ("operation", "value", "other", "expected_tangent"),
    [
        (operator.itruediv, np.asarray([2.0, 4.0]), 2.0, np.asarray([0.5, 0.5])),
        (operator.ifloordiv, np.asarray([2.5, 5.5]), 2.0, np.zeros(2)),
        (operator.imod, np.asarray([2.5, 5.5]), 2.0, np.ones(2)),
        (operator.ipow, np.asarray([2.0, 3.0]), 2.0, np.asarray([4.0, 6.0])),
        (
            operator.imatmul,
            np.eye(2),
            np.asarray([[2.0, 0.0], [0.0, 3.0]]),
            np.asarray([[2.0, 3.0], [2.0, 3.0]]),
        ),
    ],
    ids=["divide", "floor-divide", "remainder", "power", "matmul"],
)
def test_supported_augmented_operators_functionalize_without_mutating_inputs(
    operation: Callable[[Any, Any], Any],
    value: np.ndarray,
    other: object,
    expected_tangent: np.ndarray,
) -> None:
    original = value.copy()

    def apply(array: Any) -> Any:
        result = array.copy()
        operation(result, other)
        return result

    primal, tangent = ad.jvp(apply)(value, tangents=np.ones_like(value))
    expected = original.copy()
    operation(expected, other)

    np.testing.assert_allclose(primal, expected)
    np.testing.assert_allclose(tangent, expected_tangent)
    np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize(
    ("operation", "other", "name"),
    [
        (operator.iand, np.asarray([1, 6]), "bitwise_and"),
        (operator.ior, np.asarray([1, 6]), "bitwise_or"),
        (operator.ixor, np.asarray([1, 6]), "bitwise_xor"),
        (operator.ilshift, 1, "left_shift"),
        (operator.irshift, 1, "right_shift"),
    ],
)
def test_nondifferentiable_augmented_operators_report_the_canonical_operation(
    operation: Callable[[Any, Any], Any],
    other: object,
    name: str,
) -> None:
    def apply(array: Any) -> Any:
        result = array.copy()
        operation(result, other)
        return result

    value = np.asarray([3, 5])
    with pytest.raises(ad.NoJVPError, match=rf"array\.{name}"):
        ad.jvp(apply)(value, tangents=np.zeros_like(value))


def test_like_constructors_preserve_explicit_copy_dtype_and_boolean_constants() -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.linspace(0.1, 0.6, 6).reshape(2, 3)

    def construct(array: Any) -> tuple[Any, ...]:
        return (
            np.asarray(array, dtype=np.float32, copy=True, like=array),
            np.asarray([True, False], like=array),
            np.asarray(array, order="A", like=array),
        )

    primal, tangent = ad.jvp(construct)(value, tangents=direction)

    np.testing.assert_allclose(primal[0], value.astype(np.float32))
    np.testing.assert_allclose(tangent[0], direction.astype(np.float32))
    np.testing.assert_array_equal(primal[1], [True, False])
    np.testing.assert_array_equal(tangent[1], [False, False])
    np.testing.assert_allclose(primal[2], value)
    np.testing.assert_allclose(tangent[2], direction)


@pytest.mark.parametrize(
    ("operation", "error", "match"),
    [
        (
            lambda array: np.asarray(array, device="gpu", like=array),
            ValueError,
            "Only.*cpu",
        ),
        (
            lambda array: np.asarray(["not", "numeric"], like=array),
            TypeError,
            "numeric and boolean",
        ),
        (
            lambda array: np.asarray(array, dtype=np.float32, copy=False, like=array),
            ValueError,
            "avoid copy",
        ),
        (
            lambda array: np.array(array, ndmin=-1, like=array),
            ValueError,
            "ndmin must be non-negative",
        ),
    ],
    ids=["device", "non-numeric-constant", "copy-false-dtype", "negative-ndmin"],
)
def test_like_constructors_reject_invalid_public_controls(
    operation: Callable[[Any], Any],
    error: type[Exception],
    match: str,
) -> None:
    value = np.arange(4.0)
    with pytest.raises(error, match=match):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


def test_staged_asarray_rejects_unprovable_copy_free_layout_conversion() -> None:
    with pytest.raises(ad.TracingError, match="layout-constraining order"):
        ad.stage(
            lambda array: np.asarray(array, order="F", copy=False, like=array),
            specs=(ad.ArraySpec((2, 3), "float64"),),
        )


def test_list_advanced_indexing_preserves_selected_tangents() -> None:
    value = np.arange(12.0).reshape(3, 4)
    direction = np.linspace(0.1, 1.2, 12).reshape(3, 4)

    primal, tangent = ad.jvp(lambda array: array[:, [3, 1]])(
        value,
        tangents=direction,
    )

    np.testing.assert_array_equal(primal, value[:, [3, 1]])
    np.testing.assert_array_equal(tangent, direction[:, [3, 1]])


def test_traced_advanced_index_rejects_non_discrete_dtype() -> None:
    value = np.arange(6.0).reshape(3, 2)
    index = np.asarray([0.0, 1.0])

    with pytest.raises(ad.TracingError, match="integer or boolean dtype"):
        ad.jvp(lambda array, positions: array[positions], argnums=(0, 1))(
            value,
            index,
            tangents=(np.ones_like(value), np.zeros_like(index)),
        )


@pytest.mark.parametrize("update", ["add", "assign", "direct"])
def test_nested_traces_reject_functional_updates_from_an_outer_recorder(update: str) -> None:
    inner_value = np.asarray([3.0, 4.0])

    def outer(outer_value: Any) -> Any:
        def inner(value: Any) -> Any:
            result = value.copy()
            if update == "add":
                result[:1] += outer_value[:1]
            elif update == "assign":
                result[0] = outer_value[0]
            else:
                result += outer_value
            return np.sum(result)

        return ad.grad(inner)(inner_value)

    with pytest.raises(ad.TracingError, match="different trace context"):
        ad.jvp(outer)(np.asarray([1.0, 2.0]), tangents=np.ones(2))


def test_debug_mode_preserves_the_unsupported_ufunc_error() -> None:
    with ad.debug(), pytest.raises(ad.TracingError, match="Unsupported ufunc: gcd"):
        ad.jvp(lambda array: np.gcd(array, 2))(
            np.asarray([2, 3]),
            tangents=np.zeros(2, dtype=int),
        )


def test_multi_output_ufunc_rejects_out_destinations_explicitly() -> None:
    def operation(array: Any) -> Any:
        fractional = np.zeros_like(array)
        integral = np.zeros_like(array)
        return np.modf(array, out=(fractional, integral))

    with pytest.raises(ad.TracingError, match="Only single-output out="):
        ad.jvp(operation)(np.asarray([1.25, -2.5]), tangents=np.ones(2))
