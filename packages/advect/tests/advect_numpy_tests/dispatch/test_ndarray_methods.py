"""NumPy 2.3 contracts for user-visible ndarray methods."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

import advect as ad
from advect_numpy_tests._assertions import assert_staged_round_trip, assert_tree_close


@pytest.mark.parametrize("order", ["A", "C", "F", "K", "a", "c", "f", "k", None])
def test_copy_order_matches_numpy_in_dynamic_and_staged_execution(order: str | None) -> None:
    value = np.arange(12.0).reshape(3, 4)[:, ::-1]
    direction = np.linspace(0.1, 1.2, 12).reshape(3, 4)

    def apply(x: Any) -> Any:
        return x.copy(order=order)

    primal, tangent = ad.jvp(apply)(value, tangents=direction)
    expected = value.copy(order=order)
    np.testing.assert_array_equal(primal, expected)
    np.testing.assert_array_equal(tangent, direction.copy(order=order))
    assert primal.flags.c_contiguous == expected.flags.c_contiguous
    assert primal.flags.f_contiguous == expected.flags.f_contiguous

    program = ad.stage(apply, specs=(ad.ArraySpec(value.shape, value.dtype),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    for staged in (program, restored):
        result = staged(value)
        np.testing.assert_array_equal(result, expected)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous


def test_numpy_copy_preserves_order_and_rejects_subclass_contract() -> None:
    value = np.arange(6.0).reshape(2, 3)[:, ::-1]
    program = ad.stage(
        lambda x: np.copy(x, order="F"),
        specs=(ad.ArraySpec(value.shape, value.dtype),),
    )
    result = program(value)

    np.testing.assert_array_equal(result, value)
    assert result.flags.f_contiguous

    with pytest.raises(ad.TracingError, match="subok=True"):
        ad.jvp(lambda x: np.copy(x, subok=True))(
            value,
            tangents=np.ones_like(value),
        )
    with pytest.raises(ad.TracingError, match="subok=True"):
        ad.stage(
            lambda x: np.copy(x, subok=True),
            specs=(ad.ArraySpec(value.shape, value.dtype),),
        )


def test_reshape_copy_keyword_matches_numpy_2_3() -> None:
    value = np.arange(6.0).reshape(2, 3)
    direction = np.linspace(0.1, 0.6, 6).reshape(2, 3)

    def apply(x: Any) -> Any:
        return x.reshape((3, 2), copy=True)

    primal, tangent = ad.jvp(apply)(value, tangents=direction)
    np.testing.assert_array_equal(primal, value.reshape((3, 2), copy=True))
    np.testing.assert_array_equal(tangent, direction.reshape((3, 2), copy=True))

    assert_staged_round_trip(apply, value, rtol=0.0)


def test_reshape_without_shape_matches_numpy_type_error_in_both_modes() -> None:
    value = np.array([1.0])

    with pytest.raises(TypeError, match="shape"):
        ad.jvp(lambda x: x.reshape())(value, tangents=np.ones_like(value))
    with pytest.raises(TypeError, match="shape"):
        ad.stage(
            lambda x: x.reshape(),
            specs=(ad.ArraySpec(value.shape, value.dtype),),
        )


def test_sum_initial_is_a_live_differentiable_operand_in_both_modes() -> None:
    value = np.arange(6.0).reshape(2, 3)
    initial = np.array(0.25)
    value_direction = np.ones_like(value)
    initial_direction = np.array(2.0)

    def apply(x: Any, start: Any) -> Any:
        return x.sum(axis=1, initial=start)

    primal, tangent = ad.jvp(apply, argnums=(0, 1))(
        value,
        initial,
        tangents=(value_direction, initial_direction),
    )
    np.testing.assert_allclose(primal, np.sum(value, axis=1, initial=initial))
    np.testing.assert_allclose(tangent, np.full(2, 5.0))

    assert_staged_round_trip(apply, value, initial)


def test_mean_where_remains_a_numpy_owned_staged_method() -> None:
    value = np.arange(6.0).reshape(2, 3)
    mask = np.array([[True, False, True], [False, True, True]])

    def apply(x: Any) -> Any:
        return x.mean(axis=1, where=mask)

    assert_staged_round_trip(apply, value)


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

    assert_tree_close(primal, operation(value))
    assert_tree_close(tangent, operation(direction))


@pytest.mark.parametrize(
    "exponent",
    [
        2,
        3,
        True,
        pytest.param(
            2.0,
            marks=pytest.mark.skipif(
                np.lib.NumpyVersion(np.__version__) < "2.3.0",
                reason="NumPy before 2.3 squares an array for any scalar exponent of 2",
            ),
        ),
    ],
    ids=repr,
)
def test_power_operator_on_a_bool_array_matches_ndarray(exponent: object) -> None:
    # ndarray squares an array raised to the Python int 2, so a bool array
    # gives int8 where np.power gives int64; a NumPy bool scalar does not.
    def apply(array: Any) -> tuple[Any, Any, Any]:
        mask = (array > 0) ** exponent
        return mask, array * mask, (np.sum(array) > 0) ** exponent

    value = np.asarray([1.5, -2.0, 0.0], dtype=np.float32)
    primal, _tangent = ad.jvp(apply)(value, tangents=np.ones_like(value))
    assert_tree_close(primal, apply(value), rtol=0.0)
    assert_staged_round_trip(apply, value, rtol=0.0)


def test_astype_default_preserves_ndarray_subclasses() -> None:
    class ArraySubclass(np.ndarray):
        pass

    value = np.arange(4.0).view(ArraySubclass)
    primal, tangent = ad.jvp(lambda x: x.astype(np.float64))(
        value,
        tangents=np.ones_like(value),
    )

    assert isinstance(primal, ArraySubclass)
    np.testing.assert_array_equal(tangent, np.ones_like(value))
