"""Public contracts for common NumPy array-function emission paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import advect as ad
from advect_numpy_tests._assertions import (
    assert_jvp_matches_central_difference,
    assert_staged_round_trip,
    assert_tree_close,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_nested_jvp_can_use_an_operand_from_the_active_outer_trace() -> None:
    value = np.array([1.0, 2.0])
    direction = np.array([0.1, 0.2])

    def nested(outer: Any) -> Any:
        primal, tangent = ad.jvp(lambda inner: np.concatenate((inner, outer)))(
            outer,
            tangents=np.ones_like(value),
        )
        return np.sum(primal) + np.sum(tangent)

    primal, tangent = ad.jvp(nested)(value, tangents=direction)

    np.testing.assert_allclose(primal, 2 * np.sum(value) + value.size)
    np.testing.assert_allclose(tangent, 2 * np.sum(direction))


def test_clip_supports_static_array_and_one_sided_bounds() -> None:
    value = np.array([-2.0, -0.2, 0.7, 3.0])
    direction = np.array([0.3, -0.4, 0.2, 0.1])
    lower = np.array([-1.0, -0.5, 0.0, 1.0])
    upper = np.array([0.0, 0.5, 1.0, 2.0])

    def bounded(x: Any) -> Any:
        return np.clip(x, lower, upper)

    primal, tangent = ad.jvp(bounded)(value, tangents=direction)
    np.testing.assert_array_equal(primal, bounded(value))
    np.testing.assert_array_equal(
        tangent,
        np.where((value > lower) & (value < upper), direction, 0.0),
    )

    assert_staged_round_trip(bounded, value, rtol=0.0)

    def minimum_only(x: Any) -> Any:
        return np.clip(x, min=-0.5)

    def maximum_only(x: Any) -> Any:
        return np.clip(x, max=1.0)

    one_sided_cases = (
        (minimum_only, np.where(value > -0.5, direction, 0.0)),
        (maximum_only, np.where(value < 1.0, direction, 0.0)),
    )
    for operation, expected_tangent in one_sided_cases:
        result, tangent = ad.jvp(operation)(value, tangents=direction)
        np.testing.assert_array_equal(result, operation(value))
        np.testing.assert_array_equal(tangent, expected_tangent)
        assert result.shape == tangent.shape == value.shape
        assert result.dtype == tangent.dtype == value.dtype


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param((np.float64(-1.0), np.float64(1.0)), id="numpy-scalars"),
        pytest.param((np.array(-1.0), np.array(1.0)), id="0d-arrays"),
        pytest.param((np.int64(-1), None), id="numpy-integer-lower-only"),
        pytest.param((-1.0, 1.0), id="weak-python-floats"),
    ],
)
def test_clip_scalar_bounds_promote_the_result_as_numpy_does(bounds: tuple[Any, Any]) -> None:
    # Under NEP 50 only a Python scalar bound is weak; a NumPy scalar or 0-d
    # bound promotes float32 data, dynamically and inside a staged program.
    value = np.array([-2.0, 0.5, 3.0], dtype=np.float32)
    direction = np.array([0.3, -0.4, 0.2], dtype=np.float32)

    def bounded(x: Any) -> Any:
        return np.clip(x, *bounds)

    expected = bounded(value)
    expected_tangent = np.where(expected == value, direction, 0).astype(expected.dtype)
    program = assert_staged_round_trip(bounded, value, rtol=0.0)
    for function in (bounded, program):
        primal, tangent = ad.jvp(function)(value, tangents=direction)
        assert_tree_close(primal, expected, rtol=0.0)
        assert_tree_close(tangent, expected_tangent, rtol=0.0)


def test_clip_reports_complex_scalar_bounds() -> None:
    value = np.arange(3.0)
    with pytest.raises(ad.TracingError, match="got scalar complex"):
        ad.jvp(lambda x: np.clip(x, 1j, 1.0))(value, tangents=np.ones_like(value))


@pytest.mark.parametrize("operation", [np.max, np.nanmax], ids=("max", "nanmax"))
def test_extrema_differentiate_a_dynamic_initial(operation: Any) -> None:
    value = np.array([[-2.0, -1.0], [2.0, 3.0]])
    if operation is np.nanmax:
        value[0, 0] = np.nan
    direction = np.array([[0.1, 0.2], [0.3, 0.4]])
    initial = np.array(0.5)
    initial_direction = np.array(-0.25)

    def reduce(x: Any, boundary: Any) -> Any:
        return operation(x, axis=1, initial=boundary)

    assert_jvp_matches_central_difference(
        reduce, (value, initial), (direction, initial_direction), rtol=1e-6, atol=1e-6
    )
    assert_staged_round_trip(reduce, value, initial)


def test_nanmin_static_initial_and_metadata_survive_staging() -> None:
    value = np.array([[np.nan, 2.0], [0.5, 5.0]])

    def reduce(x: Any) -> Any:
        return np.nanmin(x, axis=1, keepdims=True, initial=1.0)

    primal, tangent = ad.jvp(reduce)(value, tangents=np.ones_like(value))
    np.testing.assert_allclose(primal, reduce(value))
    np.testing.assert_array_equal(tangent, [[0.0], [1.0]])

    assert_staged_round_trip(reduce, value)


def test_variance_static_dtype_and_correction_survive_staging() -> None:
    value = np.array([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]], dtype=np.float64)

    def reduce(x: Any) -> Any:
        return np.nanvar(x, axis=1, dtype=np.float32, ddof=1)

    primal, tangent = ad.jvp(reduce)(value, tangents=np.ones_like(value))
    assert primal.dtype == np.dtype(np.float32)
    assert tangent.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(primal, reduce(value))
    np.testing.assert_allclose(tangent, np.zeros(2, dtype=np.float32), atol=1e-6)

    assert_staged_round_trip(reduce, value)


def test_controlled_variance_honors_requested_accumulator_dtype() -> None:
    value = np.array([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]], dtype=np.float32)
    direction = np.array([[0.2, -0.1, 0.3], [0.4, 0.1, -0.2]], dtype=np.float32)
    mask = np.array([[True, False, True], [True, True, False]])

    def reduce(x: Any) -> Any:
        return np.var(x, axis=1, where=mask, dtype=np.float64, correction=1)

    primal, _ = assert_jvp_matches_central_difference(
        reduce, (value,), (direction,), rtol=2e-3, atol=2e-3, step=1e-4
    )
    assert primal.dtype == np.dtype(np.float64)


def test_controlled_complex_variance_uses_a_real_result_dtype() -> None:
    value = np.array([1 + 2j, 3 - 1j, 2 + 0.5j], dtype=np.complex64)
    direction = np.array([0.2 - 0.1j, -0.3 + 0.4j, 0.1 + 0.2j], dtype=np.complex64)
    mask = np.array([True, False, True])

    def reduce(x: Any) -> Any:
        return np.var(x, where=mask)

    primal, tangent = ad.jvp(reduce)(value, tangents=direction)

    assert primal.dtype == np.dtype(np.float32)
    assert tangent.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(primal, reduce(value))


def test_controlled_integer_mean_uses_numpy_float64_accumulation() -> None:
    value = np.array([[1, 2, 4], [3, 6, 8]], dtype=np.int32)
    mask = np.array([[True, False, True], [True, True, False]])

    def reduce(x: Any) -> Any:
        return np.mean(x, axis=1, where=mask)

    result, tangent = ad.jvp(reduce)(value, tangents=np.ones_like(value))

    assert result.dtype == np.dtype(np.float64)
    assert tangent.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(result, reduce(value))

    assert_staged_round_trip(reduce, value)


def _assert_every_lifetime_matches_numpy(function: Any, value: np.ndarray[Any, Any]) -> None:
    dynamic, _tangent = ad.jvp(function)(value, tangents=np.ones_like(value))
    assert_tree_close(dynamic, function(value), rtol=0.0)
    assert_staged_round_trip(function, value, rtol=0.0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ddof", [2, 3])
@pytest.mark.parametrize("operation", [np.var, np.std, np.nanvar, np.nanstd])
def test_controlled_variance_matches_numpy_without_positive_degrees_of_freedom(
    operation: Any,
    ddof: int,
    dtype: type[np.floating[Any]],
) -> None:
    value = np.array([[0.0, 1.0, 4.0], [2.0, np.nan, 5.0], [1.0, 3.0, 8.0]], dtype=dtype)
    mask = np.array([[True, True, False], [True, True, True], [False, False, False]])

    _assert_every_lifetime_matches_numpy(
        lambda x: operation(x, axis=1, ddof=ddof, where=mask),
        value,
    )
    _assert_every_lifetime_matches_numpy(
        lambda x: operation(x, axis=1, ddof=ddof + 1, mean=np.mean(x, axis=1, keepdims=True)),
        value[[0, 2]],
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("operation", "initial"),
    [(np.nanmax, -5.0), (np.nanmin, 5.0)],
    ids=("nanmax", "nanmin"),
)
def test_nan_extrema_with_where_let_initial_replace_selected_nans(
    operation: Any,
    initial: float,
) -> None:
    value = np.array([[np.nan, np.nan, np.nan], [2.0, np.nan, 1.0]])
    mask = np.array([[True, True, True], [True, True, False]])

    _assert_every_lifetime_matches_numpy(
        lambda x: operation(x, axis=1, where=mask, initial=initial),
        value,
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda x, mask: np.where(mask, x, 0.25),
        lambda x, mask: np.where(mask, 1, x),
        lambda x, mask: np.max(x, axis=1, where=mask, initial=0.25),
        lambda x, mask: np.nanmin(x, axis=1, where=mask, initial=2),
        lambda x, _mask: np.concatenate((x, 0.25), axis=None),
    ],
    ids=("where-float", "where-int", "max-initial", "nanmin-initial", "concatenate"),
)
def test_python_scalar_operands_stay_weak_like_numpy(operation: Any) -> None:
    value = np.array([[0.5, np.nan, 1.5], [2.5, -1.0, 0.0]], dtype=np.float32)
    mask = np.array([[True, False, True], [False, True, True]])

    _assert_every_lifetime_matches_numpy(lambda x: operation(x, mask), value)
    _primal, tangent = ad.jvp(lambda x: operation(x, mask))(value, tangents=np.ones_like(value))
    assert tangent.dtype == np.dtype(np.float32)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x, s: np.concatenate((x, s), axis=None),
        np.dot,
        lambda x, s: np.tensordot(x, s, axes=0),
        lambda x, s: np.stack((x[0, 0], s)),
    ],
    ids=("concatenate-weak", "dot-strong", "tensordot-strong", "stack-strong"),
)
def test_weak_scalar_inputs_promote_per_numpy_function(operation: Any) -> None:
    # Captured Python scalars are drawn in test_lifetime_parity_properties.
    value = np.array([[0.5, -1.0, 1.5], [2.5, -2.0, 0.0]], dtype=np.float32)
    specs = (ad.ArraySpec(value.shape, value.dtype), ad.ArraySpec((), "float64", weak=True))
    expected = operation(value, 0.25)
    program = ad.stage(operation, specs=specs)
    for staged in (program, ad.StagedProgram.from_dict(program.to_dict())):
        actual = staged(value, 0.25)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)

    def loss(x: Any, s: Any) -> Any:
        return np.sum(operation(x, s) ** 2)

    dynamic = ad.grad(loss, argnums=(0, 1))(value, 0.25)
    staged_grad = ad.grad(ad.stage(loss, specs=specs), argnums=(0, 1))(value, 0.25)
    assert_tree_close(staged_grad, dynamic, rtol=1e-6)


def test_like_constructor_accepts_a_scalar_shape_override() -> None:
    value = np.arange(6.0).reshape(2, 3)

    def construct(x: Any) -> Any:
        return np.zeros_like(x, shape=3)

    primal, tangent = ad.jvp(construct)(value, tangents=np.ones_like(value))
    np.testing.assert_array_equal(primal, np.zeros(3))
    np.testing.assert_array_equal(tangent, np.zeros(3))

    assert_staged_round_trip(construct, value, rtol=0.0)


@pytest.mark.parametrize("operation", [np.sum, np.nanmin])
@pytest.mark.parametrize("initial", [np.float32(0.5), np.array(0.5)], ids=("scalar", "0-d"))
def test_staged_reductions_accept_numpy_initial_values(
    operation: Callable[..., Any],
    initial: object,
) -> None:
    values = np.arange(6.0, dtype=np.float32).reshape(2, 3)

    assert_staged_round_trip(lambda array: operation(array, axis=1, initial=initial), values)


# Array API 2022.12 sums float32 in float64; NumPy's functions and methods must not.
@pytest.mark.parametrize("array_api_version", [None, "2022.12"], ids=("default", "2022.12"))
def test_controlled_float32_reductions_stage_without_dtype_creep(
    array_api_version: str | None,
) -> None:
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    mask = np.array([[True, False, True], [True, True, False]])
    functions = (
        lambda array: np.mean(array, axis=1, keepdims=True, where=mask),
        lambda array: np.var(
            array,
            axis=1,
            correction=1,
            keepdims=True,
            where=mask,
        ),
        lambda array: array.sum(1),
        lambda array: array.mean(),
    )

    for function in functions:
        program = ad.stage(
            function,
            specs=(ad.ArraySpec(value.shape, value.dtype),),
            array_api_version=array_api_version,
        )
        result = program(value)
        reference = function(value)
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_allclose(result, reference)
