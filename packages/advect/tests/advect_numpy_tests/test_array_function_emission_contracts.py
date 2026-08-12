"""Public contracts for common NumPy array-function emission paths."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad


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

    program = ad.stage(bounded, specs=(ad.ArraySpec(value.shape, value.dtype),))
    np.testing.assert_array_equal(program(value), bounded(value))

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
    ("operation", "message"),
    [
        (np.clip, "requires bounds"),
        (lambda x: np.clip(x, 0.0), "only supported during tracing as"),
        (
            lambda x: np.clip(x, 0.0, 1.0, min=0.0),
            "either positionally or via keywords",
        ),
        (lambda x: np.clip(x, 1j, 1.0), "got scalar complex"),
    ],
    ids=("missing", "incomplete-positional", "mixed", "complex"),
)
def test_clip_reports_invalid_bound_forms(operation: Any, message: str) -> None:
    value = np.arange(3.0)
    with pytest.raises(ad.TracingError, match=message):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda x: np.where(x > 0), id="where-one-argument"),
        pytest.param(lambda x: np.var(x, ddof=1, correction=1), id="variance-two-corrections"),
    ],
)
def test_emission_handlers_report_ambiguous_public_calls(operation: Any) -> None:
    value = np.arange(6.0).reshape(2, 3)
    with pytest.raises(ad.TracingError):
        ad.jvp(operation)(value, tangents=np.ones_like(value))


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

    primal, tangent = ad.jvp(reduce, argnums=(0, 1))(
        value,
        initial,
        tangents=(direction, initial_direction),
    )
    epsilon = 1e-6
    finite_difference = (
        reduce(value + epsilon * direction, initial + epsilon * initial_direction)
        - reduce(value - epsilon * direction, initial - epsilon * initial_direction)
    ) / (2 * epsilon)

    np.testing.assert_allclose(primal, reduce(value, initial))
    np.testing.assert_allclose(tangent, finite_difference, rtol=1e-6, atol=1e-6)

    program = ad.stage(
        reduce,
        specs=(ad.ArraySpec(value.shape, value.dtype), ad.ArraySpec((), initial.dtype)),
    )
    np.testing.assert_allclose(program(value, initial), reduce(value, initial))


def test_nanmin_static_initial_and_metadata_survive_staging() -> None:
    value = np.array([[np.nan, 2.0], [0.5, 5.0]])

    def reduce(x: Any) -> Any:
        return np.nanmin(x, axis=1, keepdims=True, initial=1.0)

    primal, tangent = ad.jvp(reduce)(value, tangents=np.ones_like(value))
    np.testing.assert_allclose(primal, reduce(value))
    np.testing.assert_array_equal(tangent, [[0.0], [1.0]])

    program = ad.stage(reduce, specs=(ad.ArraySpec(value.shape, value.dtype),))
    np.testing.assert_allclose(program(value), reduce(value))


def test_variance_static_dtype_and_correction_survive_staging() -> None:
    value = np.array([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]], dtype=np.float64)

    def reduce(x: Any) -> Any:
        return np.nanvar(x, axis=1, dtype=np.float32, ddof=1)

    primal, tangent = ad.jvp(reduce)(value, tangents=np.ones_like(value))
    assert primal.dtype == np.dtype(np.float32)
    assert tangent.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(primal, reduce(value))
    np.testing.assert_allclose(tangent, np.zeros(2, dtype=np.float32), atol=1e-6)

    program = ad.stage(reduce, specs=(ad.ArraySpec(value.shape, value.dtype),))
    staged_result = program(value)
    assert staged_result.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(staged_result, reduce(value))


def test_controlled_variance_honors_requested_accumulator_dtype() -> None:
    value = np.array([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]], dtype=np.float32)
    direction = np.array([[0.2, -0.1, 0.3], [0.4, 0.1, -0.2]], dtype=np.float32)
    mask = np.array([[True, False, True], [True, True, False]])

    def reduce(x: Any) -> Any:
        return np.var(x, axis=1, where=mask, dtype=np.float64, correction=1)

    primal, tangent = ad.jvp(reduce)(value, tangents=direction)
    epsilon = 1e-4
    finite_difference = (
        reduce(value + epsilon * direction) - reduce(value - epsilon * direction)
    ) / (2 * epsilon)

    assert primal.dtype == np.dtype(np.float64)
    assert tangent.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(primal, reduce(value))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-3, atol=2e-3)


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

    program = ad.stage(reduce, specs=(ad.ArraySpec(value.shape, value.dtype),))
    np.testing.assert_allclose(program(value), reduce(value))


def test_like_constructor_accepts_a_scalar_shape_override() -> None:
    value = np.arange(6.0).reshape(2, 3)

    def construct(x: Any) -> Any:
        return np.zeros_like(x, shape=3)

    primal, tangent = ad.jvp(construct)(value, tangents=np.ones_like(value))
    np.testing.assert_array_equal(primal, np.zeros(3))
    np.testing.assert_array_equal(tangent, np.zeros(3))

    program = ad.stage(construct, specs=(ad.ArraySpec(value.shape, value.dtype),))
    np.testing.assert_array_equal(program(value), np.zeros(3))
