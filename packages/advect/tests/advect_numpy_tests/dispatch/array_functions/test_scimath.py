"""Dynamic contract tests for NumPy's complex-domain math helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.lib import scimath

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


_STEP = 1e-6
_UNARY_FUNCTIONS = (
    scimath.sqrt,
    scimath.log,
    scimath.log10,
    scimath.log2,
    scimath.arcsin,
    scimath.arccos,
    scimath.arctanh,
)
_UNARY_CASES = (
    pytest.param(scimath.sqrt, (4.0, 9.0), id="sqrt-real"),
    pytest.param(scimath.sqrt, (-4.0, -9.0), id="sqrt-complex-continuation"),
    pytest.param(scimath.sqrt, (0.5 + 0.2j, -1.2 + 0.3j), id="sqrt-complex-input"),
    pytest.param(scimath.log, (2.0, 4.0), id="log-real"),
    pytest.param(scimath.log, (-2.0, -4.0), id="log-complex-continuation"),
    pytest.param(scimath.log, (0.5 + 0.2j, -1.2 + 0.3j), id="log-complex-input"),
    pytest.param(scimath.log10, (2.0, 4.0), id="log10-real"),
    pytest.param(scimath.log10, (-2.0, -4.0), id="log10-complex-continuation"),
    pytest.param(scimath.log10, (0.5 + 0.2j, -1.2 + 0.3j), id="log10-complex-input"),
    pytest.param(scimath.log2, (2.0, 4.0), id="log2-real"),
    pytest.param(scimath.log2, (-2.0, -4.0), id="log2-complex-continuation"),
    pytest.param(scimath.log2, (0.5 + 0.2j, -1.2 + 0.3j), id="log2-complex-input"),
    pytest.param(scimath.arcsin, (0.2, -0.5), id="arcsin-real"),
    pytest.param(scimath.arcsin, (2.0, -3.0), id="arcsin-complex-continuation"),
    pytest.param(scimath.arcsin, (0.5 + 0.2j, -1.2 + 0.3j), id="arcsin-complex-input"),
    pytest.param(scimath.arccos, (0.2, -0.5), id="arccos-real"),
    pytest.param(scimath.arccos, (2.0, -3.0), id="arccos-complex-continuation"),
    pytest.param(scimath.arccos, (0.5 + 0.2j, -1.2 + 0.3j), id="arccos-complex-input"),
    pytest.param(scimath.arctanh, (0.2, -0.5), id="arctanh-real"),
    pytest.param(scimath.arctanh, (2.0, -3.0), id="arctanh-complex-continuation"),
    pytest.param(scimath.arctanh, (0.5 + 0.2j, -1.2 + 0.3j), id="arctanh-complex-input"),
)
_BINARY_CASES = (
    pytest.param(scimath.logn, ((2.0, 3.0), (4.0, 9.0)), id="logn-real"),
    pytest.param(
        scimath.logn,
        ((-2.0, -3.0), (-4.0, -9.0)),
        id="logn-complex-continuation",
    ),
    pytest.param(
        scimath.logn,
        ((0.5 + 0.2j, -1.2 + 0.3j), (1.3 - 0.4j, -0.7 + 0.5j)),
        id="logn-complex-input",
    ),
    pytest.param(scimath.power, ((2.0, 3.0), (0.5, 1.5)), id="power-real"),
    pytest.param(
        scimath.power,
        ((-2.0, -3.0), (0.5, 1.5)),
        id="power-complex-continuation",
    ),
    pytest.param(
        scimath.power,
        ((0.5 + 0.2j, -1.2 + 0.3j), (0.7 - 0.1j, 1.3 + 0.2j)),
        id="power-complex-input",
    ),
)


def _direction_like(value: np.ndarray[Any, Any], *, second: bool = False) -> np.ndarray[Any, Any]:
    if np.issubdtype(value.dtype, np.complexfloating):
        raw = (-0.3 + 0.25j, 0.2 - 0.15j) if second else (0.2 + 0.1j, -0.1 + 0.2j)
    else:
        raw = (-0.3, 0.25) if second else (0.2, -0.1)
    return np.asarray(raw, dtype=value.dtype)


def _cotangent_like(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    raw = (0.7 + 0.2j, -0.4 + 0.3j) if np.iscomplexobj(value) else (0.7, -0.4)
    return np.asarray(raw, dtype=value.dtype)


def _real_inner_product(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    return float(np.vdot(left, right).real)


@pytest.mark.parametrize(("function", "raw_values"), _UNARY_CASES)
def test_scimath_unary_dynamic_contract(
    function: Callable[..., Any],
    raw_values: tuple[complex, complex],
) -> None:
    values = np.asarray(raw_values)
    direction = _direction_like(values)

    primal, tangent = ad.jvp(function)(values, tangents=direction)
    numerical = (function(values + _STEP * direction) - function(values - _STEP * direction)) / (
        2 * _STEP
    )
    expected = function(values)

    assert primal.dtype == expected.dtype
    np.testing.assert_allclose(primal, expected)
    np.testing.assert_allclose(tangent, numerical, rtol=2e-6, atol=2e-7)

    cotangent = _cotangent_like(primal)
    _value, pullback = ad.vjp(function)(values)
    input_cotangent = pullback(cotangent)
    assert input_cotangent.dtype == values.dtype
    np.testing.assert_allclose(
        _real_inner_product(cotangent, tangent),
        _real_inner_product(input_cotangent, direction),
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.parametrize(("function", "raw_inputs"), _BINARY_CASES)
def test_scimath_binary_dynamic_contract_covers_both_operands(
    function: Callable[..., Any],
    raw_inputs: tuple[tuple[complex, complex], tuple[complex, complex]],
) -> None:
    inputs = tuple(np.asarray(value) for value in raw_inputs)
    directions = tuple(
        _direction_like(value, second=index == 1) for index, value in enumerate(inputs)
    )

    for active_index in range(2):
        active_directions = tuple(
            direction if index == active_index else np.zeros_like(direction)
            for index, direction in enumerate(directions)
        )
        primal, tangent = ad.jvp(function, argnums=(0, 1))(
            *inputs,
            tangents=active_directions,
        )
        positive = list(inputs)
        negative = list(inputs)
        positive[active_index] = inputs[active_index] + _STEP * directions[active_index]
        negative[active_index] = inputs[active_index] - _STEP * directions[active_index]
        numerical = (function(*positive) - function(*negative)) / (2 * _STEP)

        np.testing.assert_allclose(primal, function(*inputs))
        np.testing.assert_allclose(tangent, numerical, rtol=3e-6, atol=3e-7)

    primal, tangent = ad.jvp(function, argnums=(0, 1))(
        *inputs,
        tangents=directions,
    )
    cotangent = _cotangent_like(primal)
    _value, pullback = ad.vjp(function, argnums=(0, 1))(*inputs)
    input_cotangents = pullback(cotangent)
    assert all(
        result.dtype == source.dtype
        for result, source in zip(input_cotangents, inputs, strict=True)
    )
    np.testing.assert_allclose(
        _real_inner_product(cotangent, tangent),
        sum(
            _real_inner_product(input_cotangent, direction)
            for input_cotangent, direction in zip(input_cotangents, directions, strict=True)
        ),
        rtol=3e-6,
        atol=3e-7,
    )


@pytest.mark.parametrize(
    ("function", "arity"),
    [
        *(pytest.param(function, 1, id=function.__name__) for function in _UNARY_FUNCTIONS),
        pytest.param(scimath.logn, 2, id="logn"),
        pytest.param(scimath.power, 2, id="power"),
    ],
)
def test_every_scimath_function_rejects_staging(
    function: Callable[..., Any],
    arity: int,
) -> None:
    specs = tuple(ad.ArraySpec((2,), np.float64) for _index in range(arity))

    with pytest.raises(ad.TracingError, match=r"dynamic-only.*output dtype"):
        ad.stage(function, specs=specs)
