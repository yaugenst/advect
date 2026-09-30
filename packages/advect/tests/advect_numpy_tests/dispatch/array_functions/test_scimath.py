"""Dynamic contract tests for NumPy's complex-domain math helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy.lib import scimath

import advect as ad
from advect_numpy_tests._assertions import assert_jvp_matches_central_difference

if TYPE_CHECKING:
    from collections.abc import Callable


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


@pytest.mark.parametrize(("function", "raw_values"), _UNARY_CASES)
def test_scimath_unary_dynamic_contract(
    function: Callable[..., Any],
    raw_values: tuple[complex, complex],
) -> None:
    values = np.asarray(raw_values)

    assert_jvp_matches_central_difference(
        function, (values,), (_direction_like(values),), rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize(("function", "raw_inputs"), _BINARY_CASES)
def test_scimath_binary_dynamic_contract_covers_both_operands(
    function: Callable[..., Any],
    raw_inputs: tuple[tuple[complex, complex], tuple[complex, complex]],
) -> None:
    inputs = tuple(np.asarray(value) for value in raw_inputs)
    first, second = (
        _direction_like(value, second=index == 1) for index, value in enumerate(inputs)
    )

    for directions in ((first, np.zeros_like(second)), (np.zeros_like(first), second)):
        assert_jvp_matches_central_difference(function, inputs, directions, rtol=3e-6, atol=3e-7)
    assert_jvp_matches_central_difference(function, inputs, (first, second), rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize(
    "function",
    [
        *(pytest.param(function, id=function.__name__) for function in _UNARY_FUNCTIONS),
        pytest.param(lambda x: scimath.logn(x, -2.0), id="logn"),
        pytest.param(lambda x: scimath.power(x, -1.5), id="power"),
    ],
)
def test_scimath_second_derivatives_read_values_through_nested_traces(
    function: Callable[..., Any],
) -> None:
    # A negative input takes each function's complex continuation branch.
    values = np.array([0.3, -0.6])
    direction = np.array([0.2, -0.1])

    def derivative(x: Any) -> Any:
        return ad.jvp(function)(x, tangents=direction)[1]

    assert_jvp_matches_central_difference(derivative, (values,), (direction,), rtol=2e-5, atol=2e-6)


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
