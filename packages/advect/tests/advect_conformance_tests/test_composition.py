"""Property tests for interactions between individually conformant rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from hypothesis.extra import numpy as hnp

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable

_COMPOSITION_EXAMPLES = max(10, min(200, settings.default.max_examples // 5))
_VALUES = hnp.arrays(
    dtype=np.float64,
    shape=(6,),
    elements=st.floats(
        min_value=-1.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
)
_OPCODES = st.sampled_from(
    (
        "add",
        "cumsum",
        "exp",
        "flip",
        "multiply",
        "mutation",
        "roll",
        "sin",
        "square",
        "tanh",
        "where",
    ),
)
_PARAMETER = st.floats(
    min_value=-0.5,
    max_value=0.5,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_PROGRAMS = st.lists(
    st.tuples(_OPCODES, _PARAMETER),
    min_size=1,
    max_size=8,
)
_DIRECTION = np.array([0.2, -0.4, 0.6, -0.8, 1.0, -0.2])
_TENSOR_VALUES = hnp.arrays(
    dtype=np.float64,
    shape=(2, 3, 4),
    elements=st.floats(
        min_value=-1.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
)
_TENSOR_DIRECTION = np.linspace(-0.7, 0.8, 24).reshape(2, 3, 4)
_MATRIX_VALUES = hnp.arrays(
    dtype=np.float64,
    shape=(3, 3),
    elements=st.floats(
        min_value=-0.75,
        max_value=0.75,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
)
_RHS_VALUES = hnp.arrays(
    dtype=np.float64,
    shape=(3,),
    elements=st.floats(
        min_value=-1.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
)
_MATRIX_DIRECTION = np.linspace(-0.4, 0.5, 9).reshape(3, 3)
_RHS_DIRECTION = np.array([0.3, -0.5, 0.7])


def _compile_program(instructions: list[tuple[str, float]]) -> Callable[[object], object]:
    def program(value: object) -> object:
        state = value
        for opcode, parameter in instructions:
            if opcode == "add":
                state = state + parameter
            elif opcode == "cumsum":
                state = np.cumsum(state) / np.arange(1.0, 7.0)
            elif opcode == "exp":
                state = np.exp(0.25 * np.tanh(state))
            elif opcode == "flip":
                state = np.flip(state)
            elif opcode == "multiply":
                state = state * (1.0 + parameter)
            elif opcode == "mutation":
                updated = state.copy()
                updated[1:-1] += 0.2 * state[:-2]
                state = updated
            elif opcode == "roll":
                state = np.roll(state, 1)
            elif opcode == "sin":
                state = np.sin(state)
            elif opcode == "square":
                state = 0.2 * np.square(np.tanh(state))
            elif opcode == "tanh":
                state = np.tanh(state)
            elif opcode == "where":
                state = np.where(np.arange(6) % 2 == 0, state, -state)
            else:  # pragma: no cover - closed strategy vocabulary
                raise AssertionError(opcode)
        return np.sum(np.sin(state) + 0.1 * state * state)

    return program


def _central_directional(
    function: Callable[[np.ndarray], object],
    value: np.ndarray,
    direction: np.ndarray,
) -> float:
    step = 1e-6
    positive = function(value + step * direction)
    negative = function(value - step * direction)
    return float((positive - negative) / (2.0 * step))


@given(instructions=_PROGRAMS, value=_VALUES)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_random_program_gradient_matches_directional_difference(
    instructions: list[tuple[str, float]],
    value: np.ndarray,
) -> None:
    program = _compile_program(instructions)
    gradient = ad.grad(program)(value)
    expected = _central_directional(program, value, _DIRECTION)
    actual = float(np.vdot(gradient, _DIRECTION).real)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@given(left=_VALUES, right=_VALUES)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_multi_argument_mutation_program_composes_reverse_mode(
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    def objective(a: object, b: object) -> object:
        state = np.sin(a) * np.tanh(b) + 0.1 * a
        updated = state.copy()
        updated[1:-1] += 0.25 * state[:-2]
        return np.sum(updated * updated)

    left_direction = _DIRECTION
    right_direction = np.flip(_DIRECTION)
    left_gradient, right_gradient = ad.grad(objective, argnums=(0, 1))(left, right)
    step = 1e-6
    positive = objective(
        left + step * left_direction,
        right + step * right_direction,
    )
    negative = objective(
        left - step * left_direction,
        right - step * right_direction,
    )
    expected = float((positive - negative) / (2.0 * step))
    actual = float(
        np.vdot(left_gradient, left_direction).real + np.vdot(right_gradient, right_direction).real,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@given(real=_VALUES, imaginary=_VALUES)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_complex_fft_composition_obeys_real_inner_product_convention(
    real: np.ndarray,
    imaginary: np.ndarray,
) -> None:
    value = real + 1j * imaginary
    direction = _DIRECTION + 1j * np.flip(_DIRECTION)

    def loss(field: object) -> object:
        spectrum = np.fft.fft(field)
        shifted = np.roll(spectrum, 1)
        return np.real(np.sum(np.conjugate(shifted) * shifted)) / 6.0

    gradient = ad.grad(loss)(value)
    expected = _central_directional(loss, value, direction)
    actual = float(np.vdot(gradient, direction).real)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@given(value=_TENSOR_VALUES)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_broadcast_reduction_chain_composes_reverse_mode(value: np.ndarray) -> None:
    weights = np.array([0.2, -0.3, 0.5])

    def objective(tensor: object) -> object:
        centered = tensor - np.mean(tensor, axis=1, keepdims=True)
        accumulated = np.cumsum(centered, axis=2)
        energy = np.sum(accumulated * accumulated, axis=(0, 2))
        return np.mean(energy * weights)

    gradient = ad.grad(objective)(value)
    expected = _central_directional(objective, value, _TENSOR_DIRECTION)
    actual = float(np.vdot(gradient, _TENSOR_DIRECTION).real)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@given(factor=_MATRIX_VALUES, right=_RHS_VALUES)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_linalg_chain_composes_multi_argument_reverse_mode(
    factor: np.ndarray,
    right: np.ndarray,
) -> None:
    identity = np.eye(3)

    def objective(matrix_factor: object, rhs: object) -> object:
        matrix = matrix_factor.T @ matrix_factor + 1.5 * identity
        solution = np.linalg.solve(matrix, rhs)
        return np.sum(np.sin(solution) + 0.1 * solution * solution) + 0.01 * np.log(
            np.linalg.det(matrix)
        )

    factor_gradient, right_gradient = ad.grad(objective, argnums=(0, 1))(factor, right)
    step = 1e-6
    positive = objective(
        factor + step * _MATRIX_DIRECTION,
        right + step * _RHS_DIRECTION,
    )
    negative = objective(
        factor - step * _MATRIX_DIRECTION,
        right - step * _RHS_DIRECTION,
    )
    expected = float((positive - negative) / (2.0 * step))
    actual = float(
        np.vdot(factor_gradient, _MATRIX_DIRECTION).real
        + np.vdot(right_gradient, _RHS_DIRECTION).real,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
