"""Property tests for interactions between individually conformant rules."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import hypothesis.strategies as st
import numpy as np
from hypothesis import example, given, settings
from hypothesis.extra import numpy as hnp

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from hypothesis.strategies import SearchStrategy

_COMPOSITION_EXAMPLES = max(10, min(200, settings.default.max_examples // 5))


def _bounded(shape: tuple[int, ...], bound: float = 1.0) -> SearchStrategy[np.ndarray]:
    elements = st.floats(
        -bound, bound, allow_nan=False, allow_infinity=False, allow_subnormal=False
    )
    return hnp.arrays(dtype=np.float64, shape=shape, elements=elements)


_VALUES = _bounded((6,))
_PARAMETER = st.floats(-0.5, 0.5, allow_nan=False, allow_infinity=False, allow_subnormal=False)


def _mutation(state: Any, _parameter: float) -> Any:
    updated = state.copy()
    updated[1:-1] += 0.2 * state[:-2]
    return updated


def _augmented(state: Any, parameter: float) -> Any:
    updated = state.copy()
    updated *= 1.0 + parameter
    return updated


def _shared_updates(state: Any, parameter: float) -> Any:
    """Two index updates read one slice, so staging deduplicates across effects."""
    first = state.copy()
    first[1:3] += state[0:2]
    second = first.copy()
    second[1:3] += parameter * state[0:2]
    return first + second


# Smooth, bounded steps keep central differences meaningful. ``fanout``,
# ``shared_updates`` and ``transposes`` give the staged optimizer common
# subexpressions and transpose pairs to rewrite. A ``scalar_`` step reads the
# program's weak Python-scalar argument instead of its own parameter.
_SMOOTH_STEPS: dict[str, Callable[[Any, float], Any]] = {
    "add": lambda state, parameter: state + parameter,
    "augmented": _augmented,
    "broadcast": lambda state, parameter: np.reshape(
        np.reshape(state, (2, 3)) * np.array([[1.0], [parameter]]), (6,)
    ),
    "cumprod": lambda state, _parameter: np.cumprod(0.5 + 0.25 * np.tanh(state)),
    "cumsum": lambda state, _parameter: np.cumsum(state) / np.arange(1.0, 7.0),
    "exp": lambda state, _parameter: np.exp(0.25 * np.tanh(state)),
    "fanout": lambda state, parameter: np.sin(state) * np.cos(state) + parameter * np.sin(state),
    "flip": lambda state, _parameter: np.flip(state),
    "multiply": lambda state, parameter: state * (1.0 + parameter),
    "mutation": _mutation,
    "roll": lambda state, _parameter: np.roll(state, 1),
    "scalar_multiply": lambda state, scale: state * scale,
    "shared_updates": _shared_updates,
    "sin": lambda state, _parameter: np.sin(state),
    "square": lambda state, _parameter: 0.2 * np.square(np.tanh(state)),
    "tanh": lambda state, _parameter: np.tanh(state),
    "transposes": lambda state, _parameter: np.reshape(
        np.transpose(np.transpose(np.reshape(state, (2, 3)))), (6,)
    ),
    "variance": lambda state, _parameter: state + 0.1 * np.var(state),
    "where": lambda state, _parameter: np.where(np.arange(6) % 2 == 0, state, -state),
}
# Kinked steps select the same branch in every mode and lifetime, so exact
# identities hold on them even where central differences do not.
_KINKED_STEPS: dict[str, Callable[[Any, float], Any]] = {
    "abs": lambda state, _parameter: np.abs(state),
    "clip": lambda state, _parameter: np.clip(state, -0.6, 0.6),
    "mask_cast": lambda state, _parameter: (state > 0).astype(state.dtype) * state,
    "maximum": np.maximum,
    "relu": lambda state, _parameter: state * (state > 0),
}
_STEPS = _SMOOTH_STEPS | _KINKED_STEPS


def _programs(vocabulary: Iterable[str]) -> SearchStrategy[list[tuple[str, float]]]:
    return st.lists(
        st.tuples(st.sampled_from(sorted(vocabulary)), _PARAMETER), min_size=1, max_size=8
    )


def _compile_program(instructions: list[tuple[str, float]]) -> Callable[[Any, float], Any]:
    """Compile a vector program of an array and a weak Python-scalar argument."""

    def program(value: Any, scale: float) -> Any:
        state = value
        for opcode, parameter in instructions:
            state = _STEPS[opcode](state, scale if opcode.startswith("scalar_") else parameter)
        return state

    return program


def _loss(state: Any) -> Any:
    return np.sum(np.sin(state) + 0.1 * state * state)


def _assert_directional_derivative(
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    directions: tuple[Any, ...],
) -> None:
    """Compare reverse gradients along drawn directions with a central difference."""
    gradients = ad.grad(function, argnums=tuple(range(len(arguments))))(*arguments)
    step = 1e-6
    positive = function(*(value + step * d for value, d in zip(arguments, directions, strict=True)))
    negative = function(*(value - step * d for value, d in zip(arguments, directions, strict=True)))
    expected = float((positive - negative) / (2.0 * step))
    actual = sum(
        float(np.vdot(gradient, direction).real)
        for gradient, direction in zip(gradients, directions, strict=True)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def _reloaded(program: ad.StagedProgram) -> ad.StagedProgram:
    """Reload a program through JSON and require the record to be a fixed point."""
    record = program.to_dict()
    restored = ad.StagedProgram.from_dict(json.loads(json.dumps(record)))
    assert restored.to_dict() == record
    return restored


@given(instructions=_programs(_SMOOTH_STEPS), value=_VALUES, scale=_PARAMETER, direction=_VALUES)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_random_program_gradient_matches_directional_difference(
    instructions: list[tuple[str, float]],
    value: np.ndarray,
    scale: float,
    direction: np.ndarray,
) -> None:
    program = _compile_program(instructions)
    _assert_directional_derivative(
        lambda state: _loss(program(state, scale)), (value,), (direction,)
    )


@given(
    instructions=_programs(_STEPS),
    value=_VALUES,
    scale=_PARAMETER,
    dtype=st.sampled_from(("float64", "float32")),
    tangent=_VALUES,
    cotangent=_VALUES,
)
# The absolute-value pullback once divided by |x|, so forward-over-reverse
# cancelled two O(1/|x|) terms and lost the curvature near zero.
@example(
    instructions=[("abs", 0.0)],
    value=np.full(6, 3.46122287e-17),
    scale=0.0,
    dtype="float64",
    tangent=np.ones(6),
    cotangent=np.zeros(6),
)
# A weak scalar that rounds to a float32 tie must select like the primal.
@example(
    instructions=[("maximum", 3.92748034904888e-48)],
    value=np.zeros(6),
    scale=0.0,
    dtype="float32",
    tangent=np.zeros(6),
    cotangent=np.zeros(6),
)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_random_program_lifetimes_agree(
    instructions: list[tuple[str, float]],
    value: np.ndarray,
    scale: float,
    dtype: str,
    tangent: np.ndarray,
    cotangent: np.ndarray,
) -> None:
    """Every mode and lifetime of one program evaluates the same derivative.

    The comparisons differ only in evaluation order, so they are bounded by
    rounding rather than by finite-difference error.
    """
    program = _compile_program(instructions)
    x, direction, seed = (array.astype(dtype) for array in (value, tangent, cotangent))
    rtol = 1e-10 if dtype == "float64" else 1e-4
    close = {"rtol": rtol, "atol": rtol}

    def loss(state: Any, weak: float) -> Any:
        return _loss(program(state, weak))

    reference = loss(x, scale)
    gradient = ad.grad(loss)(x, scale)
    assert gradient.dtype == x.dtype

    # Forward and reverse modes are adjoint on the vector program, and the
    # batched reverse Jacobian of a tall slice agrees with both.
    _value, forward = ad.jvp(program)(x, scale, tangents=direction)
    _value, pullback = ad.vjp(program)(x, scale)
    reverse = pullback(seed)
    pairings = (np.vdot(seed, forward), np.vdot(reverse, direction))
    magnitude = np.vdot(np.abs(seed), np.abs(forward)) + np.vdot(np.abs(reverse), np.abs(direction))
    np.testing.assert_allclose(*pairings, rtol=0, atol=rtol * (1.0 + magnitude))
    jacobian = ad.jacobian(lambda state: program(state, scale)[:3])(x)
    np.testing.assert_allclose(jacobian @ direction, forward[:3], **close)

    # Staged, reloaded, and dynamically traced programs agree with the dynamic run.
    specs = (ad.ArraySpec((6,), dtype), ad.ArraySpec((), "float64", weak=True))
    staged = _reloaded(ad.stage(loss, specs=specs))
    staged_gradient = ad.grad(staged)
    candidates = {
        "staged value": staged(x, scale),
        "staged grad": _reloaded(staged_gradient)(x, scale),
        "staged vjp": _reloaded(ad.vjp_program(staged))(
            x, scale, cotangent=np.ones_like(reference)
        ),
        "dynamic grad of staged": ad.grad(lambda state: staged(state, scale))(x),
        "checkpointed grad": ad.grad(ad.checkpoint(loss))(x, scale),
    }
    for label, result in candidates.items():
        expected = reference if label == "staged value" else gradient
        assert np.asarray(result).dtype == np.asarray(expected).dtype, label
        np.testing.assert_allclose(result, expected, err_msg=label, **close)

    with ad.debug():
        np.testing.assert_array_equal(ad.grad(loss)(x, scale), gradient)

    hessian = np.asarray(ad.hessian(loss)(x, scale))
    _value, curvature = ad.hvp(loss)(x, scale, vectors=direction)
    np.testing.assert_allclose(hessian, hessian.T, **close)
    np.testing.assert_allclose(curvature, hessian @ direction, **close)
    np.testing.assert_allclose(ad.hessian_diag(loss)(x, scale), np.diag(hessian), **close)

    # Checkpointing is transparent to forward and second-order transforms.
    checkpointed = ad.jvp(ad.checkpoint(program))(x, scale, tangents=direction)[1]
    np.testing.assert_allclose(checkpointed, forward, **close)
    checkpointed = ad.hvp(ad.checkpoint(loss))(x, scale, vectors=direction)[1]
    np.testing.assert_allclose(checkpointed, curvature, **close)


@given(left=_VALUES, right=_VALUES, left_direction=_VALUES, right_direction=_VALUES)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_multi_argument_mutation_program_composes_reverse_mode(
    left: np.ndarray,
    right: np.ndarray,
    left_direction: np.ndarray,
    right_direction: np.ndarray,
) -> None:
    def objective(a: Any, b: Any) -> Any:
        state = np.sin(a) * np.tanh(b) + 0.1 * a
        updated = state.copy()
        updated[1:-1] += 0.25 * state[:-2]
        return np.sum(updated * updated)

    _assert_directional_derivative(objective, (left, right), (left_direction, right_direction))


@given(real=_VALUES, imaginary=_VALUES, direction_real=_VALUES, direction_imaginary=_VALUES)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_complex_fft_composition_obeys_real_inner_product_convention(
    real: np.ndarray,
    imaginary: np.ndarray,
    direction_real: np.ndarray,
    direction_imaginary: np.ndarray,
) -> None:
    def loss(field: Any) -> Any:
        spectrum = np.fft.fft(field)
        shifted = np.roll(spectrum, 1)
        return np.real(np.sum(np.conjugate(shifted) * shifted)) / 6.0

    _assert_directional_derivative(
        loss,
        (real + 1j * imaginary,),
        (direction_real + 1j * direction_imaginary,),
    )


@given(value=_bounded((2, 3, 4)), direction=_bounded((2, 3, 4)))
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_broadcast_reduction_chain_composes_reverse_mode(
    value: np.ndarray,
    direction: np.ndarray,
) -> None:
    weights = np.array([0.2, -0.3, 0.5])

    def objective(tensor: Any) -> Any:
        centered = tensor - np.mean(tensor, axis=1, keepdims=True)
        accumulated = np.cumsum(centered, axis=2)
        energy = np.sum(accumulated * accumulated, axis=(0, 2))
        return np.mean(energy * weights)

    _assert_directional_derivative(objective, (value,), (direction,))


@given(
    factor=_bounded((3, 3), 0.75),
    right=_bounded((3,)),
    factor_direction=_bounded((3, 3)),
    right_direction=_bounded((3,)),
)
@settings(max_examples=_COMPOSITION_EXAMPLES, deadline=None)
def test_linalg_chain_composes_multi_argument_reverse_mode(
    factor: np.ndarray,
    right: np.ndarray,
    factor_direction: np.ndarray,
    right_direction: np.ndarray,
) -> None:
    identity = np.eye(3)

    def objective(matrix_factor: Any, rhs: Any) -> Any:
        matrix = matrix_factor.T @ matrix_factor + 1.5 * identity
        solution = np.linalg.solve(matrix, rhs)
        return np.sum(np.sin(solution) + 0.1 * solution * solution) + 0.01 * np.log(
            np.linalg.det(matrix)
        )

    _assert_directional_derivative(objective, (factor, right), (factor_direction, right_direction))
