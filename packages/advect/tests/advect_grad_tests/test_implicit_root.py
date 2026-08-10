"""Dynamic implicit-root differentiation contracts."""

from __future__ import annotations

from typing import Any, cast

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

from advect import ImplicitSolveError, grad, implicit_root, jvp, linearize, vjp
from advect.core import ArraySpec, TracingError, primitive, stage
from advect.core._registry import get_registry


def _newton(
    residual: Any,
    initial: np.ndarray,
) -> np.ndarray:
    value = initial.copy()
    for _iteration in range(12):
        value = value - residual(value) / (2 * value)
    return value


def _diagonal_solve(operator: Any, rhs: Any) -> Any:
    ones = np.ones_like(rhs)
    return rhs / operator(ones)


def test_scalar_root_gradient_does_not_trace_newton_iterations() -> None:
    calls = 0

    def solve(residual: Any, initial: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return _newton(residual, initial)

    root = implicit_root(
        lambda solution, params: solution**2 - params,
        solve=solve,
        linear_solve=_diagonal_solve,
    )
    params = np.array(4.0)

    gradient = grad(lambda value: root(value, initial=np.array(1.0)))(params)

    assert_allclose(gradient, 0.25)
    assert calls == 1


def test_nonlinear_iterations_are_atomic_on_the_outer_tape() -> None:
    root = implicit_root(
        lambda solution, params: solution**2 - params,
        solve=_newton,
        linear_solve=_diagonal_solve,
    )

    _value, reusable = linearize(
        lambda params: root(params, initial=np.ones_like(params)),
        np.array([1.0, 4.0]),
    )
    try:
        op_names = cast("Any", reusable)._trace.tape.op_names
        assert "custom.advect_internal.implicit_root" in op_names
        assert "array.power" not in op_names
    finally:
        reusable.close()


def test_primitive_residual_calls_do_not_escape_the_implicit_atomic_boundary() -> None:
    @primitive(name="tests.implicit.primitive_residual_identity")
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    @identity.def_jvp
    def identity_jvp(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output, primals
        tangent = tangents[0]
        assert tangent is not None
        return tangent

    @identity.def_transpose
    def identity_transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        del primals, output
        return (cotangent,)

    root = implicit_root(
        lambda solution, params: identity(solution) - params,
        solve=lambda residual, initial: initial - residual(initial),
        linear_solve=_diagonal_solve,
    )
    params = np.array([1.0, 4.0])
    initial = np.zeros_like(params)

    _value, reusable = linearize(
        lambda value: root(value, initial=initial),
        params,
    )
    try:
        op_names = cast("Any", reusable)._trace.tape.op_names
        assert "custom.advect_internal.implicit_root" in op_names
        assert identity.op_name not in op_names
    finally:
        reusable.close()

    assert_allclose(
        grad(lambda value: np.sum(root(value, initial=initial)))(params),
        np.ones_like(params),
    )


def test_implicit_root_wrappers_share_one_stable_registry_operation() -> None:
    def make_root(power: int) -> Any:
        return implicit_root(
            lambda solution, params: solution**power - params,
            solve=_newton,
            linear_solve=_diagonal_solve,
        )

    first = make_root(2)
    assert_allclose(
        grad(lambda value: first(value, initial=np.array(1.0)))(np.array(4.0)),
        0.25,
    )
    registry = get_registry()
    stable_state = len(registry._ops), registry.get_revision()

    second = implicit_root(
        lambda solution, params: solution - params,
        solve=lambda residual, initial: initial - residual(initial),
        linear_solve=lambda _operator, rhs: rhs,
    )
    assert_allclose(
        grad(lambda value: second(value, initial=np.array(0.0)))(np.array(3.0)),
        1.0,
    )

    assert (len(registry._ops), registry.get_revision()) == stable_state


def test_vector_root_jvp_uses_matrix_free_tangent_solve() -> None:
    solve_calls = 0

    def solve(residual: Any, initial: np.ndarray) -> np.ndarray:
        nonlocal solve_calls
        solve_calls += 1
        return _newton(residual, initial)

    root = implicit_root(
        lambda solution, params: solution**2 - params,
        solve=solve,
        linear_solve=_diagonal_solve,
    )
    params = np.array([1.0, 4.0, 9.0])
    tangent = np.array([0.5, -1.0, 3.0])

    value, product = jvp(root)(params, initial=np.ones(3), tangents=tangent)

    assert_allclose(value, np.sqrt(params))
    assert_allclose(product, tangent / (2 * np.sqrt(params)))
    assert solve_calls == 1


def test_root_supports_solution_and_parameter_pytrees() -> None:
    def residual(
        solution: dict[str, np.ndarray],
        params: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, np.ndarray]:
        scale, target = params
        return {
            "left": scale * solution["left"] - target,
            "right": solution["right"] - scale,
        }

    def solve(residual_at_params: Any, initial: dict[str, np.ndarray]) -> Any:
        del residual_at_params, initial
        return {
            "left": np.array([2.0, 3.0]),
            "right": np.array([2.0, 2.0]),
        }

    def linear_solve(operator: Any, rhs: dict[str, np.ndarray]) -> Any:
        basis = {
            "left": np.ones_like(rhs["left"]),
            "right": np.ones_like(rhs["right"]),
        }
        diagonal = operator(basis)
        return {
            "left": rhs["left"] / diagonal["left"],
            "right": rhs["right"] / diagonal["right"],
        }

    root = implicit_root(
        residual,
        solve=solve,
        linear_solve=linear_solve,
    )
    scale = np.array([2.0, 2.0])
    target = np.array([4.0, 6.0])

    scale_gradient, target_gradient = grad(
        lambda runtime_scale, runtime_target: np.sum(
            root(
                (runtime_scale, runtime_target),
                initial={"left": np.ones(2), "right": np.ones(2)},
            )["left"]
        ),
        argnums=(0, 1),
    )(scale, target)

    assert_allclose(scale_gradient, -target / scale**2)
    assert_allclose(target_gradient, 1 / scale)


def test_root_supports_array_api_strict_with_a_captured_initial_guess() -> None:
    initial = strict.zeros((3,), dtype=strict.float32)
    root = implicit_root(
        lambda solution, params: solution - params,
        solve=lambda residual, guess: guess - residual(guess),
        linear_solve=lambda _operator, rhs: rhs,
    )
    params = strict.asarray([1.0, -2.0, 3.0], dtype=strict.float32)

    gradient = grad(
        lambda value: value.__array_namespace__().sum(root(value, initial=initial) ** 2)
    )(params)

    assert type(gradient) is type(params)
    assert gradient.dtype == strict.float32
    assert_allclose(np.asarray(gradient), np.array([2.0, -4.0, 6.0]))


def test_root_rejects_a_linear_solve_result_from_another_provider() -> None:
    initial = strict.zeros((2,), dtype=strict.float32)
    root = implicit_root(
        lambda solution, params: solution - params,
        solve=lambda residual, guess: guess - residual(guess),
        linear_solve=lambda _operator, rhs: np.asarray(rhs),
    )
    params = strict.asarray([1.0, 2.0], dtype=strict.float32)

    with pytest.raises(TypeError, match="shape/dtype/provider"):
        jvp(root)(params, initial=initial, tangents=strict.ones_like(params))


def test_one_leaf_dictionary_state_reuses_its_public_output_structure() -> None:
    def residual(
        solution: dict[str, np.ndarray],
        params: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {"value": solution["value"] ** 2 - params}

    def solve(
        residual_at_params: Any,
        initial: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        value = initial["value"].copy()
        for _iteration in range(12):
            value = value - residual_at_params({"value": value})["value"] / (2 * value)
        return {"value": value}

    def linear_solve(
        operator: Any,
        rhs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        diagonal = operator({"value": np.ones_like(rhs["value"])})["value"]
        return {"value": rhs["value"] / diagonal}

    root = implicit_root(
        residual,
        solve=solve,
        linear_solve=linear_solve,
    )
    params = np.array([1.0, 4.0, 9.0])
    tangent = np.array([0.5, -1.0, 3.0])
    initial = {"value": np.ones_like(params)}

    value, directional = jvp(lambda runtime: root(runtime, initial=initial)["value"])(
        params,
        tangents=tangent,
    )
    gradient = grad(lambda runtime: np.sum(root(runtime, initial=initial)["value"]))(params)

    assert_allclose(value, np.sqrt(params))
    assert_allclose(directional, tangent / (2 * np.sqrt(params)))
    assert_allclose(gradient, 0.5 / np.sqrt(params))


def test_root_vjp_uses_explicit_transpose_solver() -> None:
    transpose_calls = 0
    matrix = np.array([[2.0, 1.0], [-1.0, 3.0]])

    def transpose_solve(operator: Any, rhs: np.ndarray) -> np.ndarray:
        nonlocal transpose_calls
        transpose_calls += 1
        # Exercise the supplied adjoint operator instead of closing over A.T.
        recovered = np.column_stack([operator(np.eye(2)[:, index]) for index in range(2)])
        return np.linalg.solve(recovered, rhs)

    root = implicit_root(
        lambda solution, params: matrix @ solution - params,
        solve=lambda residual, initial: np.linalg.solve(
            matrix,
            matrix @ initial - residual(initial),
        ),
        linear_solve=lambda _operator, rhs: np.linalg.solve(matrix, rhs),
        transpose_solve=transpose_solve,
    )
    params = np.array([1.0, 2.0])
    value, pullback = vjp(root)(params, initial=np.zeros(2))
    try:
        result = pullback(np.array([0.5, -1.0]))
    finally:
        pullback.close()

    assert_allclose(value, np.linalg.solve(matrix, params))
    assert_allclose(result, np.linalg.solve(matrix.T, np.array([0.5, -1.0])))
    assert transpose_calls == 1


def test_complex_root_uses_the_real_adjoint_convention() -> None:
    root = implicit_root(
        lambda solution, params: (1.0 + 2.0j) * solution - params,
        solve=lambda residual, initial: initial - residual(initial) / (1.0 + 2.0j),
        linear_solve=_diagonal_solve,
    )
    params = np.array([1.0 - 0.5j, 2.0 + 0.25j], dtype=np.complex64)

    gradient = grad(lambda value: np.real(np.sum(root(value, initial=np.zeros_like(value)))))(
        params
    )

    assert gradient.dtype == params.dtype
    assert_allclose(gradient, np.full_like(params, 1 / np.conjugate(1.0 + 2.0j)))


def test_nonholomorphic_complex_root_satisfies_real_adjoint_identity() -> None:
    coefficient = np.complex64(0.2 + 0.1j)
    denominator = np.float32(1 - abs(coefficient) ** 2)

    def inverse(rhs: np.ndarray) -> np.ndarray:
        return (rhs - coefficient * np.conj(rhs)) / denominator

    root = implicit_root(
        lambda solution, params: solution + coefficient * np.conj(solution) - params,
        solve=lambda residual, initial: initial - inverse(residual(initial)),
        linear_solve=lambda _operator, rhs: inverse(rhs),
    )
    params = np.array([1.0 + 0.5j, -0.25 + 0.75j], dtype=np.complex64)
    tangent = np.array([0.3 - 0.2j, 0.5 + 0.1j], dtype=np.complex64)
    cotangent = np.array([-0.1 + 0.4j, 0.7 - 0.3j], dtype=np.complex64)

    _value, product = jvp(root)(
        params,
        initial=np.zeros_like(params),
        tangents=tangent,
    )
    _value, pullback = vjp(root)(params, initial=np.zeros_like(params))
    try:
        adjoint_product = pullback(cotangent)
    finally:
        pullback.close()

    left = np.real(np.vdot(cotangent, product))
    right = np.real(np.vdot(adjoint_product, tangent))
    assert_allclose(left, right, rtol=1e-6, atol=1e-6)


def test_initial_guess_is_nondifferentiable() -> None:
    root = implicit_root(
        lambda solution, params: solution - params,
        solve=lambda residual, initial: initial - residual(initial),
        linear_solve=lambda _operator, rhs: rhs,
    )

    gradient = grad(lambda initial: np.sum(root(np.ones(3), initial=initial)))(np.zeros(3))

    assert_allclose(gradient, np.zeros(3))


def test_traceable_solvers_support_higher_order_differentiation() -> None:
    root = implicit_root(
        lambda solution, params: solution - params,
        solve=lambda residual, initial: initial - residual(initial),
        linear_solve=lambda _operator, rhs: rhs,
    )
    first = grad(lambda params: np.sum(root(params, initial=np.zeros_like(params)) ** 3))
    second = grad(lambda params: np.sum(first(params)))
    params = np.array([0.5, 1.5, 2.0])

    assert_allclose(first(params), 3 * params**2)
    assert_allclose(second(params), 6 * params)


def test_nonlinear_solver_may_promote_the_initial_dtype() -> None:
    root = implicit_root(
        lambda solution, params: solution - params,
        solve=lambda residual, initial: (
            initial.astype(np.float64) - residual(initial).astype(np.float64)
        ),
        linear_solve=lambda _operator, rhs: rhs,
    )

    result = root(
        np.array([1.0, 2.0], dtype=np.float32),
        initial=np.zeros(2, dtype=np.float32),
    )

    assert result.dtype == np.dtype(np.float64)
    assert_allclose(result, np.array([1.0, 2.0]))


def test_solver_failure_is_an_explicit_public_error() -> None:
    def fail(_residual: Any, _initial: np.ndarray) -> np.ndarray:
        msg = "nonlinear solve did not converge"
        raise ImplicitSolveError(msg)

    root = implicit_root(
        lambda solution, params: solution - params,
        solve=fail,
        linear_solve=lambda _operator, rhs: rhs,
    )

    with pytest.raises(ImplicitSolveError, match="did not converge"):
        root(np.ones(2), initial=np.zeros(2))


def test_root_rejects_solution_or_residual_spec_changes() -> None:
    bad_solution = implicit_root(
        lambda solution, params: solution - params,
        solve=lambda _residual, _initial: np.ones(3),
        linear_solve=lambda _operator, rhs: rhs,
    )
    with pytest.raises(TypeError, match="solution leaf 0"):
        bad_solution(np.ones(2), initial=np.zeros(2))

    bad_residual = implicit_root(
        lambda solution, params: np.sum(solution - params),
        solve=lambda _residual, initial: initial,
        linear_solve=lambda _operator, rhs: rhs,
    )
    with pytest.raises(TypeError, match="residual leaf 0"):
        bad_residual(np.ones(2), initial=np.zeros(2))


def test_root_rejects_abstract_staging_without_tracing_solver_iterations() -> None:
    solve_called = False

    def solve(residual: Any, initial: np.ndarray) -> np.ndarray:
        nonlocal solve_called
        solve_called = True
        return initial - residual(initial)

    root = implicit_root(
        lambda solution, params: solution - params,
        solve=solve,
        linear_solve=lambda _operator, rhs: rhs,
    )

    with pytest.raises(
        TracingError,
        match=r"opaque Python solver callbacks.*concrete dynamic autodiff only",
    ):
        stage(
            root,
            specs=(ArraySpec((2,), "float64"),),
            kw_specs={"initial": ArraySpec((2,), "float64")},
        )
    assert not solve_called
