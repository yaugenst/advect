"""Concrete SciPy callback qualification."""

from __future__ import annotations

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import special as scipy_special

import advect as ad
from advect.autodiff.api.implicit import ImplicitSolveError
from advect.scipy import special
from advect.scipy.optimize import root_solver
from advect.scipy.sparse.linalg import gmres_solver


def test_root_callback_preserves_array_shape_and_solves_real_system() -> None:
    target = np.array([[2.0, 3.0], [5.0, 7.0]])
    solve = root_solver()

    solution = solve(lambda value: value * value - target, np.ones_like(target))

    assert solution.shape == target.shape
    assert_allclose(solution * solution, target, rtol=1e-9, atol=1e-9)


def test_root_callback_supports_complex_systems_through_real_packing() -> None:
    target = np.array([1.0 + 2.0j, -0.5 + 0.7j])
    solve = root_solver()

    solution = solve(lambda value: value - target, np.zeros_like(target))

    assert solution.shape == target.shape
    assert np.iscomplexobj(solution)
    assert_allclose(solution, target, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("initial", "expected_type"),
    [
        (1.0, float),
        (1.0 + 0.0j, complex),
        (np.float32(1), np.float64),
        (np.float64(1), np.float64),
        (np.complex64(1), np.complex128),
        (np.complex128(1), np.complex128),
    ],
)
def test_root_callback_preserves_scalar_category(
    initial: object,
    expected_type: type[object],
) -> None:
    seen: list[type[object]] = []

    def residual(value: object) -> object:
        seen.append(type(value))
        return value - 2  # type: ignore[operator]

    solution = root_solver()(residual, initial)

    assert type(solution) is expected_type
    assert seen
    assert all(value_type is expected_type for value_type in seen)
    assert_allclose(solution, 2)


def test_root_callback_preserves_zero_dimensional_array_category() -> None:
    initial = np.array(1.0)
    seen: list[type[object]] = []

    def residual(value: object) -> object:
        seen.append(type(value))
        return value - 2  # type: ignore[operator]

    solution = root_solver()(residual, initial)

    assert type(solution) is np.ndarray
    assert np.shape(solution) == ()
    assert seen
    assert all(value_type is np.ndarray for value_type in seen)


def test_root_callback_rejects_complex_residual_for_real_state() -> None:
    solve = root_solver()

    with pytest.raises(ImplicitSolveError, match="complex values for a real state"):
        solve(lambda value: value - 1 + 1j, np.array([0.0]))


def test_root_callback_turns_nonconvergence_into_implicit_solve_error() -> None:
    solve = root_solver(options={"maxfev": 1})

    with pytest.raises(ImplicitSolveError, match="did not converge"):
        solve(lambda value: value * value + 1, np.array([1.0]))


def test_root_callback_rejects_residual_shape_changes() -> None:
    solve = root_solver()

    with pytest.raises(ImplicitSolveError, match="must return the solution shape"):
        solve(lambda _value: np.ones(3), np.ones(2))


def test_root_callback_rejects_non_numpy_provider_values() -> None:
    solve = root_solver()
    initial = strict.asarray([1.0, 1.0], dtype=strict.float64)

    with pytest.raises(ImplicitSolveError, match="concrete NumPy arrays or scalars"):
        solve(lambda value: value - 1, initial)


def test_gmres_callback_preserves_shape_and_supports_complex_values() -> None:
    matrix = np.array([[3.0 + 0.2j, 1.0 - 0.1j], [0.5 + 0.3j, 2.0 - 0.4j]])
    rhs = np.array([[1.0 + 0.5j], [2.0 - 0.25j]])
    solve = gmres_solver(rtol=1e-12, atol=1e-12)

    solution = solve(lambda value: matrix @ value, rhs)

    assert solution.shape == rhs.shape
    assert_allclose(matrix @ solution, rhs, rtol=1e-10, atol=1e-10)


def test_gmres_realifies_a_complex_real_linear_operator() -> None:
    rhs = np.array([1.5 + 0.7j, -2.1 + 1.3j])
    solve = gmres_solver(rtol=1e-12, atol=1e-12)

    solution = solve(lambda value: 2 * value + np.conj(value), rhs)

    expected = rhs.real / 3 + 1j * rhs.imag
    assert_allclose(solution, expected, rtol=1e-11, atol=1e-11)
    assert_allclose(2 * solution + np.conj(solution), rhs, rtol=1e-11, atol=1e-11)


def test_gmres_callback_handles_identity_operator_without_aliasing() -> None:
    rhs = np.array([1.5, -2.1])
    solve = gmres_solver(rtol=1e-12, atol=1e-12)

    solution = solve(lambda value: value, rhs)

    assert_allclose(solution, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("rhs", "expected_type"),
    [
        (1.0, float),
        (1.0 + 0.5j, complex),
        (np.float32(1), np.float32),
        (np.float64(1), np.float64),
        (np.complex64(1), np.complex64),
        (np.complex128(1), np.complex128),
    ],
)
def test_gmres_callback_preserves_scalar_category_and_inexact_dtype(
    rhs: object,
    expected_type: type[object],
) -> None:
    seen: list[type[object]] = []

    def operator(value: object) -> object:
        seen.append(type(value))
        return 2 * value  # type: ignore[operator]

    solution = gmres_solver(rtol=1e-12, atol=1e-12)(operator, rhs)

    assert type(solution) is expected_type
    assert seen
    assert all(value_type is expected_type for value_type in seen)
    assert_allclose(solution, rhs / 2)  # type: ignore[operator]


def test_gmres_callback_preserves_zero_dimensional_array_category() -> None:
    rhs = np.array(1.0)
    seen: list[type[object]] = []

    def operator(value: object) -> object:
        seen.append(type(value))
        return 2 * value  # type: ignore[operator]

    solution = gmres_solver(rtol=1e-12, atol=1e-12)(operator, rhs)

    assert type(solution) is np.ndarray
    assert np.shape(solution) == ()
    assert seen
    assert all(value_type is np.ndarray for value_type in seen)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_gmres_callback_preserves_inexact_rhs_dtype(dtype: object) -> None:
    rhs = np.array([1.5, -2.1], dtype=dtype)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        rhs = rhs + np.array([0.7j, 1.3j], dtype=dtype)
    solve = gmres_solver(rtol=1e-6, atol=1e-7)

    solution = solve(lambda value: 2 * value, rhs)

    assert solution.dtype == rhs.dtype
    assert_allclose(solution, rhs / 2, rtol=2e-6, atol=2e-6)


def test_gmres_callback_reports_nonconvergence() -> None:
    rhs = np.array([1.0, 2.0])
    solve = gmres_solver(rtol=0.0, atol=0.0, maxiter=1)

    with pytest.raises(ImplicitSolveError, match="did not converge"):
        solve(lambda value: np.array([value[1], 0.0]), rhs)


def test_gmres_callback_rejects_operator_shape_changes() -> None:
    solve = gmres_solver()

    with pytest.raises(ImplicitSolveError, match="must preserve"):
        solve(lambda _value: np.ones(3), np.ones(2))


def test_gmres_callback_rejects_non_numpy_provider_values() -> None:
    solve = gmres_solver()
    rhs = strict.asarray([1.0, 2.0], dtype=strict.float64)

    with pytest.raises(ImplicitSolveError, match="concrete NumPy arrays or scalars"):
        solve(lambda value: value, rhs)


def test_scipy_callbacks_drive_implicit_root_end_to_end() -> None:
    solve_root = ad.implicit_root(
        lambda solution, params: solution * solution - params,
        solve=root_solver(),
        linear_solve=gmres_solver(rtol=1e-12, atol=1e-12),
    )
    params = np.array([2.0, 3.0, 5.0])
    initial = np.ones_like(params)

    solution = solve_root(params, initial=initial)
    gradient = ad.grad(lambda runtime_params: np.sum(solve_root(runtime_params, initial=initial)))(
        params
    )

    assert_allclose(solution, np.sqrt(params), rtol=1e-10, atol=1e-10)
    assert_allclose(gradient, 0.5 / np.sqrt(params), rtol=1e-9, atol=1e-9)


def test_scipy_callbacks_drive_python_scalar_implicit_root_end_to_end() -> None:
    solve_root = ad.implicit_root(
        lambda solution, params: solution * solution - params,
        solve=root_solver(),
        linear_solve=gmres_solver(rtol=1e-12, atol=1e-12),
    )

    solution = solve_root(4.0, initial=1.0)
    value, directional = ad.jvp(lambda params: solve_root(params, initial=1.0))(4.0, tangents=1.0)
    gradient = ad.grad(lambda params: solve_root(params, initial=1.0))(4.0)

    assert type(solution) is float
    assert type(value) is float
    assert type(directional) is float
    assert type(gradient) is float
    assert_allclose((solution, value), (2.0, 2.0), rtol=1e-10, atol=1e-10)
    assert_allclose((directional, gradient), (0.25, 0.25), rtol=1e-9, atol=1e-9)


def test_scipy_special_residual_composes_with_implicit_solver_callbacks() -> None:
    solve_root = ad.implicit_root(
        lambda solution, params: special.erf(solution) - params,
        solve=root_solver(),
        linear_solve=gmres_solver(rtol=1e-12, atol=1e-12),
    )
    params = np.array([0.1, -0.3])
    initial = np.array([0.1, -0.3])

    solution = solve_root(params, initial=initial)
    gradient = ad.grad(lambda value: np.sum(solve_root(value, initial=initial)))(params)
    expected_gradient = np.sqrt(np.pi) / 2 * np.exp(solution * solution)

    assert_allclose(solution, scipy_special.erfinv(params), rtol=1e-10, atol=1e-10)
    assert_allclose(gradient, expected_gradient, rtol=1e-9, atol=1e-9)


def test_scipy_callbacks_drive_float32_implicit_root_end_to_end() -> None:
    solve_root = ad.implicit_root(
        lambda solution, params: solution * solution - params,
        solve=lambda residual, initial: initial - residual(initial) / (2 * initial),
        linear_solve=gmres_solver(rtol=1e-6, atol=1e-7),
    )
    params = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    initial = np.sqrt(params)

    gradient = ad.grad(lambda runtime_params: np.sum(solve_root(runtime_params, initial=initial)))(
        params
    )

    assert gradient.dtype == params.dtype
    assert_allclose(gradient, 0.5 / np.sqrt(params), rtol=2e-6, atol=2e-6)


def test_scipy_callbacks_preserve_real_linear_complex_implicit_jvp() -> None:
    solve_root = ad.implicit_root(
        lambda solution, params: 2 * solution + np.conj(solution) - params,
        solve=root_solver(),
        linear_solve=gmres_solver(rtol=1e-12, atol=1e-12),
    )
    params = np.array([1.5 + 0.7j, -2.1 + 1.3j])
    tangent = np.array([0.3 - 0.8j, 1.2 + 0.4j])
    initial = np.zeros_like(params)

    solution, directional = ad.jvp(
        lambda runtime_params: solve_root(runtime_params, initial=initial)
    )(params, tangents=tangent)

    assert_allclose(solution, params.real / 3 + 1j * params.imag, rtol=1e-10, atol=1e-10)
    assert_allclose(
        directional,
        tangent.real / 3 + 1j * tangent.imag,
        rtol=1e-9,
        atol=1e-9,
    )


def test_scipy_solver_callbacks_are_an_explicit_higher_order_boundary() -> None:
    solve_root = ad.implicit_root(
        lambda solution, params: solution * solution - params,
        solve=root_solver(),
        linear_solve=gmres_solver(),
    )
    initial = np.array([1.0, 1.0])
    first_derivative = ad.grad(lambda params: np.sum(solve_root(params, initial=initial)))

    with pytest.raises(ImplicitSolveError, match="first-order dynamic implicit differentiation"):
        ad.grad(lambda params: np.sum(first_derivative(params)))(np.array([2.0, 3.0]))


def test_linear_solver_failure_preserves_implicit_solve_error_through_grad() -> None:
    solve_root = ad.implicit_root(
        lambda state, params: 0 * state + (params - params),
        solve=lambda _residual, initial: initial,
        linear_solve=gmres_solver(rtol=0.0, atol=0.0, maxiter=1),
    )

    with pytest.raises(ImplicitSolveError, match="GMRES did not converge"):
        ad.grad(lambda params: np.sum(solve_root(params, initial=np.ones(2))))(np.array([2.0, 3.0]))
