"""Dynamic implicit differentiation for converged nonlinear roots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from advect.autodiff.api.forward import linearize
from advect.core._array_namespace import (
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._context import _get_active_trace_kind, is_tracing
from advect.core._errors import AdvectError, TracingError
from advect.core._primitive import primitive
from advect.core._pytree import tree_flatten, tree_map, tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._primitive import Primitive
    from advect.core._pytree import TreeDef


type ResidualFunction = Callable[[Any, Any], Any]
type RootSolver = Callable[[Callable[[Any], Any], Any], Any]
type LinearSolver = Callable[[Callable[[Any], Any], Any], Any]


class ImplicitSolveError(AdvectError):
    """A nonlinear or linear solve did not produce its promised solution."""


_RESIDUAL_ARGUMENT_COUNT = 2


@dataclass(frozen=True, slots=True)
class _ImplicitRootConfig:
    residual: ResidualFunction
    solve: RootSolver
    linear_solve: LinearSolver
    transpose_solve: LinearSolver
    params_treedef: TreeDef
    initial_treedef: TreeDef


def _leaf_spec(value: Any) -> tuple[tuple[int, ...], str, str, str | None]:
    shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = {
            bool: "bool",
            int: "int64",
            float: "float64",
            complex: "complex128",
        }.get(type(value), type(value).__name__)
    namespace = _get_array_namespace(value)
    provider = (
        "python"
        if namespace is None
        else (_get_backend_key_from_namespace(namespace) or type(namespace).__module__)
    )
    device = getattr(value, "device", None)
    return shape, str(dtype), provider, None if device is None else str(device)


def _validate_same_spec(
    actual: Any,
    expected: Any,
    *,
    label: str,
    check_dtype: bool = True,
) -> None:
    actual_leaves, actual_treedef = tree_flatten(actual)
    expected_leaves, expected_treedef = tree_flatten(expected)
    if actual_treedef != expected_treedef:
        msg = f"{label} pytree structure must match the root solution"
        raise TypeError(msg)
    for index, (actual_leaf, expected_leaf) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        actual_spec = _leaf_spec(actual_leaf)
        expected_spec = _leaf_spec(expected_leaf)
        shape_differs = actual_spec[0] != expected_spec[0]
        dtype_differs = check_dtype and actual_spec[1] != expected_spec[1]
        provider_differs = actual_spec[2:] != expected_spec[2:]
        if shape_differs or dtype_differs or provider_differs:
            expected_description = (
                f"shape/dtype/provider {expected_spec}"
                if check_dtype
                else f"shape/provider {(expected_spec[0], *expected_spec[2:])}"
            )
            actual_description = (
                f"{actual_spec}" if check_dtype else f"{(actual_spec[0], *actual_spec[2:])}"
            )
            msg = (
                f"{label} leaf {index} must match the root solution spec: "
                f"expected {expected_description}, got {actual_description}"
            )
            raise TypeError(msg)


def _zero_like(value: Any) -> Any:
    namespace = _get_array_namespace(value)
    zeros_like = getattr(namespace, "zeros_like", None) if namespace is not None else None
    if callable(zeros_like):
        return zeros_like(value)
    if isinstance(value, (bool, int, float, complex)):
        return value * 0
    msg = f"Cannot construct an implicit-root tangent for {type(value).__name__}"
    raise TypeError(msg)


def _zero_tree(value: Any) -> Any:
    return tree_map(_zero_like, value)


def _fill_missing_tangents(
    primals: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
    treedef: TreeDef,
) -> Any:
    leaves = [
        _zero_like(primal) if tangent is None else tangent
        for primal, tangent in zip(primals, tangents, strict=True)
    ]
    return tree_unflatten(treedef, leaves)


def _negate_tree(value: Any) -> Any:
    return tree_map(lambda leaf: None if leaf is None else -leaf, value)


def _materialize_cotangent(cotangent: Any, primal: Any) -> Any:
    return tree_map(
        lambda cotangent_leaf, primal_leaf: (
            _zero_like(primal_leaf) if cotangent_leaf is None else cotangent_leaf
        ),
        cotangent,
        primal,
    )


def _restore_flat_output(value: Any, treedef: TreeDef, *, label: str) -> Any:
    _leaves, actual_treedef = tree_flatten(value)
    if actual_treedef == treedef or treedef.node_type is None:
        return value
    if isinstance(value, tuple) and len(value) == treedef.num_leaves:
        return tree_unflatten(treedef, list(value))
    msg = f"{label} pytree structure does not match the root solution"
    raise TypeError(msg)


def _normalize_scalar_solution_provider(solution: Any, params: Any) -> Any:
    """Move Python scalar roots onto the differentiable parameter provider."""
    params_leaves, _params_treedef = tree_flatten(params)
    namespace = next(
        (
            candidate
            for leaf in params_leaves
            if (candidate := _get_array_namespace(leaf)) is not None
        ),
        None,
    )
    if namespace is None:
        return solution

    asarray = getattr(namespace, "asarray", None)
    if not callable(asarray):
        return solution
    solution_leaves, solution_treedef = tree_flatten(solution)
    normalized = [
        asarray(leaf) if isinstance(leaf, (bool, int, float, complex)) else leaf
        for leaf in solution_leaves
    ]
    return tree_unflatten(solution_treedef, normalized)


def _solve_root(
    residual: ResidualFunction,
    solve: RootSolver,
    *,
    params: Any,
    initial: Any,
) -> Any:
    solution = solve(lambda candidate: residual(candidate, params), initial)
    _validate_same_spec(
        solution,
        initial,
        label="implicit root solution",
        check_dtype=False,
    )
    solution = _normalize_scalar_solution_provider(solution, params)
    residual_value = residual(solution, params)
    _validate_same_spec(residual_value, solution, label="implicit residual")
    return solution


def _residual_linearization(
    residual: ResidualFunction,
    *,
    solution: Any,
    params: Any,
) -> tuple[Any, Any]:
    residual_value, residual_linear = linearize(
        residual,
        solution,
        params,
        argnums=(0, 1),
    )
    try:
        _validate_same_spec(residual_value, solution, label="implicit residual")
    except Exception:
        residual_linear.close()
        raise
    return residual_value, residual_linear


def _split_residual_pullback(value: Any) -> tuple[Any, Any]:
    if not isinstance(value, tuple) or len(value) != _RESIDUAL_ARGUMENT_COUNT:
        msg = "implicit residual pullback did not return state and parameter cotangents"
        raise TypeError(msg)
    return value


def _build_implicit_primitive() -> Primitive[..., Any]:
    @primitive(
        name="advect_internal.implicit_root",
        static_argnames=("config",),
        nondiff_argnames=("initial",),
    )
    def implementation(
        params: object,
        initial: object,
        config: _ImplicitRootConfig,
    ) -> Any:
        params_count = config.params_treedef.num_leaves
        initial_count = config.initial_treedef.num_leaves
        params_leaves, actual_params_treedef = tree_flatten(params)
        initial_leaves, actual_initial_treedef = tree_flatten(initial)
        if (
            actual_params_treedef != config.params_treedef
            or len(params_leaves) != params_count
            or actual_initial_treedef != config.initial_treedef
            or len(initial_leaves) != initial_count
        ):
            msg = "implicit_root() call structure changed during its concrete solve"
            raise TypeError(msg)
        return _solve_root(
            config.residual,
            config.solve,
            params=params,
            initial=initial,
        )

    @implementation.def_abstract
    def abstract(
        params: object,
        initial: object,
        config: _ImplicitRootConfig,
    ) -> object:
        del params, initial, config
        msg = (
            "implicit_root() contains opaque Python solver callbacks and supports "
            "concrete dynamic autodiff only. Stage explicit solver iterations or "
            "define a custom primitive with a closed abstract rule."
        )
        raise TracingError(msg)

    @implementation.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        config: _ImplicitRootConfig,
    ) -> Any:
        params_count = config.params_treedef.num_leaves
        params = tree_unflatten(config.params_treedef, list(primals[:params_count]))
        solution = _restore_flat_output(
            output,
            config.initial_treedef,
            label="implicit root output",
        )
        params_tangent = _fill_missing_tangents(
            primals[:params_count],
            tangents[:params_count],
            config.params_treedef,
        )
        _residual_value, residual_linear = _residual_linearization(
            config.residual,
            solution=solution,
            params=params,
        )
        try:
            zero_solution = _zero_tree(solution)
            zero_params = _zero_tree(params)

            def apply_state(direction: Any) -> Any:
                return residual_linear((direction, zero_params))

            forcing = residual_linear((zero_solution, params_tangent))
            solution_tangent = config.linear_solve(apply_state, _negate_tree(forcing))
            _validate_same_spec(
                solution_tangent,
                solution,
                label="implicit linear solve result",
            )
            return solution_tangent
        finally:
            residual_linear.close()

    @implementation.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        config: _ImplicitRootConfig,
    ) -> tuple[Any | None, ...]:
        params_count = config.params_treedef.num_leaves
        initial_count = config.initial_treedef.num_leaves
        params = tree_unflatten(config.params_treedef, list(primals[:params_count]))
        solution = _restore_flat_output(
            output,
            config.initial_treedef,
            label="implicit root output",
        )
        solution_cotangent = _restore_flat_output(
            cotangent,
            config.initial_treedef,
            label="implicit root cotangent",
        )
        solution_cotangent = _materialize_cotangent(
            solution_cotangent,
            solution,
        )
        residual_value, residual_linear = _residual_linearization(
            config.residual,
            solution=solution,
            params=params,
        )
        try:

            def apply_state_transpose(cotangent_value: Any) -> Any:
                state_cotangent, _parameter_cotangent = _split_residual_pullback(
                    residual_linear.pullback(cotangent_value)
                )
                return state_cotangent

            residual_cotangent = config.transpose_solve(
                apply_state_transpose,
                solution_cotangent,
            )
            _validate_same_spec(
                residual_cotangent,
                residual_value,
                label="implicit transpose solve result",
            )
            _state_cotangent, parameter_cotangent = _split_residual_pullback(
                residual_linear.pullback(residual_cotangent)
            )
            parameter_cotangent = _negate_tree(parameter_cotangent)
            parameter_leaves, parameter_gradient_treedef = tree_flatten(parameter_cotangent)
            if parameter_gradient_treedef != config.params_treedef:
                msg = "implicit parameter gradient structure changed during transpose"
                raise TypeError(msg)
            return (
                *parameter_leaves,
                *((None,) * initial_count),
            )
        finally:
            residual_linear.close()

    return implementation


_implicit_root_operation = _build_implicit_primitive()


def implicit_root(
    residual: ResidualFunction,
    *,
    solve: RootSolver,
    linear_solve: LinearSolver,
    transpose_solve: LinearSolver | None = None,
) -> Callable[..., Any]:
    """Differentiate a converged solution of ``residual(solution, params) == 0``.

    ``solve(residual_at_params, initial)`` performs the nonlinear solve without
    tracing its iterations. ``linear_solve(operator, rhs)`` solves the
    matrix-free tangent system. ``transpose_solve`` solves its real adjoint and
    defaults to ``linear_solve``.

    A successful callback return certifies convergence. Nonlinear and linear
    solver adapters must raise :class:`ImplicitSolveError` when they fail.
    ``initial`` selects a root but is explicitly nondifferentiable.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> def solve(residual_at_params, initial):
    ...     return initial - residual_at_params(initial)
    >>> def linear_solve(operator, rhs):
    ...     return rhs / operator(np.ones_like(rhs))
    >>> root = ad.implicit_root(
    ...     lambda solution, params: solution - params,
    ...     solve=solve,
    ...     linear_solve=linear_solve,
    ... )
    >>> gradient = ad.grad(lambda params: root(params, initial=np.array(0.0)))(np.array(3.0))
    >>> float(gradient)
    1.0
    """
    if not callable(residual):
        msg = "implicit_root residual must be callable"
        raise TypeError(msg)
    if not callable(solve):
        msg = "implicit_root solve must be callable"
        raise TypeError(msg)
    if not callable(linear_solve):
        msg = "implicit_root linear_solve must be callable"
        raise TypeError(msg)
    if transpose_solve is not None and not callable(transpose_solve):
        msg = "implicit_root transpose_solve must be callable or None"
        raise TypeError(msg)
    resolved_transpose_solve = linear_solve if transpose_solve is None else transpose_solve

    def root(params: Any, *, initial: Any) -> Any:
        if _get_active_trace_kind() == "stage_abstract":
            msg = (
                "implicit_root() contains opaque Python solver callbacks and supports "
                "concrete dynamic autodiff only. Stage explicit solver iterations or "
                "define a custom primitive with a closed abstract rule."
            )
            raise TracingError(msg)
        if not is_tracing():
            return _solve_root(
                residual,
                solve,
                params=params,
                initial=initial,
            )
        _params_leaves, params_treedef = tree_flatten(params)
        _initial_leaves, initial_treedef = tree_flatten(initial)
        return _implicit_root_operation._call_dynamic_only(  # noqa: SLF001
            params=params,
            initial=initial,
            config=_ImplicitRootConfig(
                residual=residual,
                solve=solve,
                linear_solve=linear_solve,
                transpose_solve=resolved_transpose_solve,
                params_treedef=params_treedef,
                initial_treedef=initial_treedef,
            ),
        )

    root.__name__ = getattr(residual, "__name__", "implicit_root")
    root.__qualname__ = getattr(residual, "__qualname__", root.__name__)
    root.__doc__ = (
        f"Implicit solution wrapper for {root.__qualname__}; call as root(params, initial=guess)."
    )
    return root


__all__ = ["ImplicitSolveError", "implicit_root"]
