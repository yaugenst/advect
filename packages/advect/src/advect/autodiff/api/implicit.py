"""Dynamic implicit differentiation for converged nonlinear roots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from advect.autodiff.api.forward import linearize
from advect.core._array_api.providers import (
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._context import _get_active_trace_kind, is_tracing
from advect.core._errors import AdvectError, TracingError
from advect.core._primitive import primitive
from advect.core._pytree import tree_flatten, tree_map, tree_unflatten

if TYPE_CHECKING:
    from advect.autodiff._ephemeral import LinearMap
    from advect.core._primitive import Primitive
    from advect.core._pytree import TreeDef


type _ResidualFunction[SolutionT, ParamsT] = Callable[[SolutionT, ParamsT], SolutionT]
type _RootSolver[SolutionT] = Callable[[Callable[[SolutionT], SolutionT], SolutionT], SolutionT]
type _LinearSolver[SolutionT] = Callable[[Callable[[SolutionT], SolutionT], SolutionT], SolutionT]


class _ImplicitRootCallable[ParamsT, SolutionT](Protocol):
    def __call__(self, params: ParamsT, *, initial: SolutionT) -> SolutionT: ...


class _Negatable(Protocol):
    def __neg__(self) -> object: ...


class ImplicitSolveError(AdvectError):
    """A nonlinear or linear solve did not produce its promised solution."""


_RESIDUAL_ARGUMENT_COUNT = 2


@dataclass(frozen=True, slots=True)
class _ImplicitRootConfig[ParamsT, SolutionT]:
    residual: _ResidualFunction[SolutionT, ParamsT]
    solve: _RootSolver[SolutionT]
    linear_solve: _LinearSolver[SolutionT]
    transpose_solve: _LinearSolver[SolutionT]
    params_treedef: TreeDef
    initial_treedef: TreeDef


def _leaf_spec(value: object) -> tuple[tuple[int, ...], str, str, str | None]:
    shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        value_type = type(value)
        if value_type is bool:
            dtype = "bool"
        elif value_type is int:
            dtype = "int64"
        elif value_type is float:
            dtype = "float64"
        elif value_type is complex:
            dtype = "complex128"
        else:
            dtype = value_type.__name__
    namespace = _get_array_namespace(value)
    provider = (
        "python"
        if namespace is None
        else (_get_backend_key_from_namespace(namespace) or type(namespace).__module__)
    )
    device = getattr(value, "device", None)
    if device is None and provider == "numpy":
        device = "cpu"
    return shape, str(dtype), provider, None if device is None else str(device)


def _validate_same_spec(
    actual: object,
    expected: object,
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


def _zero_like(value: object) -> object:
    namespace = _get_array_namespace(value)
    zeros_like = getattr(namespace, "zeros_like", None) if namespace is not None else None
    if callable(zeros_like):
        return zeros_like(value)
    if isinstance(value, (bool, int, float, complex)):
        return value * 0
    msg = f"Cannot construct an implicit-root tangent for {type(value).__name__}"
    raise TypeError(msg)


def _zero_tree(value: object) -> object:
    return tree_map(_zero_like, value)


def _negate_leaf(leaf: object) -> object:
    return None if leaf is None else -cast("_Negatable", leaf)


def _fill_missing_tangents(
    primals: tuple[object, ...],
    tangents: tuple[object | None, ...],
    treedef: TreeDef,
) -> object:
    leaves = [
        _zero_like(primal) if tangent is None else tangent
        for primal, tangent in zip(primals, tangents, strict=True)
    ]
    return tree_unflatten(treedef, leaves)


def _negate_tree(value: object) -> object:
    return tree_map(_negate_leaf, value)


def _materialize_cotangent_leaf(cotangent: object, primal: object) -> object:
    return _zero_like(primal) if cotangent is None else cotangent


def _materialize_cotangent(cotangent: object, primal: object) -> object:
    return tree_map(_materialize_cotangent_leaf, cotangent, primal)


def _restore_flat_output(value: object, treedef: TreeDef, *, label: str) -> object:
    _leaves, actual_treedef = tree_flatten(value)
    if actual_treedef == treedef or treedef.node_type is None:
        return value
    if isinstance(value, tuple) and len(value) == treedef.num_leaves:
        return tree_unflatten(treedef, list(value))
    msg = f"{label} pytree structure does not match the root solution"
    raise TypeError(msg)


def _normalize_scalar_solution_provider(solution: object, params: object) -> object:
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


def _solve_root[SolutionT, ParamsT](
    residual: _ResidualFunction[SolutionT, ParamsT],
    solve: _RootSolver[SolutionT],
    *,
    params: ParamsT,
    initial: SolutionT,
) -> SolutionT:
    solution = solve(lambda candidate: residual(candidate, params), initial)
    _validate_same_spec(
        solution,
        initial,
        label="implicit root solution",
        check_dtype=False,
    )
    solution = cast("SolutionT", _normalize_scalar_solution_provider(solution, params))
    residual_value = residual(solution, params)
    _validate_same_spec(residual_value, solution, label="implicit residual")
    return solution


def _residual_linearization(
    residual: _ResidualFunction[object, object],
    *,
    solution: object,
    params: object,
) -> tuple[object, LinearMap]:
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


def _split_residual_pullback(value: object) -> tuple[object, object]:
    if not isinstance(value, tuple) or len(value) != _RESIDUAL_ARGUMENT_COUNT:
        msg = "implicit residual pullback did not return state and parameter cotangents"
        raise TypeError(msg)
    return value


def _build_implicit_primitive() -> Primitive[..., object]:
    @primitive(
        name="advect_internal.implicit_root",
        static_argnames=("config",),
        nondiff_argnames=("initial",),
    )
    def implementation(
        params: object,
        initial: object,
        config: _ImplicitRootConfig[object, object],
    ) -> object:
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
        config: _ImplicitRootConfig[object, object],
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
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
        *,
        config: _ImplicitRootConfig[object, object],
    ) -> object:
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

            def apply_state(direction: object) -> object:
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
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
        *,
        config: _ImplicitRootConfig[object, object],
    ) -> tuple[object | None, ...]:
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

            def apply_state_transpose(cotangent_value: object) -> object:
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


def implicit_root[ParamsT, SolutionT](
    residual: _ResidualFunction[SolutionT, ParamsT],
    *,
    solve: _RootSolver[SolutionT],
    linear_solve: _LinearSolver[SolutionT],
    transpose_solve: _LinearSolver[SolutionT] | None = None,
) -> _ImplicitRootCallable[ParamsT, SolutionT]:
    """Build a dynamic transform for a converged implicit solution.

    The returned callable solves ``residual(solution, params) == 0`` and
    differentiates the defining equation without recording the nonlinear
    solver's iterations on the surrounding tape.

    Parameters
    ----------
    residual
        Trace-compatible callable with signature ``residual(solution, params)``.
        Its result must have the same pytree structure and leaf shape, dtype,
        array provider, and device as ``solution``.
    solve
        Nonlinear callback with signature
        ``solve(residual_at_params, initial) -> solution``. Advect supplies
        ``residual_at_params(candidate)`` with ``params`` fixed. Returning
        certifies convergence. The solution must match ``initial`` in pytree
        structure, leaf shape, provider, and device; its dtype may be promoted.
    linear_solve
        Matrix-free callback with signature
        ``linear_solve(operator, rhs) -> solution_tangent``. For a JVP,
        ``operator(direction)`` applies the residual's state Jacobian and
        ``rhs`` is the negative parameter-forcing tangent. The returned value
        must match the solved solution's pytree and leaf specifications.
    transpose_solve
        Matrix-free callback with signature
        ``transpose_solve(operator, rhs) -> residual_cotangent``. In reverse
        mode, ``operator`` applies the real adjoint of the residual's state
        Jacobian and ``rhs`` is the solution cotangent. The result must match
        the residual value's pytree and leaf specifications. ``None`` reuses
        ``linear_solve`` with this adjoint operator.

    Returns
    -------
    Callable
        A callable with signature ``root(params, *, initial) -> solution``.
        ``params`` and ``initial`` may be pytrees. ``initial`` selects a root
        but is excluded from the implicit derivative, so an enclosing
        derivative with respect to it is zero. A Python scalar solution is
        moved to the parameter array provider when one is available.

    Raises
    ------
    TypeError
        If a callback is not callable, or if a nonlinear solution, residual,
        tangent solve, or transpose solve violates its required pytree or leaf
        specification.
    TracingError
        If abstract staging reaches the returned root. Opaque Python solver
        callbacks have no durable staged representation.
    ImplicitSolveError
        Propagated when a nonlinear or linear callback uses this exception to
        report failure. A callback return is otherwise treated as a successful
        solve; Advect does not independently test convergence.

    Notes
    -----
    This is a concrete dynamic boundary. `stage` rejects the root before
    calling ``solve``; stage explicit solver iterations or define a custom
    primitive with a closed abstract rule when a durable program is required.
    Higher-order dynamic derivatives require ``residual`` and every solver
    callback reached by the nested transform to accept nested traced values.
    Concrete adapters such as the bundled SciPy callbacks intentionally form a
    first-order boundary.

    A derivative application creates one joint linearization of ``residual``
    at the solved value and closes it before returning, including on callback
    failure. The root wrapper retains the four callbacks for its lifetime but
    creates no user-managed resource. A `Pullback` or `LinearMap` returned by
    an enclosing transform still follows that transform's documented lifetime.

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

    def root(params: ParamsT, *, initial: SolutionT) -> SolutionT:
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
        return cast(
            "SolutionT",
            _implicit_root_operation._call_dynamic_only(  # noqa: SLF001
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
            ),
        )

    root.__name__ = getattr(residual, "__name__", "implicit_root")
    root.__qualname__ = getattr(residual, "__qualname__", root.__name__)
    root.__doc__ = (
        f"Implicit solution wrapper for {root.__qualname__}; call as root(params, initial=guess)."
    )
    return root


__all__ = ["ImplicitSolveError", "implicit_root"]
