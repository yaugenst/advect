"""Higher-order transforms built by ordinary transform composition."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from advect.autodiff._ephemeral import linearize_call
from advect.autodiff.api._scalar_boundary import (
    _is_real_python_scalar,
    _unlift_scalar_tree_by_mask,
)
from advect.autodiff.api.common import (
    _prepare_higher_order_inputs,
    _require_array_namespace_for_higher_order,
    _resolve_selected_argnums,
)
from advect.autodiff.api.forward import jvp
from advect.autodiff.api.higher_order_loops import (
    _hessian_diag_reverse_loop,
    _hessian_reverse_loop,
    _HessianLoopContext,
)
from advect.autodiff.api.inputs import _normalize_argnums_for_call, _normalize_argnums_spec
from advect.autodiff.api.reverse import grad, value_and_grad
from advect.core._pytree import tree_flatten

if TYPE_CHECKING:
    from collections.abc import Callable


def _dtype_is_complex(dtype: object) -> bool:
    return bool(getattr(dtype, "kind", None) == "c" or "complex" in str(dtype).lower())


def _hessian_context(
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    argnums: tuple[int, ...],
    single_argnum: bool,
    api_name: str,
) -> _HessianLoopContext:
    selected_argnums = _normalize_argnums_for_call(argnums, nargs=len(args))
    selected_args = tuple(args[index] for index in selected_argnums)
    array_namespace = _require_array_namespace_for_higher_order(
        args=args,
        kwargs=kwargs,
        scalar_fallback_values=selected_args,
    )
    shapes, flat_sizes, dtypes = _prepare_higher_order_inputs(
        array_ns=array_namespace,
        args=args,
        argnums=argnums,
    )
    if any(_dtype_is_complex(dtype) for dtype in dtypes):
        msg = (
            f"{api_name} requires real input leaves. Use hvp() or linearize() for "
            "complex real-linear derivatives until Advect exposes a real-block result type."
        )
        raise ValueError(msg)
    return _HessianLoopContext(
        array_ns=array_namespace,
        primal_shapes=shapes,
        primal_flat_sizes=flat_sizes,
        primal_dtypes=dtypes,
        single_argnum=single_argnum,
    )


def _unlift_hessian_result(
    value: Any,
    *,
    args: tuple[Any, ...],
    argnums: tuple[int, ...],
    diagonal: bool,
) -> Any:
    selected = _normalize_argnums_for_call(argnums, nargs=len(args))
    scalar_inputs = tuple(_is_real_python_scalar(args[index]) for index in selected)
    mask = (
        scalar_inputs
        if diagonal
        else tuple(is_scalar for _row in scalar_inputs for is_scalar in scalar_inputs)
    )
    return _unlift_scalar_tree_by_mask(
        value,
        mask=mask,
    )


def _selected_scalar_mask(
    *,
    args: tuple[Any, ...],
    argnums: tuple[int, ...],
    single_argnum: bool,
) -> tuple[bool, ...]:
    selected = _normalize_argnums_for_call(argnums, nargs=len(args))
    selected_values: Any = (
        args[selected[0]] if single_argnum else tuple(args[index] for index in selected)
    )
    leaves, _treedef = tree_flatten(selected_values)
    return tuple(_is_real_python_scalar(leaf) for leaf in leaves)


def hvp(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., tuple[Any, Any]]:
    """Return a dynamic value-and-Hessian-vector-product transform.

    The returned function evaluates ``f`` and applies its Hessian to one
    selected input-space vector without materializing the dense Hessian. It
    traces the concrete arguments on every call, evaluates ``f`` once, and
    releases the invocation-local linearization before returning.

    Parameters
    ----------
    f
        Callable producing a real scalar or a one-leaf pytree containing a
        real scalar. A `StagedProgram` is accepted, but the returned transform
        is still an ordinary dynamic callable.
    argnums
        One or more positional arguments to differentiate. An integer selects
        one input and uses its pytree directly. A tuple preserves a tuple of
        selected input pytrees in the given order, including for a one-element
        tuple. Negative indices are resolved for each call.

    Returns
    -------
    Callable
        A function called as ``transformed(*args, vectors=vectors, **kwargs)``
        that returns ``(value, product)``. The keyword-only ``vectors`` value
        must match the pytree structure and leaf shapes selected by `argnums`;
        use ``None`` for a static or otherwise untraceable leaf. ``value``
        preserves the output structure of ``f``, and ``product`` preserves the
        integer-versus-tuple selection structure described above.

    Raises
    ------
    IndexError
        If a selected positional index is out of range for the transformed
        call.
    TypeError
        If ``vectors`` is omitted, is not a tuple for a tuple selection, gives
        a non-``None`` tangent for a static leaf, or a selected input contains
        an unsupported Python complex scalar.
    ValueError
        If no argument is selected, positional selections are duplicated, the
        vector arity, pytree, or leaf shape does not match the selected input,
        or ``f`` does not produce a real scalar value.
    NoJVPError
        If an operation needed by the nested derivative has no forward-mode
        rule.
    NoVJPError
        If an operation on the differentiated path has no reverse-mode rule.
    TracingError
        If nested differentiation crosses a first-order-only primitive or
        another unsupported tracing boundary.

    Notes
    -----
    Complex provider arrays are supported under Advect's real-linear
    convention when ``f`` returns a real scalar. Python complex scalars must be
    wrapped in provider zero-dimensional arrays. Use `hessian` only for an
    explicit dense Hessian over real inputs.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> value, product = ad.hvp(lambda x: np.sum(x**2))(
    ...     np.array([1.0, 2.0]), vectors=np.array([3.0, 4.0])
    ... )
    >>> float(value), product.tolist()
    (5.0, [6.0, 8.0])
    """
    # Linearizing value_and_grad keeps the primal value and the Hessian-vector
    # product on the same nested trace.  In particular, stateful callables are
    # evaluated once rather than once for the reported value and again for the
    # differentiated gradient.
    argnums_tuple, single_argnum = _normalize_argnums_spec(argnums)
    jvp_value_and_grad = jvp(value_and_grad(f, argnums=argnums), argnums=argnums)

    @functools.wraps(f)
    def hvp_fn(*args: Any, vectors: Any, **kwargs: Any) -> tuple[Any, Any]:
        _resolve_selected_argnums(argnums=argnums_tuple, nargs=len(args))
        (value, _gradient), (_directional, product) = jvp_value_and_grad(
            *args,
            tangents=vectors,
            **kwargs,
        )
        return value, _unlift_scalar_tree_by_mask(
            product,
            mask=_selected_scalar_mask(
                args=args,
                argnums=argnums_tuple,
                single_argnum=single_argnum,
            ),
        )

    return hvp_fn


def hessian(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., Any]:
    """Return a dynamic transform that assembles an exact dense Hessian.

    Each selected positional argument is one dense real input block. For
    selected shapes ``S_i``, block ``[i][j]`` has shape ``S_i + S_j``: its rows
    index coordinates of the gradient with respect to argument ``i`` and its
    columns index coordinates of argument ``j``.

    Parameters
    ----------
    f
        Callable producing a real scalar or a one-leaf pytree containing a
        real scalar. A `StagedProgram` is accepted, but the returned transform
        is still an ordinary dynamic callable.
    argnums
        One or more positional arguments to differentiate. An integer selects
        one input and returns its dense block directly. A tuple preserves both
        selected-argument axes in the given order and returns a tuple of tuple
        blocks, including a one-by-one structure for a one-element tuple.
        Negative indices are resolved for each call.

    Returns
    -------
    Callable
        A function accepting the arguments of ``f`` and returning dense
        provider arrays. An integer selection returns one array with shape
        ``S + S``. A tuple selection returns blocks ``result[i][j]`` with shape
        ``S_i + S_j``.

    Raises
    ------
    IndexError
        If a selected positional index is out of range for the transformed
        call.
    ValueError
        If no argument is selected, positional selections are duplicated, a
        selected input is complex, or ``f`` does not produce a real scalar
        value.
    AdvectError
        If dense assembly cannot resolve a compatible runtime array namespace
        or represent the selected gradient as dense blocks.
    NoJVPError
        If an operation needed by the nested derivative has no forward-mode
        rule.
    NoVJPError
        If an operation on the differentiated path has no reverse-mode rule.
    TracingError
        If nested differentiation crosses a first-order-only primitive or
        another unsupported tracing boundary.

    Notes
    -----
    Each selected argument must be a real Python scalar or one array-like value
    with a coherent ``shape`` and ``dtype``. Generic tuple and dictionary
    pytrees are supported by `hvp`, but not by dense Hessian assembly.

    Each invocation traces ``f`` at the current values once, reuses that trace
    for the coordinate sweeps, and releases it before returning. No derivative
    graph is cached, and this transform never returns a `StagedProgram`.
    Storage dtypes are promoted with the provider's ``float64`` dtype. A
    derivative column for a selected Python ``int`` or ``float`` is restored
    to a Python scalar. Dense complex Hessians are not represented; use `hvp`
    or `linearize` for complex real-linear derivatives.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> ad.hessian(lambda x: np.sum(x**3))(np.array([1.0, 2.0])).tolist()
    [[6.0, 0.0], [0.0, 12.0]]
    """
    argnums_tuple, single_argnum = _normalize_argnums_spec(argnums)
    gradient_function = grad(f, argnums=argnums)

    @functools.wraps(f)
    def hessian_fn(*args: Any, **kwargs: Any) -> Any:
        context = _hessian_context(
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            single_argnum=single_argnum,
            api_name="hessian",
        )
        gradient, linear = linearize_call(
            gradient_function,
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            argnames=None,
            single_argnum=single_argnum,
            reverse_only=True,
        )
        with linear:
            result = _hessian_reverse_loop(
                context=context,
                linear=linear,
                grad_value=gradient,
            )
        return _unlift_hessian_result(
            result,
            args=args,
            argnums=argnums_tuple,
            diagonal=False,
        )

    return hessian_fn


def hessian_diag(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., Any]:
    """Return a dynamic transform that assembles exact Hessian diagonals.

    For each selected positional argument, the result contains the diagonal of
    that argument's self-Hessian block with the argument's original shape.
    Mixed-argument blocks are omitted.

    Parameters
    ----------
    f
        Callable producing a real scalar or a one-leaf pytree containing a
        real scalar. A `StagedProgram` is accepted, but the returned transform
        is still an ordinary dynamic callable.
    argnums
        One or more positional arguments to differentiate. An integer selects
        one input and returns its diagonal directly. A tuple returns one
        diagonal per selected input in the given order, including a one-tuple
        for a one-element tuple. Negative indices are resolved for each call.

    Returns
    -------
    Callable
        A function accepting the arguments of ``f``. For an integer selection
        with input shape ``S``, it returns one provider array with shape ``S``.
        A tuple selection returns a tuple whose entry ``i`` has shape ``S_i``.

    Raises
    ------
    IndexError
        If a selected positional index is out of range for the transformed
        call.
    ValueError
        If no argument is selected, positional selections are duplicated, a
        selected input is complex, or ``f`` does not produce a real scalar
        value.
    AdvectError
        If dense assembly cannot resolve a compatible runtime array namespace
        or represent the selected gradient as a dense block.
    NoJVPError
        If an operation needed by the nested derivative has no forward-mode
        rule.
    NoVJPError
        If an operation on the differentiated path has no reverse-mode rule.
    TracingError
        If nested differentiation crosses a first-order-only primitive or
        another unsupported tracing boundary.

    Notes
    -----
    Each selected argument must be a real Python scalar or one array-like value
    with a coherent ``shape`` and ``dtype``. Generic tuple and dictionary
    pytrees are supported by `hvp`, but not by dense diagonal assembly.

    This is an exact automatic-differentiation result, not a stochastic
    estimator; computing it may still require one coordinate sweep per
    selected scalar coordinate. Each call traces ``f`` once, releases its
    temporary linearization before returning, and never produces a
    `StagedProgram`. Storage dtypes are promoted with the provider's
    ``float64`` dtype. Selected Python ``int`` and ``float`` inputs have Python
    scalar diagonal entries. Dense complex diagonals are not represented; use
    `hvp` or `linearize` for complex real-linear derivatives.

    Examples
    --------
    >>> import advect as ad
    >>> import numpy as np
    >>> ad.hessian_diag(lambda x: np.sum(x**3))(np.array([1.0, 2.0])).tolist()
    [6.0, 12.0]
    """
    argnums_tuple, single_argnum = _normalize_argnums_spec(argnums)
    gradient_function = grad(f, argnums=argnums)

    @functools.wraps(f)
    def hessian_diag_fn(*args: Any, **kwargs: Any) -> Any:
        context = _hessian_context(
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            single_argnum=single_argnum,
            api_name="hessian_diag",
        )
        gradient, linear = linearize_call(
            gradient_function,
            args=args,
            kwargs=kwargs,
            argnums=argnums_tuple,
            argnames=None,
            single_argnum=single_argnum,
            reverse_only=True,
        )
        with linear:
            result = _hessian_diag_reverse_loop(
                context=context,
                linear=linear,
                grad_value=gradient,
            )
        return _unlift_hessian_result(
            result,
            args=args,
            argnums=argnums_tuple,
            diagonal=True,
        )

    return hessian_diag_fn


__all__ = ["hessian", "hessian_diag", "hvp"]
