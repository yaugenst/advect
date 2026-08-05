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
    """Return a function computing a value and Hessian-vector product.

    Supply the selected input-space vector through the keyword-only
    ``vectors=`` argument of the returned function.

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
    """Return an exact dense Hessian transform for real input leaves.

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
    """Return the exact Hessian diagonal for real input leaves.

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
