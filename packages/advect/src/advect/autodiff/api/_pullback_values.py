"""Backend-neutral cotangent and gradient value helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from advect.autodiff.api._scalar_boundary import (
    _is_boolean_numeric,
    _is_complex_numeric,
    _unlift_scalar_array,
)
from advect.core._array_namespace import _get_array_namespace
from advect.core._pytree import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from advect.core._pytree import TreeDef


def _sum(x: Any, *, axis: Any, keepdims: bool) -> Any:
    """Backend-agnostic sum via Array API namespace when available."""
    xp = _get_array_namespace(x)
    if xp is not None and hasattr(xp, "sum"):
        return xp.sum(x, axis=axis, keepdims=keepdims)
    if hasattr(x, "sum"):
        return x.sum(axis=axis, keepdims=keepdims)
    msg = f"Cannot sum value of type {type(x).__name__}"
    raise TypeError(msg)


def _reshape(x: Any, shape: tuple[int, ...]) -> Any:
    """Backend-agnostic reshape used while normalizing cotangents."""
    xp = _get_array_namespace(x)
    if xp is not None and hasattr(xp, "reshape"):
        return xp.reshape(x, shape)
    reshape = getattr(x, "reshape", None)
    if callable(reshape):
        return reshape(shape)
    msg = f"Cannot reshape value of type {type(x).__name__}"
    raise TypeError(msg)


def _unbroadcast(g: Any, target_shape: tuple[int, ...]) -> Any:
    """Sum ``g`` over broadcast axes to match ``target_shape``."""
    if not hasattr(g, "shape"):
        return g

    g_shape = tuple(g.shape)
    if g_shape == target_shape:
        return g

    if target_shape == ():
        return _sum(g, axis=None, keepdims=False)

    ndim_diff = len(g_shape) - len(target_shape)
    if ndim_diff > 0:
        g = _sum(g, axis=tuple(range(ndim_diff)), keepdims=False)
        g_shape = tuple(g.shape)
    elif ndim_diff < 0:
        missing_rank = -ndim_diff
        if any(dimension != 1 for dimension in target_shape[:missing_rank]):
            msg = (
                "Cotangent rank cannot be restored to the differentiated input: "
                f"cotangent shape {g_shape}, input shape {target_shape}."
            )
            raise ValueError(msg)
        g = _reshape(g, (1,) * missing_rank + g_shape)
        g_shape = tuple(g.shape)

    axes_to_sum: list[int] = []
    for i, (g_dim, t_dim) in enumerate(zip(g_shape, target_shape, strict=True)):
        if t_dim == 1 and g_dim > 1:
            axes_to_sum.append(i)

    if axes_to_sum:
        g = _sum(g, axis=tuple(axes_to_sum), keepdims=True)
    return g


def _ones_like(value: Any) -> Any:
    """Create an all-ones array matching ``value``."""
    xp = _get_array_namespace(value)
    if xp is None:
        msg = (
            "Cannot construct cotangent: no Array API namespace available for value of type "
            f"{type(value).__name__}."
        )
        raise RuntimeError(msg)
    return xp.ones_like(value)


def _zeros_like(value: Any) -> Any:
    """Create an all-zeros array matching ``value``."""
    xp = _get_array_namespace(value)
    if xp is None:
        xp = _get_array_namespace(value, api_version=None)
    if xp is None:
        msg = (
            "Cannot construct zero gradient: no Array API namespace available for value of type "
            f"{type(value).__name__}."
        )
        raise RuntimeError(msg)
    return xp.zeros_like(value)


def _flatten_output_cotangents(output_treedef: TreeDef, g: Any) -> list[Any]:
    if output_treedef.node_type is None:
        return [g]

    g_leaves, g_def = tree_flatten(g)
    if g_def != output_treedef:
        msg = "Cotangent pytree structure does not match the function output structure"
        raise ValueError(msg)
    return g_leaves


def _coerce_output_cotangent_like(cotangent: Any, primal: Any) -> Any:
    """Validate one cotangent leaf against its concrete output primal."""
    if cotangent is None:
        return None
    if _is_boolean_numeric(cotangent):
        msg = "VJP cotangents must be numeric and cannot have boolean dtype"
        raise TypeError(msg)
    if not _is_complex_numeric(primal) and _is_complex_numeric(cotangent):
        msg = "VJP cotangent for a real output cannot have complex dtype"
        raise TypeError(msg)

    primal_shape = getattr(primal, "shape", None)
    if primal_shape is None:
        cotangent_shape = getattr(cotangent, "shape", None)
        if (
            cotangent_shape is not None
            and tuple(int(dimension) for dimension in cotangent_shape) != ()
        ):
            msg = (
                "VJP cotangent shape mismatch: expected (), "
                f"got {tuple(int(dimension) for dimension in cotangent_shape)}."
            )
            raise ValueError(msg)
        return cotangent

    expected_shape = tuple(int(dimension) for dimension in primal_shape)
    cotangent_shape = getattr(cotangent, "shape", None)
    if cotangent_shape is None:
        namespace = _get_array_namespace(primal)
        asarray = None if namespace is None else getattr(namespace, "asarray", None)
        if not callable(asarray):
            msg = (
                "Cannot normalize VJP cotangent for output of type "
                f"{type(primal).__name__}: no array namespace is available."
            )
            raise TypeError(msg)
        cotangent = asarray(cotangent, dtype=getattr(primal, "dtype", None))
        cotangent_shape = getattr(cotangent, "shape", None)

    actual_shape = (
        () if cotangent_shape is None else tuple(int(dimension) for dimension in cotangent_shape)
    )
    if actual_shape != expected_shape:
        msg = f"VJP cotangent shape mismatch: expected {expected_shape}, got {actual_shape}."
        raise ValueError(msg)
    return cotangent


def _build_grad_outputs(output_node_ids: list[int], g_leaves: list[Any]) -> dict[int, Any]:
    gradients: dict[int, Any] = {}
    for node_id, g_leaf in zip(output_node_ids, g_leaves, strict=True):
        if g_leaf is None:
            continue
        previous = gradients.get(node_id)
        gradients[node_id] = g_leaf if previous is None else previous + g_leaf
    return gradients


def _build_grad_tree(
    spec: Any,
    *,
    grads: dict[int, Any],
    active_leaf_positions: frozenset[int] | None = None,
) -> Any:
    grad_leaves: list[Any] = []
    for position, leaf_spec in enumerate(spec.leaf_specs):
        if active_leaf_positions is not None and position not in active_leaf_positions:
            grad_leaves.append(None)
            continue
        node_id = leaf_spec.node_id
        if node_id is None:
            grad_leaves.append(None)
            continue

        grad_val = grads.get(node_id)
        if grad_val is None:
            primal = leaf_spec.primal
            grad_val = None if primal is None else _zeros_like(primal)
        if leaf_spec.restore_python_scalar:
            grad_val = _unlift_scalar_array(grad_val)
        grad_leaves.append(grad_val)
    return tree_unflatten(spec.treedef, grad_leaves)


def _format_backward_result(
    *,
    positional_grads: list[Any],
    named_grads: dict[str, Any],
    single_argnum: bool,
) -> Any:
    if named_grads:
        if not positional_grads:
            return named_grads
        if single_argnum and len(positional_grads) == 1:
            pos_out: Any = positional_grads[0]
        else:
            pos_out = tuple(positional_grads)
        return pos_out, named_grads

    if single_argnum:
        return positional_grads[0]
    return tuple(positional_grads)
