"""Scalar output helpers shared by reverse-mode API entrypoints."""

from __future__ import annotations

from typing import Any

from advect.autodiff.api._pullback_values import _ones_like
from advect.autodiff.api._scalar_boundary import _is_real_python_scalar
from advect.core._pytree import tree_flatten, tree_unflatten


def _is_scalar_value(value: Any) -> bool:
    """Recognize concrete Python scalar values."""
    return _is_real_python_scalar(value) or type(value) is complex


def _scalar_cotangent_for_output(
    *,
    out_leaf: Any,
    out_treedef: Any,
) -> Any:
    g_leaf = _scalar_cotangent_leaf(out_leaf)
    if out_treedef.node_type is None:
        return g_leaf
    return tree_unflatten(out_treedef, [g_leaf])


def _scalar_cotangent_leaf(out_leaf: Any) -> Any:
    """Construct the scalar seed without rebuilding output pytree structure."""
    if _is_scalar_value(out_leaf):
        return 1.0
    if getattr(out_leaf, "shape", None) == ():
        staged_seed = getattr(out_leaf, "_advect_scalar_cotangent", None)
        if callable(staged_seed):
            return staged_seed()
        scalar_type = getattr(getattr(out_leaf, "dtype", None), "type", None)
        output_provider = type(out_leaf).__module__.partition(".")[0]
        scalar_provider = str(getattr(scalar_type, "__module__", "")).partition(".")[0]
        if callable(scalar_type) and output_provider == scalar_provider:
            return scalar_type(1)
    return _ones_like(out_leaf)


def _extract_scalar_output(value: Any, *, transform_name: str) -> tuple[Any, Any]:
    leaves, out_treedef = tree_flatten(value)
    if out_treedef.node_type is None:
        out_leaf = value
    else:
        if len(leaves) != 1:
            msg = (
                f"{transform_name} requires a scalar-valued function, but output pytree has "
                f"{len(leaves)} leaves. "
                "Use vjp() for non-scalar outputs, or reduce your output to a scalar."
            )
            raise ValueError(msg)
        out_leaf = leaves[0]

    is_scalar = (
        _is_scalar_value(out_leaf)
        or (hasattr(out_leaf, "shape") and out_leaf.shape == ())
        or (hasattr(out_leaf, "ndim") and out_leaf.ndim == 0)
    )
    if not is_scalar:
        shape = getattr(out_leaf, "shape", "unknown")
        msg = (
            f"{transform_name} requires a scalar-valued function, but output has shape {shape}. "
            "Use vjp() for non-scalar outputs, or reduce your output to a scalar."
        )
        raise ValueError(msg)
    return out_leaf, out_treedef
