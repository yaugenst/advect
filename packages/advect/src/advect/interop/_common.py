"""Shared boundary mechanics for the optional host-framework bridges."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

import advect.autodiff as ad
from advect.core._pytree import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from advect.core._pytree import TreeDef


def require_dependency(module_name: str) -> ModuleType:
    """Import one optional dependency with an actionable installation error."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        message = (
            f"advect.interop.{module_name} requires the optional dependency; "
            f"install it with 'advect[{module_name}]'"
        )
        raise ModuleNotFoundError(message, name=module_name) from error


def differentiable_argnums(count: int) -> tuple[int, ...]:
    """Select every positional argument while preserving tuple-shaped gradients."""
    if count == 0:
        message = "Advect framework bridges require at least one positional argument"
        raise TypeError(message)
    return tuple(range(count))


def numeric_tree(value: Any, *, boundary: str) -> tuple[list[Any], TreeDef]:
    """Flatten a nonempty pytree of NumPy floating or complex values."""
    leaves, treedef = tree_flatten(value)
    if not leaves:
        message = f"{boundary} must contain at least one NumPy floating or complex leaf"
        raise TypeError(message)
    for index, leaf in enumerate(leaves):
        try:
            dtype = np.asarray(leaf).dtype
        except (TypeError, ValueError) as error:
            message = f"{boundary} leaf {index} is not a numeric array or scalar"
            raise TypeError(message) from error
        if dtype.kind not in {"f", "c"}:
            message = (
                f"{boundary} leaf {index} has dtype {dtype}; "
                "only NumPy floating and complex leaves are supported"
            )
            raise TypeError(message)
    return leaves, treedef


def validated_vjp(
    function: Callable[..., Any],
    values: tuple[Any, ...],
    *,
    argnums: tuple[int, ...],
) -> tuple[list[Any], TreeDef, ad.Pullback]:
    """Retain one Advect pullback after validating its public output."""
    value, pullback = ad.vjp(function, argnums=argnums)(*values)
    try:
        output_leaves, output_treedef = numeric_tree(value, boundary="Advect output")
    except BaseException:
        pullback.close()
        raise
    return output_leaves, output_treedef, pullback


def conjugate_complex_tree(value: Any) -> Any:
    """Conjugate complex leaves while preserving a framework-neutral pytree."""
    leaves, treedef = tree_flatten(value)
    converted = [np.conjugate(leaf) if np.iscomplexobj(leaf) else leaf for leaf in leaves]
    return tree_unflatten(treedef, converted)
