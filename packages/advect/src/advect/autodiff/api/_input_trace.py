"""Pytree input trace specifications and the common leaf fast path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from advect.autodiff.api._scalar_boundary import (
    _is_real_python_scalar,
    _lift_scalar_to_array,
)
from advect.autodiff.api.trace import _LEAF_TREEDEF, _wrap_input
from advect.core._array_api.profiles import LATEST_ARRAY_API_VERSION
from advect.core._array_api.providers import _get_array_namespace
from advect.core._context import _get_active_array_api_version
from advect.core._pytree import _get_node_impl

if TYPE_CHECKING:
    from advect.core._native import DynamicTape
    from advect.core._pytree import TreeDef


def _array_namespace_for_input(
    value: object,
    *,
    array_api_version: str | None = None,
) -> object | None:
    """Resolve the selected namespace or reject an incompatible protocol."""
    selected = array_api_version or _get_active_array_api_version() or LATEST_ARRAY_API_VERSION
    namespace = (
        _get_array_namespace(value)
        if array_api_version is None
        else _get_array_namespace(value, api_version=selected)
    )
    if namespace is None and callable(getattr(value, "__array_namespace__", None)):
        msg = (
            f"Advect requires Array API {selected}; "
            f"{type(value).__name__} cannot serve that version"
        )
        raise TypeError(msg)
    return namespace


@dataclass(frozen=True, slots=True)
class _LeafTraceSpec:
    node_id: int | None
    primal: object | None
    restore_python_scalar: bool


@dataclass(frozen=True, slots=True)
class _TracedInputSpec:
    treedef: TreeDef
    leaf_specs: tuple[_LeafTraceSpec, ...]


def _trace_leaf_as_input(
    graph: DynamicTape,
    value: object,
    *,
    prefix: str | None,
    xp: object | None,
) -> tuple[object, _TracedInputSpec] | None:
    """Trace an unregistered array/scalar leaf without general pytree allocation."""
    if _get_node_impl(type(value)) is not None:
        return None

    if isinstance(value, bool):
        msg = "Boolean values are not differentiable scalar primals"
        raise TypeError(msg)
    if isinstance(value, complex):
        msg = (
            "Python complex scalars are not differentiable primals. Wrap the value "
            "in a backend 0-D array before differentiation."
        )
        raise TypeError(msg)

    is_existing_traced = callable(getattr(value, "_advect_snapshot", None))
    is_traceable = is_existing_traced or _is_real_python_scalar(value)
    if not is_traceable and (
        (xp is not None and callable(getattr(value, "__array_namespace__", None)))
        or _array_namespace_for_input(value) is not None
    ):
        is_traceable = True
    if not is_traceable:
        return None

    traced, leaf_spec = _trace_leaf(graph, value, name=prefix, xp=xp)
    return traced, _TracedInputSpec(treedef=_LEAF_TREEDEF, leaf_specs=(leaf_spec,))


def _trace_leaf(
    graph: DynamicTape,
    leaf: object,
    *,
    name: str | None,
    xp: object | None,
) -> tuple[object, _LeafTraceSpec]:
    """Record one traceable leaf as an input, lifting a real Python scalar first."""
    restore_python_scalar = _is_real_python_scalar(leaf)
    primal = _lift_scalar_to_array(leaf, namespace=xp) if restore_python_scalar else leaf
    traced, node_id = _wrap_input(
        primal,
        graph,
        name=name,
        weak=restore_python_scalar or bool(getattr(leaf, "_advect_weak", False)),
    )
    return traced, _LeafTraceSpec(
        node_id=node_id,
        primal=primal,
        restore_python_scalar=restore_python_scalar,
    )
