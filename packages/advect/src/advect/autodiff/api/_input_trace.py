"""Pytree input trace specifications and the common leaf fast path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from advect.autodiff.api._scalar_boundary import (
    _is_real_python_scalar,
    _lift_scalar_to_array,
)
from advect.autodiff.api.trace import _wrap_input
from advect.core._array_api_profiles import LATEST_ARRAY_API_VERSION
from advect.core._array_namespace import _get_array_namespace
from advect.core._context import _get_active_array_api_version
from advect.core._pytree import TreeDef, _get_node_impl

if TYPE_CHECKING:
    from advect.core._native import DynamicTape


_LEAF_TREEDEF = TreeDef(node_type=None, aux_data=None, children=(), num_leaves=1)


def _array_namespace_for_input(
    value: Any,
    *,
    array_api_version: str | None = None,
) -> Any | None:
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
    primal: Any | None
    restore_python_scalar: bool


@dataclass(frozen=True, slots=True)
class _TracedInputSpec:
    treedef: TreeDef
    leaf_specs: tuple[_LeafTraceSpec, ...]


def _trace_leaf_as_input(
    graph: DynamicTape,
    value: Any,
    *,
    prefix: str | None,
    xp: Any | None,
) -> tuple[Any, _TracedInputSpec] | None:
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

    restore_python_scalar = False
    primal = value
    if (not is_existing_traced) and _is_real_python_scalar(value):
        primal = _lift_scalar_to_array(value, namespace=xp)
        restore_python_scalar = True

    traced, node_id = _wrap_input(
        primal,
        graph,
        name=prefix,
        weak=restore_python_scalar or bool(getattr(value, "_advect_weak", False)),
    )
    spec = _TracedInputSpec(
        treedef=_LEAF_TREEDEF,
        leaf_specs=(
            _LeafTraceSpec(
                node_id=node_id,
                primal=primal,
                restore_python_scalar=restore_python_scalar,
            ),
        ),
    )
    return traced, spec
