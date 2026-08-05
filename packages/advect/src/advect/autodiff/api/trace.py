"""Small helpers shared by dynamic autodiff tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from advect.autodiff.api._scalar_boundary import _is_real_python_scalar
from advect.core._backends import dispatch_input
from advect.core._pytree import TreeDef, tree_flatten_with_paths

if TYPE_CHECKING:
    from advect.core._native import DynamicTape


_LEAF_TREEDEF = TreeDef(node_type=None, aux_data=None, children=(), num_leaves=1)


def _wrap_input(
    value: Any,
    recorder: DynamicTape,
    *,
    name: str | None = None,
    active: bool = True,
    weak: bool = False,
) -> tuple[Any, int]:
    """Wrap an input value for tracing.

    Returns the traced wrapper and its node ID.
    """
    _ = recorder  # The recorder is implicit via the active trace context.
    traced = (
        dispatch_input(value, name=name)
        if active
        else dispatch_input(value, name=name, active=False)
    )
    snapshot = getattr(traced, "_advect_snapshot_in_active_trace", None)
    if callable(snapshot):
        node_id, _value = cast("tuple[int, Any]", snapshot())
        resolved_id = int(node_id)
        if weak:
            recorder.mark_weak(resolved_id)
        return traced, resolved_id
    if hasattr(traced, "node_id"):
        resolved_id = int(traced.node_id)
        if weak:
            recorder.mark_weak(resolved_id)
        return traced, resolved_id
    msg = (
        f"Backend input handler returned unsupported traced value type: {type(traced).__name__}. "
        "Expected an object with a 'node_id' attribute."
    )
    raise TypeError(msg)


def _output_node_id(result: Any, recorder: DynamicTape) -> int:
    """Resolve an output leaf to a node ID.

    Outputs can be traced arrays or constants.
    Constant outputs are lifted into an ``advect.const`` node so that autodiff
    returns well-defined zero gradients for disconnected inputs.
    """
    active_snapshot = getattr(result, "_advect_snapshot_in_active_trace", None)
    if callable(active_snapshot):
        node_id, _value = cast("tuple[int, Any]", active_snapshot())
        return int(node_id)
    if hasattr(result, "node_id"):
        return int(result.node_id)

    if _is_real_python_scalar(result):
        value = float(result)
        node_id = recorder.record_operation(
            "advect.const",
            (),
            value,
            {},
            (),
            "float64",
        )
        recorder.mark_weak(node_id)
        return node_id

    if type(result) is complex:
        node_id = recorder.record_operation(
            "advect.const",
            (),
            result,
            {},
            (),
            "complex128",
        )
        recorder.mark_weak(node_id)
        return node_id

    if hasattr(result, "shape") and hasattr(result, "dtype"):
        shape = tuple(int(d) for d in result.shape)
        return recorder.record_operation(
            "advect.const",
            (),
            result,
            {},
            shape,
            result.dtype,
        )
    msg = (
        "Autodiff functions must return a traced array or "
        f"an array/scalar constant; got {type(result).__name__}."
    )
    raise TypeError(msg)


def _mark_outputs(result: Any, recorder: DynamicTape) -> tuple[TreeDef, list[int]]:
    """Mark pytree leaves as graph outputs, returning (treedef, output_node_ids)."""
    if callable(getattr(result, "_advect_snapshot", None)):
        treedef, leaves = _LEAF_TREEDEF, (result,)
    else:
        _paths, leaves, treedef = tree_flatten_with_paths(result)
    output_node_ids = [_output_node_id(leaf, recorder) for leaf in leaves]
    for node_id in dict.fromkeys(output_node_ids):
        recorder.mark_output(node_id)
    return treedef, output_node_ids
