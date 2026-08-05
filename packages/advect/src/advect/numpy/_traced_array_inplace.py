"""In-place operation implementations for TracedArray."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from advect.core._array_protocol_helpers import literal_is_weak
from advect.core._context import _set_pending_update
from advect.core._errors import MutationError, TracingError
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._traced_array_checks import require_active_trace
from advect.numpy._traced_array_indexing import (
    _is_traced_leaf,
    _unwrap_traced_leaf,
    apply_direct_index_add,
    index_from_spec,
)
from advect.numpy._traced_array_state import PendingIndexUpdate, user_location

if TYPE_CHECKING:
    from advect.numpy._traced_array import TracedArray


def _functional_result(
    self: TracedArray,
    other: object,
    ufunc: np.ufunc,
    op_name: str,
) -> tuple[int, Any]:
    """Evaluate an in-place-shaped operation into fresh storage and emit a pure node."""
    traced_type = type(self)
    other_node_id: int | None = None
    actual_other: Any
    if isinstance(other, traced_type):
        if other.recorder is not self.recorder:
            msg = (
                "Cannot perform in-place operation with a TracedArray from a "
                "different trace context. Both arrays must belong to the same graph."
            )
            raise TracingError(msg)
        other_node_id, actual_other = other._advect_snapshot_in_active_trace()  # noqa: SLF001
    else:
        actual_other = np.asarray(other)

    receiver_node_id, receiver_value = self._advect_snapshot_in_active_trace()
    receiver_leaf = np.asarray(_unwrap_traced_leaf(receiver_value))
    other_leaf = np.asarray(_unwrap_traced_leaf(actual_other))
    concrete_result = np.empty_like(receiver_leaf)
    # Supplying a fresh destination preserves NumPy's in-place shape, dtype,
    # casting, and broadcasting checks without modifying an existing SSA value.
    ufunc(receiver_leaf, other_leaf, out=concrete_result)

    has_outer_tracer = _is_traced_leaf(receiver_value) or _is_traced_leaf(actual_other)
    result_value = (
        ufunc(receiver_value, actual_other, dtype=receiver_leaf.dtype)
        if has_outer_tracer
        else concrete_result
    )

    if other_node_id is not None:
        inputs = (receiver_node_id, other_node_id)
        node_id = self.recorder.record_operation(
            canonicalize_numpy_op(op_name),
            inputs,
            result_value,
            {"dtype": str(receiver_leaf.dtype), "_advect_backend": "numpy"},
            result_value.shape,
            result_value.dtype,
        )
    else:
        node_id = self.recorder.record_operation_with_literals(
            canonicalize_numpy_op(op_name),
            (receiver_node_id,),
            (0,),
            (actual_other,),
            result_value,
            {"dtype": str(receiver_leaf.dtype), "_advect_backend": "numpy"},
            result_value.shape,
            result_value.dtype,
            literal_weak=literal_is_weak(other),
        )
    return node_id, result_value


def inplace_op(
    self: TracedArray,
    other: object,
    ufunc: np.ufunc,
    op_name: str,
) -> TracedArray:
    """Functionalize an augmented assignment at the tracer-wrapper boundary."""
    require_active_trace(recorder=self.recorder)
    self._check_view_epoch()

    view_state = self._view_state
    if view_state is not None:
        root = view_state.root
        root._require_mutable_in_active_trace(  # noqa: SLF001
            operation=f"{ufunc.__name__} through an indexed view"
        )
        index_spec = view_state.index_spec
        if index_spec is None:
            msg = (
                "Mutation through this traced view is not supported. "
                "Update the base with a single basic index expression, or call `.copy()` first."
            )
            raise MutationError(msg)
        key = index_from_spec(index_spec)
        location = user_location(depth=3)
        if ufunc is np.add:
            apply_direct_index_add(
                root,
                key=key,
                index_attrs=key,
                operand=other,
                location=location,
            )
        else:
            replacement_node_id, replacement_value = _functional_result(self, other, ufunc, op_name)
            replacement = type(self)(
                value=replacement_value,
                node_id=replacement_node_id,
                recorder=self.recorder,
            )
            root[key] = replacement
        self._refresh_direct_view(key=key, index_spec=index_spec)
        pending = PendingIndexUpdate(
            root=root,
            root_epoch=root.epoch,
            index_spec=index_spec,
            replacement=self,
        )
        _set_pending_update(self.recorder, pending)
        return self

    self._require_mutable_in_active_trace(operation=f"augmented {ufunc.__name__}")
    node_id, value = _functional_result(self, other, ufunc, op_name)
    self._commit_current(value=value, node_id=node_id)
    return self


def inplace_matmul(
    self: TracedArray,
    other: object,
) -> TracedArray:
    """Functionalize ``@=`` through the same pure ufunc path."""
    return inplace_op(self, other, np.matmul, "numpy.matmul")
