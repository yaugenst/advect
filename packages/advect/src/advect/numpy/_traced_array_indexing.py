"""Indexing, item assignment, and augmented assignment for TracedArray."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core._array_protocol_helpers import literal_is_weak, weak_scalar_runtime_value
from advect.core._basic_index import encode_basic_index, normalize_basic_index
from advect.core._context import _set_pending_update
from advect.core._errors import MutationError, TracingError
from advect.core._protocols import _innermost, _is_traced
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._traced_array_checks import require_active_trace
from advect.numpy._traced_array_state import PendingIndexUpdate, ViewState, user_location

if TYPE_CHECKING:
    from advect.numpy._traced_array import TracedArray
    from advect.numpy._traced_array_state import SourceLocation


def _concretize_index_key(key: object) -> object:
    if _is_traced(key):
        concrete = np.asarray(_innermost(key))
        if concrete.dtype.kind not in {"b", "i", "u"}:
            msg = (
                "Traced advanced indices must have integer or boolean dtype; "
                f"got {concrete.dtype!s}."
            )
            raise TracingError(msg)
        return concrete
    if isinstance(key, tuple):
        return tuple(_concretize_index_key(item) for item in key)
    return normalize_integer_scalars(key)


def normalize_integer_scalars(key: object) -> object:
    """Read NumPy integer scalars and slice bounds as the Python ints they index with.

    Zero-dimensional integer arrays stay arrays: NumPy indexes with them as
    advanced indices, which copy instead of returning a view.
    """
    if isinstance(key, tuple):
        return tuple(normalize_integer_scalars(item) for item in key)
    if isinstance(key, np.integer):
        return int(key)
    if isinstance(key, slice):
        (key,) = normalize_basic_index(key)
    return key


def _is_basic_index(key: object) -> bool:
    # Only the outer tuple is a multi-index; NumPy reads a nested one as a sequence.
    items = key if isinstance(key, tuple) else (key,)
    return all(isinstance(item, (int, slice)) or item is None or item is Ellipsis for item in items)


def _index_array(item: object) -> np.ndarray | None:
    """Validate one sequence index component as an int64 or bool array."""
    if not isinstance(item, (np.ndarray, list, tuple)):
        return None
    index_array = np.asarray(item)
    if index_array.dtype == object:
        if any(_is_traced(value) for value in index_array.flat):
            msg = (
                "Advanced indexing with TracedArray is not yet supported. "
                "Index arrays must be concrete."
            )
            raise TracingError(msg)
        msg = "Advanced indexing with object arrays is not supported."
        raise TracingError(msg)
    if index_array.dtype.kind == "b":
        return index_array.astype(np.bool_, copy=False)
    if index_array.dtype.kind in {"i", "u"}:
        return index_array.astype(np.int64, copy=False)
    msg = (
        "Advanced indexing with arrays is only supported for integer/bool arrays. "
        f"Got dtype {index_array.dtype!s}."
    )
    raise TracingError(msg)


def index_to_attrs(key: tuple[object, ...]) -> list[dict[str, object]]:
    """Validate and serialize one normalized index key as graph attributes."""
    result: list[dict[str, object]] = []
    for item in key:
        index_array = _index_array(item)
        if index_array is None:
            result.extend(encode_basic_index((item,)))
            continue
        result.append(
            {
                "type": "array",
                "dtype": str(index_array.dtype),
                "shape": tuple(int(i) for i in index_array.shape),
                "values": index_array.tolist(),
            }
        )
    return result


def _basic_index_spec(key: object) -> tuple[tuple[object, ...], ...]:
    """Return a compact structural key for pending-update matching."""
    items = key if isinstance(key, tuple) else (key,)
    result: list[tuple[object, ...]] = []
    for item in items:
        if isinstance(item, int):
            result.append(("int", item))
        elif isinstance(item, slice):
            result.append(("slice", item.start, item.stop, item.step))
        elif item is None:
            result.append(("newaxis",))
        elif item is Ellipsis:
            result.append(("ellipsis",))
        else:
            raise AssertionError(type(item).__name__)
    return tuple(result)


def index_from_spec(index_spec: object) -> object:
    """Reconstruct one basic index from its structural matching key."""
    items: list[object] = []
    for encoded in cast("tuple[tuple[object, ...], ...]", index_spec):
        match encoded:
            case ("int", value):
                items.append(value)
            case ("slice", start, stop, step):
                items.append(slice(start, stop, step))
            case ("newaxis",):
                items.append(None)
            case ("ellipsis",):
                items.append(Ellipsis)
            case _:
                raise AssertionError(encoded)
    return items[0] if len(items) == 1 else tuple(items)


def _update_operand(self: TracedArray, operand: object, operation: str) -> tuple[int | None, Any]:
    """Resolve an update operand to a same-trace SSA parent or a backend literal."""
    if not isinstance(operand, type(self)):
        return None, np.asarray(operand)
    if operand.recorder is not self.recorder:
        msg = (
            f"Cannot use a TracedArray from a different trace context in {operation}. "
            "Both arrays must belong to the same trace recorder."
        )
        raise TracingError(msg)
    return operand._advect_snapshot_in_active_trace()  # noqa: SLF001


def _record_update(
    self: TracedArray,
    op: str,
    source_node_id: int,
    resolved: tuple[int | None, Any],
    result: Any,  # noqa: ANN401 - backend array payload
    attrs: dict[str, object],
) -> int:
    """Record ``op(source, operand)``, keeping an untraced operand as a literal."""
    operand_node_id, operand_value = resolved
    literal = operand_node_id is None
    return self.recorder.record_operation(
        op,
        (source_node_id,) if literal else (source_node_id, operand_node_id),
        result,
        attrs,
        result.shape,
        result.dtype,
        input_positions=(0,) if literal else None,
        literals=(operand_value,) if literal else (),
    )


def _apply_direct_index_add(
    self: TracedArray,
    *,
    key: object,
    operand: object,
    location: SourceLocation | None,
) -> None:
    """Emit one pure additive index update and advance the root wrapper."""
    resolved = _update_operand(self, operand, "an indexed add")
    source_node_id, source_value = self._advect_snapshot_in_active_trace()
    result = cast("Any", source_value).copy()
    result[cast("Any", key)] += resolved[1]
    node_id = _record_update(
        self,
        "advect.index_update",
        source_node_id,
        resolved,
        result,
        {"index": key, "mode": "add"},
    )
    self._commit_current(value=result, node_id=node_id, location=location)


def _consume_matching_pending(
    self: TracedArray,
    *,
    pending: object | None,
    value: object,
    index_spec: object,
) -> PendingIndexUpdate | None:
    """Consume and validate the pending half of indexed augmented assignment."""
    if pending is None:
        return None

    if not isinstance(pending, PendingIndexUpdate):
        message = getattr(pending, "unconsumed_message", None)
        if not isinstance(message, str):
            message = "A pending traced view update was redirected to the wrong assignment."
        raise TracingError(message)
    if value is not pending.replacement:
        return None

    if self.is_view:
        msg = (
            "Nested subscript mutation is not supported during tracing. "
            "Rewrite `field[i][j] += value` as `field[i, j] += value`."
        )
        raise MutationError(msg)
    if (
        pending.root is not self
        or pending.root_epoch != self.epoch
        or pending.index_spec != index_spec
    ):
        msg = (
            "The pending augmented view update does not match this base, index, or epoch. "
            "Keep the indexed augmented assignment as one expression."
        )
        raise MutationError(msg)
    return pending


def getitem(self: TracedArray, key: object) -> TracedArray:
    """Handle array indexing/slicing."""
    require_active_trace(recorder=self.recorder)
    key = _concretize_index_key(key)

    is_basic = _is_basic_index(key)
    if is_basic:
        index_spec = _basic_index_spec(key)
        index_attrs = key
    else:
        index_spec = None
        index_attrs = index_to_attrs(key if isinstance(key, tuple) else (key,))

    _source_node_id, source_value = self._advect_snapshot_in_active_trace()
    result_value = cast("Any", source_value)[cast("Any", key)]
    is_view = is_basic and isinstance(_innermost(result_value), np.ndarray)

    attrs = {"index": index_attrs}

    node_id = None

    traced_type = type(self)
    if not is_view:
        return traced_type(
            value=result_value,
            node_id=node_id,
            recorder=self.recorder,
            deferred_getitem=(self, attrs),
        )

    root = self._root_for_view()
    return traced_type(
        value=result_value,
        node_id=node_id,
        recorder=self.recorder,
        owned=False,
        view_state=ViewState(
            root=root,
            epoch=root.epoch,
            index_spec=index_spec if self is root else None,
            location=user_location(depth=3),
        ),
        deferred_getitem=(self, attrs),
    )


def setitem(self: TracedArray, key: object, value: object) -> None:
    """Functionalize item assignment into one pure ``advect.index_update`` node."""
    pending_update = require_active_trace(
        recorder=self.recorder,
        allow_pending=True,
        take_pending=True,
    )

    key = normalize_integer_scalars(key)
    if not _is_basic_index(key):
        msg = (
            "Advanced-index assignment is not supported during tracing. "
            "Use basic slicing, or express accumulation with a dedicated scatter operation."
        )
        raise TracingError(msg)

    index_spec = _basic_index_spec(key)
    pending = _consume_matching_pending(
        self,
        pending=pending_update,
        value=value,
        index_spec=index_spec,
    )
    if pending is not None:
        return

    if self.is_view:
        msg = (
            "Item assignment through a traced view is not supported. "
            "Combine indices on the base (for example, rewrite `field[i][j]` as "
            "`field[i, j]`) or call `.copy()` before assigning."
        )
        raise MutationError(msg)

    self._require_mutable_in_active_trace(operation="item assignment")

    resolved = _update_operand(self, value, "item assignment")
    source_node_id, source_value = self._advect_snapshot_in_active_trace()
    result = cast("Any", source_value).copy()
    result[cast("Any", key)] = resolved[1]
    node_id = _record_update(
        self, "advect.index_update", source_node_id, resolved, result, {"index": key}
    )
    self._commit_current(value=result, node_id=node_id)


def _functional_result(self: TracedArray, other: object, ufunc: np.ufunc) -> tuple[int, Any]:
    """Evaluate an in-place-shaped operation into fresh storage and emit a pure node."""
    resolved = _update_operand(self, other, f"augmented {ufunc.__name__}")
    other_node_id, other_value = resolved
    if other_node_id is None:
        # A Python scalar keeps NumPy's weak promotion, as in eager code.
        other_leaf = outer_other = other if literal_is_weak(other) else other_value
    else:
        # A weak scalar tracer computes as the Python scalar it stands for.
        other_leaf = weak_scalar_runtime_value(other, np.asarray(_innermost(other_value)))
        outer_other = other_value
    receiver_node_id, receiver_value = self._advect_snapshot_in_active_trace()
    receiver_leaf = np.asarray(_innermost(receiver_value))
    concrete_result = np.empty_like(receiver_leaf)
    # Supplying a fresh destination preserves NumPy's in-place shape, dtype,
    # casting, and broadcasting checks without modifying an existing SSA value.
    ufunc(receiver_leaf, other_leaf, out=concrete_result)
    result = concrete_result
    if _is_traced(receiver_value) or _is_traced(other_value):
        # An outer trace records the ufunc without loop selection; the in-place
        # result keeps the receiver's dtype.
        result = ufunc(receiver_value, outer_other)
        if result.dtype != receiver_leaf.dtype:
            result = result.astype(receiver_leaf.dtype)
    node_id = _record_update(
        self,
        canonicalize_numpy_op(f"numpy.{ufunc.__name__}"),
        receiver_node_id,
        resolved,
        result,
        {"dtype": str(receiver_leaf.dtype), "_advect_backend": "numpy"},
    )
    return node_id, result


def inplace_op(self: TracedArray, other: object, ufunc: np.ufunc) -> TracedArray:
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
        location = user_location()
        if ufunc is np.add:
            _apply_direct_index_add(root, key=key, operand=other, location=location)
        else:
            replacement_node_id, replacement_value = _functional_result(self, other, ufunc)
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
    node_id, value = _functional_result(self, other, ufunc)
    self._commit_current(value=value, node_id=node_id)
    return self
