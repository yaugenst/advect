"""Indexing and slicing support for TracedArray."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core._array_protocol_helpers import literal_is_weak
from advect.core._basic_index import encode_basic_index
from advect.core._errors import MutationError, TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._traced_array_checks import require_active_trace
from advect.numpy._traced_array_state import PendingIndexUpdate, ViewState, user_location

if TYPE_CHECKING:
    from advect.numpy._traced_array import TracedArray
    from advect.numpy._traced_array_state import SourceLocation


def _is_traced_leaf(value: object) -> bool:
    return callable(getattr(value, "_advect_snapshot", None))


def _unwrap_traced_leaf(value: object) -> object:
    current = value
    while _is_traced_leaf(current):
        _node_id, next_value = _snapshot_traced(current)
        if next_value is current:
            break
        current = next_value
    return current


def _concretize_index_key(key: object) -> object:
    if _is_traced_leaf(key):
        concrete = np.asarray(_unwrap_traced_leaf(key))
        if concrete.dtype.kind not in {"b", "i", "u"}:
            msg = (
                "Traced advanced indices must have integer or boolean dtype; "
                f"got {concrete.dtype!s}."
            )
            raise TracingError(msg)
        return concrete
    if isinstance(key, tuple):
        return tuple(_concretize_index_key(item) for item in key)
    return key


def _is_basic_index(key: object) -> bool:
    if isinstance(key, tuple):
        return all(_is_basic_index(item) for item in key)
    return isinstance(key, (int, slice)) or key is None or key is Ellipsis


def _coerce_index_array(self: TracedArray, key: object) -> np.ndarray | None:
    traced_type = type(self)

    index_array: np.ndarray | None = None
    if isinstance(key, np.ndarray):
        index_array = key
    elif isinstance(key, list):
        index_array = np.asarray(key)

    if index_array is None:
        return None

    if index_array.dtype == object:
        if any(isinstance(item, traced_type) for item in index_array.flat):
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


def validate_index_key(self: TracedArray, key: object) -> None:
    """Validate index components and reject unsupported traced/object variants."""
    traced_type = type(self)

    if isinstance(key, traced_type):
        msg = (
            "Advanced indexing with TracedArray is not yet supported. "
            "Use basic slicing (integers, slices) instead."
        )
        raise TracingError(msg)

    if _coerce_index_array(self, key) is not None:
        return

    if isinstance(key, tuple):
        for item in key:
            if _coerce_index_array(self, item) is not None:
                continue

            validate_index_key(self, item)


def normalize_index_key(key: object) -> tuple[object, ...]:
    """Normalize index key to a tuple."""
    if isinstance(key, tuple):
        return key
    return (key,)


def _serialize_index_array(index_array: np.ndarray) -> dict[str, object]:
    if index_array.dtype == object:
        msg = "Advanced indexing with object arrays is not supported."
        raise TracingError(msg)

    if index_array.dtype.kind == "b":
        index_array = index_array.astype(np.bool_, copy=False)
    elif index_array.dtype.kind in {"i", "u"}:
        index_array = index_array.astype(np.int64, copy=False)
    else:
        msg = (
            "Advanced indexing with arrays is only supported for integer/bool arrays. "
            f"Got dtype {index_array.dtype!s}."
        )
        raise TracingError(msg)

    return {
        "type": "array",
        "dtype": str(index_array.dtype),
        "shape": tuple(int(i) for i in index_array.shape),
        "values": index_array.tolist(),
    }


def index_to_attrs(key: tuple[object, ...]) -> list[dict[str, object]]:
    """Convert normalized index key to serializable attrs."""
    result: list[dict[str, object]] = []
    for item in key:
        if isinstance(item, np.ndarray):
            result.append(_serialize_index_array(item))
        elif isinstance(item, list):
            result.append(_serialize_index_array(np.asarray(item)))
        else:
            result.extend(encode_basic_index((item,)))
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


def apply_direct_index_add(
    self: TracedArray,
    *,
    key: object,
    index_attrs: object,
    operand: object,
    location: SourceLocation | None,
) -> None:
    """Emit one pure additive index update and advance the root wrapper."""
    traced_type = type(self)
    if isinstance(operand, traced_type):
        if operand.recorder is not self.recorder:
            msg = (
                "Cannot add a TracedArray from a different trace context. "
                "Both arrays must belong to the same trace recorder."
            )
            raise TracingError(msg)
        operand_node_id, actual_operand = operand._advect_snapshot_in_active_trace()  # noqa: SLF001
    else:
        actual_operand = np.asarray(operand)
        operand_node_id = None

    source_node_id, source_value = self._advect_snapshot_in_active_trace()
    result_value = cast("Any", source_value).copy()
    result_value[cast("Any", key)] += actual_operand
    if operand_node_id is None:
        node_id = self.recorder.record_operation_with_literals(
            "advect.index_update",
            (source_node_id,),
            (0,),
            (actual_operand,),
            result_value,
            {"index": index_attrs, "mode": "add"},
            result_value.shape,
            result_value.dtype,
            literal_weak=literal_is_weak(operand),
        )
    else:
        node_id = self.recorder.record_operation(
            "advect.index_update",
            (source_node_id, operand_node_id),
            result_value,
            {"index": index_attrs, "mode": "add"},
            result_value.shape,
            result_value.dtype,
        )
    self._commit_current(
        value=result_value,
        node_id=node_id,
        location=location,
    )


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
        validate_index_key(self, key)
        index_attrs = index_to_attrs(normalize_index_key(key))

    _source_node_id, source_value = self._advect_snapshot_in_active_trace()
    result_value = cast("Any", source_value)[cast("Any", key)]
    is_view = is_basic and isinstance(_unwrap_traced_leaf(result_value), np.ndarray)

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

    validate_index_key(self, key)
    if not _is_basic_index(key):
        msg = (
            "Advanced-index assignment is not supported during tracing. "
            "Use basic slicing, or express accumulation with a dedicated scatter operation."
        )
        raise TracingError(msg)

    index_spec = _basic_index_spec(key)
    index_attrs = key
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

    traced_type = type(self)
    value_node_id: int | None = None
    actual_value: Any
    if isinstance(value, traced_type):
        if value.recorder is not self.recorder:
            msg = (
                "Cannot assign a TracedArray from a different trace context. "
                "Both arrays must belong to the same graph."
            )
            raise TracingError(msg)
        value_node_id, actual_value = value._advect_snapshot_in_active_trace()  # noqa: SLF001
    else:
        actual_value = np.asarray(value)

    source_node_id, source_value = self._advect_snapshot_in_active_trace()
    result_value = cast("Any", source_value).copy()
    result_value[cast("Any", key)] = actual_value

    attrs: dict[str, object] = {"index": index_attrs}

    if value_node_id is not None:
        node_id = self.recorder.record_operation(
            "advect.index_update",
            (source_node_id, value_node_id),
            result_value,
            attrs,
            result_value.shape,
            result_value.dtype,
        )
    else:
        node_id = self.recorder.record_operation_with_literals(
            "advect.index_update",
            (source_node_id,),
            (0,),
            (actual_value,),
            result_value,
            attrs,
            result_value.shape,
            result_value.dtype,
            literal_weak=literal_is_weak(value),
        )
    self._commit_current(value=result_value, node_id=node_id)
