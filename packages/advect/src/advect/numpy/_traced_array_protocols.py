"""NumPy protocol bindings for traced arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core._abstract_helpers import PYTHON_SCALAR_TYPES, dtype_name
from advect.core._array_protocol_helpers import weak_scalar_runtime_value
from advect.core._context import _select_deepest_active_recorder, get_source_location
from advect.core._errors import TracingError
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._supported_ufuncs import _SUPPORTED_UFUNCS
from advect.numpy._traced_array_checks import require_active_trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from advect.core._native import DynamicTape
    from advect.numpy._traced_array import TracedArray


_FAST_REDUCTION_KWARGS = frozenset({"axis", "keepdims", "dtype"})
_BINARY_INPUTS = 2
NOT_HANDLED = object()
_EPHEMERAL_UFUNC_OPS = {
    ufunc: canonicalize_numpy_op(f"numpy.{ufunc.__name__}") for ufunc in _SUPPORTED_UFUNCS
}
_SUM_OP = canonicalize_numpy_op("numpy.sum")
# The NumPy scalar that holds a weak Python scalar's value at its default dtype.
_WEAK_SCALAR_VALUES = {
    python_type: np.dtype(dtype_name(python_type)).type for python_type in PYTHON_SCALAR_TYPES
}


def _ephemeral_operand(
    value: object,
    *,
    recorder: DynamicTape,
    traced_type: type[TracedArray],
) -> tuple[int | None, object]:
    if isinstance(value, traced_type):
        if value.recorder is recorder:
            node_id, payload = value._advect_snapshot_in_active_trace()  # noqa: SLF001
            return node_id, weak_scalar_runtime_value(value, payload)
        # The recorder selection already rejected an inactive trace.
        value._advect_snapshot_in_active_trace()  # noqa: SLF001
        return None, value

    snapshot = getattr(value, "_advect_snapshot", None)
    if callable(snapshot):
        if getattr(value, "recorder", None) is recorder:
            node_id, payload = cast("tuple[int, object]", snapshot())
            if bool(getattr(type(payload), "__advect_abstract_array__", False)):
                return node_id, payload
        elif bool(getattr(type(value), "__advect_abstract_array__", False)):
            # A value of the enclosing stage is a constant here.
            return None, value

    if type(value) in (bool, int, float, complex):
        return None, value

    array = np.asarray(value)
    return None, array


def _ephemeral_operation_recorder(
    self: TracedArray,
    inputs: tuple[object, ...],
) -> DynamicTape:
    """Return the common owner without paying nested-stack selection per op."""
    # Supported ufuncs take one or two inputs, and self is one of them.
    recorder = self.recorder
    if len(inputs) == _BINARY_INPUTS:
        left, right = inputs
        other = right if left is self else left
        if isinstance(other, type(self)) and other.recorder is not recorder:
            return cast("DynamicTape", _select_deepest_active_recorder((recorder, other.recorder)))
    return recorder


def run_ephemeral_simple_ufunc(
    self: TracedArray,
    ufunc: np.ufunc,
    inputs: tuple[object, ...],
) -> TracedArray:
    """Execute the common NumPy tape path without durable protocol plumbing."""
    recorder = _ephemeral_operation_recorder(self, inputs)
    traced_type = type(self)
    op = _EPHEMERAL_UFUNC_OPS.get(ufunc)
    if op is None:
        msg = f"Unsupported ufunc: {ufunc.__name__}"
        raise TracingError(msg)

    require_active_trace(recorder=recorder)
    # The selected recorder owns self or the other operand, so at least one
    # operand is a parent; the only operand of a unary ufunc is self.
    first_id, first_value = _ephemeral_operand(
        inputs[0], recorder=recorder, traced_type=traced_type
    )
    input_positions: tuple[int, ...] | None = None
    literals: tuple[object, ...] = ()
    if len(inputs) == 1:
        node_ids = (first_id,)
        values = (first_value,)
    else:
        second_id, second_value = _ephemeral_operand(
            inputs[1], recorder=recorder, traced_type=traced_type
        )
        values = (first_value, second_value)
        if first_id is None:
            node_ids, input_positions, literals = (second_id,), (1,), (first_value,)
        elif second_id is None:
            node_ids, input_positions, literals = (first_id,), (0,), (second_value,)
        else:
            node_ids = (first_id, second_id)
    result = ufunc(*values)
    node_id = recorder.record_operation(
        op,
        cast("tuple[int, ...]", node_ids),
        result,
        {"_advect_backend": "numpy"},
        tuple(result.shape),
        result.dtype,
        input_positions=input_positions,
        literals=literals,
        source_location=get_source_location(),
    )
    return traced_type(value=result, node_id=node_id, recorder=recorder)


def run_weak_python_operator(
    self: TracedArray,
    op: str,
    python_operator: Callable[..., Any],
    inputs: tuple[object, ...],
) -> TracedArray:
    """Apply Python's operator to weak operands and record a weak result (NEP 50).

    Each operand's value is a Python scalar, or an enclosing trace's tracer
    whose own operator decides its category, so the result stays weak.
    """
    recorder = _ephemeral_operation_recorder(self, inputs)
    traced_type = type(self)
    require_active_trace(recorder=recorder)
    projected = [
        _ephemeral_operand(value, recorder=recorder, traced_type=traced_type) for value in inputs
    ]
    result = python_operator(*[value for _node_id, value in projected])
    as_numpy = _WEAK_SCALAR_VALUES.get(type(result))
    if as_numpy is not None:
        result = as_numpy(result)
    literals = [value for node_id, value in projected if node_id is None]
    node_id = recorder.record_operation(
        op,
        [node_id for node_id, _value in projected if node_id is not None],
        result,
        {"_advect_backend": "numpy"},
        (),
        cast("Any", result).dtype,
        input_positions=(
            [position for position, (node, _value) in enumerate(projected) if node is not None]
            if literals
            else None
        ),
        literals=literals,
        weak=True,
        source_location=get_source_location(),
    )
    return traced_type(value=result, node_id=node_id, recorder=recorder)


def run_ephemeral_sum(
    self: TracedArray,
    func: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TracedArray | object:
    """Execute the ordinary NumPy sum path directly on an ephemeral tape."""
    recorder = self.recorder
    if (
        func is not np.sum
        or len(args) != 1
        or args[0] is not self
        or not _FAST_REDUCTION_KWARGS.issuperset(kwargs)
    ):
        return NOT_HANDLED

    require_active_trace(recorder=recorder)
    node_id, value = _ephemeral_operand(
        self,
        recorder=recorder,
        traced_type=type(self),
    )
    axis = kwargs.get("axis")
    keepdims = bool(kwargs.get("keepdims", False))
    dtype = kwargs.get("dtype")
    result = np.sum(
        cast("Any", value),
        axis=cast("Any", axis),
        dtype=cast("DTypeLike | None", dtype),
        keepdims=keepdims,
    )

    attrs: dict[str, object] = {"keepdims": keepdims}
    if axis is not None:
        attrs["axis"] = axis if isinstance(axis, tuple) else (axis,)
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(cast("DTypeLike", dtype)))
    attrs["_advect_backend"] = "numpy"
    # args[0] is self, so the operand is always an SSA parent of this recorder.
    result_node_id = recorder.record_operation(
        _SUM_OP,
        (cast("int", node_id),),
        result,
        attrs,
        tuple(result.shape),
        result.dtype,
        source_location=get_source_location(),
    )
    return type(self)(value=result, node_id=result_node_id, recorder=recorder)
