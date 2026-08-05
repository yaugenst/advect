"""NumPy protocol bindings for traced arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core._array_protocol_helpers import (
    literals_are_weak,
    weak_scalar_runtime_value,
)
from advect.core._context import (
    _is_recorder_in_active_trace_stack,
    _select_deepest_active_recorder,
    is_debug,
)
from advect.core._errors import TraceLevelError, TracingError
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._protocol_array_function import ARRAY_FUNCTION_RUNTIME
from advect.numpy._protocol_runtime import NUMPY_PROTOCOL_RUNTIME
from advect.numpy._protocol_ufunc import UFUNC_RUNTIME
from advect.numpy._traced_array_checks import require_active_trace

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from advect.core._native import DynamicTape
    from advect.numpy._traced_array import TracedArray


_UFUNC_RUNTIME = UFUNC_RUNTIME
_ARRAY_FUNCTION_RUNTIME = ARRAY_FUNCTION_RUNTIME
_RUNTIME = NUMPY_PROTOCOL_RUNTIME

_FAST_REDUCTION_KWARGS = frozenset({"axis", "keepdims", "dtype"})
_BINARY_INPUTS = 2
NOT_HANDLED = object()
_EPHEMERAL_UFUNC_OPS = {
    ufunc: canonicalize_numpy_op(f"numpy.{ufunc.__name__}")
    for ufunc in _UFUNC_RUNTIME.supported_ufuncs
}
_SUM_OP = canonicalize_numpy_op("numpy.sum")


def _ephemeral_operand(
    value: object,
    *,
    recorder: DynamicTape,
    traced_type: type[TracedArray],
) -> tuple[int | None, object]:
    if isinstance(value, traced_type):
        if value.recorder is recorder:
            node_id, payload = value._advect_snapshot_in_active_trace()  # noqa: SLF001
            return int(node_id), weak_scalar_runtime_value(value, payload)
        if not _is_recorder_in_active_trace_stack(value.recorder):
            msg = "Cannot use a NumPy tracer from an unrelated or expired trace recorder."
            raise TraceLevelError(msg)
        value._advect_snapshot_in_active_trace()  # noqa: SLF001
        return None, value

    snapshot = getattr(value, "_advect_snapshot", None)
    if callable(snapshot) and getattr(value, "recorder", None) is recorder:
        node_id, payload = cast("tuple[int, object]", snapshot())
        if bool(getattr(type(payload), "__advect_abstract_array__", False)):
            return int(node_id), payload

    if type(value) in (bool, int, float, complex):
        return None, value

    array = np.asarray(value)
    return None, array


def _split_ephemeral_operands(
    operands: tuple[tuple[int | None, object], ...],
) -> tuple[tuple[int, ...], tuple[int, ...] | None, tuple[object, ...], tuple[object, ...]]:
    """Separate differentiable SSA parents from concrete literal operands."""
    node_ids = tuple(node_id for node_id, _value in operands if node_id is not None)
    values = tuple(value for _node_id, value in operands)
    literals = tuple(value for node_id, value in operands if node_id is None)
    if not literals:
        return node_ids, None, (), values
    input_positions = tuple(
        position for position, (node_id, _value) in enumerate(operands) if node_id is not None
    )
    return node_ids, input_positions, literals, values


def _ephemeral_operation_recorder(
    self: TracedArray,
    inputs: tuple[object, ...],
) -> DynamicTape:
    """Return the common owner without paying nested-stack selection per op."""
    recorder = self.recorder
    traced_type = type(self)
    if len(inputs) == _BINARY_INPUTS:
        left, right = inputs
        other = right if left is self else left
        if isinstance(other, traced_type) and other.recorder is not recorder:
            return cast(
                "DynamicTape",
                _select_deepest_active_recorder((recorder, other.recorder)),
            )
        return recorder
    if len(inputs) == 1 and inputs[0] is self:
        return recorder
    traced_recorders = tuple(value.recorder for value in inputs if isinstance(value, traced_type))
    if any(candidate is not recorder for candidate in traced_recorders):
        return cast(
            "DynamicTape",
            _select_deepest_active_recorder(traced_recorders),
        )
    return recorder


def run_ephemeral_simple_ufunc(
    self: TracedArray,
    ufunc: np.ufunc,
    inputs: tuple[object, ...],
) -> TracedArray | object:
    """Execute the common NumPy tape path without durable protocol plumbing."""
    recorder = _ephemeral_operation_recorder(self, inputs)
    traced_type = type(self)
    if is_debug():
        return NOT_HANDLED
    op = _EPHEMERAL_UFUNC_OPS.get(ufunc)
    if op is None:
        msg = f"Unsupported ufunc: {ufunc.__name__}"
        raise TracingError(msg)

    require_active_trace(recorder=recorder)
    if len(inputs) == 1:
        first_id, first_value = _ephemeral_operand(
            inputs[0], recorder=recorder, traced_type=traced_type
        )
        node_ids = () if first_id is None else (first_id,)
        input_positions = () if first_id is None else None
        literals = (first_value,) if first_id is None else ()
        values = (first_value,)
    elif len(inputs) == _BINARY_INPUTS:
        first_id, first_value = _ephemeral_operand(
            inputs[0], recorder=recorder, traced_type=traced_type
        )
        second_id, second_value = _ephemeral_operand(
            inputs[1], recorder=recorder, traced_type=traced_type
        )
        values = (first_value, second_value)
        if first_id is not None and second_id is not None:
            node_ids = (first_id, second_id)
            input_positions = None
            literals = ()
        elif first_id is not None:
            node_ids = (first_id,)
            input_positions = (0,)
            literals = (second_value,)
        elif second_id is not None:
            node_ids = (second_id,)
            input_positions = (1,)
            literals = (first_value,)
        else:  # pragma: no cover - NumPy calls the protocol on a traced operand
            node_ids = ()
            input_positions = ()
            literals = values
    else:
        operands = tuple(
            _ephemeral_operand(value, recorder=recorder, traced_type=traced_type)
            for value in inputs
        )
        node_ids, input_positions, literals, values = _split_ephemeral_operands(operands)
    result = ufunc(*values)
    if literals:
        node_id = recorder.record_operation_with_literals(
            op,
            node_ids,
            () if input_positions is None else input_positions,
            literals,
            result,
            {"_advect_backend": "numpy"},
            tuple(result.shape),
            result.dtype,
            literal_weak=literals_are_weak(list(literals)),
        )
    else:
        node_id = recorder.record_operation(
            op,
            node_ids,
            result,
            {"_advect_backend": "numpy"},
            tuple(result.shape),
            result.dtype,
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
        or is_debug()
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
    if node_id is None:
        result_node_id = recorder.record_operation_with_literals(
            _SUM_OP,
            (),
            (),
            (value,),
            result,
            attrs,
            tuple(result.shape),
            result.dtype,
            literal_weak=True,
        )
    else:
        result_node_id = recorder.record_operation(
            _SUM_OP,
            (node_id,),
            result,
            attrs,
            tuple(result.shape),
            result.dtype,
        )
    return type(self)(value=result, node_id=result_node_id, recorder=recorder)
