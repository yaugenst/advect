"""Concrete NumPy ufunc dispatch and graph recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from advect.core._array_protocol_helpers import weak_scalar_runtime_value
from advect.core._errors import TracingError
from advect.core._protocols import ArrayLike, _snapshot_traced_in_active_trace
from advect.numpy._array_function.emission import (
    _add_backend_node,
    _LiteralOperand,
    _result_shape_and_dtype,
)
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._supported_ufuncs import _SUPPORTED_UFUNCS

if TYPE_CHECKING:
    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import _Operand


type UfuncValue = ArrayLike | tuple[ArrayLike, ...]
type UfuncNodeIDs = int | tuple[int, ...]

_PYTHON_SCALAR_TYPES = (bool, int, float, complex)
_SUPPORTED_CALL_KWARGS = frozenset(
    {
        "casting",
        "dtype",
        "order",
        "sig",
        "signature",
        "subok",
        "where",
    }
)
_LOOP_SELECTION_KWARGS = ("dtype", "sig", "signature")


class UfuncLike(Protocol):
    """Structural protocol for backend ufunc objects."""

    @property
    def __name__(self) -> str: ...

    @property
    def nin(self) -> int: ...

    @property
    def nout(self) -> int: ...

    def __call__(self, *args: object, **kwargs: object) -> object: ...


def _serializable_ufunc_attrs(kwargs: dict[str, object]) -> dict[str, object]:
    """Encode backend dtype objects at the graph attribute boundary.

    Ufuncs accept backend-specific dtype classes and objects, while the
    canonical Rust graph deliberately accepts only portable typed values.
    Keep the original kwargs for eager execution and normalize only the
    graph snapshot.  ``sig`` is NumPy's alias for ``signature``; storing
    the canonical spelling also lets the backend-neutral evaluator replay
    it without learning backend aliases.
    """
    attrs = dict(kwargs)
    if "dtype" in attrs and attrs["dtype"] is not None:
        attrs["dtype"] = str(np.dtype(cast("Any", attrs["dtype"])))

    signature = attrs.pop("sig", attrs.get("signature"))
    if signature is not None:
        if isinstance(signature, (tuple, list)):
            signature = tuple(str(np.dtype(cast("Any", item))) for item in signature)
        attrs["signature"] = signature
    return attrs


def _collect_operands(
    *,
    recorder: DynamicTape,
    traced_type: type[TracedArrayLike],
    inputs: tuple[ArrayLike | float | TracedArrayLike, ...],
) -> tuple[tuple[_Operand, ...], list[object]]:
    """Split ufunc inputs into graph operands and the values the ufunc runs on."""
    operands: list[_Operand] = []
    values: list[object] = []
    for inp in inputs:
        if isinstance(inp, traced_type):
            if cast("Any", inp).recorder is not recorder:
                # An operand of an enclosing trace is a literal of this one.
                _snapshot_traced_in_active_trace(inp)
                operands.append(_LiteralOperand(inp))
                values.append(inp)
                continue
            node_id, value = _snapshot_traced_in_active_trace(inp)
            operands.append(node_id)
            values.append(weak_scalar_runtime_value(inp, value))
            continue
        # Keep Python scalars weak. Converting them to zero-dimensional arrays
        # here would turn ``1j * float32`` into complex128 and make trace-time
        # promotion disagree with NumPy execution.
        value = (
            inp
            if type(inp) in _PYTHON_SCALAR_TYPES or isinstance(inp, np.ndarray)
            else np.asarray(inp)
        )
        operands.append(_LiteralOperand(value))
        values.append(value)
    return tuple(operands), values


def handle_ufunc(
    ufunc: UfuncLike,
    recorder: DynamicTape,
    traced_type: type[TracedArrayLike],
    inputs: tuple[ArrayLike | float | TracedArrayLike, ...],
    kwargs: dict[str, object],
) -> tuple[UfuncValue, UfuncNodeIDs]:
    """Handle one ufunc call and emit graph node(s)."""
    if ufunc not in _SUPPORTED_UFUNCS:
        msg = f"Unsupported ufunc: {ufunc.__name__}"
        raise TracingError(msg)

    unsupported_kwargs = set(kwargs) - _SUPPORTED_CALL_KWARGS
    if ufunc is np.vecdot:
        # Like linalg.vecdot, vecdot's canonical operation records its contracted axis.
        unsupported_kwargs.discard("axis")
    if unsupported_kwargs:
        msg = (
            f"{ufunc.__name__} kwargs are not supported during tracing: "
            f"{sorted(unsupported_kwargs)}"
        )
        raise TracingError(msg)
    selected_loop_controls = tuple(
        name for name in _LOOP_SELECTION_KWARGS if kwargs.get(name) is not None
    )
    if selected_loop_controls:
        rendered = ", ".join(f"{name}=" for name in selected_loop_controls)
        msg = f"{ufunc.__name__} {rendered} loop selection is not supported during differentiation"
        raise TracingError(msg)
    if "where" in kwargs:
        msg = "where= requires out= during tracing"
        raise TracingError(msg)

    operands, values = _collect_operands(recorder=recorder, traced_type=traced_type, inputs=inputs)
    result = ufunc(*values, **kwargs)
    op = canonicalize_numpy_op(f"numpy.{ufunc.__name__}")
    attrs = _serializable_ufunc_attrs(kwargs)
    if ufunc.nout == 1:
        result_value = cast("ArrayLike", result)
        node_id = _add_backend_node(
            graph=recorder, op=op, inputs=operands, value=result_value, attrs=attrs
        )
        return result_value, node_id

    outputs = tuple(cast("tuple[ArrayLike, ...]", result))
    shape, dtype = _result_shape_and_dtype(outputs[0])
    parent_id = _add_backend_node(
        graph=recorder,
        op=op,
        inputs=operands,
        value=outputs,
        attrs=attrs,
        shape=shape,
        dtype=dtype,
    )
    node_ids = tuple(
        recorder.record_operation(
            "advect.getoutput",
            (parent_id,),
            output,
            {"index": index, "num_outputs": len(outputs)},
            *_result_shape_and_dtype(output),
        )
        for index, output in enumerate(outputs)
    )
    return outputs, node_ids
