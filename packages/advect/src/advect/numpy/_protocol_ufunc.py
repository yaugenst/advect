"""Concrete NumPy ufunc dispatch and graph recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from advect.core._array_protocol_helpers import (
    literals_are_weak,
    weak_scalar_runtime_value,
)
from advect.core._context import _is_recorder_in_active_trace_stack, get_source_location
from advect.core._protocols import ArrayLike, _snapshot_traced_in_active_trace
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._protocol_array_function_common import _result_shape_and_dtype
from advect.numpy._supported_ufuncs import _SUPPORTED_UFUNCS

if TYPE_CHECKING:
    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike


class UfuncNotSupportedError(Exception):
    """Raised when a ufunc call cannot be handled by the tracer."""


type UfuncValue = ArrayLike | tuple[ArrayLike, ...]
type UfuncNodeIDs = int | tuple[int, ...]

_PYTHON_SCALAR_TYPES = (bool, int, float, complex)
_SIMPLE_ATTRS: dict[str, object] = {"_advect_backend": "numpy"}
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


def _record_operation(
    recorder: DynamicTape,
    *,
    op: str,
    operands: list[tuple[int | None, ArrayLike]],
    value: object,
    attrs: dict[str, object],
    shape: tuple[int, ...],
    dtype: object,
    source_location: str | None = None,
) -> int:
    parents = tuple(node_id for node_id, _value in operands if node_id is not None)
    literals = tuple(operand for node_id, operand in operands if node_id is None)
    if not literals:
        return recorder.record_operation(
            op,
            parents,
            value,
            attrs,
            shape,
            dtype,
            source_location=source_location,
        )
    parent_positions = tuple(
        position for position, (node_id, _value) in enumerate(operands) if node_id is not None
    )
    return recorder.record_operation_with_literals(
        op,
        parents,
        parent_positions,
        literals,
        value,
        attrs,
        shape,
        dtype,
        source_location=source_location,
        literal_weak=literals_are_weak(list(literals)),
    )


class UfuncLike(Protocol):
    """Structural protocol for backend ufunc objects."""

    @property
    def __name__(self) -> str: ...

    @property
    def nin(self) -> int: ...

    @property
    def nout(self) -> int: ...

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class UfuncRuntime:
    """NumPy ufunc runtime."""

    __slots__ = ()

    @property
    def supported_ufuncs(self) -> frozenset[UfuncLike]:
        return cast("frozenset[UfuncLike]", _SUPPORTED_UFUNCS)

    def _ufunc_name(self, ufunc: UfuncLike) -> str:
        return canonicalize_numpy_op(f"numpy.{ufunc.__name__}")

    def _with_backend_attrs(self, attrs: dict[str, object]) -> dict[str, object]:
        out = dict(attrs)
        out["_advect_backend"] = "numpy"
        return out

    def _serializable_ufunc_attrs(self, kwargs: dict[str, object]) -> dict[str, object]:
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
        self,
        *,
        recorder: DynamicTape,
        traced_type: type[TracedArrayLike],
        inputs: tuple[ArrayLike | float | TracedArrayLike, ...],
    ) -> tuple[list[tuple[int | None, ArrayLike]], list[ArrayLike]]:
        operands: list[tuple[int | None, ArrayLike]] = []
        input_values: list[ArrayLike] = []

        for inp in inputs:
            if isinstance(inp, traced_type):
                owner = cast("Any", inp).recorder
                if owner is not recorder:
                    if not _is_recorder_in_active_trace_stack(owner):
                        msg = "Cannot record a ufunc operand from an unrelated trace"
                        raise UfuncNotSupportedError(msg)
                    _snapshot_traced_in_active_trace(inp)
                    opaque = cast("ArrayLike", inp)
                    operands.append((None, opaque))
                    input_values.append(opaque)
                    continue
                if not _is_recorder_in_active_trace_stack(owner):
                    msg = "Cannot record a ufunc operand from an expired trace"
                    raise UfuncNotSupportedError(msg)
                node_id, value = _snapshot_traced_in_active_trace(inp)
                array_value = cast("ArrayLike", weak_scalar_runtime_value(inp, value))
                operands.append((int(node_id), array_value))
                input_values.append(array_value)
                continue

            if type(inp) in _PYTHON_SCALAR_TYPES:
                # Keep Python scalars weak. Converting them to zero-dimensional
                # arrays here would turn ``1j * float32`` into complex128 and
                # make trace-time promotion disagree with NumPy execution.
                scalar = cast("ArrayLike", inp)
                operands.append((None, scalar))
                input_values.append(scalar)
                continue

            if isinstance(inp, np.ndarray):
                array = cast("ArrayLike", inp)
                operands.append((None, array))
                input_values.append(array)
                continue

            arr = np.asarray(inp)
            operands.append((None, arr))
            input_values.append(arr)

        return operands, input_values

    def handle_ufunc(
        self,
        ufunc: UfuncLike,
        method: str,
        recorder: DynamicTape,
        traced_type: type[TracedArrayLike],
        inputs: tuple[ArrayLike | float | TracedArrayLike, ...],
        out: tuple[TracedArrayLike, ...] | None,
        kwargs: dict[str, object],
    ) -> tuple[UfuncValue, UfuncNodeIDs]:
        """Handle one ufunc call and emit graph node(s)."""
        if method != "__call__":
            msg = "Only __call__ is supported in Phase 1"
            raise UfuncNotSupportedError(msg)

        if ufunc not in self.supported_ufuncs:
            msg = f"Unsupported ufunc: {ufunc.__name__}"
            raise UfuncNotSupportedError(msg)
        if out is not None:
            msg = "ufunc out= must be functionalized by the traced-array protocol"
            raise UfuncNotSupportedError(msg)

        unsupported_kwargs = set(kwargs) - _SUPPORTED_CALL_KWARGS
        if unsupported_kwargs:
            msg = (
                f"{ufunc.__name__} kwargs are not supported during tracing: "
                f"{sorted(unsupported_kwargs)}"
            )
            raise UfuncNotSupportedError(msg)
        selected_loop_controls = tuple(
            name for name in _LOOP_SELECTION_KWARGS if kwargs.get(name) is not None
        )
        if selected_loop_controls:
            rendered = ", ".join(f"{name}=" for name in selected_loop_controls)
            msg = (
                f"{ufunc.__name__} {rendered} loop selection is not supported "
                "during differentiation"
            )
            raise UfuncNotSupportedError(msg)

        if not kwargs and int(ufunc.nout) == 1:
            return self.handle_simple_ufunc(
                ufunc=ufunc,
                recorder=recorder,
                traced_type=traced_type,
                inputs=inputs,
            )

        if "where" in kwargs:
            msg = "where= requires out= during tracing"
            raise UfuncNotSupportedError(msg)

        operands, input_values = self._collect_operands(
            recorder=recorder,
            traced_type=traced_type,
            inputs=inputs,
        )

        result = ufunc(*input_values, **kwargs)

        if int(ufunc.nout) == 1:
            result_value = cast("ArrayLike", result)
            result_shape, result_dtype = _result_shape_and_dtype(result_value)
            node_id = _record_operation(
                recorder,
                op=self._ufunc_name(ufunc),
                operands=operands,
                value=result_value,
                attrs=self._with_backend_attrs(self._serializable_ufunc_attrs(kwargs)),
                shape=result_shape,
                dtype=result_dtype,
                source_location=get_source_location(),
            )
            return result_value, node_id

        if not isinstance(result, tuple) or len(result) != int(ufunc.nout):
            msg = (
                f"Expected ufunc '{ufunc.__name__}' to return {ufunc.nout} outputs, "
                f"got {type(result).__name__}"
            )
            raise UfuncNotSupportedError(msg)

        outputs = tuple(result)
        output_meta = tuple(_result_shape_and_dtype(out) for out in outputs)
        output_shapes = tuple(shape for shape, _ in output_meta)
        output_dtypes = tuple(dtype for _, dtype in output_meta)

        parent_id = _record_operation(
            recorder,
            op=self._ufunc_name(ufunc),
            operands=operands,
            value=outputs,
            attrs=self._with_backend_attrs(self._serializable_ufunc_attrs(kwargs)),
            shape=output_shapes[0],
            dtype=output_dtypes[0],
            source_location=get_source_location(),
        )

        node_ids: list[int] = []
        for index, (output, shape, dtype) in enumerate(
            zip(outputs, output_shapes, output_dtypes, strict=True)
        ):
            node_id = recorder.record_operation(
                "advect.getoutput",
                (parent_id,),
                output,
                {"index": index, "num_outputs": len(outputs)},
                shape,
                dtype,
            )
            node_ids.append(node_id)

        return cast("tuple[ArrayLike, ...]", outputs), tuple(node_ids)

    def handle_simple_ufunc(
        self,
        *,
        ufunc: UfuncLike,
        recorder: DynamicTape,
        traced_type: type[TracedArrayLike],
        inputs: tuple[ArrayLike | float | TracedArrayLike, ...],
    ) -> tuple[ArrayLike, int]:
        """Record the common kwargs-free, single-output ufunc path."""
        if ufunc not in self.supported_ufuncs:
            msg = f"Unsupported ufunc: {ufunc.__name__}"
            raise UfuncNotSupportedError(msg)

        operands, input_values = self._collect_operands(
            recorder=recorder,
            traced_type=traced_type,
            inputs=inputs,
        )
        result_value = cast("ArrayLike", ufunc(*input_values))
        result_shape, result_dtype = _result_shape_and_dtype(result_value)
        node_id = _record_operation(
            recorder,
            op=self._ufunc_name(ufunc),
            operands=operands,
            value=result_value,
            attrs=_SIMPLE_ATTRS,
            shape=result_shape,
            dtype=result_dtype,
            source_location=get_source_location(),
        )
        return result_value, node_id


UFUNC_RUNTIME = UfuncRuntime()
