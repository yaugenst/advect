# ruff: noqa: ANN401, FBT001
# ANN401: NumPy's mixin requires Any operands for operator overrides.
# FBT001: __array__ follows NumPy's positional copy protocol.
"""TracedArray class for Advect.

TracedArray wraps a NumPy array and records operations to a computational graph.
It intercepts NumPy operations via __array_ufunc__ and __array_function__ protocols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, cast, override

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from advect.core._array_protocol_helpers import normalize_item_index
from advect.core._context import is_debug
from advect.core._diagnostics import summarize_value
from advect.core._errors import (
    MutationError,
    StaleViewError,
    TracingError,
    _array_conversion_error,
)
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.mutation import (
    NOT_FUNCTIONALIZED,
    functionalize_array_function_mutation,
)
from advect.numpy._constructors import NOT_A_CONSTRUCTOR, handle_traced_constructor
from advect.numpy._op_bindings import canonicalize_numpy_op, frontend_lowering
from advect.numpy._protocol_runtime import NUMPY_PROTOCOL_RUNTIME
from advect.numpy._traced_array_checks import require_active_trace
from advect.numpy._traced_array_indexing import getitem as _getitem, setitem as _setitem
from advect.numpy._traced_array_inplace import (
    inplace_matmul as _inplace_matmul,
    inplace_op as _inplace_op,
)
from advect.numpy._traced_array_protocols import (
    NOT_HANDLED,
    run_ephemeral_simple_ufunc,
    run_ephemeral_sum,
)
from advect.numpy._traced_array_state import ViewState, user_location

if TYPE_CHECKING:
    from types import NotImplementedType

    from numpy.typing import DTypeLike

    from advect.core._native import DynamicTape
    from advect.core._protocols import ArrayLike
    from advect.numpy._traced_array_state import SourceLocation


_SEMANTIC_ALIAS_FUNCTIONS = frozenset(
    {
        "array_split",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "broadcast_arrays",
        "broadcast_to",
        "diag",
        "diagonal",
        "dsplit",
        "expand_dims",
        "flip",
        "fliplr",
        "flipud",
        "hsplit",
        "imag",
        "matrix_transpose",
        "moveaxis",
        "permute_dims",
        "ravel",
        "real",
        "real_if_close",
        "reshape",
        "rollaxis",
        "rot90",
        "split",
        "sliding_window_view",
        "squeeze",
        "swapaxes",
        "transpose",
        "unstack",
        "vsplit",
    }
)


class TracedArray(NDArrayOperatorsMixin):
    """A traced array that records operations to a computational graph."""

    __slots__ = (
        "_deferred_getitem",
        "_epoch",
        "_last_update_location",
        "_node_id",
        "_owned",
        "_value",
        "_view_state",
        "recorder",
    )

    __array_priority__ = 100_000
    __advect_frontend__ = "numpy"

    def __init__(
        self,
        value: ArrayLike,
        node_id: int | None,
        recorder: DynamicTape,
        *,
        owned: bool = True,
        view_state: ViewState | None = None,
        deferred_getitem: tuple[TracedArray, dict[str, object]] | None = None,
    ) -> None:
        self._value: ArrayLike = value
        self._node_id = node_id
        self._deferred_getitem = deferred_getitem
        self.recorder = recorder
        self._owned = owned
        self._epoch = 0
        self._last_update_location: SourceLocation | None = None
        self._view_state = view_state

    @property
    def value(self) -> ArrayLike:
        """Reject public access to the trace-time payload."""
        msg = (
            "Tracer payloads are private Advect implementation details. "
            "Return a value from the traced function to materialize it."
        )
        raise TracingError(msg)

    @property
    def node_id(self) -> int:
        """Return the current SSA identifier while the owning trace is live."""
        require_active_trace(recorder=self.recorder)
        self._check_view_epoch()
        return self._materialize_deferred_getitem()

    @property
    def epoch(self) -> int:
        """Return the source-level update epoch for this wrapper cell."""
        return self._epoch

    @property
    def is_view(self) -> bool:
        """Return whether this wrapper aliases a base wrapper."""
        return self._view_state is not None

    def _check_view_epoch(self) -> None:
        view_state = self._view_state
        if view_state is None:
            return
        root = view_state.root
        if root.epoch == view_state.epoch:
            return
        created = f" The view was created at {view_state.location}." if view_state.location else ""
        last_update_location = getattr(root, "_last_update_location", None)
        updated = (
            f" The base was updated at {last_update_location}." if last_update_location else ""
        )
        msg = (
            "This traced view is stale because its base array was functionally updated. "
            "Continuing would diverge from NumPy view semantics; call `.copy()` before "
            "the update or reorder the computation."
            f"{created}{updated}"
        )
        raise StaleViewError(msg)

    def _advect_snapshot(self) -> tuple[int, ArrayLike]:
        """Return one internally validated SSA/value pair for protocol dispatch."""
        require_active_trace(recorder=self.recorder)
        self._check_view_epoch()
        return self._materialize_deferred_getitem(), self._value

    def _advect_snapshot_in_active_trace(self) -> tuple[int, ArrayLike]:
        """Return a snapshot after a protocol boundary validated this trace."""
        self._check_view_epoch()
        return self._materialize_deferred_getitem(), self._value

    def _materialize_deferred_getitem(self) -> int:
        """Emit an ephemeral getitem only when its result becomes an SSA operand."""
        node_id = self._node_id
        if node_id is not None:
            return node_id

        deferred = self._deferred_getitem
        if deferred is None:  # Constructor invariant.
            msg = "Traced array has neither an SSA node nor a deferred getitem"
            raise RuntimeError(msg)
        source, attrs = deferred
        source_node_id, _source_value = source._advect_snapshot_in_active_trace()  # noqa: SLF001
        value = self._value
        node_id = self.recorder.record_operation(
            "advect.getitem",
            (source_node_id,),
            value,
            attrs,
            value.shape,
            value.dtype,
        )
        self._node_id = node_id
        self._deferred_getitem = None
        return node_id

    def _root_for_view(self) -> TracedArray:
        view_state = self._view_state
        return self if view_state is None else view_state.root

    def _require_mutable(self, *, operation: str) -> None:
        require_active_trace(recorder=self.recorder)
        self._require_mutable_in_active_trace(operation=operation)

    def _require_mutable_in_active_trace(self, *, operation: str) -> None:
        """Validate mutation after the protocol boundary checked trace lifetime."""
        self._check_view_epoch()
        if self._view_state is not None:
            location = f" at {self._view_state.location}" if self._view_state.location else ""
            msg = (
                f"Cannot apply {operation} through a traced view{location}. "
                "Update the base with a single index expression, or call `.copy()` first."
            )
            raise MutationError(msg)
        if not self._owned:
            name = self.recorder.get_node_name(self._materialize_deferred_getitem())
            label = f" '{name}'" if name else ""
            msg = (
                f"Cannot mutate traced input{label} with {operation}. "
                "Use `x = x + value` for rebinding, or `x = x.copy()` before mutation."
            )
            raise MutationError(msg)

    def _commit_current(
        self,
        *,
        value: ArrayLike,
        node_id: int,
        location: SourceLocation | None = None,
    ) -> None:
        """Swap this wrapper to a new SSA value after a successful pure update."""
        self._value = value
        self._node_id = node_id
        self._deferred_getitem = None
        self._epoch += 1
        self._last_update_location = location if location is not None else user_location()

    def _refresh_direct_view(self, *, key: object, index_spec: object) -> None:
        """Refresh the one view that performed a functional root update."""
        view_state = self._view_state
        if view_state is None or view_state.index_spec is None:
            msg = "Only a direct basic view can be refreshed"
            raise RuntimeError(msg)
        root = view_state.root
        self._value = cast("Any", root._value)[cast("Any", key)]  # noqa: SLF001
        self._node_id = None
        self._deferred_getitem = (root, {"index": key})
        self._view_state = ViewState(
            root=root,
            epoch=root.epoch,
            index_spec=index_spec,
            location=view_state.location,
        )

    def advect_require_mutable(self, operation: str) -> None:
        """Backend-neutral protocol hook used before evaluating ``out=``."""
        self._require_mutable(operation=operation)

    def advect_replace(self, *, value: ArrayLike, node_id: int, operation: str) -> None:
        """Backend-neutral protocol hook for committing a functional write."""
        self._require_mutable(operation=operation)
        self._commit_current(value=value, node_id=node_id)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying array."""
        _node_id, value = self._advect_snapshot()
        return tuple(value.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return the dtype of the underlying array."""
        _node_id, value = self._advect_snapshot()
        return np.dtype(value.dtype)

    @property
    def _advect_weak(self) -> bool:
        """Return the weak-scalar category of the current SSA value."""
        return self.recorder.is_weak(self.node_id)

    def _advect_mark_weak(self) -> None:
        """Mark this rank-zero SSA value as a weak scalar."""
        self.recorder.mark_weak(self.node_id)

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        """Expose NumPy's negotiated namespace without detaching the tracer."""
        _node_id, value = self._advect_snapshot()
        namespace = getattr(value, "__array_namespace__", None)
        if not callable(namespace):
            msg = "The traced NumPy value does not expose __array_namespace__()"
            raise TypeError(msg)
        return namespace(api_version=api_version)

    @property
    def real(self) -> TracedArray:
        """Return the real part of the array."""
        require_active_trace(recorder=self.recorder)

        input_node_id, value = self._advect_snapshot()
        result_value = np.real(cast("Any", value))
        node_id = self.recorder.record_operation(
            canonicalize_numpy_op("numpy.real"),
            (input_node_id,),
            result_value,
            {"_advect_backend": "numpy"},
            result_value.shape,
            result_value.dtype,
        )
        root = self._root_for_view()
        return TracedArray(
            value=result_value,
            node_id=node_id,
            recorder=self.recorder,
            owned=False,
            view_state=ViewState(root, root.epoch, None, user_location()),
        )

    @property
    def imag(self) -> TracedArray:
        """Return the imaginary part of the array."""
        require_active_trace(recorder=self.recorder)

        input_node_id, value = self._advect_snapshot()
        result_value = np.imag(cast("Any", value))
        node_id = self.recorder.record_operation(
            canonicalize_numpy_op("numpy.imag"),
            (input_node_id,),
            result_value,
            {"_advect_backend": "numpy"},
            result_value.shape,
            result_value.dtype,
        )
        root = self._root_for_view()
        return TracedArray(
            value=result_value,
            node_id=node_id,
            recorder=self.recorder,
            owned=False,
            view_state=ViewState(root, root.epoch, None, user_location()),
        )

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the underlying array."""
        return len(self.shape)

    @property
    def T(self) -> TracedArray:  # noqa: N802
        """Return the transposed array."""
        return self.transpose()

    @property
    def size(self) -> int:
        """Return the total number of elements in the underlying array."""
        return int(np.prod(self.shape, dtype=np.intp))

    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray[Any, Any]:
        """Raise an error when attempting to convert to ndarray during tracing."""
        _ = dtype, copy
        require_active_trace(recorder=self.recorder)
        raise TracingError(_array_conversion_error())

    def __bool__(self) -> bool:
        """Resolve truth from the concrete trace-time value.

        This gives concrete dynamic traces define-by-run control flow. NumPy's
        normal ambiguity error is preserved for arrays with more than one
        element. Abstract staging rejects value-dependent truth instead.
        """
        require_active_trace(recorder=self.recorder)
        return bool(self._advect_snapshot()[1])

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return len(cast("Any", self._advect_snapshot()[1]))

    @frontend_lowering("advect.copy")
    def copy(self, order: str | None = "C") -> TracedArray:
        """Create a copy of the array with independent storage."""
        require_active_trace(recorder=self.recorder)
        if order is not None and not isinstance(order, str):
            msg = f"order must be str, not {type(order).__name__}"
            raise TypeError(msg)
        normalized_order = "C" if order is None else order.upper()
        if normalized_order not in {"A", "C", "F", "K"}:
            msg = f"order must be one of 'A', 'C', 'F', or 'K' (got {order!r})"
            raise ValueError(msg)

        input_node_id, value = self._advect_snapshot()
        result_value = cast("Any", value).copy(order=normalized_order)
        node_id = self.recorder.record_operation(
            "advect.copy",
            (input_node_id,),
            result_value,
            {"order": normalized_order, "_advect_backend": "numpy"},
            result_value.shape,
            result_value.dtype,
        )
        return TracedArray(value=result_value, node_id=node_id, recorder=self.recorder)

    def __copy__(self) -> TracedArray:
        """Record a real array copy, matching ``copy.copy(ndarray)`` semantics."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, object]) -> TracedArray:
        """Return a deep copy of the value, recording an `advect.copy` node."""
        _ = memo
        return self.copy()

    @frontend_lowering("array.sum")
    def sum(self, *args: Any, **kwargs: Any) -> TracedArray:
        """Sum of array elements (NumPy method-style API)."""
        return cast("TracedArray", np.sum(self, *args, **kwargs))

    @frontend_lowering("array.transpose")
    def transpose(self, *axes: Any) -> TracedArray:
        """Transpose the array (NumPy method-style API)."""
        require_active_trace(recorder=self.recorder)
        if not axes:
            axes_arg: Any = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes_arg = axes[0]
        else:
            axes_arg = axes
        return cast("TracedArray", np.transpose(self, axes=axes_arg))

    @frontend_lowering("array.reshape")
    def reshape(
        self,
        *shape: Any,
        order: Literal["A", "C", "F"] = "C",
        copy: bool | None = None,
    ) -> TracedArray:
        """Reshape the array (NumPy method-style API)."""
        require_active_trace(recorder=self.recorder)
        if not shape:
            msg = "reshape() missing required argument 'shape'"
            raise TypeError(msg)
        newshape = shape[0] if len(shape) == 1 else shape
        if copy is None:
            return cast("TracedArray", np.reshape(self, newshape, order=order))
        return cast("TracedArray", np.reshape(self, newshape, order=order, copy=copy))

    @frontend_lowering("composite")
    def item(self, *args: object) -> TracedArray:
        """Return one element as a rank-zero traced value.

        A Python scalar would detach the derivative. The transform boundary
        materializes this rank-zero value after tracing completes.
        """
        require_active_trace(recorder=self.recorder)
        index = normalize_item_index(args, ndim=self.ndim)
        if index is None:
            if self.size != 1:
                msg = "can only convert an array of size 1 to a scalar"
                raise ValueError(msg)
            if self.shape == ():
                return self
            return self[tuple(0 for _dimension in self.shape)]
        if isinstance(index, tuple):
            return self[index]
        return self.reshape((-1,))[index]

    @frontend_lowering("array.astype")
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> TracedArray:
        """Cast to a specified dtype (NumPy method-style API)."""
        require_active_trace(recorder=self.recorder)
        target_dtype = np.dtype(dtype)
        if target_dtype == self.dtype and order == "K" and casting == "unsafe" and subok:
            return self.copy(order="K") if copy else self

        input_node_id, value = self._advect_snapshot()
        result_value = cast("Any", value).astype(
            target_dtype,
            order=order,
            casting=casting,
            subok=subok,
            copy=copy,
        )
        attrs: dict[str, Any] = {
            "dtype": str(target_dtype),
            "order": order,
            "casting": casting,
            "subok": subok,
        }
        if not copy:
            attrs["copy"] = False
        attrs["_advect_backend"] = "numpy"
        node_id = self.recorder.record_operation(
            "array.astype",
            (input_node_id,),
            result_value,
            attrs,
            result_value.shape,
            result_value.dtype,
        )
        traced_type = type(self)
        return traced_type(value=result_value, node_id=node_id, recorder=self.recorder)

    def __repr__(self) -> str:
        """Return a string representation of the TracedArray."""
        node_id = self.node_id
        name = self.recorder.get_node_name(node_id)
        name_str = f" [{name}]" if name else ""
        prefix = f"TracedArray(node=%{node_id}{name_str}"
        if is_debug():
            return f"{prefix}, {summarize_value(self._value)})"
        return f"{prefix}, shape={self.shape}, dtype={self.dtype})"

    @override
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: np.ndarray[Any, Any] | TracedArray | float,
        out: tuple[np.ndarray[Any, Any] | TracedArray | None, ...] | None = None,
        **kwargs: object,
    ) -> TracedArray | tuple[TracedArray, ...] | NotImplementedType:
        simple_call = (
            method == "__call__"
            and out is None
            and not kwargs
            and len(inputs) == ufunc.nin
            and ufunc.nout == 1
        )
        if simple_call:
            fast_result = run_ephemeral_simple_ufunc(self, ufunc, inputs)
            if fast_result is not NOT_HANDLED:
                return cast(
                    "TracedArray | tuple[TracedArray, ...] | NotImplementedType",
                    fast_result,
                )
            return cast(
                "TracedArray | tuple[TracedArray, ...] | NotImplementedType",
                NUMPY_PROTOCOL_RUNTIME.run_simple_ufunc(
                    self_arr=self,
                    ufunc=ufunc,
                    inputs=inputs,
                ),
            )
        return cast(
            "TracedArray | tuple[TracedArray, ...] | NotImplementedType",
            NUMPY_PROTOCOL_RUNTIME.array_ufunc(
                self,
                ufunc,
                method,
                *inputs,
                out=out,
                **kwargs,
            ),
        )

    def __array_function__(
        self,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        constructor_result = handle_traced_constructor(self, func, args, kwargs)
        if constructor_result is not NOT_A_CONSTRUCTOR:
            return constructor_result
        mutation_result = functionalize_array_function_mutation(
            self,
            func,
            args,
            kwargs,
        )
        if mutation_result is not NOT_FUNCTIONALIZED:
            return mutation_result
        fast_result = run_ephemeral_sum(self, func, args, kwargs)
        if fast_result is not NOT_HANDLED:
            return fast_result
        result = NUMPY_PROTOCOL_RUNTIME.array_function(self, func, types, args, kwargs)
        if getattr(func, "__name__", None) not in _SEMANTIC_ALIAS_FUNCTIONS:
            return result

        location = user_location()

        def register_alias(value: object, root: TracedArray) -> object:
            if not isinstance(value, TracedArray):
                return value
            node_id, payload = _snapshot_traced(value)
            return TracedArray(
                value=payload,
                node_id=node_id,
                recorder=value.recorder,
                owned=False,
                view_state=ViewState(root, root.epoch, None, location),
            )

        def register_tree(value: object, root: TracedArray) -> object:
            if isinstance(value, list):
                return [register_tree(item, root) for item in value]
            if isinstance(value, tuple):
                items = [register_tree(item, root) for item in value]
                return tuple(items) if type(value) is tuple else cast("Any", type(value))(*items)
            return register_alias(value, root)

        if getattr(func, "__name__", None) == "broadcast_arrays" and isinstance(
            result, (tuple, list)
        ):
            roots = [
                value._root_for_view()  # noqa: SLF001 - same-class alias bookkeeping
                if isinstance(value, TracedArray)
                else self._root_for_view()
                for value in args
            ]
            items = [register_tree(value, roots[index]) for index, value in enumerate(result)]
            return items if isinstance(result, list) else tuple(items)
        return register_tree(result, self._root_for_view())

    def __getitem__(self, key: object) -> TracedArray:
        return _getitem(self, key)

    def __setitem__(self, key: object, value: object) -> None:
        _setitem(self, key, value)

    def _inplace_op(self, other: Any, ufunc: np.ufunc, op_name: str) -> Self:
        return cast("Self", _inplace_op(self, other, ufunc, op_name))

    @override
    def __iadd__(self, other: Any) -> Self:
        return self._inplace_op(other, np.add, "numpy.add")

    @override
    def __isub__(self, other: Any) -> Self:
        return self._inplace_op(other, np.subtract, "numpy.subtract")

    @override
    def __imul__(self, other: Any) -> Self:
        return self._inplace_op(other, np.multiply, "numpy.multiply")

    @override
    def __itruediv__(self, other: Any) -> Self:
        return self._inplace_op(other, np.divide, "numpy.divide")

    @override
    def __ifloordiv__(self, other: Any) -> Self:
        return self._inplace_op(other, np.floor_divide, "numpy.floor_divide")

    @override
    def __imod__(self, other: Any) -> Self:
        return self._inplace_op(other, np.mod, "numpy.mod")

    @override
    def __ipow__(self, other: Any) -> Self:
        return self._inplace_op(other, np.power, "numpy.power")

    @override
    def __imatmul__(self, other: Any) -> Self:
        return cast("Self", _inplace_matmul(self, other))

    @override
    def __iand__(self, other: Any) -> Self:
        return self._inplace_op(other, np.bitwise_and, "numpy.bitwise_and")

    @override
    def __ior__(self, other: Any) -> Self:
        return self._inplace_op(other, np.bitwise_or, "numpy.bitwise_or")

    @override
    def __ixor__(self, other: Any) -> Self:
        return self._inplace_op(other, np.bitwise_xor, "numpy.bitwise_xor")

    @override
    def __ilshift__(self, other: Any) -> Self:
        return self._inplace_op(other, np.left_shift, "numpy.left_shift")

    @override
    def __irshift__(self, other: Any) -> Self:
        return self._inplace_op(other, np.right_shift, "numpy.right_shift")
