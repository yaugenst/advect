"""Concrete NumPy protocol orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._context import _select_deepest_active_recorder
from advect.core._errors import TraceLevelError, TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.runtime import _ArrayFunctionProtocolMixin
from advect.numpy._protocol_ufunc import UFUNC_RUNTIME
from advect.numpy._traced_array_checks import require_active_trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.numpy._protocol_ufunc import UfuncLike

np: Any = _numpy


class _TracedProtocolArray(Protocol):
    value: object
    node_id: int
    recorder: object

    def advect_require_mutable(self, operation: str) -> None: ...

    def advect_replace(self, *, value: object, node_id: int, operation: str) -> None: ...


_UFUNC_OPERAND_KEY_ALIASES: dict[int, tuple[str, ...]] = {
    1: ("x1", "x", "a"),
    2: ("x2", "y", "b"),
    3: ("x3", "z", "c"),
}

_NUMPY_UFUNC_REDUCTIONS = {
    "add": "sum",
    "multiply": "prod",
}
_NUMPY_UFUNC_ACCUMULATIONS = {
    "add": "cumsum",
    "multiply": "cumprod",
}
_MISSING = object()
_BINARY_ARITY = 2
_SINGLE_OUTPUT_ARITY = 1


class ArrayProtocolRuntime(_ArrayFunctionProtocolMixin):
    """NumPy protocol runtime."""

    __slots__ = ()

    def _normalize_ufunc_inputs_and_kwargs(
        self,
        *,
        ufunc: UfuncLike,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        normalized_inputs = list(inputs)
        normalized_kwargs = dict(kwargs)

        for index in range(ufunc.nin):
            position = index + 1
            aliases = _UFUNC_OPERAND_KEY_ALIASES.get(position, (f"x{position}",))
            if index < len(normalized_inputs):
                if any(alias in normalized_kwargs for alias in aliases):
                    msg = (
                        f"{ufunc.__name__} received duplicate operand for position {index} "
                        "(both positional and keyword)."
                    )
                    raise TracingError(msg)
                continue
            for alias in aliases:
                if alias in normalized_kwargs:
                    normalized_inputs.append(normalized_kwargs.pop(alias))
                    break

        return tuple(normalized_inputs), normalized_kwargs

    @staticmethod
    def _resolve_out_arg(
        traced_type: type[_TracedProtocolArray],
        out_obj: object,
    ) -> _TracedProtocolArray | None:
        if out_obj is None:
            return None
        if not isinstance(out_obj, tuple):
            msg = "out= must be a tuple when provided"
            raise TracingError(msg)
        if len(out_obj) == 1 and out_obj[0] is None:
            return None
        if len(out_obj) != 1:
            msg = "Only single-output out= is supported during tracing"
            raise TracingError(msg)

        candidate = out_obj[0]
        if candidate is None:
            return None
        if isinstance(candidate, traced_type):
            return candidate

        msg = "out= must be a TracedArray from the active trace"
        raise TracingError(msg)

    def run_simple_ufunc(
        self,
        *,
        self_arr: object,
        ufunc: UfuncLike,
        inputs: tuple[object, ...],
    ) -> object:
        """Run the common single-output ufunc path without generic normalization."""
        owner_recorder = cast("Any", self_arr).recorder
        require_active_trace(recorder=owner_recorder)
        traced_type = cast("type[_TracedProtocolArray]", type(self_arr))
        recorder = _select_deepest_active_recorder(
            cast("Any", value).recorder for value in inputs if isinstance(value, traced_type)
        )

        try:
            result_value, node_id = UFUNC_RUNTIME.handle_simple_ufunc(
                ufunc=ufunc,
                recorder=cast("Any", recorder),
                traced_type=cast("Any", traced_type),
                inputs=cast("Any", inputs),
            )
        except Exception as exc:
            if type(exc).__name__ == "UfuncNotSupportedError":
                raise TracingError(str(exc)) from exc
            raise

        traced_ctor = cast("Callable[..., object]", traced_type)
        return traced_ctor(value=result_value, node_id=node_id, recorder=recorder)

    @staticmethod
    def _array_function_out_from_ufunc_out(
        out: tuple[object, ...] | None,
    ) -> object | None:
        if out is None:
            return None
        if len(out) != 1:
            msg = "NumPy ufunc method out= requires exactly one destination"
            raise TracingError(msg)
        return out[0]

    def _run_numpy_ufunc_method(
        self,
        *,
        ufunc: UfuncLike,
        method: str,
        inputs: tuple[object, ...],
        out: tuple[object, ...] | None,
        kwargs: dict[str, object],
    ) -> object:
        """Lower the small method subset with exact NumPy-function equivalents."""
        if len(inputs) != 1 and method in {"reduce", "accumulate"}:
            msg = f"numpy.{ufunc.__name__}.{method} expects one input array"
            raise TracingError(msg)

        clean_kwargs = dict(kwargs)
        destination = self._array_function_out_from_ufunc_out(out)

        if method == "reduce":
            function_name = _NUMPY_UFUNC_REDUCTIONS.get(ufunc.__name__)
            if function_name is None:
                msg = f"numpy.{ufunc.__name__}.reduce is not supported during tracing"
                raise TracingError(msg)
            clean_kwargs.setdefault("axis", 0)
            if destination is not None:
                clean_kwargs["out"] = destination
            return cast("Callable[..., object]", getattr(np, function_name))(
                inputs[0],
                **clean_kwargs,
            )

        if method == "accumulate":
            function_name = _NUMPY_UFUNC_ACCUMULATIONS.get(ufunc.__name__)
            if function_name is None:
                msg = f"numpy.{ufunc.__name__}.accumulate is not supported during tracing"
                raise TracingError(msg)
            clean_kwargs.setdefault("axis", 0)
            if destination is not None:
                clean_kwargs["out"] = destination
            return cast("Callable[..., object]", getattr(np, function_name))(
                inputs[0],
                **clean_kwargs,
            )

        if method == "outer":
            if (
                len(inputs) != _BINARY_ARITY
                or ufunc.nin != _BINARY_ARITY
                or ufunc.nout != _SINGLE_OUTPUT_ARITY
                or getattr(ufunc, "signature", None) is not None
            ):
                msg = (
                    f"numpy.{ufunc.__name__}.outer requires an ordinary binary, "
                    "single-output ufunc; generalized ufunc signatures are unsupported"
                )
                raise TracingError(msg)
            left, right = inputs
            left_ndim = len(cast("Any", left).shape)
            right_ndim = len(cast("Any", right).shape)
            expanded_left = np.expand_dims(
                left,
                axis=tuple(range(left_ndim, left_ndim + right_ndim)),
            )
            expanded_right = np.expand_dims(
                right,
                axis=tuple(range(left_ndim)),
            )
            return ufunc(
                expanded_left,
                expanded_right,
                out=out,
                **clean_kwargs,
            )

        msg = f"numpy.{ufunc.__name__}.{method} is not supported during tracing"
        raise TracingError(msg)

    def _validate_numpy_ufunc_out(
        self,
        *,
        ufunc: UfuncLike,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
        out_arr: _TracedProtocolArray,
        traced_type: type[_TracedProtocolArray],
    ) -> object | None:
        _out_node_id, out_value = _snapshot_traced(out_arr)
        copy_out = getattr(out_value, "copy", None)
        if not callable(copy_out):
            msg = "ufunc out= requires a destination with copy() support"
            raise TracingError(msg)
        private_out = copy_out()

        def unwrap(value: object) -> object:
            if isinstance(value, traced_type):
                concrete = _snapshot_traced(value)[1]
                if concrete is value:
                    return concrete
                return unwrap(concrete)
            if isinstance(value, tuple):
                return tuple(unwrap(item) for item in value)
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            return value

        concrete_inputs = tuple(unwrap(value) for value in inputs)
        concrete_kwargs = {key: unwrap(value) for key, value in kwargs.items()}
        ufunc(*concrete_inputs, out=private_out, **concrete_kwargs)
        return private_out

    def _functionalize_ufunc_out(
        self,
        *,
        ufunc: UfuncLike,
        recorder: object,
        traced_type: type[_TracedProtocolArray],
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
        out_arr: _TracedProtocolArray,
    ) -> object:
        validated_out = self._validate_numpy_ufunc_out(
            ufunc=ufunc,
            inputs=inputs,
            kwargs=kwargs,
            out_arr=out_arr,
            traced_type=traced_type,
        )
        pure_kwargs = dict(kwargs)
        where = pure_kwargs.pop("where", _MISSING)
        result_value, node_id = UFUNC_RUNTIME.handle_ufunc(
            ufunc=ufunc,
            recorder=cast("Any", recorder),
            traced_type=cast("Any", traced_type),
            inputs=cast("Any", inputs),
            kwargs=pure_kwargs,
        )
        if isinstance(node_id, tuple) or isinstance(result_value, tuple):
            msg = "out= is not supported for multi-output ufuncs"
            raise TracingError(msg)
        traced_ctor = cast("Callable[..., _TracedProtocolArray]", traced_type)
        replacement = traced_ctor(
            value=result_value,
            node_id=cast("int", node_id),
            recorder=recorder,
        )
        if where is not _MISSING:
            replacement = cast(
                "_TracedProtocolArray",
                np.where(where, replacement, out_arr),
            )
        _out_node_id, old_out = _snapshot_traced(out_arr)
        target_dtype = getattr(old_out, "dtype", None)
        target_shape = tuple(getattr(old_out, "shape", ()))
        if tuple(getattr(replacement, "shape", ())) != target_shape:
            msg = (
                f"ufunc result shape {getattr(replacement, 'shape', None)!r} does not "
                f"match out= shape {target_shape!r}"
            )
            raise TracingError(msg)
        if getattr(replacement, "dtype", None) != target_dtype:
            replacement = cast(
                "_TracedProtocolArray",
                cast("Any", replacement).astype(target_dtype, copy=False),
            )
        replacement_node_id, replacement_value = _snapshot_traced(replacement)
        out_arr.advect_replace(
            value=replacement_value if validated_out is None else validated_out,
            node_id=replacement_node_id,
            operation="ufunc out=",
        )
        return out_arr

    def array_ufunc(  # noqa: C901 - one protocol boundary owns dispatch validation
        self,
        self_arr: object,
        ufunc: UfuncLike,
        method: str,
        *inputs: object,
        out: tuple[object, ...] | None = None,
        **kwargs: object,
    ) -> object:
        if method != "__call__":
            return self._run_numpy_ufunc_method(
                ufunc=ufunc,
                method=method,
                inputs=inputs,
                out=out,
                kwargs=dict(kwargs),
            )

        simple_call = out is None and not kwargs and len(inputs) == ufunc.nin and ufunc.nout == 1
        if simple_call:
            return self.run_simple_ufunc(
                self_arr=self_arr,
                ufunc=ufunc,
                inputs=inputs,
            )

        owner_recorder = cast("Any", self_arr).recorder
        require_active_trace(recorder=owner_recorder)
        traced_type = cast("type[_TracedProtocolArray]", type(self_arr))
        normalized_inputs, clean_kwargs = self._normalize_ufunc_inputs_and_kwargs(
            ufunc=ufunc,
            inputs=inputs,
            kwargs=dict(kwargs),
        )
        traced_recorders = [
            cast("Any", value).recorder
            for value in normalized_inputs
            if isinstance(value, traced_type)
        ]
        where_candidate = clean_kwargs.get("where")
        if isinstance(where_candidate, traced_type):
            traced_recorders.append(cast("Any", where_candidate).recorder)
        resolved_out_candidate = clean_kwargs.get("out", out)
        if isinstance(resolved_out_candidate, tuple):
            traced_recorders.extend(
                cast("Any", value).recorder
                for value in resolved_out_candidate
                if isinstance(value, traced_type)
            )
        recorder = _select_deepest_active_recorder(traced_recorders)
        resolved_out_obj = clean_kwargs.pop("out", out)
        out_arr = self._resolve_out_arg(traced_type, resolved_out_obj)
        if out_arr is not None:
            if cast("Any", out_arr).recorder is not recorder:
                msg = "ufunc out= must belong to the current trace recorder"
                raise TraceLevelError(msg)
            out_arr.advect_require_mutable("ufunc out=")

        if out_arr is not None:
            try:
                return self._functionalize_ufunc_out(
                    ufunc=ufunc,
                    recorder=recorder,
                    traced_type=traced_type,
                    inputs=normalized_inputs,
                    kwargs=clean_kwargs,
                    out_arr=out_arr,
                )
            except Exception as exc:
                if type(exc).__name__ == "UfuncNotSupportedError":
                    raise TracingError(str(exc)) from exc
                raise

        try:
            result_value, node_id = UFUNC_RUNTIME.handle_ufunc(
                ufunc=ufunc,
                recorder=cast("Any", recorder),
                traced_type=cast("Any", traced_type),
                inputs=cast("Any", normalized_inputs),
                kwargs=clean_kwargs,
            )
        except Exception as exc:
            if type(exc).__name__ == "UfuncNotSupportedError":
                raise TracingError(str(exc)) from exc
            raise

        if isinstance(node_id, tuple):
            if not isinstance(result_value, tuple):
                msg = "Ufunc returned multiple node IDs but non-tuple value"
                raise TracingError(msg)
            if len(result_value) != len(node_id):
                msg = "Ufunc output count does not match node ID count"
                raise TracingError(msg)
            return tuple(
                cast("Callable[..., object]", traced_type)(
                    value=value,
                    node_id=out_id,
                    recorder=recorder,
                )
                for value, out_id in zip(result_value, node_id, strict=True)
            )

        if isinstance(result_value, tuple):
            msg = "Ufunc returned tuple value but single node ID"
            raise TracingError(msg)
        traced_ctor = cast("Callable[..., object]", traced_type)
        return traced_ctor(value=result_value, node_id=node_id, recorder=recorder)


NUMPY_PROTOCOL_RUNTIME = ArrayProtocolRuntime()
