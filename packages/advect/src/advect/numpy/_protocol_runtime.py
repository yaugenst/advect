"""Concrete NumPy protocol orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from advect.core._context import _select_deepest_active_recorder
from advect.core._errors import TracingError
from advect.numpy._array_function.runtime import (
    _MISSING,
    _ArrayFunctionProtocolMixin,
    _commit_out,
    _validate_out_by_numpy,
)
from advect.numpy._composite_lowering import lower_ufunc_method
from advect.numpy._protocol_ufunc import handle_ufunc
from advect.numpy._traced_array_checks import require_active_trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.numpy._protocol_ufunc import UfuncLike


class _TracedProtocolArray(Protocol):
    value: object
    node_id: int
    recorder: object

    def advect_require_mutable(self, operation: str) -> None: ...

    def advect_replace(self, *, value: object, node_id: int, operation: str) -> None: ...


class ArrayProtocolRuntime(_ArrayFunctionProtocolMixin):
    """NumPy protocol runtime."""

    __slots__ = ()

    @staticmethod
    def _resolve_out_arg(
        traced_type: type[_TracedProtocolArray],
        out_obj: tuple[object, ...] | None,
    ) -> _TracedProtocolArray | None:
        # NumPy passes out= to __array_ufunc__ as a tuple and drops an all-None one.
        if out_obj is None:
            return None
        if len(out_obj) != 1:
            msg = "Only single-output out= is supported during tracing"
            raise TracingError(msg)

        candidate = out_obj[0]
        if isinstance(candidate, traced_type):
            return candidate

        msg = "out= must be a TracedArray from the active trace"
        raise TracingError(msg)

    def array_ufunc(
        self,
        self_arr: object,
        ufunc: UfuncLike,
        method: str,
        *inputs: object,
        out: tuple[object, ...] | None = None,
        **kwargs: object,
    ) -> object:
        if method != "__call__":
            return lower_ufunc_method(ufunc, method, inputs, {"out": out, **kwargs})

        require_active_trace(recorder=cast("Any", self_arr).recorder)
        traced_type = cast("type[_TracedProtocolArray]", type(self_arr))
        recorder = _select_deepest_active_recorder(
            cast("Any", value).recorder
            for value in (*inputs, kwargs.get("where"), *(out or ()))
            if isinstance(value, traced_type)
        )
        out_arr = self._resolve_out_arg(traced_type, out)
        if out_arr is not None:
            if cast("Any", out_arr).recorder is not recorder:
                msg = "ufunc out= must belong to the current trace recorder"
                raise TracingError(msg)
            out_arr.advect_require_mutable("ufunc out=")
            validated_out = _validate_out_by_numpy(
                ufunc, inputs, kwargs, out_arr, copy_operands=False
            )
            pure_kwargs = dict(kwargs)
            where = pure_kwargs.pop("where", _MISSING)
            # NumPy requires one out entry per output, so this ufunc has one.
            result_value, node_id = handle_ufunc(
                ufunc,
                cast("Any", recorder),
                cast("Any", traced_type),
                cast("Any", inputs),
                pure_kwargs,
            )
            replacement = cast("Callable[..., object]", traced_type)(
                value=result_value, node_id=cast("int", node_id), recorder=recorder
            )
            return _commit_out(
                out_arr, replacement, where=where, validated_out=validated_out, label="ufunc"
            )

        result_value, node_ids = handle_ufunc(
            ufunc,
            cast("Any", recorder),
            cast("Any", traced_type),
            cast("Any", inputs),
            kwargs,
        )
        return self._wrap_array_function_result(
            result_value=result_value,
            node_ids=node_ids,
            traced_type=traced_type,
            recorder=recorder,
        )


NUMPY_PROTOCOL_RUNTIME = ArrayProtocolRuntime()
