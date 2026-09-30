"""Backend-neutral ``__array_function__`` protocol orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._context import (
    _select_deepest_active_recorder,
    _use_operation_recorder,
)
from advect.core._errors import TracingError
from advect.core._protocols import _innermost, _is_traced, _snapshot_traced
from advect.numpy._array_function.composite import (
    _concrete,
    _map_tree,
    _rebuild,
)
from advect.numpy._array_function.registry import (
    _STATIC_ARRAY_FUNCTIONS,
    ARRAY_FUNCTION_HANDLERS,
)
from advect.numpy._composite_lowering import operand_dtype
from advect.numpy._signature import normalize_required_positionals, positional_parameters
from advect.numpy._traced_array_checks import require_active_trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.numpy._protocol_runtime import _TracedProtocolArray

np: Any = _numpy


_LIKE_DISPATCH_CONSTRUCTORS = frozenset(
    {
        "array",
        "arange",
        "asanyarray",
        "asarray",
        "empty",
        "eye",
        "full",
        "identity",
        "ones",
        "tri",
        "zeros",
    }
)
_MISSING = object()


def _validate_out_by_numpy(
    call: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    out_arr: _TracedProtocolArray,
    *,
    copy_operands: bool,
) -> object:
    """Ask NumPy itself to validate one functionalized ``out=`` call.

    Array functions and ufuncs do not share one casting policy: reductions
    permit casts rejected by FFTs and ufunc-backed helpers, while ``stack``
    and ``einsum`` expose their own casting controls.  Reusing the upstream
    call against a private destination copy keeps those rules exact without
    embedding a second, inevitably drifting casting table in the tracer.
    The extra eager call is paid only by explicit mutation.  Array functions
    also receive private operand copies; ufuncs only ever write ``out``.
    """

    def concrete(value: object) -> object:
        leaf = _innermost(value)
        return leaf.copy() if copy_operands and isinstance(leaf, np.ndarray) else leaf

    # A concrete destination keeps the check eager inside nested traces, where
    # NumPy 2.0's clip would not dispatch on a traced out= alone.
    private_out = cast("Any", _innermost(out_arr)).copy()
    call(*_map_tree(concrete, args), **_map_tree(concrete, kwargs), out=private_out)
    return private_out


def _commit_out(
    out_arr: _TracedProtocolArray,
    replacement: object,
    *,
    where: object,
    validated_out: object,
    label: str,
) -> _TracedProtocolArray:
    """Rebind ``out_arr`` to the functional result of one ``out=`` call."""
    if where is not _MISSING:
        replacement = np.where(where, replacement, out_arr)
    old_out = cast("Any", _snapshot_traced(out_arr)[1])
    result = cast("Any", replacement)
    if result.shape != old_out.shape:
        msg = f"{label} result shape {result.shape!r} does not match out= shape {old_out.shape!r}"
        raise TracingError(msg)
    dtype = operand_dtype(old_out)
    if result.dtype != dtype:
        result = result.astype(dtype, copy=False)
    node_id, value = _snapshot_traced(result)
    # An enclosing trace needs the result's own payload, which carries its
    # dependence on the inputs; NumPy's concrete write is exact otherwise.
    payload = value if _is_traced(value) else validated_out
    out_arr.advect_replace(value=payload, node_id=node_id, operation=f"{label} out=")
    return out_arr


class _ArrayFunctionProtocolMixin:
    """Array-function half of the shared traced-array protocol runtime."""

    __slots__ = ()

    @classmethod
    def _wrap_array_function_result(
        cls,
        *,
        result_value: object,
        node_ids: object,
        traced_type: type[_TracedProtocolArray],
        recorder: object,
    ) -> object:
        if isinstance(node_ids, int):
            if isinstance(result_value, (tuple, list)):
                msg = "Array function returned tuple value but single node ID"
                raise TracingError(msg)
            traced_ctor = cast("Callable[..., object]", traced_type)
            return traced_ctor(value=result_value, node_id=node_ids, recorder=recorder)

        if not isinstance(node_ids, (tuple, list)):
            msg = "Array-function result and node-id trees do not match"
            raise TracingError(msg)
        if not isinstance(result_value, (tuple, list)):
            msg = "Array function returned multiple node IDs but non-tuple value"
            raise TracingError(msg)
        if len(result_value) != len(node_ids):
            msg = "Array function output count does not match node ID count"
            raise TracingError(msg)
        children = [
            cls._wrap_array_function_result(
                result_value=value,
                node_ids=node_id,
                traced_type=traced_type,
                recorder=recorder,
            )
            for value, node_id in zip(result_value, node_ids, strict=True)
        ]
        return _rebuild(result_value, children)

    @staticmethod
    def _normalize_array_function_args_and_kwargs(
        *,
        func: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        positional_params = positional_parameters(func)
        if positional_params is None:
            return args, dict(kwargs)
        normalized_kwargs = dict(kwargs)
        out_index = next(
            (index for index, parameter in enumerate(positional_params) if parameter.name == "out"),
            None,
        )
        if out_index is not None and len(args) > out_index:
            # NumPy's dispatcher already rejected a parameter passed both ways.
            names = (parameter.name for parameter in positional_params[out_index:])
            normalized_kwargs.update(zip(names, args[out_index:], strict=False))
            args = args[:out_index]
        return normalize_required_positionals(positional_params, args, normalized_kwargs, func=func)

    @staticmethod
    def _resolve_array_function_out_arg(
        func: object,
        traced_type: type[_TracedProtocolArray],
        out_obj: object,
    ) -> _TracedProtocolArray | None:
        if out_obj is None:
            return None

        if isinstance(out_obj, traced_type):
            return out_obj
        if isinstance(out_obj, tuple) and len(out_obj) == 1 and isinstance(out_obj[0], traced_type):
            if getattr(func, "__name__", "") == "clip":
                return out_obj[0]
            msg = (
                f"numpy.{getattr(func, '__name__', 'array_function')} out= "
                "does not accept a tuple destination"
            )
            raise TracingError(msg)
        if isinstance(out_obj, tuple):
            func_name = getattr(func, "__name__", "array function")
            msg = f"numpy.{func_name} does not accept this tuple destination for out="
            raise TracingError(msg)

        msg = "array-function out= must be one TracedArray from the active trace"
        raise TracingError(msg)

    def array_function(  # noqa: C901, PLR0912
        self,
        self_arr: object,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        _ = types
        owner_recorder = cast("Any", self_arr).recorder
        require_active_trace(recorder=owner_recorder)

        if func is np.result_type or func is np.iscomplexobj or func in _STATIC_ARRAY_FUNCTIONS:
            # Keyword operands must be unwrapped too, or func dispatches back here.
            return cast("Callable[..., object]", func)(*_concrete(args), **_concrete(kwargs))

        handler = ARRAY_FUNCTION_HANDLERS.get(cast("Callable[..., Any]", func))
        if handler is None:
            func_name = getattr(func, "__name__", str(func))
            func_module = getattr(func, "__module__", "numpy")
            msg = (
                f"Array function '{func_module}.{func_name}' is not yet supported. "
                "Rewrite it using supported array operations, or define it with "
                "@advect.primitive and derivative rules."
            )
            raise TracingError(msg)

        normalized_args, normalized_kwargs = self._normalize_array_function_args_and_kwargs(
            func=func,
            args=args,
            kwargs=kwargs,
        )
        if (
            getattr(func, "__name__", "") in _LIKE_DISPATCH_CONSTRUCTORS
            and "like" not in normalized_kwargs
        ):
            # NumPy consumes like= to select __array_function__ and omits it
            # from the forwarded call. Preserve that dispatch-only operand so
            # constructors can record a zero dependence on the active trace.
            normalized_kwargs["like"] = self_arr

        traced_type = cast("type[_TracedProtocolArray]", type(self_arr))
        # NumPy treats out=None as no destination, so handlers never see out=.
        out_arr = self._resolve_array_function_out_arg(
            func, traced_type, normalized_kwargs.pop("out", None)
        )

        traced_inputs: list[_TracedProtocolArray] = [] if out_arr is None else [out_arr]
        for arg in normalized_args:
            if isinstance(arg, traced_type):
                traced_inputs.append(arg)
            elif isinstance(arg, (list, tuple)):
                traced_inputs.extend(item for item in arg if isinstance(item, traced_type))
        for kwarg in normalized_kwargs.values():
            if isinstance(kwarg, traced_type):
                traced_inputs.append(kwarg)
            elif isinstance(kwarg, (list, tuple)):
                traced_inputs.extend(item for item in kwarg if isinstance(item, traced_type))

        recorder = _select_deepest_active_recorder(
            [owner_recorder, *(cast("Any", value).recorder for value in traced_inputs)]
        )

        validated_out: object = None
        clip_where: object = _MISSING
        clip_dtype: object | None = None
        if out_arr is not None:
            if cast("Any", out_arr).recorder is not recorder:
                msg = "array-function out= must belong to the current trace recorder"
                raise TracingError(msg)
            out_arr.advect_require_mutable("array-function out=")
            validated_out = _validate_out_by_numpy(
                cast("Callable[..., object]", func),
                normalized_args,
                normalized_kwargs,
                out_arr,
                copy_operands=True,
            )
            if getattr(func, "__name__", "") == "clip":
                selected_loop_controls = tuple(
                    name
                    for name in ("dtype", "sig", "signature")
                    if normalized_kwargs.get(name) is not None
                )
                if selected_loop_controls:
                    rendered = ", ".join(f"{name}=" for name in selected_loop_controls)
                    msg = (
                        f"numpy.clip {rendered} loop selection is not supported "
                        "during differentiation"
                    )
                    raise TracingError(msg)
                clip_where = normalized_kwargs.pop("where", _MISSING)
                clip_dtype = normalized_kwargs.pop("dtype", None)
                # These standard ufunc controls affect admissibility or
                # allocation, not the mathematical clip result recorded by
                # the primitive. NumPy validation above remains authoritative.
                for control in ("casting", "order", "subok"):
                    normalized_kwargs.pop(control, None)

        with _use_operation_recorder(recorder):
            result_value, node_id = handler(
                cast("Any", recorder),
                cast("Any", traced_type),
                normalized_args,
                normalized_kwargs,
            )

        if out_arr is not None:
            # No out=-accepting array function returns several arrays.
            replacement = cast("Callable[..., Any]", traced_type)(
                value=result_value, node_id=cast("int", node_id), recorder=recorder
            )
            if clip_dtype is not None:
                replacement = replacement.astype(clip_dtype, copy=False)
            return _commit_out(
                out_arr,
                replacement,
                where=clip_where,
                validated_out=validated_out,
                label="array-function",
            )

        return self._wrap_array_function_result(
            result_value=result_value,
            node_ids=node_id,
            traced_type=traced_type,
            recorder=recorder,
        )
