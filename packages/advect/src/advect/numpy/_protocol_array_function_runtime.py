"""Backend-neutral ``__array_function__`` protocol orchestration."""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._context import (
    _is_recorder_in_active_trace_stack,
    _select_deepest_active_recorder,
    _use_operation_recorder,
)
from advect.core._errors import TraceLevelError, TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._protocol_array_function import (
    _STATIC_ARRAY_FUNCTIONS,
    ARRAY_FUNCTION_RUNTIME,
)
from advect.numpy._signature import normalize_required_positionals
from advect.numpy._traced_array_checks import require_active_trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.numpy._protocol_runtime import _TracedProtocolArray

np: Any = _numpy


_ARRAY_FUNCTION_POS_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "clip": ("a", "a_min", "a_max"),
    "concatenate": ("arrays",),
    "stack": ("arrays",),
    "where": ("condition", "x", "y"),
    "dot": ("a", "b"),
    "inner": ("a", "b"),
    "outer": ("a", "b"),
    "kron": ("a", "b"),
    "cross": ("a", "b"),
    "tensordot": ("a", "b"),
    "interp": ("x", "xp", "fp"),
    "linspace": ("start", "stop"),
}
_UNINSPECTABLE_POSITIONAL_PARAMETERS: dict[
    str,
    tuple[tuple[str, object], ...],
] = {
    "concatenate": (
        ("arrays", inspect.Parameter.empty),
        ("axis", 0),
        ("out", None),
    ),
    "dot": (
        ("a", inspect.Parameter.empty),
        ("b", inspect.Parameter.empty),
        ("out", None),
    ),
}

_ARRAY_FUNCTION_SIGNATURE_CACHE_SIZE = 512
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


@functools.lru_cache(maxsize=_ARRAY_FUNCTION_SIGNATURE_CACHE_SIZE)
def _cached_positional_parameters(func: object) -> tuple[inspect.Parameter, ...]:
    signature = inspect.signature(cast("Any", func))
    return tuple(
        param
        for param in signature.parameters.values()
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def _positional_parameters(func: object) -> tuple[inspect.Parameter, ...]:
    try:
        return _cached_positional_parameters(func)
    except TypeError:
        # Callable instances may opt out of hashing. Preserve support without
        # retaining them or maintaining a second identity-based cache.
        signature = inspect.signature(cast("Any", func))
        return tuple(
            param
            for param in signature.parameters.values()
            if param.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )


def _unwrap_array_function_value(
    value: object,
    traced_type: type[_TracedProtocolArray],
) -> object:
    if isinstance(value, traced_type):
        concrete = _snapshot_traced(value)[1]
        if concrete is value:
            return concrete
        return _unwrap_array_function_value(concrete, traced_type)
    if isinstance(value, list):
        return [_unwrap_array_function_value(item, traced_type) for item in value]
    if isinstance(value, tuple):
        items = [_unwrap_array_function_value(item, traced_type) for item in value]
        if type(value) is tuple:
            return tuple(items)
        return cast("Callable[..., object]", type(value))(*items)
    if isinstance(value, dict):
        return {key: _unwrap_array_function_value(item, traced_type) for key, item in value.items()}
    return value


class _ArrayFunctionProtocolMixin:
    """Array-function half of the shared traced-array protocol runtime."""

    __slots__ = ()

    @staticmethod
    def _rebuild_result_container(template: object, items: list[object]) -> object:
        if isinstance(template, list):
            return items
        if type(template) is tuple:
            return tuple(items)
        return type(template)(*items)

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
        return cls._rebuild_result_container(result_value, children)

    def _normalize_array_function_args_and_kwargs(
        self,
        *,
        func: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        normalized_args = list(args)
        normalized_kwargs = dict(kwargs)

        func_name = getattr(func, "__name__", "")
        aliases = _ARRAY_FUNCTION_POS_ARG_ALIASES.get(func_name)
        if aliases is not None:
            for index, alias in enumerate(aliases):
                if index < len(normalized_args):
                    continue
                if alias in normalized_kwargs:
                    normalized_args.append(normalized_kwargs.pop(alias))

        if func_name == "einsum" and not normalized_args and "subscripts" in normalized_kwargs:
            subscripts = normalized_kwargs.pop("subscripts")
            operands: list[object] = []
            operand_index = 1
            while True:
                key = f"x{operand_index}"
                if key not in normalized_kwargs:
                    break
                operands.append(normalized_kwargs.pop(key))
                operand_index += 1
            normalized_args = [subscripts, *operands]

        try:
            positional_params = _positional_parameters(func)
        except (TypeError, ValueError):
            parameters = _UNINSPECTABLE_POSITIONAL_PARAMETERS.get(func_name)
            if parameters is None:
                return tuple(normalized_args), normalized_kwargs
            positional_params = tuple(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                )
                for name, default in parameters
            )

        out_index = next(
            (index for index, parameter in enumerate(positional_params) if parameter.name == "out"),
            None,
        )
        if out_index is not None and len(normalized_args) > out_index:
            for parameter, value in zip(
                positional_params[out_index:],
                normalized_args[out_index:],
                strict=False,
            ):
                if parameter.name in normalized_kwargs:
                    msg = f"{getattr(func, '__name__', func)!s} received {parameter.name} twice"
                    raise TypeError(msg)
                normalized_kwargs[parameter.name] = value
            del normalized_args[out_index:]

        return normalize_required_positionals(
            func,
            tuple(normalized_args),
            normalized_kwargs,
            positional=positional_params,
        )

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

    def _validate_numpy_array_function_out(
        self,
        *,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        out_arr: _TracedProtocolArray,
        traced_type: type[_TracedProtocolArray],
    ) -> object | None:
        """Ask NumPy itself to validate one functionalized ``out=`` call.

        Array functions do not share one casting policy: reductions permit
        casts rejected by FFTs and ufunc-backed helpers, while ``stack`` and
        ``einsum`` expose their own casting controls.  Reusing the upstream
        call against a private destination copy keeps those rules exact without
        embedding a second, inevitably drifting casting table in the tracer.
        The extra eager call is paid only by explicit mutation.
        """
        _out_node_id, out_value = _snapshot_traced(out_arr)
        copy_out = getattr(out_value, "copy", None)
        if not callable(copy_out):
            msg = "array-function out= requires a destination with copy() support"
            raise TracingError(msg)

        def private_validation_value(value: object) -> object:
            """Copy backend arrays before asking NumPy to validate mutation."""
            concrete = _unwrap_array_function_value(value, traced_type)
            if isinstance(concrete, np.ndarray):
                copy_value = getattr(concrete, "copy", None)
                if not callable(copy_value):
                    msg = "array-function out= validation requires copyable array operands"
                    raise TracingError(msg)
                return copy_value()
            if isinstance(concrete, list):
                return [private_validation_value(item) for item in concrete]
            if isinstance(concrete, tuple):
                items = [private_validation_value(item) for item in concrete]
                if type(concrete) is tuple:
                    return tuple(items)
                return cast("Callable[..., object]", type(concrete))(*items)
            if isinstance(concrete, dict):
                return {key: private_validation_value(item) for key, item in concrete.items()}
            return concrete

        concrete_kwargs = {key: private_validation_value(value) for key, value in kwargs.items()}
        private_out = copy_out()
        concrete_kwargs["out"] = private_out
        concrete_args = tuple(private_validation_value(value) for value in args)
        func(*concrete_args, **concrete_kwargs)
        return private_out

    def array_function(  # noqa: C901, PLR0912, PLR0915
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
            traced_type = cast("type[_TracedProtocolArray]", type(self_arr))

            def unwrap(value: object) -> object:
                if isinstance(value, traced_type):
                    return _snapshot_traced(value)[1]
                if isinstance(value, (list, tuple)):
                    return type(value)(unwrap(item) for item in value)
                return value

            unwrapped_args = [unwrap(arg) for arg in args]
            if func is np.result_type:
                return np.result_type(*unwrapped_args, **kwargs)
            if func is np.iscomplexobj and kwargs:
                msg = "numpy.iscomplexobj does not accept keyword arguments during tracing"
                raise TracingError(msg)
            if func is np.iscomplexobj and len(unwrapped_args) != 1:
                msg = "numpy.iscomplexobj expects exactly one argument during tracing"
                raise TracingError(msg)
            if func is np.iscomplexobj:
                return bool(np.iscomplexobj(unwrapped_args[0]))
            return cast("Callable[..., object]", func)(*unwrapped_args, **kwargs)

        if not callable(func) or not ARRAY_FUNCTION_RUNTIME.is_supported_array_function(func):
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
        resolved_out_obj = normalized_kwargs.get("out")
        out_arr = self._resolve_array_function_out_arg(func, traced_type, resolved_out_obj)

        traced_inputs: list[_TracedProtocolArray] = []
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
        if any(
            not _is_recorder_in_active_trace_stack(cast("Any", inp).recorder)
            for inp in traced_inputs
        ):
            msg = (
                "Cannot use a TracedArray from an unrelated or expired trace recorder "
                "in an array function call."
            )
            raise TraceLevelError(msg)

        validated_out: object | None = None
        clip_where: object = _MISSING
        clip_dtype: object | None = None
        if out_arr is not None:
            if cast("Any", out_arr).recorder is not recorder:
                msg = "array-function out= must belong to the current trace recorder"
                raise TraceLevelError(msg)
            out_arr.advect_require_mutable("array-function out=")
            normalized_kwargs.pop("out", None)
            validated_out = self._validate_numpy_array_function_out(
                func=cast("Callable[..., object]", func),
                args=normalized_args,
                kwargs=normalized_kwargs,
                out_arr=out_arr,
                traced_type=traced_type,
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

        try:
            with _use_operation_recorder(recorder):
                result_value, node_id = ARRAY_FUNCTION_RUNTIME.handle_array_function(
                    func=cast("Callable[..., object]", func),
                    recorder=cast("Any", recorder),
                    traced_type=cast("Any", traced_type),
                    args=normalized_args,
                    kwargs=dict(normalized_kwargs),
                )
        except Exception as exc:
            if type(exc).__name__ == "ArrayFunctionNotSupportedError":
                raise TracingError(str(exc)) from exc
            raise

        if out_arr is not None:
            if isinstance(node_id, tuple) or isinstance(result_value, tuple):
                msg = "out= is not supported for multi-output array functions"
                raise TracingError(msg)
            traced_ctor = cast("Callable[..., _TracedProtocolArray]", traced_type)
            replacement = traced_ctor(
                value=result_value,
                node_id=cast("int", node_id),
                recorder=recorder,
            )
            if clip_dtype is not None:
                replacement = cast(
                    "_TracedProtocolArray",
                    cast("Any", replacement).astype(clip_dtype, copy=False),
                )
            if clip_where is not _MISSING:
                replacement = cast(
                    "_TracedProtocolArray",
                    np.where(
                        clip_where,
                        replacement,
                        out_arr,
                    ),
                )
            _out_node_id, old_out = _snapshot_traced(out_arr)
            target_dtype = getattr(old_out, "dtype", None)
            target_shape = tuple(getattr(old_out, "shape", ()))
            if tuple(getattr(replacement, "shape", ())) != target_shape:
                msg = (
                    f"array-function result shape {getattr(replacement, 'shape', None)!r} "
                    f"does not match out= shape {target_shape!r}"
                )
                raise TracingError(msg)
            if getattr(replacement, "dtype", None) != target_dtype:
                replacement = cast(
                    "_TracedProtocolArray",
                    cast("Any", replacement).astype(target_dtype, copy=False),
                )
            committed_node_id, committed_value = _snapshot_traced(replacement)
            out_arr.advect_replace(
                value=committed_value if validated_out is None else validated_out,
                node_id=int(committed_node_id),
                operation="array-function out=",
            )
            return out_arr

        return self._wrap_array_function_result(
            result_value=result_value,
            node_ids=node_id,
            traced_type=traced_type,
            recorder=recorder,
        )
