# ruff: noqa: EM101, EM102, PLR2004, TRY003
"""NumPy protocol lowering for payload-free staged arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, override

from advect.core._abstract import AbstractArray, _lift, _record_abstract_op
from advect.core._errors import TracingError
from advect.numpy._abstract_calls import _empty_out, _numpy_array, apply_numpy
from advect.numpy._constructors import construct_abstract
from advect.numpy._protocol_array_function import ARRAY_FUNCTION_RUNTIME
from advect.numpy._signature import normalize_required_positionals
from advect.numpy._stage_lifecycle import stage_context

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from contextlib import AbstractContextManager

    from advect.core._native import DynamicTape
    from advect.core._protocols import ArrayLike


class _NamedProtocol(Protocol):
    __name__: str


def abstract_array_ufunc(
    self: AbstractArray,
    ufunc: _NamedProtocol,
    method: str,
    *inputs: object,
    **kwargs: object,
) -> AbstractArray:
    """Lower one NumPy ufunc call into the staged canonical graph."""
    name = str(ufunc.__name__)
    if method in {"reduce", "accumulate"}:
        if len(inputs) != 1:
            raise TracingError(f"numpy.{name}.{method} expects one input array")
        lowered = {
            ("add", "accumulate"): "cumsum",
            ("add", "reduce"): "sum",
            ("multiply", "accumulate"): "cumprod",
            ("multiply", "reduce"): "prod",
        }.get((name, method))
        if lowered is None:
            raise TracingError(f"numpy.{name}.{method} is not supported during staging")
        kwargs.setdefault("axis", 0)
        if isinstance(kwargs.get("out"), tuple):
            kwargs["_advect_ufunc_out_tuple"] = True
        return _numpy_array(self._trace, lowered, inputs, kwargs)

    if method == "outer":
        if (
            len(inputs) != 2
            or int(getattr(ufunc, "nin", 0)) != 2
            or int(getattr(ufunc, "nout", 0)) != 1
            or getattr(ufunc, "signature", None) is not None
        ):
            raise TracingError(
                f"numpy.{name}.outer requires an ordinary binary, single-output "
                "ufunc; generalized ufunc signatures are unsupported"
            )
        trace = self._trace
        left = _lift(trace, inputs[0])
        right = _lift(trace, inputs[1])
        expanded_left = _numpy_array(
            trace,
            "expand_dims",
            (left,),
            {"axis": tuple(range(left.ndim, left.ndim + right.ndim))},
        )
        expanded_right = _numpy_array(
            trace,
            "expand_dims",
            (right,),
            {"axis": tuple(range(left.ndim))},
        )
        if not _empty_out(kwargs.get("out")):
            kwargs["_advect_ufunc_call"] = True
        return _numpy_array(trace, name, (expanded_left, expanded_right), kwargs)

    if method != "__call__":
        raise TracingError(f"numpy.{name}.{method} is not supported during staging")
    if not _empty_out(kwargs.get("out")):
        kwargs["_advect_ufunc_call"] = True
    return _numpy_array(self._trace, name, inputs, kwargs)


def abstract_array_function(
    self: AbstractArray,
    func: _NamedProtocol,
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 - NumPy's protocol returns heterogeneous pytrees
    """Lower one NumPy array-function call into the staged canonical graph."""
    del types
    module = str(getattr(func, "__module__", "numpy"))
    name = str(func.__name__)
    if module == "numpy.lib.scimath":
        raise TracingError(
            f"numpy.lib.scimath.{name} is dynamic-only because its output dtype "
            "can depend on runtime values"
        )
    if module.startswith("numpy.linalg"):
        name = f"linalg.{name}"
    elif module.startswith("numpy.fft"):
        name = f"fft.{name}"
    args, kwargs = normalize_required_positionals(func, args, kwargs)
    if module.startswith("numpy") and name in {"array", "asarray", "asanyarray"}:
        return construct_abstract(name, self, args, kwargs)
    if not ARRAY_FUNCTION_RUNTIME.is_supported_array_function(cast("Callable[..., Any]", func)):
        raise TracingError(f"Array function 'numpy.{name}' is not supported during staging")
    if name == "copy":
        if not args or len(args) > 3 or set(kwargs) - {"order", "subok"}:
            raise TracingError("numpy.copy expects (a, order='K', subok=False) during staging")
        values = dict(kwargs)
        for parameter, value in zip(("order", "subok"), args[1:], strict=False):
            if parameter in values:
                raise TracingError(f"numpy.copy received {parameter} twice")
            values[parameter] = value
        if bool(values.get("subok", False)):
            raise TracingError(
                "numpy.copy(subok=True) is not supported during staging because "
                "durable programs do not preserve ndarray subclass identity"
            )
        source = cast("_NumpyAbstractArray", _lift(self._trace, args[0]))
        return source.copy(order=str(values.get("order", "K")))
    return apply_numpy(self._trace, name, args, kwargs)


class _NumpyAbstractArray(AbstractArray):
    """Payload-free staged value that owns NumPy's foreign protocols."""

    __slots__ = ()

    @staticmethod
    @override
    def _advect_stage_context(
        _captures: Sequence[tuple[str, object]],
    ) -> AbstractContextManager[None]:
        return stage_context(_captures)

    @override
    def astype(self, dtype: object, **kwargs: object) -> AbstractArray:
        if any(name in kwargs for name in ("casting", "order", "subok")):
            return cast(
                "AbstractArray",
                apply_numpy(self._trace, "astype", (self, dtype), kwargs),
            )
        return super().astype(dtype, **kwargs)

    @override
    def copy(self, order: str | None = None) -> AbstractArray:
        if order is None:
            return super().copy()
        if not isinstance(order, str):
            raise TypeError(f"order must be str, not {type(order).__name__}")
        order = order.upper()
        if order not in {"A", "C", "F", "K"}:
            raise ValueError(f"order must be one of 'A', 'C', 'F', or 'K' (got {order!r})")
        return cast(
            "AbstractArray",
            _record_abstract_op(
                self._trace,
                "advect.copy",
                (self,),
                {"order": order},
                graph_attrs={"_advect_backend": "numpy"},
            ),
        )

    @override
    def sum(self, *args: object, **kwargs: object) -> AbstractArray:
        if any(name in kwargs for name in ("initial", "out", "where")):
            return cast(
                "AbstractArray",
                apply_numpy(self._trace, "sum", (self, *args), kwargs),
            )
        return super().sum(*args, **kwargs)

    @override
    def mean(self, *args: object, **kwargs: object) -> AbstractArray:
        if any(name in kwargs for name in ("out", "where")):
            return cast(
                "AbstractArray",
                apply_numpy(self._trace, "mean", (self, *args), kwargs),
            )
        return super().mean(*args, **kwargs)

    def __array_ufunc__(
        self,
        ufunc: _NamedProtocol,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> AbstractArray:
        return abstract_array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def __array_function__(
        self,
        func: _NamedProtocol,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401 - NumPy's protocol returns heterogeneous pytrees
        return abstract_array_function(self, func, types, args, kwargs)


def as_numpy_nested(value: object) -> Any:  # noqa: ANN401 - optional protocol conversion
    """Wrap an Array API tracer whose nested payload is an abstract staged value."""
    snapshot = getattr(value, "_advect_snapshot", None)
    if not callable(snapshot):
        return NotImplemented
    node_id, wrapped = cast("tuple[int, object]", cast("Any", snapshot)())
    if not bool(getattr(type(wrapped), "__advect_abstract_array__", False)):
        return NotImplemented
    recorder = getattr(value, "recorder", None)
    if recorder is None:
        return NotImplemented
    from advect.numpy._traced_array import TracedArray  # noqa: PLC0415 - avoid init cycle

    return TracedArray(
        cast("ArrayLike", wrapped),
        node_id,
        cast("DynamicTape", recorder),
    )


def nested_array_ufunc(
    _tracer: object,
    ufunc: _NamedProtocol,
    method: str,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 - NumPy's protocol returns heterogeneous pytrees
    """Let NumPy bind a call encountered inside an Array API nested trace."""
    changed = False

    def convert(value: Any) -> Any:  # noqa: ANN401
        nonlocal changed
        nested = as_numpy_nested(value)
        if nested is not NotImplemented:
            changed = True
            return nested
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    converted_inputs = tuple(convert(value) for value in inputs)
    converted_kwargs = {key: convert(value) for key, value in kwargs.items()}
    if not changed:
        return NotImplemented
    call = cast("Callable[..., object]", getattr(ufunc, method))
    return call(*converted_inputs, **converted_kwargs)


def nested_array_function(
    _tracer: object,
    function: _NamedProtocol,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 - NumPy's protocol returns heterogeneous pytrees
    """Let NumPy bind an array function inside an Array API nested trace."""
    changed = False

    def convert(value: Any) -> Any:  # noqa: ANN401
        nonlocal changed
        nested = as_numpy_nested(value)
        if nested is not NotImplemented:
            changed = True
            return nested
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    converted_args = cast("tuple[Any, ...]", convert(args))
    converted_kwargs = cast("dict[str, Any]", convert(kwargs))
    if not changed:
        return NotImplemented
    return cast("Callable[..., object]", function)(*converted_args, **converted_kwargs)


__all__ = [
    "abstract_array_function",
    "abstract_array_ufunc",
    "as_numpy_nested",
    "nested_array_function",
    "nested_array_ufunc",
]
