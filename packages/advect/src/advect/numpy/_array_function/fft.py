"""FFT-related ``__array_function__`` handlers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._errors import TracingError
from advect.numpy._array_function.emission import (
    _add_backend_node,
    _get_node,
    _get_value,
    _result_shape_and_dtype,
)
from advect.numpy._op_bindings import canonicalize_numpy_op

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionHandler

np: Any = _numpy


def _op_name(suffix: str) -> str:
    return f"numpy.fft.{suffix}"


def _normalize_shape_tuple(value: object | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return (int(arr.item()),)
    return tuple(int(item) for item in arr.tolist())


def _normalize_axes_tuple(value: object | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return (int(arr.item()),)
    return tuple(int(item) for item in arr.tolist())


def _fft_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    np_func: Callable[..., Any],
    positional_names: tuple[str, ...],
) -> tuple[object, int]:
    op_name = _op_name(np_func.__name__)
    if not args or len(args) > len(positional_names) + 1:
        msg = (
            f"{op_name} expects one array and at most {len(positional_names)} "
            "positional metadata arguments during tracing"
        )
        raise TracingError(msg)
    unsupported = set(kwargs).difference(positional_names)
    if unsupported:
        msg = f"{op_name} kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    call_kwargs = dict(kwargs)
    for name, value in zip(positional_names, args[1:], strict=False):
        if name in call_kwargs:
            msg = f"{op_name} received {name} twice"
            raise TracingError(msg)
        call_kwargs[name] = value

    x = args[0]
    result = np_func(_get_value(x, traced_type), **call_kwargs)
    result_shape, result_dtype = _result_shape_and_dtype(result)

    attrs: dict[str, Any] = {}
    if "n" in call_kwargs and call_kwargs["n"] is not None:
        attrs["n"] = int(call_kwargs["n"])
    if "s" in call_kwargs:
        s_norm = _normalize_shape_tuple(call_kwargs["s"])
        if s_norm is not None:
            attrs["s"] = s_norm
    if "axis" in call_kwargs:
        attrs["axis"] = int(call_kwargs["axis"])
    if "axes" in call_kwargs:
        axes_norm = _normalize_axes_tuple(call_kwargs["axes"])
        if axes_norm is not None:
            attrs["axes"] = axes_norm
    if "norm" in call_kwargs and call_kwargs["norm"] is not None:
        attrs["norm"] = str(call_kwargs["norm"])

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(op_name),
        inputs=(_get_node(x, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=result_shape,
        dtype=result_dtype,
    )
    return result, node_id


_FFT_HANDLER_GROUPS = (
    (("fft", "ifft", "rfft", "irfft", "hfft", "ihfft"), ("n", "axis", "norm")),
    (
        ("fft2", "ifft2", "fftn", "ifftn", "rfft2", "rfftn", "irfft2"),
        ("s", "axes", "norm"),
    ),
    (("fftshift", "ifftshift"), ("axes",)),
)


def _irfftn(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    normalized_kwargs = dict(kwargs)
    if "s" in normalized_kwargs and normalized_kwargs.get("s") is not None:
        s_norm = _normalize_shape_tuple(normalized_kwargs["s"])
        if s_norm is not None and normalized_kwargs.get("axes") is None:
            normalized_kwargs["axes"] = tuple(range(len(s_norm)))
    return _fft_handler(
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=normalized_kwargs,
        np_func=np.fft.irfftn,
        positional_names=("s", "axes", "norm"),
    )


def register_fft_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register FFT array-function handlers."""
    for names, positional_names in _FFT_HANDLER_GROUPS:
        for name in names:
            np_func = getattr(np.fft, name)
            handlers[np_func] = partial(
                _fft_handler,
                np_func=np_func,
                positional_names=positional_names,
            )
    handlers[np.fft.irfftn] = _irfftn
