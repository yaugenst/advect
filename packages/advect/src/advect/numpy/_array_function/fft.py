"""FFT-related ``__array_function__`` handlers."""

from __future__ import annotations

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

np: Any = _numpy


def _op_name(suffix: str) -> str:
    return f"numpy.fft.{suffix}"


def _with_backend_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out = dict(attrs)
    out["_advect_backend"] = "numpy"
    return out


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
    *,
    op_name: str,
    np_func: Callable[..., Any],
    allowed_kwargs: set[str],
    positional_names: tuple[str, ...],
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    if not args or len(args) > len(positional_names) + 1:
        msg = (
            f"{op_name} expects one array and at most {len(positional_names)} "
            "positional metadata arguments during tracing"
        )
        raise TracingError(msg)
    unsupported = set(kwargs) - allowed_kwargs
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
        attrs=_with_backend_attrs(attrs),
        shape=result_shape,
        dtype=result_dtype,
    )
    return result, node_id


def _fft(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("fft"),
        np_func=np.fft.fft,
        allowed_kwargs={"n", "axis", "norm"},
        positional_names=("n", "axis", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _ifft(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("ifft"),
        np_func=np.fft.ifft,
        allowed_kwargs={"n", "axis", "norm"},
        positional_names=("n", "axis", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _fft2(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("fft2"),
        np_func=np.fft.fft2,
        allowed_kwargs={"s", "axes", "norm"},
        positional_names=("s", "axes", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _ifft2(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("ifft2"),
        np_func=np.fft.ifft2,
        allowed_kwargs={"s", "axes", "norm"},
        positional_names=("s", "axes", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _fftn(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("fftn"),
        np_func=np.fft.fftn,
        allowed_kwargs={"s", "axes", "norm"},
        positional_names=("s", "axes", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _ifftn(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("ifftn"),
        np_func=np.fft.ifftn,
        allowed_kwargs={"s", "axes", "norm"},
        positional_names=("s", "axes", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _rfft(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("rfft"),
        np_func=np.fft.rfft,
        allowed_kwargs={"n", "axis", "norm"},
        positional_names=("n", "axis", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _rfft2(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("rfft2"),
        np_func=np.fft.rfft2,
        allowed_kwargs={"s", "axes", "norm"},
        positional_names=("s", "axes", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _rfftn(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("rfftn"),
        np_func=np.fft.rfftn,
        allowed_kwargs={"s", "axes", "norm"},
        positional_names=("s", "axes", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _irfft(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("irfft"),
        np_func=np.fft.irfft,
        allowed_kwargs={"n", "axis", "norm"},
        positional_names=("n", "axis", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _irfft2(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("irfft2"),
        np_func=np.fft.irfft2,
        allowed_kwargs={"s", "axes", "norm"},
        positional_names=("s", "axes", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
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
        op_name=_op_name("irfftn"),
        np_func=np.fft.irfftn,
        allowed_kwargs={"s", "axes", "norm"},
        positional_names=("s", "axes", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=normalized_kwargs,
    )


def _hfft(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("hfft"),
        np_func=np.fft.hfft,
        allowed_kwargs={"n", "axis", "norm"},
        positional_names=("n", "axis", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _ihfft(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("ihfft"),
        np_func=np.fft.ihfft,
        allowed_kwargs={"n", "axis", "norm"},
        positional_names=("n", "axis", "norm"),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _fftshift(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("fftshift"),
        np_func=np.fft.fftshift,
        allowed_kwargs={"axes"},
        positional_names=("axes",),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _ifftshift(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[object, int]:
    return _fft_handler(
        op_name=_op_name("ifftshift"),
        np_func=np.fft.ifftshift,
        allowed_kwargs={"axes"},
        positional_names=("axes",),
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def register_fft_handlers(handlers: dict[Callable[..., Any], Callable[..., Any]]) -> None:
    """Register FFT array-function handlers."""
    handlers[np.fft.fft] = _fft
    handlers[np.fft.ifft] = _ifft
    handlers[np.fft.fft2] = _fft2
    handlers[np.fft.ifft2] = _ifft2
    handlers[np.fft.fftn] = _fftn
    handlers[np.fft.ifftn] = _ifftn
    handlers[np.fft.rfft] = _rfft
    handlers[np.fft.rfft2] = _rfft2
    handlers[np.fft.rfftn] = _rfftn
    handlers[np.fft.irfft] = _irfft
    handlers[np.fft.irfft2] = _irfft2
    handlers[np.fft.irfftn] = _irfftn
    handlers[np.fft.hfft] = _hfft
    handlers[np.fft.ihfft] = _ihfft
    handlers[np.fft.fftshift] = _fftshift
    handlers[np.fft.ifftshift] = _ifftshift
