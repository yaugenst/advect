"""Concrete NumPy ``__array_function__`` dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.numpy._array_function.emission import (
    _make_arg_reduction_handler,
    _make_atleast_handler,
    _make_axis_keepdims_reduction_handler,
    _make_binary_handler,
    _make_clip_handler,
    _make_cumulative_handler,
    _make_interp_handler,
    _make_like_handler,
    _make_multi_input_handler,
    _make_reduction_handler,
    _make_variance_handler,
    _make_where_handler,
)
from advect.numpy._array_function.families import register_family_handlers
from advect.numpy._array_function.fft import register_fft_handlers
from advect.numpy._array_function.linalg import register_linalg_handlers
from advect.numpy._array_function.shape import register_shape_handlers

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import (
        ArrayFunctionHandler,
        ArrayFunctionResult,
    )


class ArrayFunctionNotSupportedError(Exception):
    """Raised when an array function call cannot be handled by the tracer."""


@dataclass(frozen=True, slots=True)
class ArrayFunctionRuntime:
    """Resolved NumPy array-function handlers."""

    handlers: dict[Callable[..., Any], ArrayFunctionHandler]

    def get_array_function_name(self, func: Callable[..., Any]) -> str:
        module = getattr(func, "__module__", "numpy")
        name = getattr(func, "__name__", str(func))
        if module.startswith("numpy"):
            return f"{module}.{name}"
        return f"numpy.{name}"

    def is_supported_array_function(self, func: object) -> bool:
        return func in self.handlers or func in _STATIC_ARRAY_FUNCTIONS

    def handle_array_function(
        self,
        func: Callable[..., Any],
        recorder: DynamicTape,
        traced_type: type[TracedArrayLike],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ArrayFunctionResult:
        if func not in self.handlers:
            func_name = self.get_array_function_name(func)
            msg = f"Array function '{func_name}' is not supported"
            raise ArrayFunctionNotSupportedError(msg)
        return self.handlers[func](recorder, traced_type, args, kwargs)


def _require_function(path: str) -> Callable[..., Any]:
    func: object | None = np
    for part in path.split("."):
        func = getattr(func, part, None) if func is not None else None
    if func is None or not callable(func):
        msg = f"NumPy does not expose callable {path!r}"
        raise RuntimeError(msg)
    return cast("Callable[..., Any]", func)


def _op_name(suffix: str) -> str:
    return f"numpy.{suffix}"


def _register_all_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    reduction_ops: list[str] = [
        "sum",
        "mean",
        "prod",
        "cumsum",
        "cumprod",
        "nanmean",
        "nansum",
        "nanprod",
    ]
    for suffix in reduction_ops:
        func = _require_function(suffix)
        op_name = _op_name(suffix)
        if suffix in {"cumsum", "cumprod"}:
            handlers[func] = _make_cumulative_handler(func, op_name)
        else:
            handlers[func] = _make_reduction_handler(func, op_name)

    nan_minmax_ops: list[str] = [
        "max",
        "min",
        "amax",
        "amin",
        "nanmin",
        "nanmax",
    ]
    for suffix in nan_minmax_ops:
        func = _require_function(suffix)
        handlers[func] = _make_axis_keepdims_reduction_handler(func, _op_name(suffix))

    variance_ops: list[str] = [
        "var",
        "std",
        "nanvar",
        "nanstd",
    ]
    for suffix in variance_ops:
        func = _require_function(suffix)
        handlers[func] = _make_variance_handler(func, _op_name(suffix))

    register_shape_handlers(handlers)

    multi_input_ops: list[str] = ["concatenate", "stack"]
    for suffix in multi_input_ops:
        func = _require_function(suffix)
        handlers[func] = _make_multi_input_handler(func, _op_name(suffix))

    dot = _require_function("dot")
    handlers[dot] = _make_binary_handler(dot, _op_name("dot"))
    argmin = _require_function("argmin")
    handlers[argmin] = _make_arg_reduction_handler(argmin, _op_name("argmin"))
    argmax = _require_function("argmax")
    handlers[argmax] = _make_arg_reduction_handler(argmax, _op_name("argmax"))
    handlers[_require_function("where")] = _make_where_handler(_op_name("where"))
    handlers[_require_function("interp")] = _make_interp_handler(_op_name("interp"))
    handlers[_require_function("clip")] = _make_clip_handler(_op_name("clip"))

    atleast_1d = _require_function("atleast_1d")
    handlers[atleast_1d] = _make_atleast_handler(atleast_1d, "array.atleast_1d", target_ndim=1)
    atleast_2d = _require_function("atleast_2d")
    handlers[atleast_2d] = _make_atleast_handler(atleast_2d, "array.atleast_2d", target_ndim=2)
    atleast_3d = _require_function("atleast_3d")
    handlers[atleast_3d] = _make_atleast_handler(atleast_3d, "array.atleast_3d", target_ndim=3)
    register_family_handlers(handlers)

    register_fft_handlers(handlers)
    register_linalg_handlers(handlers)

    zeros_like = _require_function("zeros_like")
    handlers[zeros_like] = _make_like_handler(zeros_like, _op_name("zeros_like"))
    ones_like = _require_function("ones_like")
    handlers[ones_like] = _make_like_handler(ones_like, _op_name("ones_like"))
    empty_like = _require_function("empty_like")
    handlers[empty_like] = _make_like_handler(empty_like, _op_name("empty_like"))


_STATIC_ARRAY_FUNCTIONS = frozenset(
    {
        np.can_cast,
        np.common_type,
        np.isrealobj,
        np.ndim,
        np.shape,
        np.size,
    }
)
_HANDLERS: dict[Callable[..., Any], ArrayFunctionHandler] = {}
_register_all_handlers(_HANDLERS)
ARRAY_FUNCTION_RUNTIME = ArrayFunctionRuntime(handlers=_HANDLERS)
