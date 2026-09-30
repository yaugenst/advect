"""FFT-related ``__array_function__`` handlers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.numpy._array_function.emission import _emit, _get_value
from advect.numpy._array_function.normalization import _normalize_shape

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionHandler

np: Any = _numpy

# Each control NumPy received becomes one portable graph attr.
_ATTR_NORMALIZERS: dict[str, Callable[[Any], object]] = {
    "n": int,
    "s": _normalize_shape,
    "axis": int,
    "axes": _normalize_shape,
    "norm": str,
}
_FFT_HANDLER_GROUPS = (
    (("fft", "ifft", "rfft", "irfft", "hfft", "ihfft"), ("n", "axis", "norm")),
    (
        ("fft2", "ifft2", "fftn", "ifftn", "rfft2", "rfftn", "irfft2", "irfftn"),
        ("s", "axes", "norm"),
    ),
    (("fftshift", "ifftshift"), ("axes",)),
)


def _fft_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    np_func: Callable[..., Any],
    positional_names: tuple[str, ...],
) -> tuple[object, int]:
    controls = dict(zip(positional_names, args[1:], strict=False)) | kwargs
    if (
        np_func.__name__.endswith("fftn")
        and controls.get("s") is not None
        and controls.get("axes") is None
    ):
        # The n-D transforms apply s to the trailing axes; NumPy 2 deprecates leaving
        # that implicit, so spell it out for the primal and derivative calls alike.
        controls["axes"] = tuple(range(-len(_normalize_shape(controls["s"])), 0))
    x = args[0]
    result = np_func(_get_value(x, traced_type), **controls)
    attrs = {
        name: _ATTR_NORMALIZERS[name](value)
        for name, value in controls.items()
        if value is not None
    }
    return _emit(graph, traced_type, f"numpy.fft.{np_func.__name__}", (x,), result, attrs)


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
