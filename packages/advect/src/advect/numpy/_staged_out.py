"""Authoritative NumPy validation for payload-free staged ``out=`` calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_EXACT_C_OUTPUT_FUNCTIONS = frozenset({"dot", "linalg.multi_dot"})


def _materialize(value: object) -> object:
    if isinstance(value, np.ndarray):
        # Captured masks and indices are staged constants whose values can
        # determine the output shape. Only payload-free abstract arrays need
        # representative data.
        return value
    if isinstance(value, tuple):
        return tuple(_materialize(item) for item in value)
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _materialize(item) for key, item in value.items()}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None or isinstance(value, type):
        return value
    try:
        normalized_shape = tuple(int(size) for size in shape)
        normalized_dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        return value
    constructor = np.ones if normalized_dtype == np.dtype(bool) else np.zeros
    order = getattr(value, "_advect_layout", None)
    return constructor(
        normalized_shape,
        dtype=normalized_dtype,
        order="F" if order == "F" else "C",
    )


def _resolve(path: str) -> Callable[..., object]:
    function: object = np
    for component in path.split("."):
        function = getattr(function, component)
    if not callable(function):
        msg = f"NumPy member {path!r} is not callable"
        raise TypeError(msg)
    return function


def validate_staged_out(
    raw_name: str,
    raw_args: tuple[object, ...],
    raw_kwargs: dict[str, object],
    destination: object,
    *,
    tuple_out: bool,
) -> None:
    """Ask NumPy to validate shape, dtype, casting, and layout on dummy arrays.

    Array-function output policies are intentionally irregular. Validation is
    based only on abstract shapes, dtypes, and static attributes, so no runtime
    values are frozen into the graph.
    """
    if (
        raw_name in _EXACT_C_OUTPUT_FUNCTIONS
        and getattr(
            destination,
            "_advect_layout",
            None,
        )
        != "C"
    ):
        msg = (
            f"numpy.{raw_name} out= requires a destination with a guaranteed C layout; "
            "construct it with order='C'"
        )
        raise ValueError(msg)
    output = _materialize(destination)
    out = (output,) if tuple_out else output
    kwargs = {
        name: _materialize(value)
        for name, value in raw_kwargs.items()
        if not name.startswith("_advect_")
    }
    kwargs["out"] = out
    _resolve(raw_name)(
        *tuple(_materialize(value) for value in raw_args),
        **kwargs,
    )


__all__ = ["validate_staged_out"]
