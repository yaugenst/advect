# ruff: noqa: A002, ANN401, EM101, TRY003
# SciPy-compatible names/signatures and primitive rule schemas intentionally trigger these rules.
"""Normalize ndimage arguments and bridge public calls to private primitives.

This module owns provider checks, shape and dtype inspection, ``output=``
handling, and static configuration normalization shared by both derivative
mechanisms.  It owns no filter algorithm or primitive registration.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core import ArraySpec
from advect.scipy._frontend import (
    _array_operand,
    _is_traced_value,
    _replace_out as _replace_traced_out,
    _require_numpy_values as _require_scipy_numpy_values,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from advect.core import AbstractValue
    from advect.core._primitive import Primitive

_MODE_ALIASES = {
    "grid-constant": "constant",
    "grid-mirror": "reflect",
    "grid-wrap": "wrap",
}


def _require_numpy_values(name: str, *values: object) -> None:
    _require_scipy_numpy_values("ndimage", name, *values)


def _numpy_dtype(dtype: object) -> np.dtype[Any]:
    try:
        return np.dtype(cast("DTypeLike", dtype))
    except (TypeError, ValueError) as error:
        msg = f"advect.scipy.ndimage requires a NumPy dtype; got {dtype!r}"
        raise TypeError(msg) from error


def _operand_dtype(value: Any) -> np.dtype[Any]:
    dtype = getattr(value, "dtype", None)
    return np.asarray(value).dtype if dtype is None else _numpy_dtype(dtype)


def _traceable_astype(value: Any, dtype: object) -> Any:
    normalized = _numpy_dtype(dtype)
    if _operand_dtype(value) == normalized:
        return value
    astype = getattr(value, "astype", None)
    if callable(astype):
        return astype(normalized)
    return np.asarray(value, dtype=normalized)


def _replace_out(destination: object, replacement: object, *, operation: str) -> object:
    return _replace_traced_out(
        destination,
        replacement,
        argument="output",
        operation=operation,
    )


def _normalize_axes(axes: object, ndim: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    if np.isscalar(axes):
        raw = (operator.index(cast("Any", axes)),)
    elif isinstance(axes, Iterable):
        try:
            raw = tuple(operator.index(cast("Any", axis)) for axis in axes)
        except TypeError as error:
            msg = "axes must be an integer, iterable of integers, or None"
            raise ValueError(msg) from error
    else:
        msg = "axes must be an integer, iterable of integers, or None"
        raise ValueError(msg)
    normalized: list[int] = []
    for axis in raw:
        if axis < -ndim or axis >= ndim:
            msg = f"specified axis: {axis} is out of range"
            raise ValueError(msg)
        normalized.append(axis % ndim if axis < 0 else axis)
    if len(set(normalized)) != len(normalized):
        msg = "axes must be unique"
        raise ValueError(msg)
    return tuple(normalized)


def _normalize_axis(axis: object, ndim: int) -> int:
    normalized = operator.index(cast("Any", axis))
    if normalized < -ndim or normalized >= ndim:
        raise np.exceptions.AxisError(normalized, ndim=ndim)
    return normalized % ndim


def _static_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if _is_traced_value(value):
        msg = "ndimage configuration arguments must be concrete while tracing"
        raise TypeError(msg)
    return value


def _normalize_sequence(value: object, rank: int) -> tuple[object, ...]:
    if not isinstance(value, str) and isinstance(value, Iterable):
        normalized = tuple(_static_scalar(item) for item in value)
        if len(normalized) != rank:
            msg = "sequence argument must have length equal to input rank"
            raise RuntimeError(msg)
        return normalized
    scalar = _static_scalar(value)
    return (scalar,) * rank


def _normalize_modes(mode: object, rank: int) -> tuple[str, ...]:
    values = _normalize_sequence(mode, rank)
    return tuple(str(value) for value in values)


def _normalize_origins(origin: object, rank: int) -> tuple[int, ...]:
    return tuple(operator.index(cast("Any", item)) for item in _normalize_sequence(origin, rank))


def _shape_of(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return np.asarray(value).shape
    return tuple(int(size) for size in shape)


def _ndim_of(value: object) -> int:
    return len(_shape_of(value))


@dataclass(frozen=True, slots=True)
class _OutputChoice:
    destination: object | None
    dtype: str | None


def _normalize_output(input: object, output: object) -> _OutputChoice:
    if output is None:
        return _OutputChoice(None, None)
    if _is_traced_value(output) or isinstance(output, np.ndarray):
        _require_numpy_values("output", output)
        if _shape_of(output) != _shape_of(input):
            raise RuntimeError("output shape not correct")
        return _OutputChoice(output, _operand_dtype(output).str)
    return _OutputChoice(None, _numpy_dtype(output).str)


def _output_dtype(input: object, output: object) -> np.dtype[Any]:
    choice = _normalize_output(input, output)
    return _operand_dtype(input) if choice.dtype is None else _numpy_dtype(choice.dtype)


def _runtime_output(output_dtype: str | None) -> object:
    return None if output_dtype is None else _numpy_dtype(output_dtype)


def _sample_array(value: AbstractValue, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    sample_shape = value.spec.shape if shape is None else shape
    return np.ones(sample_shape, dtype=_numpy_dtype(value.spec.dtype))


def _result_spec(
    input: AbstractValue,
    result: object,
) -> ArraySpec:
    return ArraySpec(
        input.spec.shape,
        np.asarray(result).dtype.name,
        device=input.spec.device,
    )


def _finish_output(
    result: object,
    *,
    input: object,
    output: object,
    operation: str,
) -> object:
    choice = _normalize_output(input, output)
    dtype = _operand_dtype(input) if choice.dtype is None else _numpy_dtype(choice.dtype)
    replacement = _traceable_astype(result, dtype)
    if choice.destination is None:
        return replacement
    return _replace_out(choice.destination, replacement, operation=operation)


def _call_primitive(
    primitive_function: Primitive[..., Any],
    *,
    name: str,
    input: object,
    output: object,
    operands: dict[str, object],
    static: dict[str, object],
) -> object:
    choice = _normalize_output(input, output)
    replacement = primitive_function(
        input=_array_operand(input),
        output_dtype=choice.dtype,
        **{key: _array_operand(value) for key, value in operands.items()},
        **static,
    )
    if choice.destination is None:
        return replacement
    return _replace_out(
        choice.destination,
        replacement,
        operation=f"scipy.ndimage.{name} output=",
    )


def _mode_name(mode: str) -> str:
    return _MODE_ALIASES.get(mode, mode)


def _project_cotangent(value: Any, primal: Any, output: Any) -> Any:
    primal_dtype = _operand_dtype(primal)
    output_dtype = _operand_dtype(output)
    if not np.issubdtype(primal_dtype, np.inexact) or not np.issubdtype(
        output_dtype,
        np.inexact,
    ):
        return np.zeros_like(primal)
    if not np.issubdtype(primal_dtype, np.complexfloating) and np.issubdtype(
        _operand_dtype(value),
        np.complexfloating,
    ):
        value = np.real(value)
    return _traceable_astype(value, primal_dtype)


def _zero_tangent(primal: Any, tangent: Any | None) -> Any:
    return np.zeros_like(primal) if tangent is None else tangent


def _cast_tangent(tangent: Any, output: Any) -> Any:
    dtype = _operand_dtype(output)
    if not np.issubdtype(dtype, np.inexact):
        return np.zeros_like(output)
    return _traceable_astype(tangent, dtype)


def _validate_ufunc_output_cast(
    ufunc: np.ufunc,
    left: object,
    right: object,
    output_dtype: object,
) -> None:
    """Apply NumPy's own in-place casting check used by SciPy composites."""
    ufunc(
        np.zeros((), dtype=_operand_dtype(left)),
        np.zeros((), dtype=_operand_dtype(right)),
        out=np.zeros((), dtype=_numpy_dtype(output_dtype)),
    )
