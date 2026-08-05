# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""Differentiable functions from NumPy's classic polynomial API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_functions_extra_composite import (
    _finish,
    _first_traced,
    _lift_composite_constant,
)

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_functions_extra_composite import CompositeResult


_BINARY_ARITY = 2
_MATRIX_RANK = 2
_TERNARY_ARITY = 3


def _as_traced(
    value: object,
    *,
    anchor: TracedArrayLike,
    traced_type: type[TracedArrayLike],
) -> object:
    return value if isinstance(value, traced_type) else _lift_composite_constant(value, anchor)


def _trim_leading_coefficients(
    coefficients: Any,
    *,
    traced_type: type[TracedArrayLike],
) -> Any:
    concrete = (
        np.asarray(_snapshot_traced(coefficients)[1])
        if isinstance(coefficients, traced_type)
        else np.asarray(coefficients)
    )
    nonzero = np.flatnonzero(concrete)
    start = int(nonzero[0]) if nonzero.size else max(int(concrete.size) - 1, 0)
    return coefficients[start:]


def _poly_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = "numpy.poly expects one root vector or square matrix during tracing"
        raise TracingError(msg)
    roots = args[0]
    if roots.ndim == _MATRIX_RANK:
        if roots.shape[0] != roots.shape[1] or roots.shape[0] == 0:
            msg = "numpy.poly matrix input must be non-empty and square"
            raise TracingError(msg)
        roots = np.linalg.eigvals(roots)
    elif roots.ndim != 1:
        msg = "numpy.poly input must be one-dimensional or a square matrix"
        raise TracingError(msg)
    if roots.size == 0:
        return _finish(np.sum(roots) * 0 + 1.0, traced_type=traced_type)
    coefficients = _lift_composite_constant(
        np.ones((1,), dtype=roots.dtype),
        roots,
    )
    for index in range(int(roots.shape[0])):
        factor = np.stack((np.ones_like(roots[index]), -roots[index]))
        coefficients = np.convolve(coefficients, factor)
    return _finish(coefficients, traced_type=traced_type)


def _poly_add_sub_handler(
    *,
    subtract: bool,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        name = "polysub" if subtract else "polyadd"
        msg = f"numpy.{name} expects two coefficient vectors during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "polynomial arithmetic requires a traced operand"
        raise TracingError(msg)
    left = np.atleast_1d(_as_traced(args[0], anchor=anchor, traced_type=traced_type))
    right = np.atleast_1d(_as_traced(args[1], anchor=anchor, traced_type=traced_type))
    if left.ndim != 1 or right.ndim != 1:
        msg = "polynomial coefficients must be one-dimensional"
        raise TracingError(msg)
    if left.size < right.size:
        left = np.pad(left, (int(right.size - left.size), 0))
    elif right.size < left.size:
        right = np.pad(right, (int(left.size - right.size), 0))
    return _finish(left - right if subtract else left + right, traced_type=traced_type)


def _polyadd_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    return _poly_add_sub_handler(
        subtract=False,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _polysub_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    return _poly_add_sub_handler(
        subtract=True,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _polymul_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        msg = "numpy.polymul expects two coefficient vectors during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.polymul requires a traced operand"
        raise TracingError(msg)
    left = np.atleast_1d(_as_traced(args[0], anchor=anchor, traced_type=traced_type))
    right = np.atleast_1d(_as_traced(args[1], anchor=anchor, traced_type=traced_type))
    if left.ndim != 1 or right.ndim != 1:
        msg = "polynomial coefficients must be one-dimensional"
        raise TracingError(msg)
    left = _trim_leading_coefficients(left, traced_type=traced_type)
    right = _trim_leading_coefficients(right, traced_type=traced_type)
    return _finish(np.convolve(left, right), traced_type=traced_type)


def _polydiv_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        msg = "numpy.polydiv expects two coefficient vectors during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.polydiv requires a traced operand"
        raise TracingError(msg)
    dividend = np.atleast_1d(_as_traced(args[0], anchor=anchor, traced_type=traced_type)) + 0.0
    divisor = np.atleast_1d(_as_traced(args[1], anchor=anchor, traced_type=traced_type)) + 0.0
    if dividend.ndim != 1 or divisor.ndim != 1:
        msg = "numpy.polydiv coefficients must be one-dimensional"
        raise TracingError(msg)
    quotient_size = max(
        int(dividend.shape[0]) - int(divisor.shape[0]) + 1,
        1,
    )
    dtype = np.result_type(dividend.dtype, divisor.dtype)
    quotient = np.zeros(quotient_size, dtype=dtype) + (np.sum(dividend) + np.sum(divisor)) * 0
    remainder = np.astype(dividend, dtype)
    if int(dividend.shape[0]) >= int(divisor.shape[0]):
        scale = 1.0 / divisor[0]
        divisor_size = int(divisor.shape[0])
        for index in range(quotient_size):
            coefficient = scale * remainder[index]
            quotient[index] = coefficient
            remainder[index : index + divisor_size] -= coefficient * divisor

    _node_id, concrete_remainder = _snapshot_traced(remainder)
    concrete_array = np.asarray(concrete_remainder)
    trim = 0
    while trim < concrete_array.size - 1 and np.allclose(concrete_array[trim], 0, rtol=1e-14):
        trim += 1
    return _finish(
        (quotient, remainder[trim:]),
        traced_type=traced_type,
    )


def _polyfit_handler(  # noqa: C901, PLR0912, PLR0915 - mirrors NumPy's output contracts
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    positional_names = ("rcond", "full", "w", "cov")
    if len(args) < _TERNARY_ARITY or len(args) > _TERNARY_ARITY + len(positional_names):
        msg = "numpy.polyfit expects (x, y, deg, rcond=None, full=False, w=None, cov=False)"
        raise TracingError(msg)
    unsupported = set(kwargs) - set(positional_names)
    if unsupported:
        msg = f"numpy.polyfit kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for name, value in zip(positional_names, args[_TERNARY_ARITY:], strict=False):
        if name in values:
            msg = f"numpy.polyfit received {name} twice"
            raise TracingError(msg)
        values[name] = value
    degree_raw = args[2]
    if isinstance(degree_raw, traced_type):
        msg = "numpy.polyfit degree must be a static non-negative integer"
        raise TracingError(msg)
    degree = int(degree_raw)
    if degree < 0:
        msg = "numpy.polyfit degree must be non-negative"
        raise TracingError(msg)

    anchor = _first_traced(
        (args[0], args[1], values.get("w")),
        traced_type=traced_type,
    )
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.polyfit requires a traced x, y, or weight operand"
        raise TracingError(msg)
    x = np.atleast_1d(_as_traced(args[0], anchor=anchor, traced_type=traced_type)) + 0.0
    y = np.atleast_1d(_as_traced(args[1], anchor=anchor, traced_type=traced_type)) + 0.0
    if x.ndim != 1 or x.size == 0:
        msg = "numpy.polyfit x must be a non-empty vector"
        raise TracingError(msg)
    if y.ndim not in {1, 2} or y.shape[0] != x.shape[0]:
        msg = "numpy.polyfit y must be one- or two-dimensional and match x"
        raise TracingError(msg)
    order = degree + 1
    lhs = np.vander(x, order)
    rhs = y
    weights = values.get("w")
    if weights is not None:
        weights = np.atleast_1d(_as_traced(weights, anchor=anchor, traced_type=traced_type)) + 0.0
        if weights.ndim != 1 or weights.shape[0] != x.shape[0]:
            msg = "numpy.polyfit weights must be one-dimensional and match x"
            raise TracingError(msg)
        lhs = lhs * weights[:, None]
        rhs = rhs * (weights[:, None] if rhs.ndim == _MATRIX_RANK else weights)

    scale = np.sqrt(np.sum(lhs * lhs, axis=0))
    scaled_lhs = lhs / scale
    q_factor, r_factor = np.linalg.qr(scaled_lhs, mode="reduced")
    projected = np.matmul(np.swapaxes(np.conjugate(q_factor), -1, -2), rhs)
    coefficients = np.linalg.solve(r_factor, projected)
    coefficients = (
        coefficients / scale[:, None] if coefficients.ndim == _MATRIX_RANK else coefficients / scale
    )

    singular_values = np.linalg.svdvals(scaled_lhs)
    _node_id, concrete_singular_values = _snapshot_traced(singular_values)
    singular_array = np.asarray(concrete_singular_values)
    rcond = values.get("rcond")
    if rcond is None:
        rcond = int(x.shape[0]) * np.finfo(np.dtype(x.dtype)).eps
    if isinstance(rcond, traced_type):
        msg = "numpy.polyfit rcond must be static because it controls numerical rank"
        raise TracingError(msg)
    rank = int(np.sum(singular_array > singular_array[0] * float(rcond)))
    if rank != order:
        msg = (
            "numpy.polyfit encountered a rank-deficient design matrix; its "
            "rank-switching derivative is not supported"
        )
        raise TracingError(msg)

    residual = (
        np.matmul(
            scaled_lhs,
            coefficients * (scale[:, None] if coefficients.ndim == _MATRIX_RANK else scale),
        )
        - rhs
    )
    residuals = np.sum(np.real(np.conjugate(residual) * residual), axis=0)
    if bool(values.get("full", False)):
        rank_value = np.astype(np.sum(coefficients) * 0 + rank, np.int64)
        rcond_value = np.sum(coefficients) * 0 + float(rcond)
        return _finish(
            (coefficients, np.atleast_1d(residuals), rank_value, singular_values, rcond_value),
            traced_type=traced_type,
        )

    covariance = values.get("cov", False)
    if not covariance:
        return _finish(coefficients, traced_type=traced_type)
    base_covariance = np.linalg.inv(
        np.matmul(np.swapaxes(scaled_lhs, -1, -2), scaled_lhs)
    ) / np.outer(scale, scale)
    if covariance == "unscaled":
        factor: object = 1.0
    else:
        if int(x.shape[0]) <= order:
            msg = "numpy.polyfit covariance scaling requires more points than coefficients"
            raise TracingError(msg)
        factor = residuals / (int(x.shape[0]) - order)
    covariance_result = (
        base_covariance * factor if y.ndim == 1 else base_covariance[:, :, None] * factor
    )
    return _finish(
        (coefficients, covariance_result),
        traced_type=traced_type,
    )


def _polyder_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) not in {1, _BINARY_ARITY} or set(kwargs) - {"m"}:
        msg = "numpy.polyder expects (p, m=1) during tracing"
        raise TracingError(msg)
    if len(args) == _BINARY_ARITY and "m" in kwargs:
        msg = "numpy.polyder received m twice"
        raise TracingError(msg)
    coefficients = np.atleast_1d(args[0])
    order = int(args[1] if len(args) == _BINARY_ARITY else kwargs.get("m", 1))
    if order < 0:
        msg = "numpy.polyder order must be non-negative"
        raise TracingError(msg)
    for _ in range(order):
        degree = int(coefficients.shape[0]) - 1
        if degree <= 0:
            coefficients = coefficients[:0]
            break
        coefficients = coefficients[:-1] * np.arange(degree, 0, -1)
    return _finish(coefficients, traced_type=traced_type)


def _polyint_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) not in {1, _BINARY_ARITY, _TERNARY_ARITY} or set(kwargs) - {"m", "k"}:
        msg = "numpy.polyint expects (p, m=1, k=None) during tracing"
        raise TracingError(msg)
    values = dict(kwargs)
    for name, value in zip(("m", "k"), args[1:], strict=False):
        if name in values:
            msg = f"numpy.polyint received {name} twice"
            raise TracingError(msg)
        values[name] = value
    coefficients = np.atleast_1d(args[0])
    order = int(values.get("m", 1))
    if order < 0:
        msg = "numpy.polyint order must be non-negative"
        raise TracingError(msg)
    constants_raw = values.get("k")
    if constants_raw is None:
        constants: Any = np.zeros(order)
    else:
        constants = np.atleast_1d(constants_raw)
        if int(constants.shape[0]) == 1 and order > 1:
            constants = np.repeat(constants, order)
        if int(constants.shape[0]) < order:
            msg = "numpy.polyint k must be scalar or contain at least m constants"
            raise TracingError(msg)
    for index in range(order):
        divisors = np.arange(int(coefficients.shape[0]), 0, -1)
        constant = constants[index]
        if not isinstance(constant, traced_type):
            constant = _lift_composite_constant(constant, coefficients)
        coefficients = np.concatenate((coefficients / divisors, np.atleast_1d(constant)))
    return _finish(coefficients, traced_type=traced_type)


def _polyval_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        msg = "numpy.polyval expects (p, x) during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.polyval requires a traced operand"
        raise TracingError(msg)
    coefficients = np.atleast_1d(_as_traced(args[0], anchor=anchor, traced_type=traced_type))
    x = _as_traced(args[1], anchor=anchor, traced_type=traced_type)
    result = np.zeros_like(x)
    for index in range(int(coefficients.shape[0])):
        result = result * x + coefficients[index]
    return _finish(result, traced_type=traced_type)


def _roots_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != 1 or kwargs:
        msg = "numpy.roots expects one coefficient vector during tracing"
        raise TracingError(msg)
    coefficients = np.atleast_1d(args[0])
    if coefficients.ndim != 1:
        msg = "numpy.roots coefficients must be one-dimensional"
        raise TracingError(msg)
    _node_id, concrete = _snapshot_traced(coefficients)
    if bool(getattr(type(concrete), "__advect_abstract_array__", False)):
        msg = "staging numpy.roots requires a statically nonzero leading coefficient"
        raise TracingError(msg)
    concrete_array = np.asarray(concrete)
    nonzero = np.flatnonzero(concrete_array)
    if nonzero.size == 0 or int(nonzero[0]) == concrete_array.size - 1:
        return _finish(coefficients[:0], traced_type=traced_type)
    coefficients = coefficients[int(nonzero[0]) :]
    degree = int(coefficients.shape[0]) - 1
    first_row = -coefficients[1:] / coefficients[0]
    companion = np.eye(degree, k=-1, dtype=coefficients.dtype) + np.sum(coefficients) * 0
    companion[0, :] = first_row
    return _finish(np.linalg.eigvals(companion), traced_type=traced_type)


def register_polynomial_handlers(
    handlers: dict[Callable[..., Any], Callable[..., Any]],
) -> None:
    """Register classic polynomial functions with differentiable parameters."""
    handlers[np.poly] = _poly_handler
    handlers[np.polyadd] = _polyadd_handler
    handlers[np.polysub] = _polysub_handler
    handlers[np.polymul] = _polymul_handler
    handlers[np.polydiv] = _polydiv_handler
    handlers[np.polyfit] = _polyfit_handler
    handlers[np.polyder] = _polyder_handler
    handlers[np.polyint] = _polyint_handler
    handlers[np.polyval] = _polyval_handler
    handlers[np.roots] = _roots_handler
