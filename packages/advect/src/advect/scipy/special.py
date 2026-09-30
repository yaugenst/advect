# ruff: noqa: ANN401  # Primitive rules are intentionally array-provider generic.
"""Traceable high-value counterparts to ``scipy.special``."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import special as _scipy_special

from advect.core import ArraySpec, primitive
from advect.core._context import is_tracing
from advect.scipy._frontend import (
    _array_operand,
    _concrete_scipy,
    _numpy_dtype,
    _operand_dtype,
    _replace_out as _replace_traced_out,
    _require_numpy_values as _require_scipy_numpy_values,
    _traceable_astype,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from advect.core import AbstractValue
    from advect.core._primitive import Primitive

_UFUNC_OPTION_NAMES = frozenset({"casting", "dtype", "order", "sig", "signature", "subok", "where"})
type _UfuncOptions = tuple[
    str | bytes,
    str | bytes | None,
    str | None,
    bool,
    str | bytes | tuple[str | None, ...] | None,
]
_DEFAULT_UFUNC_OPTIONS: _UfuncOptions = ("same_kind", "K", None, True, None)
_REFLECTION_BOUNDARY = 0.5
_ERFCX_ASYMPTOTIC_BOUNDARY = 8.0
_scipy_parity = _concrete_scipy(_scipy_special)


def gammaln(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the logarithm of the absolute gamma function."""
    return _call_unary("gammaln", x, out, kwargs)


def digamma(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the logarithmic derivative of the gamma function."""
    return _call_unary("digamma", x, out, kwargs)


@_scipy_parity
def polygamma(n: object, x: object) -> object:
    """Compute the ``n``-th derivative of ``digamma`` with SciPy broadcasting."""
    return _polygamma_primitive(n=_array_operand(n), x=_array_operand(x))


def erf(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the error function."""
    return _call_unary("erf", x, out, kwargs)


def erfc(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the complementary error function."""
    return _call_unary("erfc", x, out, kwargs)


def erfcx(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the scaled complementary error function."""
    return _call_unary("erfcx", x, out, kwargs)


def erfinv(y: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the inverse error function."""
    return _call_unary("erfinv", y, out, kwargs)


def expit(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the logistic sigmoid."""
    return _call_unary("expit", x, out, kwargs)


def log_expit(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the logarithm of the logistic sigmoid."""
    return _call_unary("log_expit", x, out, kwargs)


def ndtr(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the standard normal cumulative distribution function."""
    return _call_unary("ndtr", x, out, kwargs)


def log_ndtr(x: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the logarithm of the standard normal cumulative distribution."""
    return _call_unary("log_ndtr", x, out, kwargs)


def ndtri(p: object, /, out: object = None, **kwargs: object) -> object:
    """Compute the inverse standard normal cumulative distribution."""
    return _call_unary("ndtri", p, out, kwargs)


@_scipy_parity
def logsumexp(
    a: object,
    axis: object = None,
    b: object = None,
    keepdims: bool = False,  # noqa: FBT001, FBT002 - SciPy-compatible spelling.
    return_sign: bool = False,  # noqa: FBT001, FBT002 - SciPy-compatible spelling.
) -> object:
    """Compute SciPy-compatible weighted, optionally signed log-sum-exp."""
    has_b = b is not None
    result, sign = _logsumexp_primitive(
        a=_array_operand(a),
        b=1.0 if b is None else _array_operand(b),
        axis=_static_axis(axis),
        has_b=has_b,
        keepdims=keepdims,
        return_sign=return_sign,
    )
    return (result, sign) if return_sign else result


@_scipy_parity
def softmax(x: object, axis: object = None) -> object:
    """Compute the softmax function along ``axis``."""
    return _softmax_primitive(
        x=_array_operand(x),
        axis=_static_axis(axis, name="softmax"),
    )


@_scipy_parity
def log_softmax(x: object, axis: object = None) -> object:
    """Compute the logarithm of the softmax function along ``axis``."""
    return _log_softmax_primitive(
        x=_array_operand(x),
        axis=_static_axis(axis, name="log_softmax"),
    )


def _require_numpy_values(name: str, *values: object) -> None:
    _require_scipy_numpy_values("special", name, *values)


def _normalize_out(out: object) -> object | None:
    if not isinstance(out, tuple):
        return out
    if len(out) != 1:
        msg = "The 'out' tuple must have exactly one entry per ufunc output"
        raise ValueError(msg)
    return out[0]


def _normalize_dtype(dtype: object) -> str | None:
    if dtype is None:
        return None
    return np.dtype(cast("DTypeLike", dtype)).str


def _normalize_signature(value: object) -> str | bytes | tuple[str | None, ...]:
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.bytes_):
        return bytes(value)
    if isinstance(value, str | bytes):
        return value
    if not isinstance(value, tuple):
        msg = "the signature object to ufunc must be a string or a tuple"
        raise TypeError(msg)
    return tuple(None if item is None else np.dtype(cast("DTypeLike", item)).str for item in value)


def _normalize_ufunc_text_option(
    value: object,
    *,
    name: str,
    allow_none: bool = False,
) -> str | bytes | None:
    if value is None and allow_none:
        return None
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.bytes_):
        return bytes(value)
    if isinstance(value, str | bytes):
        return value
    msg = f"{name} must be str, not {type(value).__name__}"
    raise TypeError(msg)


def _normalize_ufunc_options(kwargs: dict[str, object]) -> dict[str, object]:
    unknown = set(kwargs).difference(_UFUNC_OPTION_NAMES)
    if unknown:
        name = min(unknown)
        msg = f"got an unexpected keyword argument {name!r}"
        raise TypeError(msg)
    if "signature" in kwargs and "sig" in kwargs:
        msg = "cannot specify both 'sig' and 'signature'"
        raise TypeError(msg)

    signature_value = kwargs.get("signature", kwargs.get("sig"))
    if ("signature" in kwargs or "sig" in kwargs) and signature_value is None:
        msg = "the signature object to ufunc must be a string or a tuple"
        raise TypeError(msg)

    subok = kwargs.get("subok", True)
    if type(subok) is not bool:
        msg = "'subok' must be a boolean"
        raise TypeError(msg)
    casting = kwargs.get("casting", "same_kind")
    order = kwargs.get("order", "K")
    return {
        "where": _array_operand(kwargs.get("where", True)),
        "options": (
            _normalize_ufunc_text_option(casting, name="casting"),
            _normalize_ufunc_text_option(order, name="order", allow_none=True),
            _normalize_dtype(kwargs.get("dtype")),
            subok,
            None if signature_value is None else _normalize_signature(signature_value),
        ),
    }


def _replace_out(destination: object, replacement: object) -> object:
    return _replace_traced_out(
        destination,
        replacement,
        argument="out",
        operation="scipy.special out=",
    )


def _call_unary(name: str, x: object, out: object, kwargs: dict[str, object]) -> object:
    if not is_tracing():
        _require_numpy_values(name, x, out, kwargs.get("where"))
        return getattr(_scipy_special, name)(x, out=out, **kwargs)

    destination = _normalize_out(out)
    has_out = destination is not None
    options = _normalize_ufunc_options(kwargs)
    replacement = _UNARY_PRIMITIVES[name](
        x=_array_operand(x),
        destination=_array_operand(x if destination is None else destination),
        has_out=has_out,
        **options,
    )
    return replacement if destination is None else _replace_out(destination, replacement)


def _static_axis(
    axis: object,
    *,
    name: str = "logsumexp",
) -> int | tuple[int, ...] | None:
    if axis is None:
        return None
    if isinstance(axis, bool):
        msg = f"{name} axis must be an integer, a tuple of integers, or None"
        raise TypeError(msg)
    try:
        return operator.index(cast("Any", axis))
    except TypeError:
        pass
    if not isinstance(axis, tuple):
        msg = f"{name} axis must be an integer, a tuple of integers, or None"
        raise TypeError(msg)
    normalized: list[int] = []
    for item in axis:
        if isinstance(item, bool):
            msg = f"{name} axis entries must be integers"
            raise TypeError(msg)
        try:
            normalized.append(operator.index(item))
        except TypeError as error:
            msg = f"{name} axis entries must be integers"
            raise TypeError(msg) from error
    return tuple(normalized)


def _normalized_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    raw = tuple(range(ndim)) if axis is None else ((axis,) if isinstance(axis, int) else axis)
    normalized: list[int] = []
    for item in raw:
        if ndim == 0 and item in (0, -1):
            normalized.append(0)
            continue
        if item < -ndim or item >= ndim:
            msg = f"axis {item} is out of bounds for array of dimension {ndim}"
            raise np.exceptions.AxisError(msg)
        normalized.append(item + ndim if item < 0 else item)
    if len(set(normalized)) != len(normalized):
        msg = f"duplicate axes are not allowed: {axis!r}"
        raise ValueError(msg)
    return tuple(normalized)


def _reduction_shape(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
    *,
    keepdims: bool,
) -> tuple[int, ...]:
    axes = set(_normalized_axes(axis, len(shape)))
    if not shape:
        return (1,) if keepdims or axis == () else ()
    if keepdims:
        return tuple(1 if index in axes else size for index, size in enumerate(shape))
    return tuple(size for index, size in enumerate(shape) if index not in axes)


def _ufunc_runtime_kwargs(
    *,
    options: _UfuncOptions,
    where: Any,
) -> dict[str, Any]:
    casting, order, dtype, subok, signature = options
    kwargs: dict[str, Any] = {
        "casting": casting,
        "order": order,
        "subok": subok,
        "where": where,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if signature is not None:
        kwargs["signature"] = signature
    return kwargs


def _is_inexact_dtype(dtype: np.dtype[Any]) -> bool:
    return np.issubdtype(dtype, np.inexact)


def _cotangent_at_primal_precision(cotangent: Any, primal: Any) -> Any:
    """Cast ``cotangent`` to an inexact primal's dtype.

    SciPy may compute in a wider float than the primal. An integer or boolean
    primal keeps that working precision instead of a truncating cast.
    """
    dtype = _operand_dtype(primal)
    return _traceable_astype(cotangent, dtype) if _is_inexact_dtype(dtype) else cotangent


def _unary_loop_dtypes(
    function: Callable[..., Any],
    x: Any,
    destination: Any,
    *,
    has_out: bool,
    options: _UfuncOptions,
) -> tuple[np.dtype[Any], np.dtype[Any], np.dtype[Any]]:
    casting, _order, dtype, _subok, signature = options
    output_dtype = _operand_dtype(destination) if has_out else None
    resolve_dtypes = getattr(function, "resolve_dtypes", None)
    if not callable(resolve_dtypes):
        msg = f"{function!r} does not expose NumPy ufunc dtype resolution"
        raise TypeError(msg)
    resolve_kwargs: dict[str, Any] = {"casting": casting}
    if signature is not None:
        resolve_kwargs["signature"] = signature
    elif dtype is not None:
        resolve_kwargs["signature"] = (None, dtype)
    input_dtype, loop_output_dtype = cast(
        "tuple[object, object]",
        resolve_dtypes(
            (_operand_dtype(x), output_dtype),
            **resolve_kwargs,
        ),
    )
    result_dtype = output_dtype if output_dtype is not None else loop_output_dtype
    return (
        _numpy_dtype(input_dtype),
        _numpy_dtype(loop_output_dtype),
        _numpy_dtype(result_dtype),
    )


def _unary_impl(
    x: Any,
    destination: Any,
    where: Any,
    *,
    function: Callable[..., Any],
    has_out: bool,
    options: _UfuncOptions = _DEFAULT_UFUNC_OPTIONS,
) -> Any:
    kwargs = _ufunc_runtime_kwargs(
        options=options,
        where=where,
    )
    if not has_out:
        # SciPy 1.18 follows NumPy 2.4 in warning when ``where`` is supplied
        # without an explicit output choice. ``None`` preserves allocation
        # semantics while making the initialized-output contract explicit.
        return function(x, out=None, **kwargs)
    fresh_out = destination.copy()
    return function(x, out=fresh_out, **kwargs)


def _unary_abstract(
    x: AbstractValue,
    destination: AbstractValue,
    where: AbstractValue,
    *,
    function: Callable[..., Any],
    has_out: bool,
    options: _UfuncOptions = _DEFAULT_UFUNC_OPTIONS,
) -> ArraySpec:
    broadcast_shapes = [x.spec.shape, where.spec.shape]
    if has_out:
        broadcast_shapes.append(destination.spec.shape)
    output_shape = np.broadcast_shapes(*broadcast_shapes)
    if has_out and destination.spec.shape != output_shape:
        msg = (
            "non-broadcastable output operand with shape "
            f"{destination.spec.shape!r} does not match the broadcast shape {output_shape!r}"
        )
        raise ValueError(msg)

    x_sample = np.ones((), dtype=_numpy_dtype(x.spec.dtype))
    where_sample = np.ones((), dtype=_numpy_dtype(where.spec.dtype))
    destination_sample = np.empty((), dtype=_numpy_dtype(destination.spec.dtype))
    result = _unary_impl(
        x_sample,
        destination_sample,
        where_sample,
        function=function,
        has_out=has_out,
        options=options,
    )
    return ArraySpec(
        output_shape,
        np.asarray(result).dtype.name,
        device=destination.spec.device if has_out else x.spec.device,
    )


def _install_unary(name: str, derivative: Callable[[Any], Any]) -> Primitive[..., Any]:
    implementation = getattr(_scipy_special, name)

    @primitive(
        name=f"scipy.special.{name}",
        static_argnames=("has_out", "options"),
        nondiff_argnames=("where",),
    )
    def concrete(
        x: Any,
        destination: Any,
        where: Any,
        *,
        has_out: bool = False,
        options: _UfuncOptions = _DEFAULT_UFUNC_OPTIONS,
    ) -> Any:
        _require_numpy_values(name, x, destination, where)
        return _unary_impl(
            x,
            destination,
            where,
            function=implementation,
            has_out=has_out,
            options=options,
        )

    @concrete.def_abstract
    def abstract(
        x: AbstractValue,
        destination: AbstractValue,
        where: AbstractValue,
        *,
        has_out: bool = False,
        options: _UfuncOptions = _DEFAULT_UFUNC_OPTIONS,
    ) -> ArraySpec:
        return _unary_abstract(
            x,
            destination,
            where,
            function=implementation,
            has_out=has_out,
            options=options,
        )

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        has_out: bool = False,
        options: _UfuncOptions = _DEFAULT_UFUNC_OPTIONS,
    ) -> Any:
        x, destination, where = primals
        tangent, destination_tangent, _where_tangent = tangents
        input_dtype, loop_output_dtype, result_dtype = _unary_loop_dtypes(
            implementation,
            x,
            destination,
            has_out=has_out,
            options=options,
        )
        input_is_differentiable = _is_inexact_dtype(_operand_dtype(x))
        output_is_differentiable = _is_inexact_dtype(result_dtype)
        if tangent is None or not input_is_differentiable or not output_is_differentiable:
            active = np.zeros_like(output)
        else:
            loop_x = _traceable_astype(x, input_dtype)
            loop_tangent = _traceable_astype(tangent, input_dtype)
            active = _traceable_astype(
                derivative(loop_x) * loop_tangent,
                loop_output_dtype,
            )
            active = _traceable_astype(active, result_dtype)
        if has_out:
            inactive = np.zeros_like(output) if destination_tangent is None else destination_tangent
            return np.where(where, active, inactive)
        return np.where(where, active, 0)

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        has_out: bool = False,
        options: _UfuncOptions = _DEFAULT_UFUNC_OPTIONS,
    ) -> tuple[Any, Any | None, None]:
        del output
        x, destination, where = primals
        active_cotangent = np.where(where, cotangent, 0)
        input_dtype, loop_output_dtype, result_dtype = _unary_loop_dtypes(
            implementation,
            x,
            destination,
            has_out=has_out,
            options=options,
        )
        if not _is_inexact_dtype(_operand_dtype(x)) or not _is_inexact_dtype(result_dtype):
            x_cotangent = np.zeros_like(x)
        else:
            loop_x = _traceable_astype(x, input_dtype)
            loop_cotangent = _traceable_astype(active_cotangent, loop_output_dtype)
            x_cotangent = _traceable_astype(
                np.conj(derivative(loop_x)) * loop_cotangent,
                _operand_dtype(x),
            )
        destination_cotangent = np.where(where, 0, cotangent) if has_out else None
        return x_cotangent, destination_cotangent, None

    return concrete


def _complex_trigamma(x: Any) -> Any:
    """Evaluate trigamma traceably for real or complex ``x``.

    Reflection moves the left half-plane to ``Re(z) >= 1/2``. A fixed
    recurrence then moves the argument into the asymptotic region, where the
    Bernoulli expansion is accurate to roughly double precision.
    """
    reflected = np.real(x) < _REFLECTION_BOUNDARY
    base = np.where(reflected, 1 - x, x)
    shifted = base + 16
    reciprocal = 1 / shifted
    reciprocal_squared = reciprocal * reciprocal
    asymptotic = reciprocal + 0.5 * reciprocal_squared
    power = reciprocal * reciprocal_squared
    for coefficient in (
        1 / 6,
        -1 / 30,
        1 / 42,
        -1 / 30,
        5 / 66,
        -691 / 2730,
        7 / 6,
        -3617 / 510,
    ):
        asymptotic = asymptotic + coefficient * power
        power = power * reciprocal_squared
    recurrence = 0
    for offset in range(16):
        recurrence = recurrence + 1 / ((base + offset) ** 2)
    positive = asymptotic + recurrence
    reflected_value = (math.pi / np.sin(math.pi * x)) ** 2 - positive
    return np.where(reflected, reflected_value, positive)


def _expit_derivative(x: Any) -> Any:
    # ``1 - expit(x)`` cancels in the upper tail; ``expit(-x)`` stays exact.
    return cast("Any", expit(x)) * cast("Any", expit(-x))


def _erfcx_derivative(x: Any) -> Any:
    direct = 2 * x * cast("Any", erfcx(x)) - 2 / math.sqrt(math.pi)
    use_asymptotic = np.real(x) > _ERFCX_ASYMPTOTIC_BOUNDARY
    safe_x = np.where(use_asymptotic, x, 1)
    inverse_squared = 1 / (safe_x * safe_x)
    series = 1
    power = 1
    coefficient = 1.0
    for order in range(1, 16):
        coefficient *= -(2 * order + 1) / 2
        power = power * inverse_squared
        series = series + coefficient * power
    asymptotic = -inverse_squared * series / math.sqrt(math.pi)
    return np.where(use_asymptotic, asymptotic, direct)


def _erfinv_derivative(x: Any) -> Any:
    value = cast("Any", erfinv(x))
    return math.sqrt(math.pi) / 2 * np.exp(value * value)


def _log_ndtr_derivative(x: Any) -> Any:
    # Scaling by erfcx keeps the lower tail exact, but erfcx overflows in the
    # real upper tail, where the nested derivative of its reciprocal would
    # divide infinities. There ndtr(x) = erfc(-x / sqrt(2)) / 2 stays near one
    # and exp(-x**2 / 2) underflows smoothly instead.
    scale = math.sqrt(2 / math.pi)
    if _operand_dtype(x).kind == "c":
        return scale / cast("Any", erfcx(-x / math.sqrt(2)))
    upper = x > 0
    lower_x, upper_x = np.where(upper, 0, x), np.where(upper, x, 0)
    lower_tail = scale / cast("Any", erfcx(-lower_x / math.sqrt(2)))
    upper_tail = (
        scale * np.exp(-0.5 * upper_x * upper_x) / cast("Any", erfc(-upper_x / math.sqrt(2)))
    )
    return np.where(upper, upper_tail, lower_tail)


def _ndtri_derivative(x: Any) -> Any:
    value = cast("Any", ndtri(x))
    return math.sqrt(2 * math.pi) * np.exp(0.5 * value * value)


def _polygamma_impl(n: Any, x: Any) -> Any:
    _require_numpy_values("polygamma", n, x)
    return _scipy_special.polygamma(n, x)


def _polygamma_abstract(n: AbstractValue, x: AbstractValue) -> ArraySpec:
    output_shape = np.broadcast_shapes(n.spec.shape, x.spec.shape)
    n_sample = np.zeros((), dtype=_numpy_dtype(n.spec.dtype))
    x_sample = np.ones((), dtype=_numpy_dtype(x.spec.dtype))
    result = _scipy_special.polygamma(n_sample, x_sample)
    return ArraySpec(
        output_shape,
        np.asarray(result).dtype.name,
        device=x.spec.device,
    )


def _polygamma_jvp(
    output: Any,
    primals: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
) -> Any:
    del output
    n, x = primals
    tangent = tangents[1]
    coefficient = cast("Any", polygamma(n + 1, x))
    return coefficient * (0 if tangent is None else tangent)


def _polygamma_transpose(
    cotangent: Any,
    primals: tuple[Any, ...],
    output: Any,
) -> tuple[None, Any]:
    del output
    n, x = primals
    x_cotangent = np.conj(cast("Any", polygamma(n + 1, x))) * cotangent
    return None, _cotangent_at_primal_precision(x_cotangent, x)


def _logsumexp_impl(
    a: Any,
    b: Any,
    *,
    axis: int | tuple[int, ...] | None = None,
    has_b: bool = False,
    keepdims: bool = False,
    return_sign: bool = False,
) -> tuple[Any, Any]:
    _require_numpy_values("logsumexp", a, b)
    result = _scipy_special.logsumexp(
        a,
        axis=axis,
        b=b if has_b else None,
        keepdims=keepdims,
        return_sign=return_sign,
    )
    if return_sign:
        return cast("tuple[Any, Any]", result)
    sign = np.ones_like(result)
    return result, sign


def _logsumexp_abstract(
    a: AbstractValue,
    b: AbstractValue,
    *,
    axis: int | tuple[int, ...] | None = None,
    has_b: bool = False,
    keepdims: bool = False,
    return_sign: bool = False,
) -> tuple[ArraySpec, ArraySpec]:
    input_shape = np.broadcast_shapes(a.spec.shape, b.spec.shape) if has_b else a.spec.shape
    output_shape = _reduction_shape(input_shape, axis, keepdims=keepdims)
    a_sample = np.ones((), dtype=_numpy_dtype(a.spec.dtype))
    b_sample = np.ones((), dtype=_numpy_dtype(b.spec.dtype))
    result, sign = _logsumexp_impl(
        a_sample,
        b_sample,
        axis=None,
        has_b=has_b,
        keepdims=False,
        return_sign=return_sign,
    )
    return (
        ArraySpec(output_shape, np.asarray(result).dtype.name, device=a.spec.device),
        ArraySpec(output_shape, np.asarray(sign).dtype.name, device=a.spec.device),
    )


def _expand_reduced(
    value: Any,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> Any:
    if keepdims or axis is None or axis == ():
        return value
    return np.expand_dims(value, axis=axis)


def _logsumexp_weights(
    a: Any,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    reduced: Any,
    sign: Any,
    return_sign: bool,
) -> Any:
    expanded = _expand_reduced(reduced, axis=axis, keepdims=keepdims)
    weights = np.exp(a - expanded)
    if return_sign:
        weights = weights * np.conj(_expand_reduced(sign, axis=axis, keepdims=keepdims))
    return weights


def _logsumexp_jvp(
    output: Any,
    primals: tuple[Any, ...],
    tangents: tuple[Any | None, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
    has_b: bool = False,
    keepdims: bool = False,
    return_sign: bool = False,
) -> tuple[Any, Any]:
    a, b = primals
    a_tangent, b_tangent = tangents
    reduced, sign = output
    scaled_exponential = _logsumexp_weights(
        a,
        axis=axis,
        keepdims=keepdims,
        reduced=reduced,
        sign=sign,
        return_sign=return_sign,
    )
    summand_tangent = b * (0 if a_tangent is None else a_tangent)
    if has_b:
        summand_tangent = summand_tangent + (0 if b_tangent is None else b_tangent)
    ratio_tangent = np.sum(
        scaled_exponential * summand_tangent,
        axis=axis,
        keepdims=keepdims,
    )
    if not return_sign:
        return ratio_tangent, np.zeros_like(sign)
    real_ratio = np.real(ratio_tangent)
    return real_ratio, sign * (ratio_tangent - real_ratio)


def _logsumexp_transpose(
    cotangent: tuple[Any, Any],
    primals: tuple[Any, ...],
    output: tuple[Any, Any],
    *,
    axis: int | tuple[int, ...] | None = None,
    has_b: bool = False,
    keepdims: bool = False,
    return_sign: bool = False,
) -> tuple[Any, Any | None]:
    a, b = primals
    reduced, sign = output
    reduced_cotangent, sign_cotangent = cotangent
    reduced_cotangent = 0 if reduced_cotangent is None else reduced_cotangent
    sign_cotangent = 0 if sign_cotangent is None else sign_cotangent
    weights = _logsumexp_weights(
        a,
        axis=axis,
        keepdims=keepdims,
        reduced=reduced,
        sign=sign,
        return_sign=return_sign,
    )
    if return_sign:
        ratio_cotangent = np.real(reduced_cotangent) - 1j * (np.conj(sign_cotangent) * sign).imag
    else:
        ratio_cotangent = reduced_cotangent
    expanded_cotangent = _expand_reduced(
        ratio_cotangent,
        axis=axis,
        keepdims=keepdims,
    )
    a_cotangent = _cotangent_at_primal_precision(np.conj(weights * b) * expanded_cotangent, a)
    if not has_b:
        return a_cotangent, None
    return a_cotangent, _cotangent_at_primal_precision(np.conj(weights) * expanded_cotangent, b)


def _install_normalization(name: str, *, logarithmic: bool) -> Primitive[..., Any]:
    implementation = getattr(_scipy_special, name)

    @primitive(
        name=f"scipy.special.{name}",
        static_argnames=("axis",),
    )
    def concrete(
        x: Any,
        *,
        axis: int | tuple[int, ...] | None,
    ) -> Any:
        _require_numpy_values(name, x)
        return implementation(x, axis=axis)

    @concrete.def_abstract
    def abstract(
        x: AbstractValue,
        *,
        axis: int | tuple[int, ...] | None,
    ) -> ArraySpec:
        sample_shape = () if not x.spec.shape else (1,) * len(x.spec.shape)
        sample = np.ones(sample_shape, dtype=_numpy_dtype(x.spec.dtype))
        result = implementation(sample, axis=axis)
        return ArraySpec(x.spec.shape, np.asarray(result).dtype.name, device=x.spec.device)

    @concrete.def_jvp
    def jvp_rule(
        output: Any,
        primals: tuple[Any, ...],
        tangents: tuple[Any | None, ...],
        *,
        axis: int | tuple[int, ...] | None,
    ) -> Any:
        (x,) = primals
        (tangent,) = tangents
        output_dtype = _operand_dtype(output)
        if tangent is None or not _is_inexact_dtype(_operand_dtype(x)):
            return np.zeros_like(output)
        work_dtype = np.dtype(np.float32) if output_dtype == np.dtype(np.float16) else output_dtype
        work_output = _traceable_astype(output, work_dtype)
        work_tangent = _traceable_astype(tangent, work_dtype)
        if logarithmic:
            probabilities = np.exp(work_output)
            result = work_tangent - np.sum(
                probabilities * work_tangent,
                axis=axis,
                keepdims=True,
            )
        else:
            result = work_output * (
                work_tangent
                - np.sum(
                    work_tangent * work_output,
                    axis=axis,
                    keepdims=True,
                )
            )
        return _traceable_astype(result, output_dtype)

    @concrete.def_transpose
    def transpose_rule(
        cotangent: Any,
        primals: tuple[Any, ...],
        output: Any,
        *,
        axis: int | tuple[int, ...] | None,
    ) -> tuple[Any]:
        (x,) = primals
        output_dtype = _operand_dtype(output)
        if not _is_inexact_dtype(_operand_dtype(x)) or not _is_inexact_dtype(output_dtype):
            return (np.zeros_like(x),)
        work_dtype = np.dtype(np.float32) if output_dtype == np.dtype(np.float16) else output_dtype
        work_output = _traceable_astype(output, work_dtype)
        work_cotangent = _traceable_astype(cotangent, work_dtype)
        if logarithmic:
            result = work_cotangent - np.conj(np.exp(work_output)) * np.sum(
                work_cotangent,
                axis=axis,
                keepdims=True,
            )
        else:
            conjugate = np.conj(work_output)
            result = conjugate * (
                work_cotangent
                - np.sum(
                    conjugate * work_cotangent,
                    axis=axis,
                    keepdims=True,
                )
            )
        return (_traceable_astype(result, _operand_dtype(x)),)

    return concrete


_polygamma_primitive = primitive(
    _polygamma_impl,
    name="scipy.special.polygamma",
    nondiff_argnames=("n",),
)
_polygamma_primitive.def_abstract(_polygamma_abstract)
_polygamma_primitive.def_jvp(_polygamma_jvp)
_polygamma_primitive.def_transpose(_polygamma_transpose)

_logsumexp_primitive = primitive(
    _logsumexp_impl,
    name="scipy.special.logsumexp",
    static_argnames=("axis", "has_b", "keepdims", "return_sign"),
)
_logsumexp_primitive.def_abstract(_logsumexp_abstract)
_logsumexp_primitive.def_jvp(_logsumexp_jvp)
_logsumexp_primitive.def_transpose(_logsumexp_transpose)

_UNARY_DERIVATIVES: dict[str, Callable[[Any], Any]] = {
    "gammaln": digamma,
    "digamma": _complex_trigamma,
    "erf": lambda x: (2.0 / math.sqrt(math.pi)) * np.exp(-(x * x)),
    "erfc": lambda x: (-2.0 / math.sqrt(math.pi)) * np.exp(-(x * x)),
    "erfcx": _erfcx_derivative,
    "erfinv": _erfinv_derivative,
    "expit": _expit_derivative,
    "log_expit": lambda x: cast("Any", expit(-x)),
    "ndtr": lambda x: np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi),
    "log_ndtr": _log_ndtr_derivative,
    "ndtri": _ndtri_derivative,
}
_UNARY_PRIMITIVES = {
    name: _install_unary(name, derivative) for name, derivative in _UNARY_DERIVATIVES.items()
}
_softmax_primitive = _install_normalization("softmax", logarithmic=False)
_log_softmax_primitive = _install_normalization("log_softmax", logarithmic=True)


__all__ = [
    "digamma",
    "erf",
    "erfc",
    "erfcx",
    "erfinv",
    "expit",
    "gammaln",
    "log_expit",
    "log_ndtr",
    "log_softmax",
    "logsumexp",
    "ndtr",
    "ndtri",
    "polygamma",
    "softmax",
]
