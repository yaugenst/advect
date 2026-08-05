"""Small traceable JVP formulas for common array primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from advect.autodiff.rules.array_family._backend_runtime import _scalar_like, xp
from advect.autodiff.rules.array_family._impl.jvp.common import (
    _normalize_output_tangent,
    _zeros_output_tangent,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _elementwise_tangent(
    answer: object,
    tangents: tuple[object | None, ...],
    contribution: object | None,
) -> object | None:
    return (
        None if contribution is None else _normalize_output_tangent(answer, tangents, contribution)
    )


def _unary_tangent(
    answer: object,
    inputs: tuple[object, ...],
    tangents: tuple[object | None, ...],
    attrs: dict[str, object],
    factor: object,
) -> object | None:
    tangent = tangents[0] if tangents else None
    contribution = None if tangent is None else cast("Any", tangent) * cast("Any", factor)
    return _elementwise_tangent(
        answer,
        tangents,
        contribution,
    )


def _jvp_negative(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    _ = inputs, attrs
    tangent = tangents[0] if tangents else None
    return _elementwise_tangent(
        answer,
        tangents,
        None if tangent is None else -cast("Any", tangent),
    )


def _jvp_positive(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    _ = inputs, attrs
    tangent = tangents[0] if tangents else None
    return _elementwise_tangent(
        answer,
        tangents,
        tangent,
    )


def _jvp_copy(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    return _jvp_positive(answer, *inputs, tangents=tangents, **attrs)


def _jvp_sin(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    return _unary_tangent(answer, inputs, tangents, attrs, xp.cos(cast("Any", inputs[0])))


def _jvp_cos(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    return _unary_tangent(answer, inputs, tangents, attrs, -xp.sin(cast("Any", inputs[0])))


def _jvp_exp(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    return _unary_tangent(answer, inputs, tangents, attrs, answer)


def _jvp_log(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    x = cast("Any", inputs[0])
    return _unary_tangent(answer, inputs, tangents, attrs, _scalar_like(1.0, x) / x)


def _jvp_sqrt(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    answer_array = cast("Any", answer)
    factor = _scalar_like(1.0, answer_array) / (_scalar_like(2.0, answer_array) * answer_array)
    return _unary_tangent(answer, inputs, tangents, attrs, factor)


def _jvp_tanh(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    **attrs: object,
) -> object | None:
    answer_array = cast("Any", answer)
    factor = _scalar_like(1.0, answer_array) - answer_array * answer_array
    return _unary_tangent(answer, inputs, tangents, attrs, factor)


def _reduction_tangent(
    operation: Callable[..., object],
    answer: object,
    tangents: tuple[object | None, ...],
    *,
    axis: int | tuple[int, ...] | None,
    dtype: object | None,
    keepdims: bool,
    attrs: dict[str, object],
) -> object:
    if any(attrs.get(name) is not None for name in ("out", "where")):
        msg = "reduction derivatives do not support where/out control operands"
        raise NotImplementedError(msg)
    tangent = tangents[0] if tangents else None
    if tangent is None:
        return _zeros_output_tangent(answer, tangents)
    kwargs: dict[str, object] = {"axis": axis, "keepdims": keepdims}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return operation(tangent, **kwargs)


def _jvp_sum(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    axis: int | tuple[int, ...] | None = None,
    dtype: object | None = None,
    keepdims: bool = False,
    **attrs: object,
) -> object:
    return _reduction_tangent(
        xp.sum,
        answer,
        tangents,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        attrs=attrs,
    )


def _jvp_mean(
    answer: object,
    *inputs: object,
    tangents: tuple[object | None, ...],
    axis: int | tuple[int, ...] | None = None,
    dtype: object | None = None,
    keepdims: bool = False,
    **attrs: object,
) -> object:
    return _reduction_tangent(
        xp.mean,
        answer,
        tangents,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        attrs=attrs,
    )
