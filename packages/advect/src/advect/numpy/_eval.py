"""Data-driven forward interpreter for NumPy operations.

This module provides a registry-aligned evaluator that:
1. Maps op names to NumPy functions dynamically
2. Filters attrs to valid kwargs via signature inspection
3. Uses special handlers for complex ops (reshape, getitem, setitem)

The goal is to replace the hard-coded lambda table with a more maintainable
data-driven approach that automatically handles new ops.
"""

# ruff: noqa: ANN401  # Evaluator returns Any to handle diverse NumPy results

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import numpy as np

from advect.numpy._abstract_protocol import as_numpy_nested
from advect.numpy._protocol_eval import NUMPY_EVAL_RUNTIME

__all__ = ["bind_evaluator", "evaluate_op", "has_evaluator"]

Evaluator = Callable[[tuple[np.ndarray, ...], dict[str, Any]], Any]
type _ArrayOrder = Literal["A", "C", "F", "K"]
type _CastingRule = Literal["equiv", "no", "safe", "same_kind", "unsafe"]

_RUNTIME = NUMPY_EVAL_RUNTIME
_BINARY_ARITY = 2


def _coerce_eval_result(value: Any) -> Any:
    """Return ndarray for raw results while preserving traced wrappers."""
    return _RUNTIME._coerce_eval_result(value)  # noqa: SLF001 - thin binding layer


def _evaluator(op: str) -> Callable[[Evaluator], Evaluator]:
    """Co-locate an exceptional evaluator with the operation it handles."""

    def register(evaluator: Evaluator) -> Evaluator:
        _RUNTIME.register_evaluator(op, cast("Any", evaluator))
        return evaluator

    return register


def bind_evaluator(
    op: str,
    attrs: dict[str, Any],
) -> Callable[[tuple[object, ...]], object] | None:
    """Bind NumPy evaluator dispatch for a stable graph node."""
    return _RUNTIME.bind_evaluator(op, cast("dict[str, object]", attrs))


def has_evaluator(op: str) -> bool:
    """Return whether the live NumPy evaluator can route a canonical operation."""
    return _RUNTIME.bind_evaluator(op, {}) is not None


# -----------------------------------------------------------------------------
# Special evaluators for ops that need custom logic
# -----------------------------------------------------------------------------


@_evaluator("numpy.reshape")
def _eval_reshape(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate reshape with shape/newshape compatibility."""
    if "shape" in attrs:
        shape = attrs["shape"]
    elif "newshape" in attrs:
        shape = attrs["newshape"]
    else:
        msg = "reshape requires shape attribute"
        raise ValueError(msg)
    order = attrs.get("order", "C")
    kwargs: dict[str, Any] = {"order": order}
    if attrs.get("copy") is not None:
        kwargs["copy"] = bool(attrs["copy"])
    return np.reshape(inputs[0], shape, **kwargs)


@_evaluator("numpy.diff")
def _eval_diff(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate diff with optional differentiable prepend/append operands."""
    values = list(inputs)
    source = values.pop(0)
    kwargs: dict[str, Any] = {
        "axis": attrs.get("axis", -1),
        "n": attrs.get("n", 1),
    }
    if attrs.get("_advect_diff_prepend_input") is True:
        kwargs["prepend"] = values.pop(0)
    elif "prepend" in attrs:
        kwargs["prepend"] = attrs["prepend"]
    if attrs.get("_advect_diff_append_input") is True:
        kwargs["append"] = values.pop(0)
    elif "append" in attrs:
        kwargs["append"] = attrs["append"]
    if values:
        msg = "numpy.diff received unexpected dynamic operands"
        raise ValueError(msg)
    return np.diff(source, **kwargs)


@_evaluator("numpy.bincount")
def _eval_bincount(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate a weighted bincount with discrete indices."""
    if len(inputs) != _BINARY_ARITY:
        msg = f"numpy.bincount evaluator expects indices and weights (got {len(inputs)})"
        raise ValueError(msg)
    return np.bincount(
        inputs[0],
        weights=inputs[1],
        minlength=int(attrs.get("minlength", 0)),
    )


@_evaluator("numpy.eye")
def _eval_eye(_inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate canonical eye dimensions through NumPy's parameter names."""
    return np.eye(
        int(attrs["n_rows"]),
        None if attrs.get("n_cols") is None else int(attrs["n_cols"]),
        k=int(attrs.get("k", 0)),
        dtype=attrs.get("dtype"),
    )


@_evaluator("numpy.arange")
def _eval_arange(_inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Replay canonical positional bounds through NumPy's arange signature."""
    start = attrs["start"]
    stop = attrs.get("stop")
    step = attrs.get("step", 1)
    kwargs = {name: attrs[name] for name in ("device", "dtype") if attrs.get(name) is not None}
    if stop is None:
        return np.arange(start, **kwargs)
    return np.arange(start, stop, step, **kwargs)


@_evaluator("advect.getitem")
def _eval_getitem(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> Any:
    """Evaluate getitem with deserialized index."""
    idx = attrs.get("index")
    return _coerce_eval_result(inputs[0][idx])


@_evaluator("advect.index_update")
def _eval_index_update(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate a pure functional basic-index update."""
    result = inputs[0].copy()
    idx = attrs.get("index")
    mode = attrs.get("mode", "set")
    if mode == "add":
        result[idx] += inputs[1]
    elif mode == "set":
        result[idx] = inputs[1]
    else:
        msg = f"Unsupported index_update mode {mode!r}"
        raise ValueError(msg)
    return result


@_evaluator("advect.copy")
def _eval_copy(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate copy."""
    order = cast("_ArrayOrder", str(attrs.get("order", "K")))
    value = as_numpy_nested(inputs[0])
    return cast("Any", inputs[0] if value is NotImplemented else value).copy(order=order)


@_evaluator("numpy.linalg.qr_r")
def _eval_qr_r(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> Any:
    """Evaluate qr_r (R-only) via np.linalg.qr(..., mode='r')."""
    mode = attrs.get("mode", "r")
    if mode != "r":
        msg = f"numpy.linalg.qr_r only supports mode='r' (got {mode!r})"
        raise ValueError(msg)
    return _coerce_eval_result(np.linalg.qr(inputs[0], mode="r"))


@_evaluator("numpy.broadcast_to")
def _eval_broadcast_to(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate broadcast_to."""
    return np.broadcast_to(inputs[0], attrs["shape"])


@_evaluator("numpy.swapaxes")
def _eval_swapaxes(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate swapaxes."""
    return np.swapaxes(inputs[0], attrs["axis1"], attrs["axis2"])


@_evaluator("numpy.concatenate")
def _eval_concatenate(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate concatenate - inputs are the arrays to concatenate."""
    kwargs: dict[str, Any] = {"axis": attrs.get("axis", 0)}
    if attrs.get("dtype") is not None:
        kwargs["dtype"] = attrs["dtype"]
    if attrs.get("casting") is not None:
        kwargs["casting"] = attrs["casting"]
    return np.concatenate(inputs, **kwargs)


@_evaluator("numpy.stack")
def _eval_stack(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate stack - inputs are the arrays to stack."""
    kwargs: dict[str, Any] = {"axis": attrs.get("axis", 0)}
    if attrs.get("dtype") is not None:
        kwargs["dtype"] = attrs["dtype"]
    if attrs.get("casting") is not None:
        kwargs["casting"] = attrs["casting"]
    return np.stack(inputs, **kwargs)


def _like_creation_kwargs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Forward only non-default metadata when replay itself is being traced."""
    kwargs = {
        name: attrs[name] for name in ("device", "dtype", "shape") if attrs.get(name) is not None
    }
    if attrs.get("order", "K") != "K":
        kwargs["order"] = attrs["order"]
    if not bool(attrs.get("subok", True)):
        kwargs["subok"] = False
    return kwargs


@_evaluator("numpy.zeros_like")
def _eval_zeros_like(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate zeros_like."""
    return np.zeros_like(inputs[0], **_like_creation_kwargs(attrs))


@_evaluator("numpy.ones_like")
def _eval_ones_like(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate ones_like."""
    return np.ones_like(inputs[0], **_like_creation_kwargs(attrs))


@_evaluator("numpy.full_like")
def _eval_full_like(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate full_like."""
    fill_value = inputs[1] if len(inputs) > 1 else attrs.get("fill_value", 0)
    return np.full_like(inputs[0], fill_value, **_like_creation_kwargs(attrs))


@_evaluator("numpy.empty_like")
def _eval_empty_like(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate empty_like."""
    return np.empty_like(inputs[0], **_like_creation_kwargs(attrs))


@_evaluator("numpy.astype")
def _eval_astype(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate astype recorded as a traced op."""
    dtype = attrs.get("dtype")
    if dtype is None:
        msg = "numpy.astype requires dtype attr"
        raise ValueError(msg)
    copy = bool(attrs.get("copy", True))
    order_value = attrs.get("order", "K")
    if not isinstance(order_value, str) or order_value not in {"A", "C", "F", "K"}:
        msg = f"numpy.astype received invalid order {order_value!r}"
        raise ValueError(msg)
    order = cast("_ArrayOrder", order_value)
    casting_value = attrs.get("casting", "unsafe")
    if not isinstance(casting_value, str) or casting_value not in {
        "equiv",
        "no",
        "safe",
        "same_kind",
        "unsafe",
    }:
        msg = f"numpy.astype received invalid casting rule {casting_value!r}"
        raise ValueError(msg)
    casting = cast("_CastingRule", casting_value)
    subok = bool(attrs.get("subok", False))
    value = inputs[0]
    if type(value) in {bool, complex, float, int}:
        return np.asarray(value, dtype=np.dtype(dtype))
    nested_value = as_numpy_nested(value)
    if nested_value is not NotImplemented:
        value = nested_value
    return cast("Any", value).astype(
        np.dtype(dtype),
        order=order,
        casting=casting,
        subok=subok,
        copy=copy,
    )


@_evaluator("numpy.full")
def _eval_full(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate full using traced fill value input."""
    shape = attrs.get("shape")
    if shape is None:
        msg = "numpy.full requires shape attr"
        raise ValueError(msg)
    dtype = attrs.get("dtype")
    order = attrs.get("order", "C")
    fill_value: object = inputs[0]
    nested = as_numpy_nested(fill_value)
    if nested is not NotImplemented:
        fill_value = nested
        like: object | None = nested
    else:
        like = attrs.get("like")
    call_kwargs: dict[str, Any] = {"dtype": dtype, "order": order}
    if attrs.get("device") is not None:
        call_kwargs["device"] = attrs["device"]
    if like is not None:
        call_kwargs["like"] = like
    return np.full(shape, fill_value, **call_kwargs)


@_evaluator("numpy.clip")
def _eval_clip(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> np.ndarray:
    """Evaluate clip with traced/static bound variants."""
    if not inputs:
        msg = "numpy.clip requires at least one input value"
        raise ValueError(msg)

    min_is_input = bool(attrs.get("_advect_clip_min_is_input", False))
    max_is_input = bool(attrs.get("_advect_clip_max_is_input", False))

    cursor = 1
    if min_is_input:
        if len(inputs) <= cursor:
            msg = "numpy.clip expected traced a_min input but none was provided"
            raise ValueError(msg)
        a_min: Any = inputs[cursor]
        cursor += 1
    else:
        a_min = attrs.get("a_min")

    if max_is_input:
        if len(inputs) <= cursor:
            msg = "numpy.clip expected traced a_max input but none was provided"
            raise ValueError(msg)
        a_max: Any = inputs[cursor]
        cursor += 1
    else:
        a_max = attrs.get("a_max")

    if cursor != len(inputs):
        msg = (
            "numpy.clip received unexpected extra inputs during evaluation "
            f"(expected {cursor}, got {len(inputs)})"
        )
        raise ValueError(msg)

    return np.clip(inputs[0], a_min, a_max)


@_evaluator("numpy.einsum")
def _eval_einsum(inputs: tuple[np.ndarray, ...], attrs: dict[str, Any]) -> Any:
    """Evaluate einsum with traced operands."""
    subscripts = attrs.get("subscripts")
    if not isinstance(subscripts, str):
        msg = "numpy.einsum requires a subscripts string attr"
        raise TypeError(msg)
    optimize = attrs.get("optimize")
    call_kwargs: dict[str, Any] = {
        "optimize": optimize,
        "order": attrs.get("order", "K"),
        "casting": attrs.get("casting", "safe"),
    }
    if attrs.get("dtype") is not None:
        call_kwargs["dtype"] = attrs["dtype"]
    return _coerce_eval_result(
        np.einsum(
            subscripts,
            *inputs,
            **call_kwargs,
        )
    )


def _eval_unique(
    inputs: tuple[np.ndarray, ...],
    attrs: dict[str, Any],
    *,
    op: str,
    returns: tuple[bool, bool, bool],
) -> Any:
    """Evaluate one canonical numpy.unique result selection."""
    if len(inputs) != 1:
        msg = f"{op} evaluator expects exactly one input (got {len(inputs)})"
        raise ValueError(msg)
    axis_raw = attrs.get("axis")
    return_index, return_inverse, return_counts = returns
    return np.unique(
        inputs[0],
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=None if axis_raw is None else int(cast("Any", axis_raw)),
        equal_nan=bool(attrs.get("equal_nan", True)),
        sorted=bool(attrs.get("sorted", True)),
    )


for _unique_op, _unique_returns in (
    ("numpy.unique", (False, False, False)),
    ("numpy.unique_index", (True, False, False)),
    ("numpy.unique_inverse", (False, True, False)),
    ("numpy.unique_counts", (False, False, True)),
    ("numpy.unique_index_inverse", (True, True, False)),
    ("numpy.unique_index_counts", (True, False, True)),
    ("numpy.unique_inverse_counts", (False, True, True)),
    ("numpy.unique_index_inverse_counts", (True, True, True)),
):
    _RUNTIME.register_evaluator(
        _unique_op,
        cast("Any", partial(_eval_unique, op=_unique_op, returns=_unique_returns)),
    )


def evaluate_op(
    op: str,
    inputs: tuple[np.ndarray, ...],
    attrs: dict[str, Any],
) -> Any:
    """Evaluate one operation by delegating to the shared runtime router."""
    return _RUNTIME.evaluate_op(op, inputs, attrs)
