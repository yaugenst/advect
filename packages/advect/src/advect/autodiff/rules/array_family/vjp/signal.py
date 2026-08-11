"""Direct real adjoints for one-dimensional signal products."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.autodiff.rules.array_family._signal import native_signal_product


def _signal_crop(
    left_size: int,
    right_size: int,
    *,
    mode: str,
    correlate: bool,
) -> tuple[int, int]:
    full_size = left_size + right_size - 1
    if mode == "full":
        return 0, full_size

    shorter = min(left_size, right_size)
    if mode == "valid":
        return shorter - 1, max(left_size, right_size) - shorter + 1
    if mode == "same":
        start = shorter // 2 if correlate and left_size < right_size else (shorter - 1) // 2
        return start, max(left_size, right_size)
    msg = f"signal product mode must be full, same, or valid (got {mode!r})"
    raise ValueError(msg)


def _full_signal_cotangent(
    cotangent: xp.ndarray,
    *,
    left_size: int,
    right_size: int,
    mode: str,
    correlate: bool,
) -> xp.ndarray:
    start, output_size = _signal_crop(
        left_size,
        right_size,
        mode=mode,
        correlate=correlate,
    )
    expected_shape = (output_size,)
    if tuple(cotangent.shape) != expected_shape:
        msg = f"signal product expects cotangent shape {expected_shape}, got {cotangent.shape}"
        raise TypeError(msg)
    full_size = left_size + right_size - 1
    if output_size == full_size:
        return cotangent
    zero = xp.sum(cotangent[:1] * 0)
    parts: list[xp.ndarray] = []
    if start:
        parts.append(xp.broadcast_to(zero, (start,)))
    parts.append(cotangent)
    trailing = full_size - start - output_size
    if trailing:
        parts.append(xp.broadcast_to(zero, (trailing,)))
    return xp.concatenate(tuple(parts), axis=0)


def _vjp_signal_binary(
    ans: xp.ndarray,
    left: xp.ndarray,
    right: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    mode: str,
    correlate: bool,
    active_input_indices: tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    _ = ans, rest, attrs
    active = frozenset((0, 1)) if active_input_indices is None else frozenset(active_input_indices)
    if not active <= {0, 1}:
        msg = f"signal product active input indices are invalid: {sorted(active)}"
        raise ValueError(msg)

    full_cotangent = _full_signal_cotangent(
        g,
        left_size=int(left.shape[0]),
        right_size=int(right.shape[0]),
        mode=mode,
        correlate=correlate,
    )
    if not correlate:
        grad_left = (
            native_signal_product(
                full_cotangent,
                right,
                mode="valid",
                correlate=True,
            )
            if 0 in active
            else None
        )
        grad_right = (
            native_signal_product(
                full_cotangent,
                left,
                mode="valid",
                correlate=True,
            )
            if 1 in active
            else None
        )
        return grad_left, grad_right

    kernel = xp.conj(xp.flip(right))
    grad_left = (
        native_signal_product(
            full_cotangent,
            kernel,
            mode="valid",
            correlate=True,
        )
        if 0 in active
        else None
    )
    grad_right = None
    if 1 in active:
        kernel_gradient = native_signal_product(
            full_cotangent,
            left,
            mode="valid",
            correlate=True,
        )
        grad_right = xp.conj(xp.flip(kernel_gradient))
    return grad_left, grad_right


_vjp_convolve = partial(_vjp_signal_binary, mode="full", correlate=False)
_vjp_correlate = partial(_vjp_signal_binary, mode="valid", correlate=True)


for _selective_vjp in (_vjp_convolve, _vjp_correlate):
    cast("Any", _selective_vjp).__advect_vjp_for_input_indices__ = _selective_vjp
