"""Shared execution helpers for NumPy's one-dimensional signal operations."""

from __future__ import annotations

from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import xp
from advect.core._abstract_helpers import ABSTRACT_NAMESPACE_NAME


def _is_advect_abstract(value: object) -> bool:
    namespace_function = getattr(value, "__array_namespace__", None)
    if not callable(namespace_function):
        return False
    namespace = namespace_function()
    return getattr(namespace, "__name__", None) == ABSTRACT_NAMESPACE_NAME


def native_signal_product(
    left: xp.ndarray,
    right: xp.ndarray,
    *,
    mode: str,
    correlate: bool,
) -> xp.ndarray:
    """Call the provider operation, including NumPy-specific abstract replay."""
    name = "correlate" if correlate else "convolve"
    try:
        operation = getattr(xp, name)
    except AttributeError:
        if not (_is_advect_abstract(left) or _is_advect_abstract(right)):
            raise
        # The abstract namespace intentionally exposes only the Array API.
        # NumPy protocol dispatch still records this NumPy-only extension op.
        import numpy as np  # noqa: PLC0415

        operation = getattr(np, name)
    return cast("xp.ndarray", cast("Any", operation)(left, right, mode=mode))


__all__ = ["native_signal_product"]
