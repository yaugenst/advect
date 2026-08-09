"""Lazy autodiff API for dynamic and staged transforms.

Staged reverse transforms include `grad`, `value_and_grad`, and `vjp_program`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from advect._autodiff_exports import AUTODIFF_EXPORT_MODULES


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public transform
    """Load the public autodiff API only when a transform is requested."""
    if name not in AUTODIFF_EXPORT_MODULES:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module("advect.autodiff.api"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(AUTODIFF_EXPORT_MODULES))


__all__ = list(AUTODIFF_EXPORT_MODULES)
