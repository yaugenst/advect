"""Public and internal API layers for autodiff."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from advect._autodiff_exports import AUTODIFF_EXPORT_MODULES


def __getattr__(name: str) -> Any:
    module = AUTODIFF_EXPORT_MODULES.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(f"advect.autodiff.api.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


__all__ = list(AUTODIFF_EXPORT_MODULES)
