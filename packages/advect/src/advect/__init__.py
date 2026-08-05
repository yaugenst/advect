"""Advect: a small, extensible automatic-differentiation core."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from advect import pytree
from advect._array import array, asarray, is_traced, stop_gradient
from advect._autodiff_exports import AUTODIFF_EXPORT_MODULES
from advect.core import (
    AbstractValue,
    AdvectError,
    ArraySpec,
    ConstantRecord,
    EscapedTracerError,
    MissingPrimitiveRuleError,
    MutationError,
    NoJVPError,
    NoVJPError,
    NumericsError,
    OptimizationPass,
    OptimizationReport,
    PrimitiveResult,
    StagedProgram,
    StagedTrace,
    StaleViewError,
    StaticSpec,
    TracedNode,
    TracingError,
    debug,
    primitive,
    stage,
)
from advect.support import support_catalog

# NumPy is the required base frontend, so its handlers are deterministic
# process state rather than an ambient plugin side effect. The Array API
# compatibility fallback follows the same rule: array-api-compat is a base
# dependency, so configuration here stays independent of install state.
import_module("advect.numpy")
import_module("advect._array_api_compat")

try:
    __version__ = version("advect")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public transform
    """Load automatic-differentiation transforms on first use."""
    if name not in AUTODIFF_EXPORT_MODULES:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    autodiff = import_module("advect.autodiff")
    value = getattr(autodiff, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the deliberately small public surface."""
    return sorted(set(globals()).union(AUTODIFF_EXPORT_MODULES))


__all__ = [
    "AbstractValue",
    "AdvectError",
    "ArraySpec",
    "ConstantRecord",
    "EscapedTracerError",
    "MissingPrimitiveRuleError",
    "MutationError",
    "NoJVPError",
    "NoVJPError",
    "NumericsError",
    "OptimizationPass",
    "OptimizationReport",
    "PrimitiveResult",
    "StagedProgram",
    "StagedTrace",
    "StaleViewError",
    "StaticSpec",
    "TracedNode",
    "TracingError",
    "__version__",
    "array",
    "asarray",
    "debug",
    "is_traced",
    "primitive",
    "pytree",
    "stage",
    "stop_gradient",
    "support_catalog",
]
__all__ += list(AUTODIFF_EXPORT_MODULES)
