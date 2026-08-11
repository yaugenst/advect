"""NumPy-native automatic differentiation and reusable staged programs.

Start with ``grad``, ``value_and_grad``, ``jvp``, or ``vjp`` on an ordinary
NumPy-backed callable. ``stage`` captures a reusable ``StagedProgram`` that
can be serialized with ``to_dict()`` and restored with
``StagedProgram.from_dict()``.

Extension and integration entry points are deliberately explicit:

* ``primitive`` defines an atomic operation. Prefer a traceable
  ``@handle.def_jvp`` rule, add ``@handle.def_abstract`` for staging, and
  validate it with ``from advect.testing import check_primitive``.
* ``from advect.interop.jax import wrap`` (or ``torch`` / ``autograd``) puts
  an Advect callable inside a first-order host-framework transform.
* ``import advect.xarray`` registers labeled containers for dynamic
  transforms; stage raw ``.data`` arrays instead.
* ``support_catalog()`` reports supported public forms, while ``debug()``
  adds source locations to tracing errors.

Use ``help(advect.grad)``, ``help(advect.stage)``, or
``help(advect.primitive)`` for the complete local contracts.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from advect import pytree
from advect._array import array, asarray, is_traced, stop_gradient
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

_AUTODIFF_EXPORT_MODULES = {
    "ImplicitSolveError": "implicit",
    "LinearMap": "forward",
    "Pullback": "reverse",
    "checkpoint": "checkpoint",
    "grad": "reverse",
    "hessian": "higher_order",
    "hessian_diag": "higher_order",
    "hvp": "higher_order",
    "implicit_root": "implicit",
    "jacobian": "forward",
    "jvp": "forward",
    "linearize": "forward",
    "value_and_grad": "reverse",
    "vjp": "reverse",
    "vjp_program": "reverse",
}


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public transform
    """Load automatic-differentiation transforms on first use."""
    module = _AUTODIFF_EXPORT_MODULES.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(f"advect.autodiff.api.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the deliberately small public surface."""
    return sorted(set(globals()).union(_AUTODIFF_EXPORT_MODULES))


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
__all__ += list(_AUTODIFF_EXPORT_MODULES)
