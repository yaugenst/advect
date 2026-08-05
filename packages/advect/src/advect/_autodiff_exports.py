"""Single source of truth for lazy public autodiff exports."""

from __future__ import annotations

AUTODIFF_EXPORT_MODULES = {
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

__all__ = ["AUTODIFF_EXPORT_MODULES"]
