"""Concrete NumPy values and container preservation for SciPy callbacks."""

from __future__ import annotations

import numpy as np

from advect.autodiff.api.implicit import ImplicitSolveError


def _as_concrete_array(value: object, *, operation: str) -> np.ndarray:
    if not isinstance(value, (np.ndarray, np.generic)) and type(value) not in (
        bool,
        int,
        float,
        complex,
    ):
        msg = (
            f"SciPy {operation} requires concrete NumPy arrays or scalars; "
            f"got {type(value).__name__}. Convert provider arrays to NumPy before "
            "entering the solver boundary. This callback supports first-order "
            "dynamic implicit differentiation only."
        )
        raise ImplicitSolveError(msg)
    try:
        return np.asarray(value)
    except (RuntimeError, TypeError, ValueError) as error:
        msg = (
            f"SciPy {operation} requires concrete NumPy values and supports "
            "first-order dynamic implicit differentiation only. Use a traceable "
            "callback for higher-order dynamic differentiation; stage explicit "
            "iterations or define a closed custom primitive for durable programs."
        )
        raise ImplicitSolveError(msg) from error


def _restore_container(value: object, template: object) -> object:
    """Restore Python scalar, NumPy scalar, or ndarray shape from ``template``."""
    restored = np.asarray(value).reshape(np.asarray(template).shape)
    if type(template) in (bool, int, float, complex):
        return restored.item()
    if isinstance(template, np.generic):
        return restored[()]
    return restored


__all__ = ["_as_concrete_array", "_restore_container"]
