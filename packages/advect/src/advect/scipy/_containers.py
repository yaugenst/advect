"""Concrete NumPy values and container preservation for SciPy callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True, slots=True)
class _RealPacking:
    """Pack a real or complex state as the real vector a SciPy solver iterates on.

    Complex states store real parts before imaginary parts. ``operation``
    names the callback in errors, and ``requirement`` states its shape rule.
    """

    template: object
    state: np.ndarray
    dtype: np.dtype[Any]
    operation: str
    requirement: str

    def unpack(self, flat: np.ndarray) -> object:
        if np.iscomplexobj(self.state):
            flat = flat[: self.state.size] + 1j * flat[self.state.size :]
        return _restore_container(flat, self.template)

    def pack(self, value: object) -> np.ndarray:
        array = _as_concrete_array(value, operation=self.operation)
        if array.shape != self.state.shape:
            msg = (
                f"SciPy {self.operation} must {self.requirement} "
                f"{self.state.shape!r}, got {array.shape!r}"
            )
            raise ImplicitSolveError(msg)
        if np.iscomplexobj(self.state):
            parts = (array.real, array.imag)
            return np.concatenate([np.asarray(part, dtype=self.dtype).ravel() for part in parts])
        if np.iscomplexobj(array):
            msg = f"SciPy {self.operation} returned complex values for a real state"
            raise ImplicitSolveError(msg)
        return np.array(array, dtype=self.dtype).ravel()


__all__ = ["_RealPacking", "_as_concrete_array", "_restore_container"]
