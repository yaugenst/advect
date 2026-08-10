"""Concrete SciPy nonlinear-solver callbacks for implicit differentiation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize as _scipy_optimize

from advect.autodiff.api.implicit import ImplicitSolveError
from advect.scipy._containers import _restore_container

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


type ResidualFunction = Callable[[object], object]
type RootSolver = Callable[[ResidualFunction, object], object]


def _as_concrete_array(value: object, *, operation: str) -> np.ndarray:
    if not isinstance(value, (np.ndarray, np.generic)) and type(value) not in (
        bool,
        int,
        float,
        complex,
    ):
        msg = (
            f"SciPy {operation} adapters require concrete NumPy arrays or scalars; "
            f"got {type(value).__name__}. Convert provider arrays to NumPy before "
            "entering the solver boundary. These callbacks support first-order "
            "dynamic implicit differentiation only."
        )
        raise ImplicitSolveError(msg)
    try:
        return np.asarray(value)
    except (RuntimeError, TypeError, ValueError) as error:
        msg = (
            f"SciPy {operation} adapters require concrete NumPy values and support "
            "first-order dynamic implicit differentiation only. Use a traceable "
            "callback for higher-order dynamic differentiation; stage explicit "
            "iterations or define a closed custom primitive for durable programs."
        )
        raise ImplicitSolveError(msg) from error


def root_solver(
    *,
    method: str | None = None,
    options: Mapping[str, object] | None = None,
) -> RootSolver:
    """Build a SciPy nonlinear solver for ``advect.implicit_root``.

    Parameters
    ----------
    method
        Solver method forwarded to ``scipy.optimize.root``. ``None`` uses
        SciPy's default.
    options
        Method-specific options forwarded to SciPy. The mapping is copied when
        this solver is created.

    Returns
    -------
    RootSolver
        A callback accepting ``(residual, initial)``. It preserves the shape
        and scalar container category of ``initial`` and supports real and
        complex NumPy values.

    Raises
    ------
    ImplicitSolveError
        Raised by the returned callback when its values cross the concrete
        NumPy boundary incorrectly, the residual changes shape, or SciPy does
        not converge.

    Notes
    -----
    This is an opaque, first-order dynamic callback. Stage explicit traceable
    iterations or a closed custom primitive when a durable program is needed.
    """
    captured_options = None if options is None else dict(options)

    def solve(residual: ResidualFunction, initial: object) -> object:
        initial_array = _as_concrete_array(initial, operation="root")
        shape = initial_array.shape
        is_complex = np.iscomplexobj(initial_array)
        flat_size = initial_array.size

        def unpack(flat: np.ndarray) -> object:
            if not is_complex:
                unpacked = flat.reshape(shape)
            else:
                unpacked = (flat[:flat_size] + 1j * flat[flat_size:]).reshape(shape)
            return _restore_container(unpacked, initial)

        def pack(value: object) -> np.ndarray:
            array = _as_concrete_array(value, operation="root residual")
            if array.shape != shape:
                msg = (
                    "SciPy root residual must return the solution shape "
                    f"{shape!r}, got {array.shape!r}"
                )
                raise ImplicitSolveError(msg)
            if not is_complex:
                if np.iscomplexobj(array):
                    msg = "SciPy root residual returned complex values for a real state"
                    raise ImplicitSolveError(msg)
                return np.asarray(array, dtype=float).reshape(-1)
            return np.concatenate(
                (
                    np.asarray(array.real, dtype=float).reshape(-1),
                    np.asarray(array.imag, dtype=float).reshape(-1),
                )
            )

        packed_initial = pack(initial_array)
        solve_kwargs: dict[str, object] = {}
        if method is not None:
            solve_kwargs["method"] = method
        if captured_options is not None:
            solve_kwargs["options"] = dict(captured_options)

        def packed_residual(flat: np.ndarray) -> np.ndarray:
            return pack(residual(unpack(flat)))

        result = _scipy_optimize.root(
            packed_residual,
            packed_initial,
            **solve_kwargs,
        )
        if not result.success:
            msg = f"SciPy root solve did not converge: {result.message}"
            raise ImplicitSolveError(msg)
        return unpack(np.asarray(result.x))

    return solve


__all__ = ["RootSolver", "root_solver"]
