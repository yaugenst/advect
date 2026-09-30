"""Concrete SciPy nonlinear-solver callbacks for implicit differentiation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize as _scipy_optimize

from advect.autodiff.api.implicit import ImplicitSolveError
from advect.scipy._containers import _as_concrete_array, _RealPacking

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


type ResidualFunction = Callable[[object], object]
type RootSolver = Callable[[ResidualFunction, object], object]


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
        packing = _RealPacking(
            initial,
            _as_concrete_array(initial, operation="root"),
            np.dtype(float),
            operation="root residual",
            requirement="return the solution shape",
        )
        packed_initial = packing.pack(packing.state)
        solve_kwargs: dict[str, object] = {}
        if method is not None:
            solve_kwargs["method"] = method
        if captured_options is not None:
            solve_kwargs["options"] = dict(captured_options)

        def packed_residual(flat: np.ndarray) -> np.ndarray:
            return packing.pack(residual(packing.unpack(flat)))

        result = _scipy_optimize.root(
            packed_residual,
            packed_initial,
            **solve_kwargs,
        )
        if not result.success:
            msg = f"SciPy root solve did not converge: {result.message}"
            raise ImplicitSolveError(msg)
        return packing.unpack(np.asarray(result.x))

    return solve


__all__ = ["RootSolver", "root_solver"]
