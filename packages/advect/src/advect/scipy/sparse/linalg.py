"""Concrete SciPy linear-solver callbacks for implicit differentiation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.sparse import linalg as _scipy_sparse_linalg

from advect.autodiff.api.implicit import ImplicitSolveError
from advect.scipy._containers import _as_concrete_array, _RealPacking, _restore_container

if TYPE_CHECKING:
    from collections.abc import Callable


type LinearOperator = Callable[[object], object]
type LinearSolver = Callable[[LinearOperator, object], object]


def gmres_solver(
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> LinearSolver:
    """Build a SciPy GMRES solver for implicit differentiation.

    Parameters
    ----------
    rtol
        Relative convergence tolerance forwarded to
        ``scipy.sparse.linalg.gmres``.
    atol
        Absolute convergence tolerance forwarded to SciPy.
    maxiter
        Maximum iteration count. ``None`` uses SciPy's default.

    Returns
    -------
    LinearSolver
        A callback accepting ``(operator, rhs)``. It preserves the shape and
        scalar container category of ``rhs`` and realifies complex
        real-linear operators before calling SciPy.

    Raises
    ------
    ValueError
        If either tolerance is negative or ``maxiter`` is not positive.
    ImplicitSolveError
        Raised by the returned callback when its values cross the concrete
        NumPy boundary incorrectly, the operator changes shape, or SciPy does
        not converge.

    Notes
    -----
    This is an opaque, first-order dynamic callback. It restores an inexact
    right-hand-side dtype after solving. Stage explicit traceable iterations
    or a closed custom primitive when a durable program is needed.
    """
    if rtol < 0 or atol < 0:
        msg = "GMRES tolerances must be non-negative"
        raise ValueError(msg)
    if maxiter is not None and maxiter < 1:
        msg = "GMRES maxiter must be positive"
        raise ValueError(msg)

    def solve(operator: LinearOperator, rhs: object) -> object:
        rhs_array = _as_concrete_array(rhs, operation="GMRES")
        inexact = np.issubdtype(rhs_array.dtype, np.inexact)
        packing = _RealPacking(
            rhs,
            rhs_array,
            np.finfo(rhs_array.dtype).dtype if inexact else np.dtype(np.float64),
            operation="GMRES operator",
            requirement="preserve the right-hand-side shape",
        )

        def matvec(flat: np.ndarray) -> np.ndarray:
            return packing.pack(operator(packing.unpack(flat)))

        packed_rhs = packing.pack(rhs_array)

        linear_operator_factory = cast("Any", _scipy_sparse_linalg.LinearOperator)
        linear_operator = linear_operator_factory(
            (packed_rhs.size, packed_rhs.size),
            matvec=matvec,
            dtype=packed_rhs.dtype,
        )
        solution, info = _scipy_sparse_linalg.gmres(
            linear_operator,
            packed_rhs,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
        )
        if info != 0:
            reason = (
                f"iteration limit reached after {info} iterations"
                if info > 0
                else f"solver breakdown (info={info})"
            )
            msg = f"SciPy GMRES did not converge: {reason}"
            raise ImplicitSolveError(msg)
        result = packing.unpack(solution)
        if inexact:
            return _restore_container(
                np.asarray(result, dtype=rhs_array.dtype),
                rhs,
            )
        return result

    return solve


__all__ = ["LinearOperator", "LinearSolver", "gmres_solver"]
