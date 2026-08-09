"""Concrete SciPy linear-solver callbacks for implicit differentiation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.sparse import linalg as _scipy_sparse_linalg

from advect.autodiff.api.implicit import ImplicitSolveError
from advect.scipy._containers import _restore_container

if TYPE_CHECKING:
    from collections.abc import Callable


type LinearOperator = Callable[[object], object]
type LinearSolver = Callable[[LinearOperator, object], object]


def _as_concrete_array(value: object) -> np.ndarray:
    if not isinstance(value, (np.ndarray, np.generic)) and type(value) not in (
        bool,
        int,
        float,
        complex,
    ):
        msg = (
            "SciPy GMRES requires concrete NumPy arrays or scalars; "
            f"got {type(value).__name__}. Convert provider arrays to NumPy before "
            "entering the solver boundary. This callback supports first-order "
            "dynamic implicit differentiation only."
        )
        raise ImplicitSolveError(msg)
    try:
        return np.asarray(value)
    except (RuntimeError, TypeError, ValueError) as error:
        msg = (
            "SciPy GMRES requires concrete NumPy values and supports first-order "
            "dynamic implicit differentiation only. Use a traceable linear solver "
            "for higher-order dynamic differentiation; stage explicit iterations or "
            "define a closed custom primitive for durable programs."
        )
        raise ImplicitSolveError(msg) from error


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
        rhs_array = _as_concrete_array(rhs)
        shape = rhs_array.shape
        size = rhs_array.size
        is_complex = np.iscomplexobj(rhs_array)
        if np.issubdtype(rhs_array.dtype, np.complexfloating):
            packed_dtype = np.empty((), dtype=rhs_array.dtype).real.dtype
        elif np.issubdtype(rhs_array.dtype, np.floating):
            packed_dtype = rhs_array.dtype
        else:
            packed_dtype = np.dtype(np.float64)

        def unpack(flat: np.ndarray) -> object:
            if not is_complex:
                unpacked = flat.reshape(shape)
            else:
                unpacked = (flat[:size] + 1j * flat[size:]).reshape(shape)
            return _restore_container(unpacked, rhs)

        def pack(value: object) -> np.ndarray:
            result = _as_concrete_array(value)
            if result.shape != shape:
                msg = (
                    "SciPy GMRES operator must preserve the right-hand-side shape "
                    f"{shape!r}, got {result.shape!r}"
                )
                raise ImplicitSolveError(msg)
            if not is_complex:
                if np.iscomplexobj(result):
                    msg = "SciPy GMRES operator returned complex values for a real state"
                    raise ImplicitSolveError(msg)
                return np.asarray(result, dtype=packed_dtype).reshape(-1).copy()
            return np.concatenate(
                (
                    np.asarray(result.real, dtype=packed_dtype).reshape(-1),
                    np.asarray(result.imag, dtype=packed_dtype).reshape(-1),
                )
            )

        def matvec(flat: np.ndarray) -> np.ndarray:
            return pack(operator(unpack(flat)))

        packed_rhs = pack(rhs_array)

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
        result = unpack(solution)
        if np.issubdtype(rhs_array.dtype, np.inexact):
            return _restore_container(
                np.asarray(result, dtype=rhs_array.dtype),
                rhs,
            )
        return result

    return solve


__all__ = ["LinearOperator", "LinearSolver", "gmres_solver"]
