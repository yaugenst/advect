# Solver Callbacks

The [implicit differentiation tutorial](https://yaugenst.github.io/advect/0.1.0/tutorials/implicit-differentiation/#use-the-scipy-callbacks) shows both callbacks together. They plug into [`implicit_root`](https://yaugenst.github.io/advect/0.1.0/api/transforms/#advect.implicit_root) and keep opaque solver iterations outside the derivative trace.

## Nonlinear solver callback

The callback follows the contract of [`scipy.optimize.root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html).

## optimize

Concrete SciPy nonlinear-solver callbacks for implicit differentiation.

### root_solver

```python
root_solver(
    *,
    method: str | None = None,
    options: Mapping[str, object] | None = None,
) -> RootSolver
```

Build a SciPy nonlinear solver for `advect.implicit_root`.

Parameters:

- **`method`** (`str | None`, default: `None` ) – Solver method forwarded to scipy.optimize.root. None uses SciPy's default.
- **`options`** (`Mapping[str, object] | None`, default: `None` ) – Method-specific options forwarded to SciPy. The mapping is copied when this solver is created.

Returns:

- `RootSolver` – A callback accepting (residual, initial). It preserves the shape and scalar container category of initial and supports real and complex NumPy values.

Raises:

- `ImplicitSolveError` – Raised by the returned callback when its values cross the concrete NumPy boundary incorrectly, the residual changes shape, or SciPy does not converge.

Notes

This is an opaque, first-order dynamic callback. Stage explicit traceable iterations or a closed custom primitive when a durable program is needed.

## Linear solver callback

The callback follows the contract of [`scipy.sparse.linalg.gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html).

## linalg

Concrete SciPy linear-solver callbacks for implicit differentiation.

### gmres_solver

```python
gmres_solver(
    *,
    rtol: float = 1e-05,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> LinearSolver
```

Build a SciPy GMRES solver for implicit differentiation.

Parameters:

- **`rtol`** (`float`, default: `1e-05` ) – Relative convergence tolerance forwarded to scipy.sparse.linalg.gmres.
- **`atol`** (`float`, default: `0.0` ) – Absolute convergence tolerance forwarded to SciPy.
- **`maxiter`** (`int | None`, default: `None` ) – Maximum iteration count. None uses SciPy's default.

Returns:

- `LinearSolver` – A callback accepting (operator, rhs). It preserves the shape and scalar container category of rhs and realifies complex real-linear operators before calling SciPy.

Raises:

- `ValueError` – If either tolerance is negative or maxiter is not positive.
- `ImplicitSolveError` – Raised by the returned callback when its values cross the concrete NumPy boundary incorrectly, the operator changes shape, or SciPy does not converge.

Notes

This is an opaque, first-order dynamic callback. It restores an inexact right-hand-side dtype after solving. Stage explicit traceable iterations or a closed custom primitive when a durable program is needed.
