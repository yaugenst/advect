# Implicit Differentiation

Many algorithms iterate only to find a state that satisfies an equation. Differentiating every iteration then ties the derivative to the particular solver rather than the equation being solved. [`implicit_root`](https://yaugenst.github.io/advect/0.2.0/api/transforms/#advect.implicit_root) differentiates the converged equation instead.

## Differentiate the equation, not the iterations

Suppose the solution is defined by `solution**2 - parameters == 0`. The nonlinear callback finds a root; the linear callback solves with the [state Jacobian](https://yaugenst.github.io/advect/0.2.0/tutorials/linear-maps/#materialize-the-jacobian-only-when-useful) needed by the implicit derivative.

```python
import numpy as np

import advect as ad


def newton_sqrt(residual, initial):
    value = initial.copy()
    for _ in range(12):
        value = value - residual(value) / (2 * value)
    return value


def diagonal_solve(operator, right_hand_side):
    diagonal = operator(np.ones_like(right_hand_side))
    return right_hand_side / diagonal


square_root = ad.implicit_root(
    lambda solution, parameters: solution**2 - parameters,
    solve=newton_sqrt,
    linear_solve=diagonal_solve,
)

parameters = np.array([1.0, 4.0, 9.0])
initial = np.ones_like(parameters)
gradient = ad.grad(
    lambda values: np.sum(square_root(values, initial=initial))
)(parameters)

np.testing.assert_allclose(gradient, 0.5 / np.sqrt(parameters))
print("root:", square_root(parameters, initial=initial))
print("gradient:", gradient)
```

The iteration count does not appear on the outer trace. `initial` selects a root but is nondifferentiable; the solution and parameters may be built-in [pytrees](https://yaugenst.github.io/advect/0.2.0/api/pytree/index.md). Advect trusts a successful solver return as convergence, so a callback must raise [`ImplicitSolveError`](https://yaugenst.github.io/advect/0.2.0/api/errors/#advect.ImplicitSolveError) when it fails.

Reverse mode applies the real adjoint of the state Jacobian. Pass a separate [`transpose_solve`](https://yaugenst.github.io/advect/0.2.0/api/transforms/#advect.implicit_root) when the same linear solver cannot handle that adjoint; otherwise Advect reuses `linear_solve` with the adjoint operator.

## Use the SciPy callbacks

The `advect[scipy]` extra provides ready-made NumPy [`root_solver`](https://yaugenst.github.io/advect/0.2.0/api/scipy/solvers/#advect.scipy.optimize.root_solver) and [`gmres_solver`](https://yaugenst.github.io/advect/0.2.0/api/scipy/solvers/#advect.scipy.sparse.linalg.gmres_solver) callbacks:

```python
import numpy as np

import advect as ad
from advect.scipy.optimize import root_solver
from advect.scipy.sparse.linalg import gmres_solver


square_root = ad.implicit_root(
    lambda solution, parameters: solution**2 - parameters,
    solve=root_solver(),
    linear_solve=gmres_solver(rtol=1e-10, atol=1e-12),
)

parameters = np.array([1.0, 4.0, 9.0])
initial = np.ones_like(parameters)
gradient = ad.grad(
    lambda values: np.sum(square_root(values, initial=initial))
)(parameters)
print("SciPy-backed gradient:", gradient)
# SciPy-backed gradient: [0.5        0.25       0.16666667]
```

The bundled callbacks accept one NumPy array or scalar and form a first-order dynamic boundary. A custom traceable callback can support nested dynamic derivatives when every operation it executes is itself traceable.

`implicit_root` is dynamic because opaque Python solver callbacks have no serializable graph form. If the iterations themselves are the computation, trace them normally; if a solver must be staged, write its iterations as [stageable array code](https://yaugenst.github.io/advect/0.2.0/tutorials/staging/index.md) or package the closed behavior as a [custom primitive](https://yaugenst.github.io/advect/0.2.0/tutorials/primitives/index.md).
