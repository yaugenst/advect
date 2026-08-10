# Dynamic Control Flow and Mutation

Advect's dynamic transforms are *define by run*: every call executes the
ordinary Python function with the current inputs and differentiates only the
path that ran. There is no special conditional or loop API.

## Each call follows its own branch

A Python condition may depend on a traced value. Calling the same gradient
function with different values traces different branches:

```{.python .run}
import numpy as np

import advect as ad


def piecewise_loss(value):
    if np.sum(value) > 0:
        return np.sum(np.sin(value))
    return np.sum(value * value)


gradient = ad.grad(piecewise_loss)
positive = np.array([0.2, 0.4])
negative = np.array([-0.2, -0.4])

print("positive branch:", gradient(positive))
print("negative branch:", gradient(negative))
```

The first result differentiates `sin`; the second differentiates the square.
This is a pathwise derivative. Advect does not differentiate the discrete
decision that selected the branch, so derivatives can jump where the
condition changes.

## Loops execute until Python stops them

Iteration counts may also depend on concrete traced values. Auxiliary outputs
are useful when the algorithm should report what happened without
differentiating that report:

```{.python .run}
def settle_loss(value):
    state = value
    steps = 0
    while np.max(np.abs(state)) > 0.25:
        state = 0.5 * state
        steps += 1
    return np.sum(state * state), steps


gradient, steps = ad.grad(settle_loss, has_aux=True)(
    np.array([2.0, -1.0])
)
print(f"{steps} iterations:", gradient)
```

The loop is unrolled into this invocation's dynamic trace. A later call may
run a different number of iterations and gets a fresh trace. Helper functions
behave the same way: Advect records the supported numerical operations they
execute, not the Python call boundary.

## Mutation is local and functional

Advect records supported mutation syntax as immutable SSA updates. Inputs are
not implicitly writable: copy first, then update the owned local value.

```{.python .run}
def stencil_loss(field):
    updated = field.copy()
    laplacian = field[2:] - 2 * field[1:-1] + field[:-2]
    updated[1:-1] += 0.1 * laplacian
    return np.sum(updated * updated)


field = np.linspace(0.0, 1.0, 128)
dfield = ad.grad(stencil_loss)(field)
print(dfield[[0, len(dfield) // 2, -1]])
```

Basic indexed updates and a direct named basic-slice view are supported by the
NumPy frontend. Mutation of an input, advanced-index updates, and arbitrary
mutation through transformed views raise with a suggested rewrite; see
[Debugging](debugging.md) for the full taxonomy.

## Do not trace iterations that only find a root

Sometimes a loop is an implementation detail rather than the computation you
want to differentiate. `implicit_root` differentiates the equation defining a
converged state instead of recording the solver's iterations:

```{.python .run}
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
print(gradient)
```

The state and parameter may be built-in pytrees. `initial` selects a root but
is nondifferentiable. A successful solver return certifies convergence; failed
solves raise `ImplicitSolveError`. The derivative uses the matrix-free
state-Jacobian and its real adjoint, so complex and non-holomorphic residuals
follow the same convention.

Opaque callback-based roots are dynamic. `stage` rejects them before invoking
the solver. A traceable solver callback may support nested dynamic derivatives,
but Advect never selects that path by catching a provider failure.

With `advect[scipy]` installed, ready-made SciPy root and linear-solver
callbacks can supply the same interface:

```python
from advect.scipy.optimize import root_solver
from advect.scipy.sparse.linalg import gmres_solver

scipy_square_root = ad.implicit_root(
    lambda solution, parameters: solution**2 - parameters,
    solve=root_solver(),
    linear_solve=gmres_solver(rtol=1e-10, atol=1e-12),
)
print(scipy_square_root(parameters, initial=initial))
```

These callbacks turn SciPy nonconvergence into `ImplicitSolveError` and form a
first-order dynamic boundary. See the [SciPy API](../api/scipy/index.md) for
the solver, special-function, and image-filter surfaces.
