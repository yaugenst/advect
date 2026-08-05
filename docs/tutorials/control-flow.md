# Control Flow and Mutation

Dynamic transforms trace the actual Python call, so loops, branches, and
helper functions need no special forms. The boundaries appear where code
mutates arrays or hides iteration behind an opaque callback — this page covers
both.

## Mutation is local and functional

Advect records supported mutation syntax as immutable SSA updates. Inputs are
not implicitly writable: copy first, then update the owned local value.

```{.python .run}
import numpy as np

import advect as ad


def stencil_loss(field):
    updated = field.copy()
    laplacian = field[2:] - 2 * field[1:-1] + field[:-2]
    updated[1:-1] += 0.1 * laplacian
    return np.sum(updated * updated)


field = np.linspace(0.0, 1.0, 128)
dfield = ad.grad(stencil_loss)(field)
```

Basic indexed updates and a direct named basic-slice view are supported by the
NumPy frontend. Mutation of an input, advanced-index updates, and arbitrary
mutation through transformed views raise with a suggested rewrite; see
[Debugging](debugging.md) for the full taxonomy.

## Differentiate a converged solve

`implicit_root` differentiates the equation defining a converged state rather
than recording solver iterations:

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
```

The state and parameter may be built-in pytrees. `initial` selects a root but
is nondifferentiable. A successful solver return certifies convergence; failed
solves raise `ImplicitSolveError`. The derivative uses the matrix-free
state-Jacobian and its real adjoint, so complex and non-holomorphic residuals
follow the same convention.

Opaque callback-based roots are dynamic. `stage` rejects them before invoking
the solver. A traceable solver callback may support nested dynamic derivatives,
but Advect never selects that path by catching a provider failure.

## Use SciPy callbacks, special functions, and image filters

The built-in `advect.scipy` module contains a deliberately small function
surface when `advect[scipy]` is installed:

```python
from advect.scipy import ndimage, special
from advect.scipy.optimize import root_solver
from advect.scipy.sparse.linalg import gmres_solver


def likelihood_loss(value):
    normalized = special.logsumexp(value, axis=-1, keepdims=True)
    return np.sum(special.gammaln(value) + special.ndtr(value) - normalized)


gradient = ad.grad(likelihood_loss)(np.array([[0.7, 1.4, 2.8]]))

smoothed_gradient = ad.grad(
    lambda value: np.sum(ndimage.gaussian_filter(value, 1.2, mode="reflect"))
)(np.array([[0.7, 1.4, 2.8]]))

scipy_square_root = ad.implicit_root(
    lambda solution, parameters: solution**2 - parameters,
    solve=root_solver(),
    linear_solve=gmres_solver(rtol=1e-10, atol=1e-12),
)
```

The supported special surface covers gamma/error/normal-distribution functions,
logistic transforms, `logsumexp`, and softmax forms. The `ndimage` surface
covers Gaussian and uniform filters, convolution/correlation, derivative
filters, extrema and rank filters, and greyscale morphology. These NumPy-backed
calls preserve their SciPy 1.18 signatures, output forms, modes, axes, and
neighborhood configuration, and may be staged and serialized.
`root_solver` and `gmres_solver` accept NumPy arrays, NumPy scalars, and Python
numeric scalars. They preserve array shape and scalar container category, turn
SciPy nonconvergence into `ImplicitSolveError`, and form a first-order dynamic
boundary. `root_solver` follows SciPy's dtype promotion; `gmres_solver`
restores the inexact right-hand-side dtype. A traceable callback can support
higher-order dynamic differentiation. Staging requires explicit traceable
iterations or a closed custom primitive, not an opaque callback.
