# ADR: Implicit Differentiation and Scientific Frontends

**Date:** 2026-07-26
**Status:** Accepted
**Implementation status:** Implemented

## Context

The core reset deliberately removed broad SciPy and xarray packages. That was
the right deletion: the former integrations carried their own tracing and
policy machinery, while the replacement core was still settling its primitive,
pytree, and lifetime contracts.

The remaining user need is now clearer:

- scientific programs often obtain a state from a nonlinear equation and need
  derivatives of the converged state, not derivatives of solver iterations;
- a small set of SciPy special functions cannot be expressed faithfully from
  ordinary NumPy operations;
- xarray users need labels to survive differentiation without making labels
  part of numerical tracing.

These capabilities fit the reset only if they reuse the existing semantic
concepts. They must not add a solver IR, a second derivative engine, an xarray
graph format, or a broad compatibility namespace.

## Decision

### One implicit-root transform

`implicit_root` is the general implicit-differentiation surface:

```python
root = ad.implicit_root(
    residual,
    solve=nonlinear_solve,
    linear_solve=linear_solve,
)
solution = root(parameters, initial=initial_guess)
```

`residual(solution, parameters)` has the same pytree structure and leaf shapes
as `solution`. The initial guess selects a root but is nondifferentiable.
Returning from a solver callback certifies convergence; a failed or uncertain
solve raises `ImplicitSolveError`.

For `F(z, p) = 0`, the transform records one joint matrix-free `LinearMap` and
uses it as `A = dF/dz` and `B = dF/dp`. Its JVP solves
`A dz = -B dp`. Its transpose solves
`A* lambda = z_bar` and returns `-B* lambda`. The adjoint is Advect's real
adjoint, so the same rule covers complex and non-holomorphic residuals without
a separate Wirtinger implementation. The joint trace also keeps state and
parameter arrays under one provider contract.

The nonlinear solve is one atomic operation on an enclosing dynamic tape.
Solver iterations, factorization choices, and convergence logic do not become
Advect nodes. State and parameter pytrees are reconstructed at the callback
boundary rather than flattened into a public vector convention.

The first contract is dynamic. Opaque Python/SciPy callbacks are not closed
graph attributes, so abstract staging rejects the boundary before invoking a
solver. A future staged solver must be an explicit traceable program or stable
custom primitive; Advect does not silently unroll callbacks or serialize Python
closures.

The existing residual mechanism remains the extension point for a custom
solver primitive that retains an invocation-local factorization,
preconditioner, device allocation, or remote handle. Such opaque state remains
first-order and nonserializable. `implicit_root` itself does not introduce a
second residual store.

### Derivative rules receive the primal output

The unified primitive contract makes the already-computed output available to
authored derivative rules:

```python
@primitive.def_jvp
def jvp(output, primals, tangents, **static_attrs): ...

@primitive.def_transpose
def transpose(cotangent, primals, output, **static_attrs): ...
```

A residual transpose additionally receives `residual` after `output`. There is
one signature rather than output-aware and output-unaware authoring paths.
This lets expensive primitives reuse a solve or stable normalization instead
of recomputing it. An explicit custom transpose conservatively retains its
output until reverse; structurally transposed JVP-only primitives keep their
existing inferred lifetime requirements.

### SciPy is a bounded optional primitive frontend

The built-in `advect.scipy` module, enabled by the `scipy` extra, contributes:

- `gammaln`, `digamma`, `polygamma`, `erf`, `expit`, `ndtr`, and `logsumexp`;
- a shape-preserving `root_solver` callback factory over `scipy.optimize.root`;
- a matrix-free `gmres_solver` callback factory over
  `scipy.sparse.linalg.gmres`.

Each special function is one stable custom primitive with concrete SciPy
evaluation, abstract output evaluation, and traceable derivative rules. The
listed functions stage and serialize because their names, schemas, output
specifications, and static arguments are closed. Unsupported SciPy options
raise at the wrapper rather than passing a tracer into SciPy. NumPy is the
admitted provider. Each named callable implements its complete SciPy 1.18
contract: the unary functions support their ufunc controls and functionalized
`out=`, `polygamma` broadcasts array-valued orders and inputs, and
`logsumexp` supports differentiable weights and signed real or complex
results. Complex `digamma` uses a traceable trigamma rule rather than relying
on SciPy's real-only `polygamma` implementation. A loading process imports
`advect.scipy` before deserializing an artifact that uses these primitive
schemas.

The solver factories are intentionally different: they are concrete callbacks
for first-order dynamic `implicit_root`. `root_solver` and `gmres_solver`
accept NumPy arrays, NumPy scalars, and Python numeric scalars; preserve array
shape and scalar container category; and raise on nonconvergence. `root_solver`
follows SciPy's dtype promotion, while `gmres_solver` restores the inexact
right-hand-side dtype. A traceable callback may support higher-order dynamic
differentiation; staging requires explicit iterations or a closed custom
primitive. Complex SciPy solves use a doubled real representation so the
linear operator may be real-linear rather than complex-linear.

Base Advect neither imports nor depends on SciPy. There is no claim that
`advect.scipy` mirrors the full SciPy namespace.

The initial seven-function boundary in this section is extended by
[Bounded SciPy Filter Coverage](2026-08-01-scipy-filter-coverage.md); the
explicit-namespace and optional-dependency decisions remain unchanged.

### xarray is static metadata around differentiable leaves

The built-in `advect.xarray` module, enabled by the `xarray` extra, registers
`DataArray` and `Dataset` as custom pytrees when the user imports it.
`DataArray` subclasses inherit that registration; `Dataset` registration stays
exact until a concrete subclass integration requires broader behavior.

- floating- and complex-valued data buffers are differentiable children, while
  integer, boolean, string, and object buffers reject;
- dimensions, coordinates, names, variable order, and attributes are copied
  into equality-safe static metadata;
- unflattening reconstructs the original labeled container around transformed
  data leaves.

xarray owns alignment, named indexing, transposition, and reduction semantics.
Advect owns differentiation of the duck-array operations that xarray emits.
Coordinates and labels never become tangents or graph constants.

The first integration is dynamic. Durable staging still accepts only built-in
pytree nodes and `Static`; passing an xarray container to `stage` raises with a
rewrite to stage its raw array kernel and restore labels outside the staged
program. We do not create an xarray-specific artifact envelope.

`advect.xarray` is not registered as an array backend. Explicit import performs
the pytree registration, avoiding provider-discovery side effects and keeping
the dependency direction visible.

## Critical boundaries

- `implicit_root` requires a locally nonsingular state derivative at the chosen
  solution. Multiple roots, branch switching, and nonsmooth residuals are
  properties the user solver/model must resolve.
- `root_solver` and `gmres_solver` support NumPy arrays and NumPy or Python
  numeric scalars, not arbitrary pytrees or other providers. Custom callbacks
  retain the general pytree contract.
- Only the SciPy functions named here or by a later accepted extension decision
  are supported.
- xarray coordinates are static. Data-dependent coordinates, MultiIndex,
  groupby/resample topology, Dask execution, and broad interpolation or rolling
  support are outside the first contract.
- Opaque solver state never enters `RawArena` or `GraphStore`.
- No Rust changes or compiler subsystem are required.

## Consequences

Advect gains a useful differentiable-solver abstraction and two everyday
scientific integrations while preserving one derivative engine and one staged
artifact model. A solver can be arbitrarily sophisticated without making its
iterations part of reverse mode; special functions can participate in durable
programs; labeled containers can preserve user meaning around dynamic
gradients.

The price is an explicit lifetime boundary. External solvers and xarray
containers remain dynamic unless their numerical kernels are separately
staged. This is preferable to artifacts that silently freeze callbacks,
coordinates, or provider resources.
