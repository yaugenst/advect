# Solver Callbacks

The [implicit differentiation tutorial](../../tutorials/implicit-differentiation.md#use-the-scipy-callbacks)
shows both callbacks together. They plug into
[`implicit_root`](../transforms.md#advect.implicit_root) and keep opaque solver
iterations outside the derivative trace.

## Nonlinear solver callback

The callback follows the contract of
[`scipy.optimize.root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html).

::: advect.scipy.optimize

## Linear solver callback

The callback follows the contract of
[`scipy.sparse.linalg.gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html).

::: advect.scipy.sparse.linalg
