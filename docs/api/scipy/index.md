# SciPy

`advect.scipy` is a bounded, NumPy-backed optional frontend installed with the
`scipy` extra. Its public functions follow their documented SciPy signatures;
direct calls to `scipy.*` are not intercepted. The generated
[SciPy compatibility page](../../compatibility/scipy.md) is the exact callable
and lifetime inventory.

Special functions and image filters may participate in dynamic transforms and
in the staged lifetimes declared by the catalog. The root and GMRES factories
are concrete, first-order dynamic callbacks for `implicit_root`; their opaque
solver iterations are not staged.

- [Special functions](special.md)
- [Image filters](ndimage.md)
- [Solver callbacks](solvers.md)
