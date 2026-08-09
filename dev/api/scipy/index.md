# SciPy

`advect.scipy` is a bounded, NumPy-backed optional frontend installed with the `scipy` extra. Its public functions follow their documented SciPy signatures; direct calls to `scipy.*` are not intercepted. The generated [SciPy compatibility page](https://yaugenst.github.io/advect/dev/compatibility/scipy/index.md) is the exact callable and lifetime inventory.

Special functions and image filters may participate in dynamic transforms and in the staged lifetimes declared by the catalog. The root and GMRES factories are concrete, first-order dynamic callbacks for `implicit_root`; their opaque solver iterations are not staged.

- [Special functions](https://yaugenst.github.io/advect/dev/api/scipy/special/index.md)
- [Image filters](https://yaugenst.github.io/advect/dev/api/scipy/ndimage/index.md)
- [Solver callbacks](https://yaugenst.github.io/advect/dev/api/scipy/solvers/index.md)
