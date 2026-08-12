# SciPy

`advect.scipy` provides differentiable, NumPy-backed versions of selected [SciPy](https://docs.scipy.org/doc/scipy/reference/) functions. Install it with the `advect[scipy]` extra and import the functions from `advect.scipy`; direct calls to `scipy.*` continue to use SciPy itself. The [Scientific Python tutorial](https://yaugenst.github.io/advect/0.1.1/tutorials/scientific-python/#use-the-differentiable-scipy-namespace) shows the pattern in a complete derivative. The generated [SciPy compatibility page](https://yaugenst.github.io/advect/0.1.1/compatibility/scipy/index.md) lists the available functions and where they can run.

Special functions and image processing may participate in dynamic transforms and in the staged lifetimes declared by the catalog. The root and GMRES factories are concrete, first-order dynamic callbacks for [`implicit_root`](https://yaugenst.github.io/advect/0.1.1/api/transforms/#advect.implicit_root); their opaque solver iterations are not staged. See the [implicit differentiation tutorial](https://yaugenst.github.io/advect/0.1.1/tutorials/implicit-differentiation/#use-the-scipy-callbacks) for both callbacks together.

- [Special functions](https://yaugenst.github.io/advect/0.1.1/api/scipy/special/index.md)
- [Image processing](https://yaugenst.github.io/advect/0.1.1/api/scipy/ndimage/index.md)
- [Solver callbacks](https://yaugenst.github.io/advect/0.1.1/api/scipy/solvers/index.md)
