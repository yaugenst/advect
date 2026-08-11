# SciPy

`advect.scipy` provides differentiable, NumPy-backed versions of selected
[SciPy](https://docs.scipy.org/doc/scipy/reference/) functions. Install it with
the `advect[scipy]` extra and import the functions from `advect.scipy`; direct
calls to `scipy.*` continue to use SciPy itself. The
[Scientific Python tutorial](../../tutorials/scientific-python.md#use-the-differentiable-scipy-namespace)
shows the pattern in a complete derivative. The generated
[SciPy compatibility page](../../compatibility/scipy.md) lists the available
functions and where they can run.

Special functions and image processing may participate in dynamic transforms and
in the staged lifetimes declared by the catalog. The root and GMRES factories
are concrete, first-order dynamic callbacks for
[`implicit_root`](../transforms.md#advect.implicit_root); their opaque solver
iterations are not staged. See the
[implicit differentiation tutorial](../../tutorials/implicit-differentiation.md#use-the-scipy-callbacks)
for both callbacks together.

- [Special functions](special.md)
- [Image processing](ndimage.md)
- [Solver callbacks](solvers.md)
