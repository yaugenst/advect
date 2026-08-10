# SciPy

`advect.scipy` is a bounded, NumPy-backed optional frontend installed with the
`advect[scipy]` extra. Its public functions follow their documented
[SciPy](https://docs.scipy.org/doc/scipy/reference/) signatures; direct calls to
`scipy.*` are not intercepted. The
[Scientific Python tutorial](../../tutorials/scientific-python.md#use-the-differentiable-scipy-namespace)
shows the import boundary in a complete derivative. The generated
[SciPy compatibility page](../../compatibility/scipy.md) is the exact callable
and lifetime inventory.

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
