# API Reference

Advect's public API has a small root namespace and a few explicit modules. The [tutorials](https://yaugenst.github.io/advect/dev/tutorials/index.md) teach complete workflows; these pages render the installed signatures and docstrings.

| Public surface                                                                      | Responsibility                                                                      |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [Transforms](https://yaugenst.github.io/advect/dev/api/transforms/index.md)         | Dynamic differentiation, higher-order transforms, checkpointing, and implicit roots |
| [Staging](https://yaugenst.github.io/advect/dev/api/staging/index.md)               | Immutable programs, serialization, and staged differentiation                       |
| [Primitives](https://yaugenst.github.io/advect/dev/api/primitives/index.md)         | Public custom-operation authoring                                                   |
| [Arrays](https://yaugenst.github.io/advect/dev/api/arrays/index.md)                 | Provider-preserving construction and tracer boundaries                              |
| [Pytrees](https://yaugenst.github.io/advect/dev/api/pytree/index.md)                | Structured inputs, outputs, and custom node registration                            |
| [Testing utilities](https://yaugenst.github.io/advect/dev/api/testing/index.md)     | Numerical checks for composed functions and custom primitives                       |
| [Support catalog](https://yaugenst.github.io/advect/dev/api/support/index.md)       | Machine-readable operation and lifetime claims                                      |
| [Errors](https://yaugenst.github.io/advect/dev/api/errors/index.md)                 | Public diagnostics and exception hierarchy                                          |
| [NumPy frontend](https://yaugenst.github.io/advect/dev/api/numpy/index.md)          | Explicit constructors and transparent NumPy namespace behavior                      |
| [SciPy](https://yaugenst.github.io/advect/dev/api/scipy/index.md)                   | Optional special functions, image filters, and solver callbacks                     |
| [xarray](https://yaugenst.github.io/advect/dev/api/xarray/index.md)                 | Optional labeled-container pytree registration                                      |
| [Host autodiff interop](https://yaugenst.github.io/advect/dev/api/interop/index.md) | Optional JAX, PyTorch, and HIPS Autograd VJP bridges                                |

## Shared semantics

Dynamic transforms trace concrete values for each call and preserve the Python control flow that ran. `stage` instead compiles one shape-and-dtype signature into an immutable graph; staged and serialized support are therefore separate claims from dynamic support.

Selected real Python scalars are lifted to zero-dimensional `float64` arrays and their derivative results return as Python scalars. Structured inputs and outputs use pytrees. Complex differentiation is real-linear; use `jvp`, `vjp`, or `linearize` when the output is complex.

The root `advect` import installs the required NumPy frontend and Array API compatibility bridge. SciPy, xarray, and host-framework integrations remain explicit optional imports. Consult the generated [compatibility catalog](https://yaugenst.github.io/advect/dev/compatibility/index.md) for exact callable and lifetime coverage; the presence of an internal registered operation is not a public support claim.
