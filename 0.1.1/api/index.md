# API Reference

Most work begins with functions imported directly from `advect`. Optional NumPy, SciPy, xarray, and host-framework integrations live in their own modules. The [tutorials](https://yaugenst.github.io/advect/0.1.1/tutorials/index.md) teach complete workflows; these pages collect the installed signatures and exact API contracts.

| Public API                                                                            | Responsibility                                                                      |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [Transforms](https://yaugenst.github.io/advect/0.1.1/api/transforms/index.md)         | Dynamic differentiation, higher-order transforms, checkpointing, and implicit roots |
| [Staging](https://yaugenst.github.io/advect/0.1.1/api/staging/index.md)               | Immutable programs, serialization, and staged differentiation                       |
| [Primitives](https://yaugenst.github.io/advect/0.1.1/api/primitives/index.md)         | Public custom-operation authoring                                                   |
| [Arrays](https://yaugenst.github.io/advect/0.1.1/api/arrays/index.md)                 | Provider-preserving construction and tracer boundaries                              |
| [Pytrees](https://yaugenst.github.io/advect/0.1.1/api/pytree/index.md)                | Structured inputs, outputs, and custom node registration                            |
| [Testing utilities](https://yaugenst.github.io/advect/0.1.1/api/testing/index.md)     | Numerical checks for composed functions and custom primitives                       |
| [Support catalog](https://yaugenst.github.io/advect/0.1.1/api/support/index.md)       | Machine-readable operation and lifetime claims                                      |
| [Errors](https://yaugenst.github.io/advect/0.1.1/api/errors/index.md)                 | Public diagnostics and exception hierarchy                                          |
| [NumPy frontend](https://yaugenst.github.io/advect/0.1.1/api/numpy/index.md)          | Explicit constructors and transparent NumPy namespace behavior                      |
| [SciPy](https://yaugenst.github.io/advect/0.1.1/api/scipy/index.md)                   | Optional special functions, image processing, and solver callbacks                  |
| [xarray](https://yaugenst.github.io/advect/0.1.1/api/xarray/index.md)                 | Optional labeled-container pytree registration                                      |
| [Host autodiff interop](https://yaugenst.github.io/advect/0.1.1/api/interop/index.md) | Optional JAX, PyTorch, and HIPS Autograd VJP bridges                                |

## Shared semantics

Dynamic [transforms](https://yaugenst.github.io/advect/0.1.1/api/transforms/index.md) trace concrete values for each call and preserve the Python control flow that ran. [`stage`](https://yaugenst.github.io/advect/0.1.1/api/staging/#advect.stage) instead compiles one shape-and-dtype signature into an immutable graph; staged and serialized support are therefore separate claims from dynamic support.

Selected real Python scalars are lifted to zero-dimensional `float64` arrays and their derivative results return as Python scalars. Structured inputs and outputs use [pytrees](https://yaugenst.github.io/advect/0.1.1/api/pytree/index.md). Complex differentiation is real-linear; use [`jvp`](https://yaugenst.github.io/advect/0.1.1/api/transforms/#advect.jvp), [`vjp`](https://yaugenst.github.io/advect/0.1.1/api/transforms/#advect.vjp), or [`linearize`](https://yaugenst.github.io/advect/0.1.1/api/transforms/#advect.linearize) when the output is complex.

Importing `advect` is enough for NumPy and Array API code. SciPy, xarray, and host-framework integrations use their own optional imports. The generated [compatibility catalog](https://yaugenst.github.io/advect/0.1.1/compatibility/index.md) lists which calls can run dynamically, be staged, be saved, and be differentiated.
