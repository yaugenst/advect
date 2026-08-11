# API Reference

Most work begins with functions imported directly from `advect`. Optional
NumPy, SciPy, xarray, and host-framework integrations live in their own
modules. The [tutorials](../tutorials/index.md) teach complete workflows; these
pages collect the installed signatures and exact API contracts.

| Public API | Responsibility |
| --- | --- |
| [Transforms](transforms.md) | Dynamic differentiation, higher-order transforms, checkpointing, and implicit roots |
| [Staging](staging.md) | Immutable programs, serialization, and staged differentiation |
| [Primitives](primitives.md) | Public custom-operation authoring |
| [Arrays](arrays.md) | Provider-preserving construction and tracer boundaries |
| [Pytrees](pytree.md) | Structured inputs, outputs, and custom node registration |
| [Testing utilities](testing.md) | Numerical checks for composed functions and custom primitives |
| [Support catalog](support.md) | Machine-readable operation and lifetime claims |
| [Errors](errors.md) | Public diagnostics and exception hierarchy |
| [NumPy frontend](numpy.md) | Explicit constructors and transparent NumPy namespace behavior |
| [SciPy](scipy/index.md) | Optional special functions, image processing, and solver callbacks |
| [xarray](xarray.md) | Optional labeled-container pytree registration |
| [Host autodiff interop](interop/index.md) | Optional JAX, PyTorch, and HIPS Autograd VJP bridges |

## Shared semantics

Dynamic [transforms](transforms.md) trace concrete values for each call and
preserve the Python control flow that ran. [`stage`](staging.md#advect.stage)
instead compiles one shape-and-dtype signature into an immutable graph; staged
and serialized support are therefore separate claims from dynamic support.

Selected real Python scalars are lifted to zero-dimensional `float64` arrays
and their derivative results return as Python scalars. Structured inputs and
outputs use [pytrees](pytree.md). Complex differentiation is real-linear; use
[`jvp`](transforms.md#advect.jvp), [`vjp`](transforms.md#advect.vjp), or
[`linearize`](transforms.md#advect.linearize) when the output is complex.

Importing `advect` is enough for NumPy and Array API code. SciPy, xarray, and
host-framework integrations use their own optional imports. The generated
[compatibility catalog](../compatibility/index.md) lists which calls can run
dynamically, be staged, be saved, and be differentiated.
