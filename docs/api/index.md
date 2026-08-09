# API Reference

Advect's public API has a small root namespace and a few explicit modules. The
[tutorials](../tutorials/index.md) teach complete workflows; these pages render
the installed signatures and docstrings.

| Public surface | Responsibility |
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
| [SciPy](scipy/index.md) | Optional special functions, image filters, and solver callbacks |
| [xarray](xarray.md) | Optional labeled-container pytree registration |
| [Host autodiff interop](interop/index.md) | Optional JAX, PyTorch, and HIPS Autograd VJP bridges |

## Shared semantics

Dynamic transforms trace concrete values for each call and preserve the Python
control flow that ran. `stage` instead compiles one shape-and-dtype signature
into an immutable graph; staged and serialized support are therefore separate
claims from dynamic support.

Selected real Python scalars are lifted to zero-dimensional `float64` arrays
and their derivative results return as Python scalars. Structured inputs and
outputs use pytrees. Complex differentiation is real-linear; use `jvp`, `vjp`,
or `linearize` when the output is complex.

The root `advect` import installs the required NumPy frontend and Array API
compatibility bridge. SciPy, xarray, and host-framework integrations remain
explicit optional imports. Consult the generated
[compatibility catalog](../compatibility/index.md) for exact callable and
lifetime coverage; the presence of an internal registered operation is not a
public support claim.
