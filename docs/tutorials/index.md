# Tutorials

Start with a gradient, then build one mental model at a time: dynamic Python,
local linear maps, higher-order derivatives, implicit equations, and reusable
staged programs. These six pages form the core path.

Most examples can run directly in the browser. Each page is one Python session,
so pressing `[ run ]` also runs any earlier runnable blocks on that page.

| Step | Tutorial | What it teaches |
| ---: | --- | --- |
| 1 | [Gradients and pytrees](gradients.md) | `grad`, `value_and_grad`, auxiliary results, argument selection, and structured parameters |
| 2 | [Dynamic control flow and mutation](control-flow.md) | Data-dependent branches and loops, helper functions, and owned local updates |
| 3 | [JVPs, VJPs, and linear maps](linear-maps.md) | Forward and reverse products, reusable linearization, Jacobians, and complex derivatives |
| 4 | [Higher-order differentiation](advanced-differentiation.md) | Nested derivatives, HVPs, Hessians, and checkpointing |
| 5 | [Implicit differentiation](implicit-differentiation.md) | Differentiate a converged equation instead of solver iterations |
| 6 | [Staging and serialization](staging.md) | Exact signatures, reusable derivative programs, and saved artifacts |

## Connect and extend

- [Arrays and Scientific Python](scientific-python.md) explains providers,
  differentiable SciPy functions, and xarray labels.
- [Host autodiff frameworks](host-frameworks.md) wraps an Advect function as one
  PyTorch, JAX, or HIPS Autograd operation.
- [Custom primitives](primitives.md) adds an atomic operation with its abstract
  and derivative rules.
- [Troubleshooting](debugging.md) turns common tracing, numerical, and staging
  failures into concrete rewrites.
