# Tutorials

Start with one NumPy gradient, then build the model a piece at a time: how
Advect follows Python, how derivatives act as linear maps, and when to use
higher-order, implicit, or staged differentiation. These six pages form the
main path through the library.

Most examples can run directly in the browser. Each page is one Python session,
so pressing `[ run ]` also runs any earlier runnable blocks on that page.

| Step | Tutorial | What it teaches |
| ---: | --- | --- |
| 1 | [Gradients and pytrees](gradients.md) | [`grad`](../api/transforms.md#advect.grad), [`value_and_grad`](../api/transforms.md#advect.value_and_grad), auxiliary results, argument selection, and structured parameters |
| 2 | [Dynamic control flow and mutation](control-flow.md) | Data-dependent branches and loops, helper functions, and owned local updates |
| 3 | [JVPs, VJPs, and linear maps](linear-maps.md) | [`jvp`](../api/transforms.md#advect.jvp), [`vjp`](../api/transforms.md#advect.vjp), reusable [`LinearMap`](../api/transforms.md#advect.LinearMap) objects, Jacobians, and complex derivatives |
| 4 | [Higher-order differentiation](advanced-differentiation.md) | Nested derivatives, [`hvp`](../api/transforms.md#advect.hvp), Hessians, and [`checkpoint`](../api/transforms.md#advect.checkpoint) |
| 5 | [Implicit differentiation](implicit-differentiation.md) | Differentiate a converged equation with [`implicit_root`](../api/transforms.md#advect.implicit_root) |
| 6 | [Staging and serialization](staging.md) | Exact signatures, reusable derivative programs, and [`StagedProgram`](../api/staging.md#advect.StagedProgram) artifacts |

## Connect and extend

- [Arrays and Scientific Python](scientific-python.md) explains providers,
  differentiable SciPy functions, and xarray labels.
- [Host autodiff frameworks](host-frameworks.md) wraps an Advect function as one
  PyTorch, JAX, or HIPS Autograd operation.
- [Custom primitives](primitives.md) adds an atomic operation with its abstract
  and derivative rules.
- [Troubleshooting](debugging.md) turns common tracing, numerical, and staging
  failures into concrete rewrites.
