# Transforms

Most transforms on this page run the callable and trace the path taken by its
concrete inputs. [`grad`](transforms.md#advect.grad) and
[`value_and_grad`](transforms.md#advect.value_and_grad) also accept a
[`StagedProgram`](staging.md#advect.StagedProgram); they return another staged
program with the same input signature and Array API revision.
[`vjp_program`](staging.md#advect.vjp_program) builds a reusable staged
pullback. The other transforms operate dynamically.

The [gradient](../tutorials/gradients.md),
[linear-map](../tutorials/linear-maps.md),
[higher-order](../tutorials/advanced-differentiation.md), and
[implicit-differentiation](../tutorials/implicit-differentiation.md) tutorials
connect these transforms through complete examples.

Library adapters may use `transform_state` for namespaced bookkeeping that
lives only while one concrete transform is tracing. Differentiable primitive
inputs and backward residuals must remain explicit.

::: advect.grad

::: advect.value_and_grad

::: advect.jvp

::: advect.vjp

::: advect.linearize

::: advect.jacobian

::: advect.hvp

::: advect.hessian

::: advect.hessian_diag

::: advect.checkpoint

::: advect.implicit_root

::: advect.Pullback

::: advect.LinearMap

::: advect.transform_state

::: advect.transform_states
