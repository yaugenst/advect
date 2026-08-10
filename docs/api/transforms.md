# Transforms

Most entries on this page are dynamic: an ordinary callable is traced from its
concrete values on each invocation. [`grad`](transforms.md#advect.grad) and
[`value_and_grad`](transforms.md#advect.value_and_grad) also accept a
[`StagedProgram`](staging.md#advect.StagedProgram); in that form they compile and
return another immutable staged program with the same input signature and Array
API revision. [`vjp_program`](staging.md#advect.vjp_program) is the explicit
staged pullback transform. The remaining transforms do not produce staged
programs.

The [gradient](../tutorials/gradients.md),
[linear-map](../tutorials/linear-maps.md),
[higher-order](../tutorials/advanced-differentiation.md), and
[implicit-differentiation](../tutorials/implicit-differentiation.md) tutorials
connect these transforms through complete examples.

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
