# Transforms

Most entries on this page are dynamic: an ordinary callable is traced from its
concrete values on each invocation. `grad` and `value_and_grad` also accept a
`StagedProgram`; in that form they compile and return another immutable staged
program with the same input signature and Array API revision. `vjp_program` is
the explicit staged pullback transform and is documented with
[staging](staging.md). The remaining transforms do not produce staged programs.

See the [API overview](index.md) for semantics shared across transforms and the
[target API](https://github.com/yaugenst/advect/blob/main/design/target-api.md)
for normative examples.

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
