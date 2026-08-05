# Primitives

Authoring surface for custom operations: one implementation, abstract
evaluation, a JVP, and an optional explicit transpose. See the
[conformance testing ADR](https://github.com/yaugenst/advect/blob/main/design/decisions/2026-07-27-primitive-conformance-testing.md)
for the law battery every registered primitive must pass.

::: advect.primitive

::: advect.PrimitiveResult

::: advect.AbstractValue

::: advect.testing.check_primitive

## Composed functions

Use the whole-function checker first when a finite gradient looks suspicious.
It performs a directional finite-difference sweep and checks the reverse
gradient; `check_primitive` remains the narrower authoring contract for one
extension.

::: advect.testing.check_gradient
