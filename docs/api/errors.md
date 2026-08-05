# Errors

Advect fails loudly at the offending line rather than returning a silently
wrong derivative. All exceptions derive from `AdvectError`.

## Diagnostic scope

::: advect.debug

::: advect.AdvectError

::: advect.TracingError

::: advect.EscapedTracerError

::: advect.MutationError

::: advect.StaleViewError

::: advect.NumericsError

::: advect.NoJVPError

::: advect.NoVJPError

::: advect.MissingPrimitiveRuleError

::: advect.ImplicitSolveError
