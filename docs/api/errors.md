# Errors

When Advect cannot differentiate a call safely, it raises an error at the
offending line instead of returning a plausible but wrong derivative. All
exceptions derive from
[`AdvectError`](errors.md#advect.AdvectError).
The [troubleshooting guide](../tutorials/debugging.md) shows how to use
[`debug`](errors.md#advect.debug) and act on the common tracing, numerical, and
staging failures.

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
