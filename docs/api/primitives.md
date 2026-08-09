# Primitives

A primitive makes one closed implementation atomic to Advect. The decorator
returns the callable authoring handle; that same handle owns abstract staging,
an optional JVP, and any explicit transpose. There is no separately imported
or constructed primitive class. A JVP is the preferred derivative rule because
it supports forward mode and structural transposition. An explicit transpose
may instead supply reverse mode without a JVP; a traceable non-residual rule can
participate in reverse-over-reverse differentiation. Residual-bearing
transposes are the first-order-only boundary.

Concrete and abstract calls retain the implementation's named parameters and
pytrees. JVP and transpose rules operate on the dynamic array/scalar leaves in
one stable flattened order. Static arguments remain named configuration;
nondifferentiable arguments remain dynamic values but have no derivative
contribution.

Use the [testing utilities](testing.md) to validate a custom primitive and a
representative composition. The [custom primitive tutorial](../tutorials/primitives.md)
shows the complete public authoring workflow.

## Define the operation

::: advect.primitive

## Attach rules to the returned handle

The following methods are used on the object returned by `advect.primitive`;
their source location is private so application code has only one public
authoring entry point.

::: advect.core._primitive.Primitive.def_abstract

::: advect.core._primitive.Primitive.def_jvp

::: advect.core._primitive.Primitive.def_transpose

## Exact residuals

Set `residual=True` only when reverse mode needs opaque data from the exact
forward invocation. A direct call, JVP, or plain staged replay releases it
before returning; a reusable linear map retains it until the map is closed.
Residual primitives require an explicit transpose and are first-order
boundaries: their primal can be staged, but a staged or higher-order derivative
cannot retain the opaque residual. They may omit a JVP when only reverse mode is
supported.

::: advect.PrimitiveResult

## Abstract values

::: advect.AbstractValue
