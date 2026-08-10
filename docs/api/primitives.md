# Primitives

A primitive makes one closed implementation atomic to Advect. The decorator
returns a callable authoring handle that owns its abstract, JVP, and optional
transpose rules. Prefer a JVP because it supports forward mode and structural
transposition. An explicit transpose can instead provide reverse mode when no
JVP is available.

Concrete and abstract calls retain the implementation's named parameters and
pytrees. JVP and transpose rules operate on the dynamic array/scalar leaves in
one stable flattened order. Static arguments remain named configuration;
nondifferentiable arguments remain dynamic values but have no derivative
contribution.

The [custom primitive tutorial](../tutorials/primitives.md) shows the common
JVP-first workflow. Use the [testing utilities](testing.md) to validate both the
primitive and a representative composition.

## Define the operation

::: advect.primitive

## Attach rules to the returned handle

These methods belong to the object returned by `advect.primitive`:

::: advect.core._primitive.Primitive.def_abstract

::: advect.core._primitive.Primitive.def_jvp

::: advect.core._primitive.Primitive.def_transpose

## Exact residuals

Set `residual=True` only when reverse mode needs exact opaque data from the
forward invocation. Residual primitives require an explicit transpose and form
a first-order boundary; the object docstring below defines their lifetime and
cleanup contract.

::: advect.PrimitiveResult

## Abstract values

::: advect.AbstractValue
