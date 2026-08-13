# Primitives

[`primitive`](primitives.md#advect.primitive) makes one function appear as a
single operation to Advect. The decorator returns the callable with methods for
adding its abstract, JVP, and optional transpose rules. Prefer a
[JVP](transforms.md#advect.jvp) because it supports forward mode and structural
transposition. An explicit transpose can instead provide
[reverse mode](transforms.md#advect.vjp) when no JVP is available.

Concrete and abstract calls retain the implementation's named parameters and
pytrees. JVP and transpose rules operate on the dynamic array/scalar leaves in
one stable flattened order. Static arguments remain named configuration;
nondifferentiable arguments remain dynamic values but have no derivative
contribution.

Output arity is fixed by default. Set `variable_output_arity=True` only when a
concrete invocation determines its number of output leaves. Advect records that
invocation's output pytree for differentiation; variable-arity primitives cannot
be staged.

The [custom primitive tutorial](../tutorials/primitives.md) shows the common
JVP-first workflow. Use
[`check_primitive`](testing.md#advect.testing.check_primitive) and
[`check_gradient`](testing.md#advect.testing.check_gradient) to validate both
the primitive and a representative composition.

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
