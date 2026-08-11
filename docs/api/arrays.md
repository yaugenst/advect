# Arrays

Advect preserves the provider selected by traced inputs.
[`array`](arrays.md#advect.array) and [`asarray`](arrays.md#advect.asarray) are
construction helpers for live tracers and rectangular nested tracer sequences.
Ordinary NumPy code can keep using NumPy constructors, including `like=`
dispatch. Use
[`is_traced`](arrays.md#advect.is_traced) only when code genuinely needs to
distinguish a live trace, and
[`stop_gradient`](arrays.md#advect.stop_gradient) to remove one dependency
explicitly.

For [Array API inputs](../compatibility/array-api.md), a dynamic call selects
the newest supported revision common to every input. Mixed providers fail.
Staging records one explicit revision in the program rather than inferring a
minimum from the operations used. See the [NumPy frontend](numpy.md),
[pytree utilities](pytree.md), and [support catalog](support.md) for their
separate public contracts.

::: advect.array

::: advect.asarray

::: advect.is_traced

::: advect.stop_gradient
