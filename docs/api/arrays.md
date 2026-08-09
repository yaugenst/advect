# Arrays

Advect preserves the provider selected by traced inputs. `array` and `asarray`
are explicit construction helpers for direct tracers and rectangular nested
tracer sequences; ordinary NumPy code should normally use NumPy constructors,
including `like=` dispatch.

For Array API inputs, a dynamic call selects the newest supported revision
common to every input. Mixed providers fail. Staging records one explicit
revision in the program rather than inferring a minimum from the operations
used. See the [NumPy frontend](numpy.md), [pytree utilities](pytree.md), and
[support catalog](support.md) for their separate public contracts.

::: advect.array

::: advect.asarray

::: advect.is_traced

::: advect.stop_gradient
