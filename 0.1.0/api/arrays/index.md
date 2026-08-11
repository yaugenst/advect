# Arrays

Advect preserves the provider selected by traced inputs. [`array`](https://yaugenst.github.io/advect/0.1.0/api/arrays/#advect.array) and [`asarray`](https://yaugenst.github.io/advect/0.1.0/api/arrays/#advect.asarray) are construction helpers for live tracers and rectangular nested tracer sequences. Ordinary NumPy code can keep using NumPy constructors, including `like=` dispatch. Use [`is_traced`](https://yaugenst.github.io/advect/0.1.0/api/arrays/#advect.is_traced) only when code genuinely needs to distinguish a live trace, and [`stop_gradient`](https://yaugenst.github.io/advect/0.1.0/api/arrays/#advect.stop_gradient) to remove one dependency explicitly.

For [Array API inputs](https://yaugenst.github.io/advect/0.1.0/compatibility/array-api/index.md), a dynamic call selects the newest supported revision common to every input. Mixed providers fail. Staging records one explicit revision in the program rather than inferring a minimum from the operations used. See the [NumPy frontend](https://yaugenst.github.io/advect/0.1.0/api/numpy/index.md), [pytree utilities](https://yaugenst.github.io/advect/0.1.0/api/pytree/index.md), and [support catalog](https://yaugenst.github.io/advect/0.1.0/api/support/index.md) for their separate public contracts.

## array

```python
array(
    obj: object,
    dtype: object | None = None,
    *,
    copy: bool = True,
) -> Any
```

Construct an owned array while preserving traced dependencies.

This is the explicit traced counterpart of the common `numpy.array(obj, dtype=..., copy=...)` forms. It intentionally does not mirror NumPy's complete constructor signature.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def total(value):
...     return np.sum(ad.array([value[0], value[1]]))
>>> ad.grad(total)(np.array([2.0, 3.0])).tolist()
[1.0, 1.0]
```

## asarray

```python
asarray(
    obj: object,
    dtype: object | None = None,
    *,
    copy: bool | None = None,
) -> Any
```

Construct an array without detaching Advect tracers.

Direct tracers and rectangular nested tracer sequences remain differentiable. This is the provider-neutral explicit alternative to NumPy's standard `numpy.asarray(..., like=tracer)` dispatch. Ordinary non-traced values retain their provider when they expose the pinned Array API namespace and otherwise use NumPy.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def total(value):
...     return np.sum(ad.asarray([value[0], value[1]]))
>>> ad.grad(total)(np.array([2.0, 3.0])).tolist()
[1.0, 1.0]
```

## is_traced

```python
is_traced(value: object) -> bool
```

Return whether `value` is an Advect tracer.

This check does not read the trace-time payload and remains safe for an escaped tracer. It tests the value itself rather than recursively searching an arbitrary object graph.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> ad.is_traced(np.array([1.0]))
False
>>> def contains_tracer(value):
...     assert ad.is_traced(value)
...     return np.sum(value**2)
>>> ad.grad(contains_tracer)(np.array([2.0])).tolist()
[4.0]
```

## stop_gradient

```python
stop_gradient(value: T) -> T
```

Return a concrete copy of traced leaves, explicitly stopping gradients.

Registered pytree structure is preserved. The operation is available only during concrete dynamic tracing; staging rejects it because an abstract value has no concrete primal to validate or serialize.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def loss(value):
...     return np.sum(value * ad.stop_gradient(value))
>>> ad.grad(loss)(np.array([2.0, 3.0])).tolist()
[2.0, 3.0]
```
