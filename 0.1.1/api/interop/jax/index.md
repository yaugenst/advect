# JAX

The [shared bridge contract](https://yaugenst.github.io/advect/0.1.1/api/interop/#shared-contract) and [host-framework tutorial](https://yaugenst.github.io/advect/0.1.1/tutorials/host-frameworks/index.md) explain what crosses this boundary.

## wrap

```python
wrap(
    function: Callable[..., object],
    *,
    has_aux: bool = False,
    result_shape_dtypes: Any | None = None,
) -> Callable[..., object]
```

Wrap a pure NumPy-backed callable as a first-order JAX operation.

Floating or complex JAX array pytrees may be passed positionally or by keyword; every leaf is differentiable. Static configuration should be closed over by `function`. With `has_aux=True`, `function` returns `(value, aux)` and only `value` participates in the Advect VJP. Eager calls infer their outputs by executing `function` directly. JIT compilation and abstract shape evaluation require `result_shape_dtypes`, a JAX pytree of objects with `shape` and `dtype` attributes, normally `jax.ShapeDtypeStruct` objects. Reverse mode replays `function` to build and consume an Advect pullback, so it must be pure and deterministic.

Without a result specification, the adapter executes eagerly and observes the actual output structure, shapes, and dtypes:

```python
import jax
import jax.numpy as jnp
import numpy as np

from advect.interop.jax import wrap


def numpy_energy(value):
    return np.sum(np.sin(value) ** 2)


energy = wrap(numpy_energy)
x = jnp.linspace(0, 1, 8)
value, gradient = jax.value_and_grad(energy)(x)
print(value, gradient)
```

For auxiliary outputs, pass `has_aux=True` to both the bridge and the [`jax.value_and_grad`](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html) transformation. Only the first result participates in the Advect [VJP](https://yaugenst.github.io/advect/0.1.1/api/transforms/#advect.vjp):

```python
def energy_and_metrics(value):
    return numpy_energy(value), {"iterations": np.asarray(3, dtype=np.int32)}


energy_with_metrics = wrap(energy_and_metrics, has_aux=True)
(value, metrics), gradient = jax.value_and_grad(
    energy_with_metrics,
    has_aux=True,
)(x)
print(value, metrics, gradient)
```

Auxiliary leaves may be JAX-compatible floating, complex, integer, or boolean arrays and scalars. They remain nondifferentiable even when floating. Opaque Python objects such as strings cannot cross this JAX operation boundary.

JAX needs output shapes and dtypes during JIT compilation and abstract shape evaluation. Supply that contract to use [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) or [`jax.eval_shape`](https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html):

```python
compiled_energy = wrap(
    energy_and_metrics,
    has_aux=True,
    result_shape_dtypes=(
        jax.ShapeDtypeStruct((), np.float32),
        {"iterations": jax.ShapeDtypeStruct((), np.int32)},
    ),
)
(value, metrics), gradient = jax.jit(
    jax.value_and_grad(compiled_energy, has_aux=True)
)(x)
print(value, metrics, gradient)
```

If staging starts without `result_shape_dtypes`, the bridge raises a `TypeError` that explains how to enable the callback path. It does not cache a contract from an earlier eager call. A supplied specification must match the callable's output structure, shapes, and dtypes exactly; the bridge does not cast callback results. The callback path uses a [pure host callback](https://docs.jax.dev/en/latest/external-callbacks.html), so the callable must be deterministic and free of externally visible effects. Reverse mode replays it once at the saved primal values to construct and immediately consume an Advect pullback. [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) remains unsupported; an output specification does not add batching semantics.
