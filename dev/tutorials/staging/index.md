# Staging and Serialization

## Stage repeated work

`stage` traces abstract shape and dtype values, builds a durable graph, and runs Advect's fixed optimization pipeline once:

```python
import numpy as np

import advect as ad


def loss(x):
    return np.sum(np.sin(x) ** 2)


x = np.linspace(-0.5, 0.5, 8)

program = ad.stage(loss, x)
gradient_program = ad.grad(program)

value = program(x)
gradient = gradient_program(x)
print(gradient_program.optimization)
```

`gradient_program` is itself a `StagedProgram`. A warm call executes its prebound derivative graph; it does not record a dynamic tape or perform a reverse sweep. `program.graph` exposes the optimized nodes — the [architecture page](https://yaugenst.github.io/advect/dev/architecture/index.md) walks them, and the playground draws them live.

For reusable derivative stitching, compile a VJP program. Its cotangent has the same pytree, shape, and dtype contract as the primal output:

```python
field_program = ad.stage(lambda value: np.sin(value), x)
pullback_program = ad.vjp_program(field_program)

cotangent = np.linspace(1.0, 2.0, 8)
input_cotangent = pullback_program(x, cotangent=cotangent)
np.testing.assert_allclose(input_cotangent, cotangent * np.cos(x))
```

Unlike the one-shot `Pullback` returned by dynamic `vjp`, `pullback_program` contains no invocation-local tape. It is reusable, optimized once, and can be serialized directly.

A positional example supplies its shape, dtype, device, and scalar weakness, but its data is not available to the abstract trace. Use `specs=(ad.ArraySpec(...),)` when no representative value is available. A different signature is compiled into a different `StagedProgram`; the program object is never also a hidden multi-signature cache.

## Serialize a staged derivative

Staged primals, scalar gradients, and VJP programs share the same artifact format:

```python
import json

payload = json.dumps(gradient_program.to_dict(), sort_keys=True)
restored = ad.StagedProgram.from_dict(json.loads(payload))
np.testing.assert_allclose(restored(x), gradient_program(x))
```

A loaded program executes its one serialized signature. Any custom primitive referenced by the artifact must be linked by name and schema version in the loading process. In particular, import `advect.scipy` before calling `StagedProgram.from_dict()` for an artifact that contains an `advect.scipy` primitive.
