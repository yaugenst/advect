# Staging and Serialization

Dynamic transforms trace each concrete call.
[`stage`](../api/staging.md#advect.stage) instead compiles one exact input
signature, optimizes the graph once, and returns an immutable
[`StagedProgram`](../api/staging.md#advect.StagedProgram) for repeated execution.

## Stage once, call many times

```{.python .run}
import numpy as np

import advect as ad


def loss(x):
    return np.sum(np.sin(x) ** 2)


sample = np.linspace(-0.5, 0.5, 8)
program = ad.stage(loss, sample)

print(f"sample loss: {program(sample):.6f}")
print(f"shifted loss: {program(sample + 0.1):.6f}")
print("Array API target:", program.array_api_version)
print(
    "optimized nodes:",
    f"{program.optimization.nodes_before} -> {program.optimization.nodes_after}",
)
```

The example fixes the positional pytree, shape, dtype, device, and Python
scalar category. Calls that change that contract fail instead of compiling a
hidden second program. The original Python function is not rerun during warm
calls.

## Declare a signature without example data

Use [`ArraySpec`](../api/staging.md#advect.ArraySpec) when no representative
value is available. The [`kw_specs`](../api/staging.md#advect.stage) argument
declares keyword inputs, and
[`StaticSpec`](../api/staging.md#advect.StaticSpec) snapshots a compile-time
Python value. Static values can control Python branches because they are known
while staging:

```{.python .run}
@ad.stage(
    specs=(ad.ArraySpec((4,), "float64"),),
    kw_specs={
        "scale": ad.ArraySpec((), "float64"),
        "center": ad.StaticSpec(True),
    },
)
def transform(x, *, scale, center):
    if center:
        x = x - np.mean(x)
    return scale * x


values = np.array([1.0, 2.0, 4.0, 5.0])
result = transform(
    values,
    scale=np.asarray(2.0),
    center=True,
)
print("staged transform:", result)
```

The static value is part of the signature: calling this program with
`center=False` is a contract mismatch. Data-dependent Python branches remain
dynamic-only because an abstract staged value has no data to test.

## Differentiate the program once

[`grad`](../api/transforms.md#advect.grad) and
[`value_and_grad`](../api/transforms.md#advect.value_and_grad) accept a staged
program and return another staged program.
[`vjp_program`](../api/staging.md#advect.vjp_program) adds an explicit cotangent
input for a reusable pullback:

```{.python .run}
value_and_gradient = ad.value_and_grad(program)
value, gradient = value_and_gradient(sample)

field_program = ad.stage(lambda x: np.sin(x), sample)
pullback_program = ad.vjp_program(field_program)
cotangent = np.linspace(1.0, 2.0, sample.size)
input_cotangent = pullback_program(sample, cotangent=cotangent)

np.testing.assert_allclose(input_cotangent, cotangent * np.cos(sample))
print(f"staged loss: {value:.6f}")
print("staged gradient:", gradient)
print("reusable pullback:", input_cotangent)
```

Warm derivative calls execute their prebuilt graphs. They do not create a
dynamic tape or run a reverse sweep. This is the reusable counterpart to the
one-shot pullback returned by dynamic
[`vjp`](../api/transforms.md#advect.vjp).

## Save and restore the program

```{.python .run}
import json

payload = json.dumps(value_and_gradient.to_dict(), sort_keys=True)
restored = ad.StagedProgram.from_dict(json.loads(payload))
restored_value, restored_gradient = restored(sample)

np.testing.assert_allclose(restored_gradient, gradient)
print(f"restored loss: {restored_value:.6f}")
print("serialized bytes:", len(payload.encode()))
```

The artifact contains the graph and its exact call contract, not Python code.
Captured arrays and static values are snapshotted at compile time. A custom
[primitive](../api/primitives.md) referenced by the graph must be imported or
registered under the same stable name before loading, with an implementation
that matches the saved program.

Provider-neutral functions written through `x.__array_namespace__()` can be
staged against an explicit
[Array API revision](../compatibility/array-api.md) and replayed by a compatible
provider. NumPy-authored functions retain the separate
[NumPy frontend](../api/numpy.md) contract.
Serialized formats may change before 1.0, so matching Advect versions are the
safest choice for saved programs.
