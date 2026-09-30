# Staging and Serialization

Dynamic transforms follow every call through Python. When the same computation will run repeatedly with matching inputs, [`stage`](https://yaugenst.github.io/advect/0.3.0/api/staging/#advect.stage) compiles that input signature, optimizes the graph once, and returns an immutable [`StagedProgram`](https://yaugenst.github.io/advect/0.3.0/api/staging/#advect.StagedProgram) for repeated execution.

## Stage once, call many times

```python
import numpy as np

import advect as ad


def loss(x):
    return np.sum(np.sin(x) ** 2)


sample = np.linspace(-0.5, 0.5, 8)
program = ad.stage(loss, sample)

print(f"sample loss: {program(sample):.6f}")
print(f"shifted loss: {program(sample + 0.1):.6f}")
```

The example fixes the positional pytree, shape, dtype, device, and Python scalar category. Calls that change that contract fail instead of compiling a hidden second program. The original Python function is not rerun during warm calls. The optimizer merges repeated computations, so two outputs of one call may share storage even where eager NumPy would return separate arrays; copy an output before writing to it in place if another output must stay unchanged.

## Declare a signature without example data

Use [`ArraySpec`](https://yaugenst.github.io/advect/0.3.0/api/staging/#advect.ArraySpec) when no representative value is available. The [`kw_specs`](https://yaugenst.github.io/advect/0.3.0/api/staging/#advect.stage) argument declares keyword inputs, and [`StaticSpec`](https://yaugenst.github.io/advect/0.3.0/api/staging/#advect.StaticSpec) snapshots a compile-time Python value. Static values can control Python branches because they are known while staging:

```python
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

The static value is part of the signature: calling this program with `center=False` is a contract mismatch. Data-dependent Python branches remain dynamic-only because an abstract staged value has no data to test.

Dtypes are known while staging, so they can select a branch too. Inside a staged function, `x.dtype` is the dtype object that your array provider's arrays report, and a test such as `x.dtype == np.float32` takes the branch that eager and dynamic code take. The same holds for the dtypes of derived values and, with `xp = x.__array_namespace__()`, for `xp.float32`, `xp.result_type`, and `xp.finfo(x).dtype`. Staging from examples presents their provider's dtype objects, while `specs=` alone stages against NumPy, so pass examples to see another provider's dtypes. A program derived from a staged or restored program, such as `grad(program)` or `vjp_program(program)`, records no provider, so a custom primitive rule that its derivation runs sees NumPy dtypes. Staged values have `bool` or one of the numeric dtypes `int8` to `int64`, `uint8` to `uint64`, `float16` to `float64`, `complex64`, and `complex128`; staging rejects any other dtype, such as `object`, bytes, or `float128`, with a `TypeError` that names it.

## Differentiate the program once

[`grad`](https://yaugenst.github.io/advect/0.3.0/api/transforms/#advect.grad) and [`value_and_grad`](https://yaugenst.github.io/advect/0.3.0/api/transforms/#advect.value_and_grad) accept a staged program and return another staged program. [`vjp_program`](https://yaugenst.github.io/advect/0.3.0/api/staging/#advect.vjp_program) adds an explicit cotangent input for a reusable pullback:

```python
value_and_gradient = ad.value_and_grad(program)
value, gradient = value_and_gradient(sample)

field_program = ad.stage(np.sin, sample)
pullback_program = ad.vjp_program(field_program)
cotangent = np.linspace(1.0, 2.0, sample.size)
input_cotangent = pullback_program(sample, cotangent=cotangent)

np.testing.assert_allclose(input_cotangent, cotangent * np.cos(sample))
print(f"staged loss: {value:.6f}")
print("staged gradient:", gradient)
print("reusable pullback:", input_cotangent)
```

Warm derivative calls execute their prebuilt graphs. They do not create a dynamic tape or run a reverse sweep. This is the reusable counterpart to the one-shot pullback returned by dynamic [`vjp`](https://yaugenst.github.io/advect/0.3.0/api/transforms/#advect.vjp).

## Save and restore the program

```python
import json

payload = json.dumps(value_and_gradient.to_dict(), sort_keys=True)
restored = ad.StagedProgram.from_dict(json.loads(payload))
restored_value, restored_gradient = restored(sample)

np.testing.assert_allclose(restored_gradient, gradient)
print(f"restored loss: {restored_value:.6f}")
print("serialized bytes:", len(payload.encode()))
```

The artifact contains the graph and its exact call contract, not Python code. Captured arrays and static values are snapshotted at compile time. A captured array that the program returns is read-only on NumPy and a fresh copy on providers without read-only arrays, so writing to it cannot change later calls. A custom [primitive](https://yaugenst.github.io/advect/0.3.0/api/primitives/index.md) referenced by the graph must be imported or registered under the same stable name before loading, with an implementation that matches the saved program.

Provider-neutral functions written through `x.__array_namespace__()` can be staged against an explicit [Array API revision](https://yaugenst.github.io/advect/0.3.0/compatibility/array-api/index.md) and replayed by a compatible provider. NumPy-authored functions retain the separate [NumPy frontend](https://yaugenst.github.io/advect/0.3.0/api/numpy/index.md) contract. Staged from another provider's examples, they lower to the same graph as from NumPy examples unless they branch on `x.dtype`, which is that provider's dtype object; a NumPy call that receives such a dtype object raises NumPy's `TypeError` while staging. Save and load a program with the same Advect version.

Advect 0.3.0 uses saved-program schema version 3. Programs saved with schema version 2 by earlier releases must be staged again from their original callable and signature, then saved with the new release. Changing the version number in the JSON is insufficient: version 3 changes the meaning of weak scalar metadata to preserve NumPy's scalar promotion rules. Schema versions describe the saved format and are independent of package release numbers.
