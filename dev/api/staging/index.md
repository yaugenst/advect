# Staging

[`stage`](https://yaugenst.github.io/advect/dev/api/staging/#advect.stage) compiles one input signature into an immutable, optimized program that can be called repeatedly, differentiated, saved, and loaded. The [staging tutorial](https://yaugenst.github.io/advect/dev/tutorials/staging/index.md) develops the complete workflow.

Advect stages against one explicit [Array API contract](https://data-apis.org/array-api/latest/). The supported targets are `2022.12`, `2023.12`, and `2024.12`:

```python
import numpy as np

import advect as ad


def energy(x):
    return np.sum(x**2)


example = np.array([1.0, 2.0])
program = ad.stage(energy, example, array_api_version="2023.12")
print(program.array_api_version)
# 2023.12
```

When examples are supplied without a target, [`stage`](https://yaugenst.github.io/advect/dev/api/staging/#advect.stage) selects the newest revision every example provider can serve. With [`specs=`](https://yaugenst.github.io/advect/dev/api/staging/#advect.stage) and no concrete provider, it defaults to `2024.12`. The selected target is stored in the graph, preserved by [`grad`](https://yaugenst.github.io/advect/dev/api/transforms/#advect.grad) and [`vjp_program`](https://yaugenst.github.io/advect/dev/api/staging/#advect.vjp_program), and enforced before runtime graph evaluation. Choosing an older target is the deliberate way to build a more portable artifact; Advect does not infer a minimum revision from the operations used by the function.

## ArraySpec

```python
ArraySpec(
    shape: tuple[int, ...],
    dtype: Any,
    device: str | None = None,
    weak: bool = False,
)
```

Shape/dtype contract for one staged array input or result.

Examples:

```pycon
>>> import advect as ad
>>> spec = ad.ArraySpec((2, 3), "float64")
>>> spec.shape, spec.dtype
((2, 3), 'float64')
```

## StaticSpec

```python
StaticSpec(value: Any)
```

An explicit compile-time Python value in a staged call signature.

Examples:

```pycon
>>> import advect as ad
>>> ad.StaticSpec("sum").value
'sum'
```

## stage

```python
stage(
    function: Callable[..., Any] | None = None,
    *examples: Any,
    specs: tuple[Any, ...] | None = None,
    kw_specs: dict[str, Any] | None = None,
    array_api_version: str | None = None,
) -> (
    StagedProgram
    | Callable[[Callable[..., Any]], StagedProgram]
)
```

Compile one callable signature into an immutable staged program.

Use the direct form with a callable, or omit `function` to create a decorator. Applying the decorator compiles the function and replaces it with a `StagedProgram`. Compilation traces the Python callable once with abstract values; later calls execute the graph without running or retracing the Python callable.

Declare the positional signature in exactly one of two ways: pass concrete `examples` to infer its array leaves, or pass `specs` containing `ArraySpec` and `StaticSpec` leaves. Keyword arguments have no example form and are declared with `kw_specs` in either case.

Parameters:

- **`function`** (`Callable[..., Any] | None`, default: `None` ) – Callable to compile. If omitted, return a decorator that compiles the callable it receives.
- **`*examples`** (`Any`, default: `()` ) – Concrete positional arguments whose pytree structure, shapes, dtypes, devices, and Python-scalar categories define the compiled signature. Wrap a non-array compile-time leaf in StaticSpec. Mutually exclusive with specs.
- **`specs`** (`tuple[Any, ...] | None`, default: `None` ) – Positional argument specification tree. Every leaf must be an ArraySpec or StaticSpec. Mutually exclusive with examples.
- **`kw_specs`** (`dict[str, Any] | None`, default: `None` ) – Mapping from keyword argument names to specification trees whose leaves are ArraySpec or StaticSpec. The mapping is combined with the positional signature declared by examples or specs.
- **`array_api_version`** (`str | None`, default: `None` ) – Array API revision to compile and store in the graph. With concrete examples and no explicit revision, Advect selects the newest supported revision served by their common array provider. With specs alone, it selects Advect's latest supported revision. An explicit revision must be supported by Advect and by the provider of every array example.

Returns:

- `StagedProgram or callable` – A fully compiled, single-signature StagedProgram when function is supplied; otherwise, a decorator that returns such a program. The program snapshots static inputs and captured constants, can be serialized, and never grows a polymorphic cache or retraces.

Raises:

- `TypeError` – If neither examples nor specs is supplied, if both are supplied, if an example is neither array-like nor a supported Python scalar nor wrapped in StaticSpec, if a specification contains another leaf type, or if the concrete array examples cannot use one common provider at the selected Array API revision.
- `ValueError` – If array_api_version is not a supported revision, or if abstract tracing finds incompatible shapes, dtypes, or operation semantics.

Notes

A returned program accepts only its compiled call pytree and leaf contract. At execution time, a changed call structure, non-array leaf, or static value raises `TypeError`; an incompatible array shape, dtype, device, or Python-scalar category raises `ValueError`.

Examples:

Infer a direct-call signature from a concrete array:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def add_one(x):
...     return x + 1
>>> example_input = np.array([1.0, 2.0])
>>> program = ad.stage(add_one, example_input)
>>> program(np.array([3.0, 4.0])).tolist()
[4.0, 5.0]
```

The decorator form uses explicit positional and keyword specifications:

```pycon
>>> @ad.stage(
...     specs=(ad.ArraySpec((2,), "float32"),),
...     kw_specs={"scale": ad.ArraySpec((), "float32")},
... )
... def scale(x, *, scale):
...     return x * scale
>>> scale(
...     np.array([1.0, 2.0], dtype=np.float32),
...     scale=np.asarray(2.0, dtype=np.float32),
... ).tolist()
[2.0, 4.0]
```

## StagedProgram

```python
StagedProgram(
    function: Callable[..., Any],
    *,
    specs: tuple[Any, ...],
    kw_specs: dict[str, Any],
    array_api_version: str,
)
```

One callable input signature compiled into one immutable durable graph.

Use `stage` to create a program. Its dictionary representation does not contain Python code and can be loaded after required primitives link.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> program = ad.stage(lambda x: x + 1, np.array([1.0, 2.0]))
>>> loaded = ad.StagedProgram.from_dict(program.to_dict())
>>> loaded(np.array([3.0, 4.0])).tolist()
[4.0, 5.0]
```

### graph

```python
graph: GraphStore
```

Return this program's immutable graph.

### signature

```python
signature: tuple[tuple[Any, ...], dict[str, Any]]
```

Return a detached positional and keyword input-specification tree.

### compile_seconds

```python
compile_seconds: float
```

Return the time spent compiling this in-process program.

### constants

```python
constants: tuple[ConstantRecord, ...]
```

Return the concrete values captured by this program.

### optimization

```python
optimization: OptimizationReport
```

Return this program's fixed-pipeline optimization report.

### trace

```python
trace: StagedTrace | None
```

Return the staging tape and optimizer mapping for in-process programs.

`None` for programs loaded from a durable artifact: the trace is a staging byproduct and is deliberately not serialized.

### array_api_version

```python
array_api_version: str
```

Return the Array API revision required by this program.

### __repr__

```python
__repr__() -> str
```

Return a compact program summary for notebooks and debuggers.

### __str__

```python
__str__() -> str
```

Render the program's optimized operation sequence.

### to_dict

```python
to_dict() -> dict[str, object]
```

Serialize this program without serializing Python code.

### from_dict

```python
from_dict(payload: object) -> StagedProgram
```

Load a versioned staged artifact after linking custom primitives.

## vjp_program

```python
vjp_program(
    f: StagedProgram,
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
) -> StagedProgram
```

Compile a reusable staged pullback program.

Parameters:

- **`f`** (`StagedProgram`) – Primal StagedProgram to transpose. Ordinary callables are not accepted; use vjp for a concrete dynamic pullback.
- **`argnums`** (`int | tuple[int, ...] | None`, default: `None` ) – Positional inputs to differentiate. An integer returns that input's gradient pytree directly, while a tuple returns a tuple in the given order. None selects input zero unless argnames is provided, in which case it selects no positional inputs. Negative indices are resolved against the program's positional signature.
- **`argnames`** (`tuple[str, ...] | None`, default: `None` ) – Keyword inputs from the staged signature to differentiate. Their gradients are returned in a dictionary keyed by name. When positional and named inputs are both selected, the result is (positional_gradients, named_gradients); the positional part follows the integer-versus-tuple rule above.

Returns:

- `StagedProgram` – An immutable, serializable program with the primal call signature plus a reserved keyword-only cotangent input. The cotangent has the primal output's pytree and leaf specifications. The program preserves the primal program's Array API revision.

Raises:

- `IndexError` – If a positional selection is out of range for the staged signature.
- `TypeError` – If f is not a StagedProgram, a selected input leaf is static, or a selected weak scalar signature is not real floating-point.
- `ValueError` – If positional selections are duplicated, a selected input is absent from the staged signature, or the primal signature already reserves the cotangent keyword.
- `NoVJPError` – If an operation on the differentiated path has no reverse-mode rule.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> square = ad.stage(lambda x: x**2, np.array([1.0, 2.0]))
>>> pullback = ad.vjp_program(square)
>>> pullback(np.array([3.0, 4.0]), cotangent=np.ones(2)).tolist()
[6.0, 8.0]
```

## ConstantRecord

```python
ConstantRecord(
    value_id: int,
    origin: str,
    location: str | None,
    shape: tuple[int, ...],
    dtype: str,
    bytes: int,
    digest: str,
    name: str | None = None,
)
```

Inspectable provenance for one concrete value captured while staging.

Attributes:

- **`value_id`** (`int`) – Identifier of the constant-producing node. Records on a StagedProgram use optimized graph numbering; records on a StagedTrace use pre-optimization tape numbering.
- **`origin`** (`str`) – Capture category: "closure", "global", or "created".
- **`location`** (`str | None`) – Source location associated with the capture, when available.
- **`shape`** (`tuple[int, ...]`) – Captured array shape.
- **`dtype`** (`str`) – Canonical dtype name stored in the durable artifact.
- **`bytes`** (`int`) – Number of bytes in the captured value payload.
- **`digest`** (`str`) – Content digest used to identify the captured value.
- **`name`** (`str | None`) – Source-level name associated with the capture, when available.

## OptimizationReport

```python
OptimizationReport(
    nodes_before: int,
    nodes_after: int,
    rewritten_nodes: int,
    passes: tuple[OptimizationPass, ...],
)
```

Inspectable result of the fixed staged optimization pipeline.

Attributes:

- **`nodes_before`** (`int`) – Graph node count before the first required pass.
- **`nodes_after`** (`int`) – Graph node count after the final required pass.
- **`rewritten_nodes`** (`int`) – Total number of nodes rewritten across all passes.
- **`passes`** (`tuple[OptimizationPass, ...]`) – Ordered diagnostics for each required optimization pass.

## OptimizationPass

```python
OptimizationPass(
    name: str,
    nodes_before: int,
    nodes_after: int,
    removed_nodes: int,
    rewritten_nodes: int,
)
```

Diagnostics for one required pass in the staged compiler.

Attributes:

- **`name`** (`str`) – Stable pass name.
- **`nodes_before`** (`int`) – Graph node count before the pass.
- **`nodes_after`** (`int`) – Graph node count after the pass.
- **`removed_nodes`** (`int`) – Number of input nodes that have no output representative.
- **`rewritten_nodes`** (`int`) – Number of input nodes removed or mapped to a different node.

## StagedTrace

```python
StagedTrace(
    nodes: tuple[TracedNode, ...],
    old_to_new: tuple[int | None, ...],
    constants: tuple[ConstantRecord, ...],
)
```

The staging tape before cleanup and its mapping onto the final graph.

`old_to_new[node.id]` is the optimized node that carries the traced node's value, or `None` when the cleanup pipeline removed it. Several traced nodes mapping onto one optimized node were merged by cse. `constants` holds the captured constant records in tape numbering. The trace is an in-process staging byproduct: it is not serialized, and it is `None` on programs loaded from a durable artifact.

## TracedNode

```python
TracedNode(
    id: int,
    op: str,
    inputs: tuple[int, ...],
    name: str | None,
)
```

One pre-optimization tape entry captured while staging.

Attributes:

- **`id`** (`int`) – Position of the value-producing entry in the staging tape.
- **`op`** (`str`) – Canonical registered operation identifier.
- **`inputs`** (`tuple[int, ...]`) – Tape identifiers consumed by the operation.
- **`name`** (`str | None`) – Source-level input or constant name, when available.
