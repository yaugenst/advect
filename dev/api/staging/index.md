# Staging

The explicit durable-graph boundary: compile one signature into one immutable, optimized, serializable program.

Advect stages against one explicit Array API contract. The supported targets are `2022.12`, `2023.12`, and `2024.12`:

```python
program = ad.stage(function, example, array_api_version="2023.12")
assert program.array_api_version == "2023.12"
```

When examples are supplied without a target, `stage` selects the newest revision every example provider can serve. With `specs=` and no concrete provider, it defaults to `2024.12`. The selected target is stored in the graph, preserved by `grad` and `vjp_program`, and enforced before runtime graph evaluation. Choosing an older target is the deliberate way to build a more portable artifact; Advect does not infer a minimum revision from the operations used by the function.

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

Compile one inferred or explicitly declared signature into a staged program.

Pass concrete examples positionally to infer shape and dtype, or pass an explicit `specs=` tree. The result accepts exactly that one signature.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> program = ad.stage(lambda x: x + 1, np.array([1.0, 2.0]))
>>> program(np.array([3.0, 4.0])).tolist()
[4.0, 5.0]
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

Use :func:`stage` to create a program. Its dictionary representation does not contain Python code and can be loaded after required primitives link.

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

Compile a reusable staged pullback with a keyword-only `cotangent=` input.

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
