# Staging

The explicit durable-graph boundary: compile one signature into one immutable,
optimized, serializable program.

Advect stages against one explicit Array API contract. The supported targets
are `2022.12`, `2023.12`, and `2024.12`:

```python
program = ad.stage(function, example, array_api_version="2023.12")
assert program.array_api_version == "2023.12"
```

When examples are supplied without a target, `stage` selects the newest
revision every example provider can serve. With `specs=` and no concrete
provider, it defaults to `2024.12`. The selected target is stored in the graph,
preserved by `grad` and `vjp_program`, and enforced before runtime graph
evaluation. Choosing an older target is the deliberate way to build a more
portable artifact; Advect does not infer a minimum revision from the operations
used by the function.

::: advect.ArraySpec

::: advect.StaticSpec

::: advect.stage

::: advect.StagedProgram

::: advect.vjp_program

::: advect.ConstantRecord

::: advect.OptimizationReport

::: advect.OptimizationPass

::: advect.StagedTrace

::: advect.TracedNode
