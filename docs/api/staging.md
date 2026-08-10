# Staging

Compile one signature into one immutable, optimized, serializable program.
The [staging tutorial](../tutorials/staging.md) develops the complete workflow
from a first call through differentiation and serialization.

Advect stages against one explicit
[Array API contract](https://data-apis.org/array-api/latest/). The supported
targets are `2022.12`, `2023.12`, and `2024.12`:

```{.python .run}
import numpy as np

import advect as ad


def energy(x):
    return np.sum(x**2)


example = np.array([1.0, 2.0])
program = ad.stage(energy, example, array_api_version="2023.12")
print(program.array_api_version)
# 2023.12
```

When examples are supplied without a target,
[`stage`](staging.md#advect.stage) selects the newest revision every example
provider can serve. With [`specs=`](staging.md#advect.stage) and no concrete
provider, it defaults to `2024.12`.
The selected target is stored in the graph, preserved by
[`grad`](transforms.md#advect.grad) and
[`vjp_program`](staging.md#advect.vjp_program), and enforced before runtime
graph evaluation. Choosing an older target is the deliberate way to build a
more portable artifact; Advect does not infer a minimum revision from the
operations used by the function.

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
