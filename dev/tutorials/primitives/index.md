# Custom Primitives

A primitive starts as an ordinary implementation function. Decorating it makes the function an atomic Advect operation and provides the rule decorators:

```python
import numpy as np

import advect as ad


@ad.primitive
def cube(value):
    return value * value * value


@cube.def_abstract
def cube_abstract(value):
    return value.spec


@cube.def_jvp
def cube_jvp(output, primals, tangents):
    del output
    (value,), (tangent,) = primals, tangents
    if tangent is None:
        return value * 0
    return 3 * value * value * tangent
```

The JVP is ordinary traceable code. Advect validates real-linearity when it derives the transpose. The optional authoring checks exercise that path before publication:

```python
from advect.testing import check_primitive

sample = np.array([1.0, 2.0, 3.0])
check_primitive(
    cube,
    primals=(sample,),
    check=("abstract", "jvp", "transpose", "nested", "stage"),
)

dcube = ad.grad(lambda value: np.sum(cube(value)))(sample)
np.testing.assert_allclose(dcube, 3 * sample**2)
```

The default checker tuple—`abstract`, `jvp`, and `transpose`—is a first-order smoke check. For a serializable primitive, run the full tuple above separately for every materially different shape, dtype, and static-argument form. Add a `complex` check with complex primals when complex values are supported. The stage check executes the compiled and restored program, compares exact output metadata, and verifies that inputs remain unchanged.

The operation name defaults to the function's module and qualified name. Use `@ad.primitive(name="example.cube")` only when a library needs a stable public identity for serialized programs. There is one implementation and no user-managed schema version or provider-key registry; portable implementations can use Array API operations directly, while backend-specific operations should validate their accepted inputs in the function. Advect does not infer whether a changed implementation is semantically compatible with an old artifact; the loading environment must provide the matching code.
