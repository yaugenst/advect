# Custom Primitives

Create a [primitive](https://yaugenst.github.io/advect/0.1.1/api/primitives/#advect.primitive) when a numerical operation is opaque to Advect or needs one stable atomic identity in a [staged program](https://yaugenst.github.io/advect/0.1.1/api/staging/index.md). Ordinary traceable array code does not need a wrapper; Advect can already see and differentiate its operations.

The cube below is intentionally simple so the authoring contract stays visible. A primitive starts with its concrete implementation, then adds only the rules it promises to support.

```python
import numpy as np

import advect as ad


@ad.primitive
def cube(x):
    return x * x * x


@cube.def_abstract
def cube_abstract(x):
    return x.spec


@cube.def_jvp
def cube_jvp(output, primals, tangents):
    del output
    (x,), (tangent,) = primals, tangents
    if tangent is None:
        return np.zeros_like(x)
    return 3 * x * x * tangent


print("cube:", cube(np.array([1.0, 2.0, 3.0])))
```

The implementation handles ordinary calls. The [abstract rule](https://yaugenst.github.io/advect/0.1.1/api/primitives/#attach-rules-to-the-returned-handle) describes output shape and dtype for staging. The [JVP](https://yaugenst.github.io/advect/0.1.1/tutorials/linear-maps/#push-a-direction-forward) is ordinary traceable code, so Advect can use it for forward mode, derive its transpose for reverse mode, and compose it under higher-order transforms.

## Check the promised capabilities

```python
from advect.testing import check_primitive

sample = np.array([1.0, 2.0, 3.0])
check_primitive(
    cube,
    primals=(sample,),
    check=("abstract", "jvp", "transpose", "nested", "stage"),
)

gradient = ad.grad(lambda x: np.sum(cube(x)))(sample)
np.testing.assert_allclose(gradient, 3 * sample**2)
print("gradient:", gradient)
```

The default [`check_primitive`](https://yaugenst.github.io/advect/0.1.1/api/testing/#advect.testing.check_primitive) covers the first-order abstract, JVP, and transpose paths. Request only the extra capabilities the primitive claims, and run materially different shape, dtype, static-argument, and complex cases separately. Follow the primitive check with [`check_gradient`](https://yaugenst.github.io/advect/0.1.1/api/testing/#advect.testing.check_gradient) on a representative composition.

The operation name defaults to the function's module and qualified name. Give it an explicit name such as `example.cube` only when saved programs need an identity independent of the Python module path. Loading still requires the matching implementation to be imported under that name.

Some operations need an explicit transpose, exact forward residuals, static arguments, or intentionally first-order behavior. Those are real contracts, but they are extension-author reference material rather than prerequisites for the common JVP-first path; see the [primitive API](https://yaugenst.github.io/advect/0.1.1/api/primitives/index.md).
