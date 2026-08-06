# Primitives

Authoring surface for custom operations: one implementation, abstract evaluation, a JVP, and an optional explicit transpose. See the [conformance testing ADR](https://github.com/yaugenst/advect/blob/main/design/decisions/2026-07-27-primitive-conformance-testing.md) for the law battery every registered primitive must pass.

## primitive

```python
primitive(
    function: Callable[CallP, ResultT],
    /,
    *,
    name: str | None = None,
    static_argnames: tuple[str, ...] = (),
    nondiff_argnames: tuple[str, ...] = (),
    residual: bool = False,
) -> Primitive[CallP, ResultT]
```

```python
primitive(
    function: None = None,
    /,
    *,
    name: str | None = None,
    static_argnames: tuple[str, ...] = (),
    nondiff_argnames: tuple[str, ...] = (),
    residual: bool = False,
) -> Callable[
    [Callable[CallP, ResultT]], Primitive[CallP, ResultT]
]
```

```python
primitive(
    function: Callable[CallP, ResultT] | None = None,
    /,
    *,
    name: str | None = None,
    static_argnames: tuple[str, ...] = (),
    nondiff_argnames: tuple[str, ...] = (),
    residual: bool = False,
) -> (
    Primitive[CallP, ResultT]
    | Callable[
        [Callable[CallP, ResultT]],
        Primitive[CallP, ResultT],
    ]
)
```

Define one atomic operation from its concrete implementation.

Derivative rules are ordinary traceable functions attached to the returned primitive handle.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> @ad.primitive(name="examples.cube")
... def cube(value):
...     return value**3
>>> @cube.def_jvp
... def cube_jvp(output, primals, tangents):
...     del output
...     (value,), (tangent,) = primals, tangents
...     return 3 * value**2 * tangent
>>> ad.grad(lambda value: np.sum(cube(value)))(np.array([2.0])).tolist()
[12.0]
```

## PrimitiveResult

```python
PrimitiveResult(
    output: R,
    residual: Any,
    release: Callable[[Any], None] | None = None,
)
```

A primitive's public output and private same-invocation residual.

`output` is the only value returned to the caller. Advect retains `residual` for the matching derivative invocation and calls `release` exactly once when that invocation state is discarded. The output must remain valid after the residual is released.

Examples:

```pycon
>>> import advect as ad
>>> result = ad.PrimitiveResult(output=3.0, residual="cached state")
>>> result.output
3.0
```

## AbstractValue

```python
AbstractValue(spec: ArraySpec)
```

A payload-free value passed to custom primitive abstract rules.

Examples:

```pycon
>>> import advect as ad
>>> abstract = ad.AbstractValue(ad.ArraySpec((4,), "float32"))
>>> abstract.spec.shape
(4,)
```

## check_primitive

```python
check_primitive(
    primitive: Primitive[Any, Any],
    *,
    primals: tuple[Any, ...],
    static: Mapping[str, Any] | None = None,
    tangents: tuple[Any, ...] | None = None,
    cotangent: Any | None = None,
    check: tuple[str, ...] = (
        "abstract",
        "jvp",
        "transpose",
    ),
    epsilon: float = 0.0001,
    atol: float = 1e-05,
    rtol: float = 0.0001,
) -> None
```

Run selected author checks for one representative primitive invocation.

The default `("abstract", "jvp", "transpose")` is a first-order smoke check. Authors of a serializable primitive should normally run `("abstract", "jvp", "transpose", "nested", "stage")` for every materially different shape, dtype, and static-argument form. Add `"complex"` in a separate call whose primals are complex when the primitive supports complex values.

The stage check executes both the compiled and serialized program, compares output structure, shape, and dtype exactly, and verifies that inputs remain unchanged. Repository-wide support still requires the conformance inventory; this helper intentionally does not import Hypothesis or claim exhaustive coverage from one sample.

## Composed functions

Use the whole-function checker first when a finite gradient looks suspicious. It performs a directional finite-difference sweep and checks the reverse gradient; `check_primitive` remains the narrower authoring contract for one extension.

## check_gradient

```python
check_gradient(
    function: Callable[..., Any],
    primal: Any,
    *,
    tangent: Any | None = None,
    epsilons: Sequence[float] = (
        0.01,
        0.001,
        0.0001,
        1e-05,
    ),
    atol: float = 1e-05,
    rtol: float = 0.0001,
) -> None
```

Check a unary composed function against directional differences.

The check compares Advect's whole-function JVP with a central finite- difference sweep, then checks the reverse gradient against the same directional derivative. It checks consistency with the function that actually ran, not whether that function encodes the intended mathematics.
