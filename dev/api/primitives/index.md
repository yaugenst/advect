# Primitives

[`primitive`](https://yaugenst.github.io/advect/dev/api/primitives/#advect.primitive) makes one function appear as a single operation to Advect. The decorator returns the callable with methods for adding its abstract, JVP, and optional transpose rules. Prefer a [JVP](https://yaugenst.github.io/advect/dev/api/transforms/#advect.jvp) because it supports forward mode and structural transposition. An explicit transpose can instead provide [reverse mode](https://yaugenst.github.io/advect/dev/api/transforms/#advect.vjp) when no JVP is available.

Concrete and abstract calls retain the implementation's named parameters and pytrees. JVP and transpose rules operate on the dynamic array/scalar leaves in one stable flattened order. Static arguments remain named configuration; nondifferentiable arguments remain dynamic values but have no derivative contribution.

Output arity is fixed by default. Set `variable_output_arity=True` only when a concrete invocation determines its number of output leaves. Advect records that invocation's output pytree for differentiation; variable-arity primitives cannot be staged.

The [custom primitive tutorial](https://yaugenst.github.io/advect/dev/tutorials/primitives/index.md) shows the common JVP-first workflow. Use [`check_primitive`](https://yaugenst.github.io/advect/dev/api/testing/#advect.testing.check_primitive) and [`check_gradient`](https://yaugenst.github.io/advect/dev/api/testing/#advect.testing.check_gradient) to validate both the primitive and a representative composition.

## Define the operation

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
    variable_output_arity: bool = False,
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
    variable_output_arity: bool = False,
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
    variable_output_arity: bool = False,
) -> (
    Primitive[CallP, ResultT]
    | Callable[
        [Callable[CallP, ResultT]],
        Primitive[CallP, ResultT],
    ]
)
```

Define one atomic operation from its concrete implementation.

The implementation must have fixed named parameters: positional-or-keyword and keyword-only parameters are supported, while positional-only parameters, `*args`, and `**kwargs` are rejected. Calls still follow the implementation's normal Python signature.

`static_argnames` removes complete named arguments from tracing and stores them as operation attributes. `nondiff_argnames` keeps complete arguments as dynamic operands but supplies `None` tangents and suppresses their transpose contributions. The two sets must be disjoint. Derivative rules receive all remaining dynamic array/scalar leaves flattened in implementation-parameter and pytree order.

With `residual=True`, the implementation must return `advect.PrimitiveResult`; callers still receive only its `output`. With `variable_output_arity=True`, each concrete dynamic invocation owns its output leaf count; abstract staging remains unsupported. Rules are attached to the returned handle.

Parameters:

- **`function`** (`Callable[CallP, ResultT] | None`, default: `None` ) – Concrete implementation, when the decorator is applied directly.
- **`name`** (`str | None`, default: `None` ) – Operation identity without the internal custom. prefix. By default Advect uses the implementation's module and qualified name. Use a stable explicit name for serialized artifacts.
- **`static_argnames`** (`tuple[str, ...]`, default: `()` ) – Complete implementation arguments treated as concrete configuration.
- **`nondiff_argnames`** (`tuple[str, ...]`, default: `()` ) – Complete dynamic arguments excluded from differentiation.
- **`residual`** (`bool`, default: `False` ) – Whether the implementation returns an invocation-local PrimitiveResult for an exact transpose.
- **`variable_output_arity`** (`bool`, default: `False` ) – Whether concrete invocations may return different numbers of output leaves. Such primitives cannot be staged.

Returns:

- `Primitive or callable` – A callable authoring handle, or a decorator that creates one.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> @ad.primitive(name="examples.cube")
... def cube(value):
...     return value**3
>>> @cube.def_abstract
... def cube_abstract(value):
...     return value.spec
>>> @cube.def_jvp
... def cube_jvp(output, primals, tangents):
...     del output
...     (value,), (tangent,) = primals, tangents
...     return np.zeros_like(value) if tangent is None else 3 * value**2 * tangent
>>> from advect.testing import check_primitive
>>> sample = np.array([2.0])
>>> check_primitive(
...     cube,
...     primals=(sample,),
...     check=("abstract", "jvp", "transpose", "nested", "stage"),
... )
>>> ad.grad(lambda value: np.sum(cube(value)))(sample).tolist()
[12.0]
```

## Attach rules to the returned handle

These methods belong to the object returned by `advect.primitive`:

## def_abstract

```python
def_abstract(fn: Callable[..., Any]) -> Callable[..., Any]
```

Attach the primitive's abstract staging rule.

The rule has the implementation's fixed named parameters. Advect preserves each dynamic argument's pytree while replacing its array/scalar leaves with `advect.AbstractValue`; declared static arguments arrive unchanged. Return the concrete output pytree with `advect.ArraySpec` or `AbstractValue` leaves.

The function is returned unchanged so this method can be used as a decorator.

## def_jvp

```python
def_jvp(fn: Callable[..., Any]) -> Callable[..., Any]
```

Attach `fn(output, primals, tangents, **static_attrs)` as the JVP.

`output` has the implementation's public output pytree. `primals` and `tangents` are flat tuples with one entry per dynamic array/scalar leaf, in implementation-parameter and pytree order. Tangents may be `None` for inactive leaves and are always `None` for leaves of a declared nondifferentiable argument. Static arguments are passed by name. Return a tangent with the output pytree.

Write the rule as traceable, real-linear code so Advect can transpose it structurally and differentiate it again. The function is returned unchanged for decorator use.

## def_transpose

```python
def_transpose(fn: Callable[..., Any]) -> Callable[..., Any]
```

Attach an ordinary or exact-residual transpose rule.

Ordinary primitives receive `(cotangent, primals, output, **static_attrs)`. A primitive declared with `residual=True` receives `(cotangent, primals, output, residual, **static_attrs)`. `cotangent` and `output` have the public output pytree; `primals` is the same flattened dynamic-leaf tuple used by the JVP. Return a flat tuple with one contribution per dynamic leaf in that order. Advect suppresses contributions for declared nondifferentiable arguments.

A rule may accept the optional keyword-only `active_input_indices=None` and return `None` for inactive contributions to avoid unnecessary work. Add an explicit transpose only when structural transposition cannot express the correct real adjoint, when an exact residual is required, or when measurement justifies a direct rule. The function is returned unchanged for decorator use.

## Exact residuals

Set `residual=True` only when reverse mode needs exact opaque data from the forward invocation. Residual primitives require an explicit transpose and form a first-order boundary; the object docstring below defines their lifetime and cleanup contract.

## PrimitiveResult

```python
PrimitiveResult(
    output: R,
    residual: Any,
    release: Callable[[Any], None] | None = None,
)
```

A primitive's public output and private same-invocation residual.

`output` is the only value returned to the caller. Advect retains `residual` for the matching derivative invocation and calls `release` exactly once when that invocation state is discarded. The output must remain valid after the residual is released. A JVP never receives the residual; an explicit transpose on a primitive declared with `residual=True` receives it after `output`. A plain call or plain staged replay releases before returning. A one-shot reverse trace releases after consumption; a reusable linear map retains the residual until the map is closed.

Parameters:

- **`output`** (`R`) – Public primitive result returned to the caller.
- **`residual`** (`Any`) – Opaque invocation-local data retained for reverse mode.
- **`release`** (`Callable[[Any], None] | None`, default: `None` ) – Optional cleanup callback invoked exactly once with residual when Advect releases the invocation state.

Examples:

```pycon
>>> import advect as ad
>>> result = ad.PrimitiveResult(output=3.0, residual="cached state")
>>> result.output
3.0
```

## Abstract values

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
