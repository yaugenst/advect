# Transforms

Most transforms on this page run the callable and trace the path taken by its concrete inputs. [`grad`](https://yaugenst.github.io/advect/0.2.0/api/transforms/#advect.grad) and [`value_and_grad`](https://yaugenst.github.io/advect/0.2.0/api/transforms/#advect.value_and_grad) also accept a [`StagedProgram`](https://yaugenst.github.io/advect/0.2.0/api/staging/#advect.StagedProgram); they return another staged program with the same input signature and Array API revision. [`vjp_program`](https://yaugenst.github.io/advect/0.2.0/api/staging/#advect.vjp_program) builds a reusable staged pullback. The other transforms operate dynamically.

The [gradient](https://yaugenst.github.io/advect/0.2.0/tutorials/gradients/index.md), [linear-map](https://yaugenst.github.io/advect/0.2.0/tutorials/linear-maps/index.md), [higher-order](https://yaugenst.github.io/advect/0.2.0/tutorials/advanced-differentiation/index.md), and [implicit-differentiation](https://yaugenst.github.io/advect/0.2.0/tutorials/implicit-differentiation/index.md) tutorials connect these transforms through complete examples.

Library adapters may use `transform_state` for namespaced bookkeeping that lives only while one concrete transform is tracing. Differentiable primitive inputs and backward residuals must remain explicit.

## grad

```python
grad(
    f: StagedProgram,
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> StagedProgram
```

```python
grad(
    f: Callable[CallP, tuple[object, AuxT]],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: Literal[True],
) -> Callable[CallP, tuple[object, AuxT]]
```

```python
grad(
    f: Callable[CallP, object],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: Literal[False] = False,
) -> Callable[CallP, object]
```

```python
grad(
    f: Callable[CallP, object],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool,
) -> Callable[CallP, object]
```

```python
grad(
    f: Callable[CallP, object],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> Callable[CallP, object] | StagedProgram
```

Differentiate a scalar-valued function with reverse mode.

An ordinary callable is traced from concrete inputs on every invocation; no trace or graph is cached between calls. Passing a `StagedProgram` instead compiles and returns another immutable staged program. Warm calls to that program execute its prebuilt graph without a dynamic tape or reverse sweep.

Parameters:

- **`f`** (`Callable[CallP, object]`) – Callable whose differentiated value is a real scalar or a one-leaf pytree containing a real scalar. With has_aux=True, it instead returns (value, auxiliary), where only value is differentiated. A StagedProgram is also accepted.
- **`argnums`** (`int | tuple[int, ...] | None`, default: `None` ) – Positional arguments to differentiate. An integer returns that argument's gradient pytree directly, while a tuple returns a tuple in the given order. None selects argument zero unless argnames is provided, in which case it selects no positional arguments. Negative indices are resolved for each call.
- **`argnames`** (`tuple[str, ...] | None`, default: `None` ) – Named arguments to differentiate. Their gradients are returned in a dictionary keyed by name. For an ordinary callable, a selected name may be passed positionally or by keyword; staged named inputs must be passed by keyword. When positional and named inputs are both selected, the result is (positional_gradients, named_gradients), with the positional part following the integer-versus-tuple rule above.
- **`has_aux`** (`bool`, default: `False` ) – Whether f returns (value, auxiliary). The auxiliary value is excluded from differentiation. A dynamic call materializes it as a concrete sidecar; a staged transform records it as an ordinary staged output.

Returns:

- `Callable or StagedProgram` – For an ordinary callable, a concrete-tracing gradient function. For a staged input, an immutable, serializable derivative program with the same input signature and Array API revision. Its result has the gradient structure selected by argnums and argnames; when has_aux=True, it returns (gradient, auxiliary).

Raises:

- `IndexError` – If a positional selection is out of range for the transformed call.
- `TypeError` – If a selected input is an unsupported Python complex scalar, or a selected staged input leaf is static, or a selected staged weak-scalar signature is not real floating-point.
- `ValueError` – If positional selections are duplicated, an ordinary callable argument is selected both positionally and by name, a selected name is unavailable, or the differentiated output is not a real scalar.
- `NoVJPError` – If an operation on the differentiated path has no reverse-mode rule.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def squared_norm(x):
...     return np.sum(x**2)
>>> x = np.array([1.0, 2.0, 3.0])
>>> gradient = ad.grad(squared_norm)
>>> gradient(x).tolist()
[2.0, 4.0, 6.0]
```

## value_and_grad

```python
value_and_grad(
    f: StagedProgram,
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> StagedProgram
```

```python
value_and_grad(
    f: Callable[CallP, ResultT],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: Literal[False] = False,
) -> Callable[CallP, tuple[ResultT, object]]
```

```python
value_and_grad(
    f: Callable[CallP, tuple[ResultT, AuxT]],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: Literal[True],
) -> Callable[CallP, tuple[ResultT, object, AuxT]]
```

```python
value_and_grad(
    f: Callable[CallP, object],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool,
) -> Callable[CallP, tuple[object, ...]]
```

```python
value_and_grad(
    f: Callable[CallP, object],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> Callable[CallP, tuple[object, ...]] | StagedProgram
```

Compute a scalar value and its reverse-mode gradient together.

An ordinary callable is traced from concrete inputs on every invocation; no trace or graph is cached between calls. Passing a `StagedProgram` instead compiles and returns another immutable staged program. Warm calls to that program execute its prebuilt graph without a dynamic tape or reverse sweep.

Parameters:

- **`f`** (`Callable[CallP, object]`) – Callable whose differentiated value is a real scalar or a one-leaf pytree containing a real scalar. With has_aux=True, it instead returns (value, auxiliary), where only value is differentiated. A StagedProgram is also accepted.
- **`argnums`** (`int | tuple[int, ...] | None`, default: `None` ) – Positional arguments to differentiate. An integer returns that argument's gradient pytree directly, while a tuple returns a tuple in the given order. None selects argument zero unless argnames is provided, in which case it selects no positional arguments. Negative indices are resolved for each call.
- **`argnames`** (`tuple[str, ...] | None`, default: `None` ) – Named arguments to differentiate. Their gradients are returned in a dictionary keyed by name. For an ordinary callable, a selected name may be passed positionally or by keyword; staged named inputs must be passed by keyword. When positional and named inputs are both selected, the gradient is (positional_gradients, named_gradients), with the positional part following the integer-versus-tuple rule above.
- **`has_aux`** (`bool`, default: `False` ) – Whether f returns (value, auxiliary). The auxiliary value is excluded from differentiation. A dynamic call materializes it as a concrete sidecar; a staged transform records it as an ordinary staged output.

Returns:

- `Callable or StagedProgram` – For an ordinary callable, a concrete-tracing function returning (value, gradient). For a staged input, an immutable, serializable program with the same input signature and Array API revision returning the same structure. With has_aux=True, either lifetime returns (value, gradient, auxiliary).

Raises:

- `IndexError` – If a positional selection is out of range for the transformed call.
- `TypeError` – If a selected input is an unsupported Python complex scalar, or a selected staged input leaf is static, or a selected staged weak-scalar signature is not real floating-point.
- `ValueError` – If positional selections are duplicated, an ordinary callable argument is selected both positionally and by name, a selected name is unavailable, or the differentiated output is not a real scalar.
- `NoVJPError` – If an operation on the differentiated path has no reverse-mode rule.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def squared_norm(x):
...     return np.sum(x**2)
>>> x = np.array([1.0, 2.0, 3.0])
>>> value_and_gradient = ad.value_and_grad(squared_norm)
>>> value, gradient = value_and_gradient(x)
>>> float(value), gradient.tolist()
(14.0, [2.0, 4.0, 6.0])
```

## jvp

```python
jvp(
    f: Callable[..., R], argnums: int | tuple[int, ...] = 0
) -> Callable[..., tuple[R, object]]
```

Return a concrete-tracing Jacobian-vector product transform.

Parameters:

- **`f`** (`Callable[..., R]`) – Callable to differentiate. Its output may be any supported array or pytree. Passing a StagedProgram executes it inside a concrete trace; this transform does not compile a new staged program.
- **`argnums`** (`int | tuple[int, ...]`, default: `0` ) – Positional arguments to differentiate. An integer expects one tangent pytree directly. A tuple expects a tuple of tangent pytrees in the given order, including for a one-element tuple. Negative indices are resolved for each transformed call.

Returns:

- `Callable` – A function called as transformed(\*args, tangents=..., \*\*kwargs). tangents must match the selected primal pytree or pytrees. The result is (value, output_tangent); both entries preserve f's output pytree, and disconnected output leaves receive zero tangents.

Raises:

- `IndexError` – If a positional selection is out of range for the transformed call.
- `TypeError` – If a selected input is an unsupported Python complex scalar, multiple selected arguments are not given a tuple of tangents, or a tangent is supplied for a static or untraceable leaf.
- `ValueError` – If positional selections are duplicated, or the tangent arity, pytree, or leaf shape does not match the selected primals.
- `NoJVPError` – If an operation on the differentiated path has no JVP rule.

Notes

Each invocation traces the concrete values once, applies the JVP once, and releases the temporary `LinearMap` and any retained primitive residuals before returning. No trace is cached between calls.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def square(x):
...     return x**2
>>> x = np.array([1.0, 2.0])
>>> direction = np.ones_like(x)
>>> directional_derivative = ad.jvp(square)
>>> value, tangent = directional_derivative(x, tangents=direction)
>>> value.tolist(), tangent.tolist()
([1.0, 4.0], [2.0, 4.0])
```

## vjp

```python
vjp(
    f: Callable[CallP, ResultT],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[CallP, tuple[ResultT, Pullback]]
```

Return a concrete value and a one-shot reverse pullback.

`vjp` is always a dynamic transform. Each call traces the selected concrete inputs and returns a `Pullback` that owns that invocation's tape, retained provider values, and primitive residuals. This remains true when `f` is a `StagedProgram`; use `vjp_program` to compile a reusable staged pullback.

Parameters:

- **`f`** (`Callable[CallP, ResultT]`) – Callable to linearize. Its output may be any supported array or pytree.
- **`argnums`** (`int | tuple[int, ...]`, default: `0` ) – Positional arguments to differentiate. An integer makes the pullback return that argument's gradient pytree directly; a tuple makes it return a tuple in the given order. Negative indices are resolved for each call.

Returns:

- `Callable` – A concrete-tracing function returning (value, pullback). value preserves the callable's output pytree. Call pullback(cotangent) with a cotangent matching that pytree to obtain the selected input gradients.

Raises:

- `IndexError` – If a positional selection is out of range for the transformed call.
- `TypeError` – If a selected input is an unsupported Python complex scalar, or a cotangent leaf has an invalid numeric category.
- `ValueError` – If positional selections are duplicated or the cotangent pytree or leaf shape does not match the output.
- `NoVJPError` – If an operation on the differentiated path has no reverse-mode rule.
- `RuntimeError` – If the pullback is applied after it has already been consumed or closed.

Notes

Applying the pullback consumes it and releases its retained trace. Call `close()` to release it without applying it, or use it as a context manager when deterministic cleanup matters.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> value, pullback = ad.vjp(lambda x: x**2)(np.array([1.0, 2.0]))
>>> value.tolist()
[1.0, 4.0]
>>> pullback(np.ones(2)).tolist()
[2.0, 4.0]
```

## linearize

```python
linearize(
    f: Callable[..., R],
    *primals: object,
    argnums: int | tuple[int, ...] = 0,
    **kwargs: object,
) -> tuple[R, LinearMap]
```

Linearize one concrete call and return its reusable real-linear map.

The call is traced immediately from `primals` and `kwargs`. The returned `LinearMap` owns that invocation's tape, retained provider values, and primitive residuals; it is not a cached or durable program.

Parameters:

- **`f`** (`Callable[..., R]`) – Callable to linearize. Its output may be any supported array or pytree. A StagedProgram is accepted, but the surrounding linearization is still concrete and invocation-local.
- **`*primals`** (`object`, default: `()` ) – Positional arguments for this call. Only the arguments selected by argnums are tangent inputs; the others remain primal coefficients.
- **`argnums`** (`int | tuple[int, ...]`, default: `0` ) – Positional arguments to differentiate. An integer makes linear(tangents) accept that argument's tangent pytree directly; a tuple makes it accept a tuple of tangent pytrees in the given order. Negative indices are resolved against primals.
- **`**kwargs`** (`object`, default: `{}` ) – Keyword arguments forwarded to f. linearize does not select keyword arguments for differentiation.

Returns:

- `value` – The concrete output of f, with its pytree structure preserved.
- `linear` – A reusable LinearMap. Calling linear(tangents) applies the JVP and returns a tangent with the output pytree. Calling linear.pullback(cotangent) or linear.transpose()(cotangent) applies the real adjoint and returns the selected input structure.

Raises:

- `IndexError` – If a positional selection is out of range.
- `TypeError` – If a selected input contains an unsupported Python complex scalar, or a tangent has an invalid structure or numeric category.
- `ValueError` – If positional selections are duplicated, or a tangent pytree or leaf shape does not match its selected primal.
- `NoJVPError` – If an operation on the differentiated path has no JVP rule. Public primitives are rejected before their concrete implementation runs.
- `NoVJPError` – If the returned map is transposed through an operation without an explicit or structurally derivable transpose rule.
- `RuntimeError` – If the map is applied after it has been closed.

Notes

The map remains reusable until `close()`. Close it explicitly, or use it as a context manager, to release retained concrete values and residuals deterministically.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> value, linear = ad.linearize(lambda x: x**2, np.array([1.0, 2.0]))
>>> value.tolist()
[1.0, 4.0]
>>> with linear:
...     linear(np.ones(2)).tolist()
[2.0, 4.0]
```

## jacobian

```python
jacobian(
    f: Callable[P, object],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
) -> Callable[P, object]
```

Return a shape-preserving dense Jacobian for real pytree inputs and outputs.

Parameters:

- **`f`** (`Callable[P, object]`) – Callable to differentiate. Its selected inputs and output may be pytrees of real arrays or real Python scalars. A StagedProgram is accepted, but each call still uses a concrete, invocation-local linearization.
- **`argnums`** (`int | tuple[int, ...] | None`, default: `None` ) – Positional arguments to differentiate. An integer represents that input directly in every output block; a tuple represents the selected inputs as a tuple in the given order. None selects argument zero unless argnames is provided, in which case it selects no positional arguments. Negative indices are resolved for each call.
- **`argnames`** (`tuple[str, ...] | None`, default: `None` ) – Named arguments to differentiate. Each output leaf contains their derivative blocks in a dictionary keyed by name. For an ordinary callable, a selected name may be passed positionally or by keyword; staged named inputs must be passed by keyword. With both positional and named selections, each output leaf contains (positional_blocks, named_blocks).

Returns:

- `Callable` – A concrete-tracing function returning an output-shaped pytree of input-shaped derivative blocks. For each output leaf and selected input leaf, the dense block shape is output_leaf.shape + input_leaf.shape; neither side is flattened. Static or untraceable selected input leaves have None blocks.

Raises:

- `IndexError` – If a positional selection is out of range for the transformed call.
- `TypeError` – If a selected input contains an unsupported Python complex scalar.
- `ValueError` – If positional selections are duplicated, an argument is selected both positionally and by name, a selected name is unavailable, or an input or output leaf is complex.
- `NoJVPError` – If forward assembly is required through an operation without a JVP rule.
- `NoVJPError` – If reverse assembly is required through an operation without an explicit or structurally derivable transpose rule.
- `RuntimeError` – If the provider cannot assemble a dense block or a derivative rule changes the expected pytree structure between basis seeds.

Notes

Advect chooses forward or reverse assembly from the traced coordinate counts and available rule direction. The temporary `LinearMap` is always closed before the call returns or raises, releasing retained values and primitive residuals.

A general real-linear complex map needs two complex blocks, or one real `2m x 2n` block, so a single complex matrix would be ambiguous. Complex callers should use `linearize` instead.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> jacobian = ad.jacobian(lambda x: x**2)(np.array([1.0, 2.0]))
>>> jacobian.tolist()
[[2.0, 0.0], [0.0, 4.0]]
```

## hvp

```python
hvp(
    f: Callable[..., R], argnums: int | tuple[int, ...] = 0
) -> _HvpTransform[R]
```

Return a dynamic value-and-Hessian-vector-product transform.

The returned function evaluates `f` and applies its Hessian to one selected input-space vector without materializing the dense Hessian. It traces the concrete arguments on every call, evaluates `f` once, and releases the invocation-local linearization before returning.

Parameters:

- **`f`** (`Callable[..., R]`) – Callable producing a real scalar or a one-leaf pytree containing a real scalar. A StagedProgram is accepted, but the returned transform is still an ordinary dynamic callable.
- **`argnums`** (`int | tuple[int, ...]`, default: `0` ) – One or more positional arguments to differentiate. An integer selects one input and uses its pytree directly. A tuple preserves a tuple of selected input pytrees in the given order, including for a one-element tuple. Negative indices are resolved for each call.

Returns:

- `Callable` – A function called as transformed(\*args, vectors=vectors, \*\*kwargs) that returns (value, product). The keyword-only vectors value must match the pytree structure and leaf shapes selected by argnums; use None for a static or otherwise untraceable leaf. value preserves the output structure of f, and product preserves the integer-versus-tuple selection structure described above.

Raises:

- `IndexError` – If a selected positional index is out of range for the transformed call.
- `TypeError` – If vectors is omitted, is not a tuple for a tuple selection, gives a non-None tangent for a static leaf, or a selected input contains an unsupported Python complex scalar.
- `ValueError` – If no argument is selected, positional selections are duplicated, the vector arity, pytree, or leaf shape does not match the selected input, or f does not produce a real scalar value.
- `NoJVPError` – If an operation needed by the nested derivative has no forward-mode rule.
- `NoVJPError` – If an operation on the differentiated path has no reverse-mode rule.
- `TracingError` – If nested differentiation crosses a first-order-only primitive or another unsupported tracing boundary.

Notes

Complex provider arrays are supported under Advect's real-linear convention when `f` returns a real scalar. Python complex scalars must be wrapped in provider zero-dimensional arrays. Use `hessian` only for an explicit dense Hessian over real inputs.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> value, product = ad.hvp(lambda x: np.sum(x**2))(
...     np.array([1.0, 2.0]), vectors=np.array([3.0, 4.0])
... )
>>> float(value), product.tolist()
(5.0, [6.0, 8.0])
```

## hessian

```python
hessian(
    f: Callable[P, object],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[P, object]
```

Return a dynamic transform that assembles an exact dense Hessian.

Each selected positional argument is one dense real input block. For selected shapes `S_i`, block `[i][j]` has shape `S_i + S_j`: its rows index coordinates of the gradient with respect to argument `i` and its columns index coordinates of argument `j`.

Parameters:

- **`f`** (`Callable[P, object]`) – Callable producing a real scalar or a one-leaf pytree containing a real scalar. A StagedProgram is accepted, but the returned transform is still an ordinary dynamic callable.
- **`argnums`** (`int | tuple[int, ...]`, default: `0` ) – One or more positional arguments to differentiate. An integer selects one input and returns its dense block directly. A tuple preserves both selected-argument axes in the given order and returns a tuple of tuple blocks, including a one-by-one structure for a one-element tuple. Negative indices are resolved for each call.

Returns:

- `Callable` – A function accepting the arguments of f and returning dense provider arrays. An integer selection returns one array with shape S + S. A tuple selection returns blocks result[i][j] with shape S_i + S_j.

Raises:

- `IndexError` – If a selected positional index is out of range for the transformed call.
- `ValueError` – If no argument is selected, positional selections are duplicated, a selected input is complex, or f does not produce a real scalar value.
- `AdvectError` – If dense assembly cannot resolve a compatible runtime array namespace or represent the selected gradient as dense blocks.
- `NoJVPError` – If an operation needed by the nested derivative has no forward-mode rule.
- `NoVJPError` – If an operation on the differentiated path has no reverse-mode rule.
- `TracingError` – If nested differentiation crosses a first-order-only primitive or another unsupported tracing boundary.

Notes

Each selected argument must be a real Python scalar or one array-like value with a coherent `shape` and `dtype`. Generic tuple and dictionary pytrees are supported by `hvp`, but not by dense Hessian assembly.

Each invocation traces `f` at the current values once, reuses that trace for the coordinate sweeps, and releases it before returning. No derivative graph is cached, and this transform never returns a `StagedProgram`. Storage dtypes are promoted with the provider's `float64` dtype. A derivative column for a selected Python `int` or `float` is restored to a Python scalar. Dense complex Hessians are not represented; use `hvp` or `linearize` for complex real-linear derivatives.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> ad.hessian(lambda x: np.sum(x**3))(np.array([1.0, 2.0])).tolist()
[[6.0, 0.0], [0.0, 12.0]]
```

## hessian_diag

```python
hessian_diag(
    f: Callable[P, object],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[P, object]
```

Return a dynamic transform that assembles exact Hessian diagonals.

For each selected positional argument, the result contains the diagonal of that argument's self-Hessian block with the argument's original shape. Mixed-argument blocks are omitted.

Parameters:

- **`f`** (`Callable[P, object]`) – Callable producing a real scalar or a one-leaf pytree containing a real scalar. A StagedProgram is accepted, but the returned transform is still an ordinary dynamic callable.
- **`argnums`** (`int | tuple[int, ...]`, default: `0` ) – One or more positional arguments to differentiate. An integer selects one input and returns its diagonal directly. A tuple returns one diagonal per selected input in the given order, including a one-tuple for a one-element tuple. Negative indices are resolved for each call.

Returns:

- `Callable` – A function accepting the arguments of f. For an integer selection with input shape S, it returns one provider array with shape S. A tuple selection returns a tuple whose entry i has shape S_i.

Raises:

- `IndexError` – If a selected positional index is out of range for the transformed call.
- `ValueError` – If no argument is selected, positional selections are duplicated, a selected input is complex, or f does not produce a real scalar value.
- `AdvectError` – If dense assembly cannot resolve a compatible runtime array namespace or represent the selected gradient as a dense block.
- `NoJVPError` – If an operation needed by the nested derivative has no forward-mode rule.
- `NoVJPError` – If an operation on the differentiated path has no reverse-mode rule.
- `TracingError` – If nested differentiation crosses a first-order-only primitive or another unsupported tracing boundary.

Notes

Each selected argument must be a real Python scalar or one array-like value with a coherent `shape` and `dtype`. Generic tuple and dictionary pytrees are supported by `hvp`, but not by dense diagonal assembly.

This is an exact automatic-differentiation result, not a stochastic estimator; computing it may still require one coordinate sweep per selected scalar coordinate. Each call traces `f` once, releases its temporary linearization before returning, and never produces a `StagedProgram`. Storage dtypes are promoted with the provider's `float64` dtype. Selected Python `int` and `float` inputs have Python scalar diagonal entries. Dense complex diagonals are not represented; use `hvp` or `linearize` for complex real-linear derivatives.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> ad.hessian_diag(lambda x: np.sum(x**3))(np.array([1.0, 2.0])).tolist()
[6.0, 12.0]
```

## checkpoint

```python
checkpoint(function: Callable[P, R]) -> Callable[P, R]
```

Return a dynamic rematerialization wrapper for `function`.

An ordinary call invokes `function` directly. During concrete autodiff, Advect records the whole call as one operation on the outer tape and recomputes its body when applying a JVP or transpose instead of retaining the body's interior trace.

Parameters:

- **`function`** (`Callable[P, R]`) – Pure callable to rematerialize. Its positional arguments, keyword arguments, and result may be pytrees. Replaying the same explicit inputs must produce the same result; observed mutable state and side effects are therefore outside the contract.

Returns:

- `Callable` – A wrapper with the apparent signature and metadata of function. Calling it as wrapped(\*args, \*\*kwargs) returns the same output pytree as function(\*args, \*\*kwargs).

Raises:

- `TypeError` – If function is not callable, or if a traced invocation changes its input pytree structure while the rematerialized region is executing.
- `TracingError` – If abstract staging reaches the wrapper, or if recomputation reaches a residual-bearing primitive whose opaque residual cannot cross the checkpoint boundary.

Notes

Checkpointing is a concrete dynamic transform; it does not create a durable staged region, and `stage` rejects a checkpointed call. Nested dynamic derivatives are supported when the recomputed callable and all derivative rules on its path remain traceable at the nested level.

The returned wrapper retains `function` and its closure for the wrapper's lifetime. Each JVP or transpose application owns and releases its temporary inner trace before returning; checkpointing exposes no additional resource handle to close.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> @ad.checkpoint
... def square(value):
...     return value**2
>>> ad.grad(lambda value: np.sum(square(value)))(np.array([1.0, 2.0, 3.0])).tolist()
[2.0, 4.0, 6.0]
```

## implicit_root

```python
implicit_root(
    residual: _ResidualFunction[SolutionT, ParamsT],
    *,
    solve: _RootSolver[SolutionT],
    linear_solve: _LinearSolver[SolutionT],
    transpose_solve: _LinearSolver[SolutionT] | None = None,
) -> _ImplicitRootCallable[ParamsT, SolutionT]
```

Build a dynamic transform for a converged implicit solution.

The returned callable solves `residual(solution, params) == 0` and differentiates the defining equation without recording the nonlinear solver's iterations on the surrounding tape.

Parameters:

- **`residual`** (`_ResidualFunction[SolutionT, ParamsT]`) – Trace-compatible callable with signature residual(solution, params). Its result must have the same pytree structure and leaf shape, dtype, array provider, and device as solution.
- **`solve`** (`_RootSolver[SolutionT]`) – Nonlinear callback with signature solve(residual_at_params, initial) -> solution. Advect supplies residual_at_params(candidate) with params fixed. Returning certifies convergence. The solution must match initial in pytree structure, leaf shape, provider, and device; its dtype may be promoted.
- **`linear_solve`** (`_LinearSolver[SolutionT]`) – Matrix-free callback with signature linear_solve(operator, rhs) -> solution_tangent. For a JVP, operator(direction) applies the residual's state Jacobian and rhs is the negative parameter-forcing tangent. The returned value must match the solved solution's pytree and leaf specifications.
- **`transpose_solve`** (`_LinearSolver[SolutionT] | None`, default: `None` ) – Matrix-free callback with signature transpose_solve(operator, rhs) -> residual_cotangent. In reverse mode, operator applies the real adjoint of the residual's state Jacobian and rhs is the solution cotangent. The result must match the residual value's pytree and leaf specifications. None reuses linear_solve with this adjoint operator.

Returns:

- `Callable` – A callable with signature root(params, \*, initial) -> solution. params and initial may be pytrees. initial selects a root but is excluded from the implicit derivative, so an enclosing derivative with respect to it is zero. A Python scalar solution is moved to the parameter array provider when one is available.

Raises:

- `TypeError` – If a callback is not callable, or if a nonlinear solution, residual, tangent solve, or transpose solve violates its required pytree or leaf specification.
- `TracingError` – If abstract staging reaches the returned root. Opaque Python solver callbacks have no durable staged representation.
- `ImplicitSolveError` – Propagated when a nonlinear or linear callback uses this exception to report failure. A callback return is otherwise treated as a successful solve; Advect does not independently test convergence.

Notes

This is a concrete dynamic boundary. `stage` rejects the root before calling `solve`; stage explicit solver iterations or define a custom primitive with a closed abstract rule when a durable program is required. Higher-order dynamic derivatives require `residual` and every solver callback reached by the nested transform to accept nested traced values. Concrete adapters such as the bundled SciPy callbacks intentionally form a first-order boundary.

A derivative application creates one joint linearization of `residual` at the solved value and closes it before returning, including on callback failure. The root wrapper retains the four callbacks for its lifetime but creates no user-managed resource. A `Pullback` or `LinearMap` returned by an enclosing transform still follows that transform's documented lifetime.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> def solve(residual_at_params, initial):
...     return initial - residual_at_params(initial)
>>> def linear_solve(operator, rhs):
...     return rhs / operator(np.ones_like(rhs))
>>> root = ad.implicit_root(
...     lambda solution, params: solution - params,
...     solve=solve,
...     linear_solve=linear_solve,
... )
>>> gradient = ad.grad(lambda params: root(params, initial=np.array(0.0)))(np.array(3.0))
>>> float(gradient)
1.0
```

## Pullback

```python
Pullback(linear: LinearMap)
```

One-shot reverse linearization returned by `vjp`.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> _, pullback = ad.vjp(lambda x: x**2)(np.array([1.0, 2.0]))
>>> pullback(np.ones(2)).tolist()
[2.0, 4.0]
```

### __call__

```python
__call__(cotangent: object) -> object
```

Apply the pullback once and release its retained trace.

### close

```python
close() -> None
```

Release the retained trace without applying the pullback.

### __enter__

```python
__enter__() -> Self
```

Enter an ownership scope for the pending pullback.

### __exit__

```python
__exit__(*_exc_info: object) -> None
```

Release the pullback when leaving its ownership scope.

## LinearMap

```python
LinearMap(trace: TraceResult, *, single_argnum: bool)
```

Reusable real-linear map captured by one concrete trace.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> _, linear = ad.linearize(lambda x: x**2, np.array([1.0, 2.0]))
>>> with linear:
...     linear(np.ones(2)).tolist()
[2.0, 4.0]
```

### close

```python
close() -> None
```

Release retained concrete values and primitive residuals.

## transform_state

```python
transform_state(
    namespace: object, factory: Callable[[], T]
) -> T | None
```

Return namespaced state owned by the active dynamic transform.

Libraries can use this to retain ordinary Python bookkeeping for exactly one define-by-run transform invocation without wrapping the transform or keeping process-global state. Repeated calls with the same namespace return the same object. Nested transforms have independent state, and Advect drops the state when its owning trace exits, including on exceptions.

State is not a hidden differentiable input or a backward residual. Pass active leaves explicitly to primitives, and retain backward data with `PrimitiveResult`.

Outside a transform this returns `None`. Abstract staging rejects the operation because staged programs cannot retain invocation-local Python state.

Parameters:

- **`namespace`** (`object`) – Hashable library-owned key identifying the state.
- **`factory`** (`Callable[[], T]`) – Zero-argument callable used once to create the state.

Returns:

- `object or None` – The invocation-local state, or None outside dynamic tracing.

## transform_states

```python
transform_states(namespace: object) -> tuple[T, ...]
```

Return existing namespaced states from inner to outer dynamic transforms.

This lets a library resolve state owned by an enclosing transform while a nested transform is active. It never creates state. Outside a transform it returns an empty tuple; abstract staging rejects the operation.
