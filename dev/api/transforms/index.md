# Transforms

Dynamic differentiation entry points. Each transform traces concrete values per invocation; see the [API overview](https://yaugenst.github.io/advect/dev/api/index.md) for semantics shared across transforms and the [target API](https://github.com/yaugenst/advect/blob/main/design/target-api.md) for normative examples.

## grad

```python
grad(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> Callable[..., Any]
```

Differentiate a scalar-valued function with reverse mode.

A normal callable is traced from its concrete inputs on every invocation. Passing a :class:`StagedProgram` instead returns another staged program.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> x = np.array([1.0, 2.0, 3.0])
>>> ad.grad(lambda value: np.sum(value**2))(x).tolist()
[2.0, 4.0, 6.0]
```

## value_and_grad

```python
value_and_grad(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
    has_aux: bool = False,
) -> Callable[..., tuple[Any, ...]]
```

Compute a scalar value and its reverse-mode gradient together.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> x = np.array([1.0, 2.0, 3.0])
>>> value, gradient = ad.value_and_grad(lambda v: np.sum(v**2))(x)
>>> float(value), gradient.tolist()
(14.0, [2.0, 4.0, 6.0])
```

## jvp

```python
jvp(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., tuple[Any, Any]]
```

Return a concrete-tracing Jacobian-vector product transform.

Call the returned function with the primal arguments and a keyword-only `tangents=` pytree matching the arguments selected by `argnums`.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> value, tangent = ad.jvp(lambda x: x**2)(np.array([1.0, 2.0]), tangents=np.ones(2))
>>> value.tolist(), tangent.tolist()
([1.0, 4.0], [2.0, 4.0])
```

## vjp

```python
vjp(
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., tuple[Any, Pullback]]
```

Return a concrete value and a one-shot, tape-owning pullback.

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
    f: Callable[..., Any],
    *primals: Any,
    argnums: int | tuple[int, ...] = 0,
    **kwargs: Any,
) -> tuple[Any, LinearMap]
```

Linearize one concrete call and return its reusable real-linear map.

Close the returned map, or use it as a context manager, to release the concrete values retained by its trace.

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
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] | None = None,
    *,
    argnames: tuple[str, ...] | None = None,
) -> Callable[..., Any]
```

Return a shape-preserving dense Jacobian for real pytree inputs and outputs.

A general real-linear complex map needs two complex blocks (or one real `2m x 2n` block), so a single complex matrix would be ambiguous. Complex callers use :func:`linearize` instead.

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
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., tuple[Any, Any]]
```

Return a function computing a value and Hessian-vector product.

Supply the selected input-space vector through the keyword-only `vectors=` argument of the returned function.

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
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., Any]
```

Return an exact dense Hessian transform for real input leaves.

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
    f: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., Any]
```

Return the exact Hessian diagonal for real input leaves.

Examples:

```pycon
>>> import advect as ad
>>> import numpy as np
>>> ad.hessian_diag(lambda x: np.sum(x**3))(np.array([1.0, 2.0])).tolist()
[6.0, 12.0]
```

## checkpoint

```python
checkpoint(
    function: Callable[..., Any],
) -> Callable[..., Any]
```

Recompute `function` during reverse mode instead of saving its interior.

The initial transform is dynamic-only. The function must be pure and deterministic from explicit array/scalar inputs. Opaque residual primitives are barriers.

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
    residual: ResidualFunction,
    *,
    solve: RootSolver,
    linear_solve: LinearSolver,
    transpose_solve: LinearSolver | None = None,
) -> Callable[..., Any]
```

Differentiate a converged solution of `residual(solution, params) == 0`.

`solve(residual_at_params, initial)` performs the nonlinear solve without tracing its iterations. `linear_solve(operator, rhs)` solves the matrix-free tangent system. `transpose_solve` solves its real adjoint and defaults to `linear_solve`.

A successful callback return certifies convergence. Nonlinear and linear solver adapters must raise :class:`ImplicitSolveError` when they fail. `initial` selects a root but is explicitly nondifferentiable.

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

One-shot reverse linearization returned by :func:`vjp`.

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
__call__(cotangent: Any) -> Any
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
