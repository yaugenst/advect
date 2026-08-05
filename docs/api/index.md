# API Reference

Advect's public API is intentionally small. The [tutorials](../tutorials/index.md)
show it in use; the normative semantics live in the repository's
[target API](https://github.com/yaugenst/advect/blob/main/design/target-api.md).
Reference pages:

- [transforms(3)](transforms.md) — dynamic differentiation entry points
- [staging(3)](staging.md) — the durable-graph boundary
- [primitives(3)](primitives.md) — custom-operation authoring
- [arrays(3)](arrays.md) — construction, introspection, pytrees
- [host interop](interop.md) — first-order PyTorch, JAX, and HIPS Autograd VJP bridges
- [errors(3)](errors.md) — the exception hierarchy

## Dynamic differentiation

- `advect.grad` and `advect.value_and_grad`
- `advect.jvp`, `advect.vjp`, and `advect.linearize`
- `advect.jacobian`, `advect.hvp`, `advect.hessian`, and `advect.hessian_diag`
- `advect.checkpoint`
- `advect.debug`
- `advect.implicit_root`
- `advect.Pullback` and `advect.LinearMap`

These transforms trace concrete values per invocation. The `Pullback` returned
by `vjp` is one-shot and releases its invocation payloads when called;
`LinearMap` is reusable and explicitly closed.
`grad`, `value_and_grad`, and `jacobian` support `argnums` and `argnames`; the
remaining dynamic transforms use `argnums`. `checkpoint(f)` rematerializes a
pure callable during dynamic differentiation and is intentionally rejected by
`stage`. `implicit_root` differentiates a converged root through matrix-free
tangent and real-adjoint solves; the initial guess and opaque solver iterations
are not differentiated.

With `has_aux=True`, the transformed function must return `(value, aux)`.
`grad(f, has_aux=True)` returns `(gradient, aux)`, while
`value_and_grad(f, has_aux=True)` returns `(value, gradient, aux)`. The
auxiliary value is a transparent sidecar and is excluded from differentiation.

## Staging

- `advect.ArraySpec` and `advect.StaticSpec`
- `advect.stage` and `advect.StagedProgram`
- `advect.vjp_program`
- `advect.ConstantRecord`, `advect.OptimizationReport`, and
  `advect.OptimizationPass`

`stage` accepts concrete example arguments or explicit `specs=` and compiles
exactly one signature into one immutable graph. It is the only public
durable-graph boundary. Former manual
`trace`/`input`/`output`/`execute` and `Graph` APIs are not public.
`vjp_program(program)` derives an optimized, serializable staged pullback whose
keyword-only `cotangent` input matches the primal output pytree.
An optional `array_api_version=` target selects Array API 2022.12, 2023.12, or
2024.12. Example-based staging otherwise negotiates the newest common revision;
spec-only staging defaults to 2024.12. The selected target is available as
`program.array_api_version` and is preserved by derived programs and
serialization.

## Primitive authoring

- `advect.primitive` and `advect.PrimitiveResult`
- `advect.testing.check_gradient` for ordinary composed functions
- `advect.testing.check_primitive` for atomic extension contracts

A primitive is defined from one implementation function and may add abstract
evaluation, a JVP, and an optional explicit transpose. Advect owns graph schema
revisions; authors may supply an explicit stable name but no schema version or
provider dispatch table. A residual-bearing implementation returns
`PrimitiveResult(output, residual, release=None)`; callers receive only
`output`, while Advect owns the residual and optional release callback for the
matching derivative invocation. A primitive name links an artifact to code; it
does not make Advect infer semantic compatibility after that code changes.
The original decorated handle is the authoring surface; Advect does not expose
a second string-based lookup API or mutable capability record.

## Frontends and structure

- `advect.support_catalog()` returns the runtime-derived, machine-readable
  primitive matrix and separate lists of supported Array API, NumPy, and SciPy
  extension functions. Dynamic, staged, and serialized are separate support
  claims; registration alone is not presented as support.
- `advect.array` and `advect.asarray` explicitly construct provider-preserving
  arrays from direct tracers or rectangular nested tracer sequences.
- Ordinary NumPy `array`, `asarray`, and `asanyarray` accept a traced `like=`
  anchor. This is the preferred constructor path. `advect.numpy` is a secondary
  namespace that overrides those constructors and delegates every other
  attribute directly to NumPy.
- `advect.is_traced` checks one value without reading its payload.
- `advect.stop_gradient` preserves pytree structure while replacing dynamic
  traced leaves with defensive concrete copies; abstract staging rejects it.
- Traced arrays implement differentiable size-one, flat-index, and tuple-index
  `.item()` forms by keeping the result rank zero during tracing.
- Array API code uses the namespace returned by a traced array's
  `__array_namespace__` method.
- A private built-in bridge resolves arrays supported by `array-api-compat`, a
  base dependency. Its fixed fallback is configured when `advect` is imported;
  users do not import or register a provider module. CuPy requires the separate
  manual GPU qualification for each supported revision.
- NumPy code is intercepted through NumPy's array protocols, including its
  standard `like=` creation dispatch.
- Registered single-output NumPy array functions with `out=` functionalize an
  owned tracer using the installed supported NumPy minor's own shape and
  casting validation.
  `add`/`multiply` reductions and accumulations and binary single-output ufunc
  `outer` calls are explicit supported method forms; other ufunc methods do
  not inherit support from the ufunc call.
- `advect.pytree` registers and manipulates structured inputs and outputs.
  Application model base classes may instead provide inherited
  `__advect_tree_flatten__` and `__advect_tree_unflatten__` hooks.
- The built-in `advect.scipy` module adds a bounded set of stable special
  functions and image filters, plus concrete `root_solver`/`gmres_solver`
  callback factories, when `advect[scipy]` is installed.
- Explicitly importing the built-in `advect.xarray` module registers floating- or
  complex-valued `DataArray` and `Dataset` objects as dynamic pytrees when
  `advect[xarray]` is installed; their labels are static.
- Provider modules may register input handlers explicitly when imported;
  installed packages are never discovered implicitly.
- Transform entry points lift selected real Python scalars to zero-dimensional
  `float64` arrays and return corresponding derivative results as Python
  scalars. There is no separate scalar-operation namespace.

## Errors

- `advect.AdvectError`
- `advect.ImplicitSolveError`
- `advect.TracingError` and `advect.EscapedTracerError`
- `advect.MutationError` and `advect.StaleViewError`
- `advect.NoJVPError`, `advect.NoVJPError`, and `advect.MissingPrimitiveRuleError`
- `advect.NumericsError`

## SciPy special functions (`scipy` extra)

The built-in `advect.scipy.special` module exports NumPy-backed primitives when
SciPy is installed:

- `gammaln(x)`
- `digamma(x)`
- `polygamma(n, x)`
- `erf(x)`
- `erfc(x)`, `erfcx(x)`, and `erfinv(x)`
- `expit(x)`
- `log_expit(x)`
- `ndtr(x)`, `log_ndtr(x)`, and `ndtri(x)`
- `logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False)`
- `softmax(x, axis=None)` and `log_softmax(x, axis=None)`

These callables expose SciPy 1.18-compatible functions for the admitted NumPy
provider. Unary ufuncs accept the standard keyword arguments, including
functionalized `out=` on an owned tracer. `polygamma` broadcasts array-valued
`n` and `x`; `logsumexp` supports weights and signed results.

## SciPy image filters (`scipy` extra)

`advect.scipy.ndimage` exports the following SciPy 1.18-compatible functions:

- Gaussian and uniform filters: `gaussian_filter`, `gaussian_filter1d`,
  `uniform_filter`, and `uniform_filter1d`;
- linear filters: `convolve`, `convolve1d`, `correlate`, `correlate1d`,
  `laplace`, `gaussian_laplace`, `sobel`, and `prewitt`;
- order filters: `maximum_filter`, `maximum_filter1d`, `minimum_filter`,
  `minimum_filter1d`, `median_filter`, `rank_filter`, and `percentile_filter`;
- greyscale morphology: `grey_dilation`, `grey_erosion`, `grey_opening`,
  `grey_closing`, `morphological_gradient`, `morphological_laplace`,
  `white_tophat`, and `black_tophat`.

All named calls are differentiable and support dynamic execution, staging, and
serialization. The installed extension and exact lowering mode are inspectable
through `advect.support_catalog()`.

## Optional SciPy solver callbacks

- `advect.scipy.optimize.root_solver(*, method=None, options=None)`
- `advect.scipy.sparse.linalg.gmres_solver(*, rtol=1e-5, atol=0.0, maxiter=None)`

Both factories return shape- and scalar-category-preserving callbacks for
`implicit_root`. They accept NumPy arrays, NumPy scalars, and Python numeric
scalars and are a first-order dynamic boundary.

## xarray integration (`xarray` extra)

The built-in `advect.xarray` module is usable when xarray is installed. Importing
it registers the supported containers; calling `advect.xarray.register()`
explicitly performs the same idempotent registration. Floating- and
complex-valued data buffers become pytree leaves; labels remain static metadata.
