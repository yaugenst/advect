# Target API

This is the public numerical-transformation surface. Examples are
normative where they express semantics. APIs omitted here are not commitments.

## Dynamic transforms

Dynamic transforms trace concrete values on every call. Ordinary Python
control flow therefore follows the current inputs.

```python
import numpy as np
import advect as ad

def loss(x, scale=1.0):
    y = np.sin(x) * scale
    return np.sum(y * y)

g = ad.grad(loss, argnums=0)
value, gradient = ad.value_and_grad(loss)(np.arange(4.0), scale=2.0)

value, tangent = ad.jvp(loss)(
    np.arange(4.0),
    scale=2.0,
    tangents=np.ones(4),
)

value, pullback = ad.vjp(lambda x: np.sin(x))(np.arange(4.0))
cotangent = pullback(np.ones(4))

value, linear = ad.linearize(lambda x: np.sin(x), np.arange(4.0))
tangent = linear(np.ones(4))
cotangent = linear.transpose()(np.ones(4))
```

With `has_aux=True`, the original callable returns `(value, aux)`. The
user-facing transformed results are:

```text
grad(f, has_aux=True)(...) -> (gradient, aux)
value_and_grad(f, has_aux=True)(...) -> (value, gradient, aux)
```

The auxiliary value is excluded from differentiation. Dynamic transforms
materialize it as a concrete sidecar. For a `StagedProgram`, it is an ordinary
staged output and must therefore satisfy the durable output contract.

`LinearMap` is the reusable derivative object. Reverse mode applies its real
adjoint; it is not a second numerical interpretation of the primitive.
`vjp` returns a one-shot `Pullback`: applying it releases retained provider
values and residuals, and a second application errors. `LinearMap` retains its
trace for repeated applications until `close()`; it is a context manager when
deterministic ownership matters.

`jacobian` accepts the same positional and named selection model as `grad`,
including input and output pytrees. A block from one input leaf to one output
leaf has shape `output_shape + input_shape`; neither side is flattened.
Advect counts selected input and output scalar coordinates after tracing. It
assembles columns with bounded forward JVP groups when the input is smaller.
For equal dimensions it chooses forward mode if any traced operation requires
structural JVP transposition, and reverse mode otherwise. A transpose-only
primitive forces reverse mode even for a wide Jacobian because no JVP exists.
Larger inputs use bounded reverse VJP groups. This is an internal cost-based
choice, not a public batching transform. The reverse path creates one
device-local basis allocation per output leaf and crosses the pytree boundary
only when returning the completed blocks.
`hvp`, `hessian`, and `hessian_diag` are conveniences over the linear
transforms. Exact `hessian_diag` may require one HVP per input coordinate.
Stochastic diagonal estimation is a separately named operation. Dense complex
Jacobians and Hessians are rejected until Advect has an explicit real-block
result type.

There is no `cache=` argument on a transform. A dynamic transform means one
concrete trace per invocation.

`debug` is the single scoped diagnostic entry point:

```python
with ad.debug():
    gradient = ad.grad(loss)(x)

with ad.debug(numerics=True):
    gradient = ad.grad(loss)(x)
```

The first form records per-operation user locations and expands live tracer
representations with bounded value summaries. The numerical form additionally
raises `NumericsError` at the first non-finite primal, JVP, or VJP value found
by a dynamic transform. Both are thread-local and restore prior state on exit;
neither exposes tracer payloads programmatically or creates a durable dynamic
graph.

`advect.testing.check_gradient` is the application-level check for an ordinary
composed scalar function. It compares a whole-function JVP with a central
finite-difference sweep and its reverse gradient. It reports custom primitives
on a failing path. `check_primitive` remains the atomic extension-author check.

`checkpoint` makes a pure callable atomic on the outer dynamic tape and
recomputes it when applying its JVP or transpose:

```python
@ad.checkpoint
def expensive_block(x):
    return np.sin(x) ** 2
```

Inputs may be pytrees. The callable must be deterministic from its explicit
inputs, and residual-bearing primitives are barriers. `stage` rejects a
checkpoint boundary because the current transform has no durable region
representation.

## Implicit differentiation

`implicit_root` differentiates a converged solution from its defining residual
without tracing solver iterations:

```python
root = ad.implicit_root(
    lambda solution, parameters: solution**2 - parameters,
    solve=nonlinear_solve,
    linear_solve=matrix_free_solve,
)

solution = root(parameters, initial=initial_guess)
gradient = ad.grad(lambda p: np.sum(root(p, initial=initial_guess)))(parameters)
```

The callback contracts are:

```text
solve(residual_at_parameters, initial) -> solution
linear_solve(operator, right_hand_side) -> solution
```

`transpose_solve` may be supplied separately and otherwise defaults to
`linear_solve` applied to the real-adjoint operator. State and parameter
pytrees are preserved. The residual and solution structures and leaf shapes
must match; the initial guess is nondifferentiable.

A solver return certifies convergence. Provider adapters raise
`ImplicitSolveError` rather than returning an unconverged iterate. JVP and
transpose rules reuse the exact solved output and one joint matrix-free
`LinearMap` for the state and parameter derivatives, then solve the tangent or
adjoint system once.

The transform is dynamic. Opaque Python callbacks reject during abstract
staging before solver execution. Nested dynamic differentiation works when the
supplied solve rules are themselves traceable; there is no exception-driven
fallback. Multiple-root selection, singular state Jacobians, and nonsmooth
residuals remain model/solver responsibilities.

## Complex differentiation

`grad` accepts a real scalar output. For a real loss and complex input it
returns the descent-ready real gradient encoded as a complex array:

```python
z = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
g = ad.grad(lambda z: np.sum(np.abs(z) ** 2))(z)
assert np.allclose(g, 2 * z)
```

The convention is

```text
g = dL/dx + i dL/dy = 2 dL/dconj(z)
dL = real(vdot(g, dz))
```

Consequently the transpose of multiplication by a complex coefficient uses
its conjugate, and cotangents are projected back into each input's real tangent
space. `real`, `imag`, `conj`, `abs`, and real-to-complex casts obey the same
rule. A complex output passed to `grad` raises with a suggestion to use
`linearize`, `jvp`, or `vjp`. The initial API has no `holomorphic=True` mode.

## Array construction and integration

Ordinary NumPy remains the primary numerical namespace. Its creation functions
dispatch through the standard `like=` argument when their object contains live
traced values:

```python
def stencil_row(x):
    row = np.array(
        [x[0], 2 * x[1], x[2]],
        dtype=x.dtype,
        like=x,
    )
    return row.sum().item()


gradient = ad.grad(stencil_row)(np.arange(3.0))
```

The anchor selects Advect's constructor handling and the array provider; it is
not a differentiable operand merely because it appears in `like=`. The
constructed object may contain the anchor or any other live values from the
same trace. `np.array` preserves its owned-copy behavior, while `np.asarray`
preserves a direct tracer when dtype, layout, and copy requirements permit it.
`np.asanyarray` follows the same numeric tracing path. Numeric rectangular
lists and tuples, dtype conversion, copy controls, order, device, and `ndmin`
participate in dynamic and staged differentiation. Object arrays and durable
ndarray-subclass identity remain outside the traced contract.

Calling `np.array(tracer)` or `np.asarray(tracer)` without `like=` cannot
dispatch before NumPy attempts concrete coercion. That failure points directly
to the `like=` rewrite; coercion inside another library instead identifies
that operation as tracer-incompatible. Advect does not patch these array
constructors. Its only process-visible NumPy patch is the scoped ambient-RNG
tripwire during abstract staging.

The provider-neutral `advect.array` and `advect.asarray` constructors remain
available when an explicit Advect operation is preferable. Their signatures
cover `obj`, `dtype`, and `copy`.

Constructor-heavy migration code may instead opt into a deliberately thin
namespace:

```python
import advect.numpy as np

row = np.array([x[0], 2 * x[1], x[2]])
```

That module overrides only `array`, `asarray`, and `asanyarray`. Every other
attribute is returned directly from the installed NumPy module, so, for
example, `advect.numpy.sin is numpy.sin`. This is a secondary convenience rather
than the advertised default.

Traced `.item()` accepts the NumPy size-one, flat-index, and tuple-index forms.
It returns a rank-zero tracer while tracing so the dependency is not detached.
The final transform boundary may unlift a rank-zero result.

Two helpers make application integration explicit:

```python
if ad.is_traced(value):
    validated_value = ad.stop_gradient(value)
```

`is_traced` inspects only the value itself and remains safe after a tracer has
escaped, although using that escaped tracer still errors. `stop_gradient`
preserves registered pytree structure and replaces traced leaves with
defensive concrete copies. It is deliberately dynamic-only: staging rejects
it because an abstract value has no concrete primal to validate or serialize.

## Functionalized source mutation

The initial mutation tier belongs to the NumPy frontend; generic Array API
providers may reject source mutation. For participating owned traced values,
mutation syntax is recorded as pure SSA updates in both concrete transforms and
abstract staging:

```python
def step(u, dt):
    u = u.copy()                         # input mutation is never implicit
    lap = u[2:] - 2 * u[1:-1] + u[:-2]
    u[1:-1] += dt * lap                 # pure index_update node
    return u

du = ad.grad(lambda u: np.sum(step(u, 0.1)))(np.arange(16.0))
```

Python aliases to the same wrapper see pointer swaps:

```python
owned = x.copy()
y = owned
owned += 1
assert y is owned
```

Inputs and non-replayable views cannot be mutated. A direct named basic-slice
view of an owned root is replayed onto that root and remains usable:

```python
owned = x.copy()
interior = owned[1:-1]
interior += update
use(owned, interior)
```

Previously created sibling views become stale after their root is updated:

```python
owned = x.copy()
v = owned[::2]
owned += 1
use(v)  # StaleViewError: copy the view or reorder the update
```

Whole-wrapper epochs are deliberately conservative. Even a view of a disjoint
region becomes stale after a root update. Aliasing operations are classified
by the frontend profile, not by a provider's runtime layout.

Python implements `x[index] += value` as getitem, in-place update, then
setitem. Advect applies the functional root update during the in-place call and
stores one thread-local acknowledgement. The generated setitem consumes that
acknowledgement as a no-op; a named-view statement simply lets it expire on the
next operation or trace finalization. Matching uses view identity, root
identity, epoch, and structural index. Chained `x[i][j] += value` is rejected
with a suggestion to write `x[i, j] += value`.

Basic indexing is supported initially. Advanced indexed assignment is rejected
until duplicate-index semantics have a deliberate API, such as a future
`.at[index].add(...)` operation.

Supported `out=` calls also update an owned tracer wrapper while emitting only
pure nodes. Masked output is defined by the differentiable select
`where(mask, new, old_out)`. Destination dtype, shape, casting, and return
identity follow the frontend contract.

## Array API frontend

Generic Array API code traces without importing an Advect-specific numerical
namespace:

```python
def normalized_energy(x):
    xp = x.__array_namespace__()
    centered = x - xp.mean(x)
    return xp.sum(centered * centered)

g = ad.grad(normalized_energy)(array_api_array)
```

Advect supports Array API 2022.12, 2023.12, and 2024.12 as ordered profiles.
At the start of a dynamic transform it requests them newest first and selects
the newest revision every array input from one provider can serve. Mixed
providers and inputs that cannot serve any supported revision fail before
tracing. A provider may report a newer revision after accepting the request;
Advect keeps that provider metadata while exposing only the selected contract
through the trace-aware namespace.

Staging makes the portability target inspectable and durable:

```python
program = ad.stage(
    normalized_energy,
    array_api_array,
    array_api_version="2023.12",
)
assert program.array_api_version == "2023.12"
```

Without an explicit target, example-based staging negotiates the newest common
revision. Specification-only staging defaults to 2024.12. The target is stored
in the `advect-array-1` graph header, preserved by staged differentiation and
serialization, and requested from runtime inputs before graph evaluation.

The returned namespace emits canonical `array.*` primitives admitted by the
selected profile. Concrete promotion, dtype, device, and shape behavior comes
from the conforming provider and is recorded by Advect; Advect does not carry a
competing array type system. Weak-scalar cases important to scientific code are
differential tested, including preserving complex64 when a Python complex
scalar is combined with float32 data.

An operation outside Advect's explicit table rejects at the trace boundary.
Data-dependent result shapes need an explicit dynamic rule; abstract staging
rejects them unless the operation has static output semantics.

The portable scientific gate uses the same staged primal and serialized staged
derivative on NumPy and `array-api-strict` for every supported revision. CuPy
uses the same manual gate; the recorded configurations cover all three
supported revisions. The admitted staged
extension surface contains the ten `fft`/`ifft` real, complex, n-dimensional,
and shift forms plus `linalg.solve`. The exact provider and official-suite
matrix is in
[Array Provider Qualification](implementation/array-provider-qualification.md).

NumPy remains a separate first-class frontend through `__array_ufunc__` and
`__array_function__`. Advect does not patch NumPy array constructors; its only
process-visible NumPy patch is scoped to the ambient-RNG staging policy.
Differences between the concrete NumPy frontend and the provider-neutral Array
API frontend are explicit and tested. They share canonical operations and
derivative semantics, not foreign calling conventions.

NumPy 2.0 through 2.5 use a separate first-class profile selected from the
installed minor. NumPy 2.0 defaults to Array API 2022.12, 2.1-2.2 to 2023.12,
and 2.3-2.5 to 2024.12. A registered single-output array function with an
upstream `out=` parameter updates one owned tracer through pure SSA. The
dynamic path asks that installed NumPy minor to validate the exact
function-specific shape and casting rules against a private destination, then
records the corresponding pure result, mask, and cast. Staging records the
same semantics for functions with abstract rules. Ordinary calls without
`out=` retain their direct fast path.

Ufunc methods are independent call forms. Advect lowers `add.reduce`,
`multiply.reduce`, `add.accumulate`, and `multiply.accumulate` to their
equivalent reduction or cumulative functions, and lowers any supported binary
single-output `outer` through broadcasting and the ordinary ufunc call.
`reduceat`, `at`, and unsupported generalized-ufunc controls raise by method or
parameter name. The extension catalog lists only explicitly supported method
forms rather than letting `__call__` imply them.

## Optional SciPy frontend

The `advect.scipy` module ships with Advect. Installing `advect[scipy]` supplies its
SciPy dependency and enables this deliberately bounded namespace:

```python
from advect.scipy import ndimage, special


def log_likelihood(x):
    return np.sum(
        special.gammaln(x)
        + special.ndtr(x)
        - special.logsumexp(x, axis=-1, keepdims=True)
    )


def smooth_design(x):
    blurred = ndimage.gaussian_filter(x, sigma=1.2, mode="reflect")
    return ndimage.grey_closing(blurred, size=3)
```

The admitted special functions are `gammaln`, `digamma`, `polygamma`, `erf`,
`erfc`, `erfcx`, `erfinv`, `expit`, `log_expit`, `ndtr`, `log_ndtr`, `ndtri`,
`logsumexp`, `softmax`, and `log_softmax`. Each function has concrete,
abstract, JVP, and reverse-mode semantics through a stable primitive identity.
The unary functions accept their complete SciPy 1.18 ufunc controls, including
masked functionalized `out=`; `polygamma` broadcasts array-valued orders and
inputs; and `logsumexp` supports differentiable weights and signed real or
complex results.

The admitted `ndimage` surface is:

- `gaussian_filter`, `gaussian_filter1d`, `uniform_filter`, and
  `uniform_filter1d`;
- `convolve`, `correlate`, `convolve1d`, `correlate1d`, `laplace`,
  `gaussian_laplace`, `sobel`, and `prewitt`;
- `maximum_filter`, `minimum_filter`, `maximum_filter1d`, `minimum_filter1d`,
  `median_filter`, `rank_filter`, and `percentile_filter`;
- `grey_dilation`, `grey_erosion`, `grey_opening`, `grey_closing`,
  `morphological_gradient`, `morphological_laplace`, `white_tophat`, and
  `black_tophat`.

Forward calls preserve SciPy 1.18 signatures and semantics, including output
arrays and dtype specifications. Linear rules use exact boundary-aware
stencils and differentiate convolution/correlation weights. Extrema, rank, and
greyscale morphology use an equal-share subgradient across equal winning
window slots; structures and constant boundary values are live operands.
All supported SciPy functions are NumPy-backed. Import `advect.scipy` before
loading a serialized artifact that references their primitive identities.

`root_solver` and `gmres_solver` are callback factories for `implicit_root`:

```python
from advect.scipy.optimize import root_solver
from advect.scipy.sparse.linalg import gmres_solver

root = ad.implicit_root(
    residual,
    solve=root_solver(),
    linear_solve=gmres_solver(rtol=1e-10),
)
```

They accept NumPy arrays, NumPy scalars, and Python numeric scalars for
first-order dynamic differentiation; they are not staged SciPy operations.
Array shape and scalar container category are preserved. `root_solver` follows
SciPy's dtype promotion, while `gmres_solver` restores the inexact
right-hand-side dtype. Nonconvergence raises `ImplicitSolveError`, and complex
real-linear operators use a doubled real representation. Base Advect neither
imports nor depends on SciPy.

## Optional xarray pytrees

The `advect.xarray` module also ships with Advect. Installing `advect[xarray]`
supplies xarray; importing the module then registers `xarray.DataArray` and
`xarray.Dataset` as dynamic pytrees:

```python
import advect.xarray

gradient = ad.grad(
    lambda field: ((field - field.mean("x")) ** 2).sum()
)(field)
```

Floating- and complex-valued data buffers are differentiable children.
Integer, boolean, string, and object data variables reject at the pytree
boundary. Dimensions, coordinates, names, variable order, and attributes are
equality-safe static metadata restored around transformed leaves. xarray
continues to own alignment and named-axis semantics; Advect differentiates the
array operations it emits.

The integration is explicit and dynamic. It is not an array backend, and
custom xarray pytree nodes do not cross the current durable artifact boundary.
Stage the raw array kernel and reconstruct labels outside the program when
durable reuse is required.

## Optional host-autodiff bridges

The built-in `advect.interop` package contains three explicitly imported VJP
bridges. Each module is enabled by its matching extra and exports one `wrap`
function:

```python
from advect.interop.autograd import wrap as autograd_wrap
from advect.interop.jax import wrap as jax_wrap
from advect.interop.torch import wrap as torch_wrap
```

All three frameworks accept `wrap(function)`. JAX also accepts an optional
`result_shape_dtypes=` pytree, normally made from `jax.ShapeDtypeStruct`, and
`has_aux=True`. Without a result specification, concrete eager calls and eager
reverse mode execute directly; JIT compilation or abstract shape evaluation
raises an actionable error. Supplying it enables the callback path needed by
`jax.jit` and `jax.eval_shape`; it does not enable `jax.vmap`. With
`has_aux=True`, the callable returns `(value, aux)`; `value` is the nonempty
differentiable output and `aux` is a nondifferentiable JAX-compatible pytree.
Wrapped functions take one or more positional tuple, list, or dictionary
pytrees containing standard NumPy floating or complex values; all input leaves
are differentiable, while static configuration belongs in the function
closure. Custom containers require matching Advect and host registrations.

These are first-order, dynamic VJP boundaries rather than providers. They do
not make direct JAX or PyTorch arrays valid inputs to `advect.grad`, and they do
not stage host-framework programs. PyTorch crosses through host NumPy and
consumes one retained pullback. HIPS Autograd retains the same invocation's
pullback. JAX executes concrete calls directly, uses a pure callback when a
static output signature enables staging, and replays the callable during
backward. The adapters translate their respective complex cotangent conventions
at these boundaries.

## Abstract staging

`stage` is the opt-in durable boundary:

```python
program = ad.stage(normalized_energy, x)

result = program(x)
print(program.graph)
print(program.signature)
print(program.constants)
print(program.compile_seconds)
print(program.optimization.nodes_before, program.optimization.nodes_after)
```

`stage(f, *examples, kw_specs=...)` infers the positional shape, dtype, device,
and scalar-weakness signature from representative values. The values
themselves are not passed to the abstract trace or captured by the program.
`stage(f, specs=..., kw_specs=...)` declares the same signature without
representative values. One form is required because the returned
`StagedProgram` always owns one fully compiled graph; staging is not an
implicit multi-signature JIT cache.

Python branching on an abstract value, iteration over value-dependent extents,
ambient randomness, and other intercepted stateful provider operations raise
with a staged rewrite. Use an array control-flow primitive or pass explicit
state.

Transforms compose with staged programs:

```python
program = ad.stage(loss, specs=(spec,))
gradient_program = ad.grad(program)
value_gradient_program = ad.value_and_grad(program)
pullback_program = ad.vjp_program(program)
```

All three results are `StagedProgram` objects. Advect differentiates the source
program's one signature, optimizes the complete derivative graph with the
normal staged pipeline, and reuses a prebound execution plan on warm calls.
There is no dynamic tape or fresh reverse sweep at runtime, and each derivative
program round-trips through `to_dict()` and `from_dict()` like any other staged
artifact.

The VJP program accepts the primal arguments plus one explicit keyword-only
cotangent with the primal output structure:

```python
y_bar = np.ones_like(program(x))
x_bar = pullback_program(x, cotangent=y_bar)
```

Multi-argument selection, pytrees, keyword `argnames`, complex real gradients,
functionalized mutation, and traceable custom primitive rules retain their
ordinary `grad` semantics. Opaque residual primitives are a compile-time
barrier because invocation-local residuals are deliberately absent from
`GraphStore`. Ordinary `vjp` and `linearize` still return invocation-local
callable objects and therefore remain dynamic transforms.

The compiled signature owns one prebound native execution plan. Repeated calls
reuse its structure and evaluator bindings while allocating fresh invocation
values. A different signature is a different staged program.

## Constants

Ordinary closed-over arrays and literal array operands become constant nodes:

```python
kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4
program = ad.stage(lambda x: x * kernel, specs=(spec,))

for record in program.constants:
    print(record.origin, record.location, record.shape, record.dtype,
          record.bytes, record.digest)
```

Concrete capture requires no wrapper and is inspectable rather than forbidden.
Known ambient RNG calls raise while staging; random programs take explicit
keys/state as inputs. Static arguments and pytree metadata are copied into the
closed artifact value model before tracing. Captured arrays are detached at
compile time; the program materializes them once per runtime namespace and
device and reuses them on warm calls.

The live artifact stores canonical raw numeric bytes. `to_dict()` emits one
`advect.ssa-program` envelope at version 2. Its nested graph artifact version 2
represents those bytes as lowercase hexadecimal and records
`required_array_api_version`.

## Custom primitives

A primitive starts with its concrete implementation:

```python
@ad.primitive(
    static_argnames=("config",),
)
def solve(a, b, *, config): ...

@solve.def_abstract
def solve_abstract(a, b, *, config): ...

@solve.def_jvp
def solve_jvp(output, primals, tangents, *, config): ...

# Optional when structural transposition of the JVP is insufficient.
@solve.def_transpose
def solve_transpose(cotangent, primals, output, *, config): ...
```

Advect infers the operation identity from the implementation's module and
qualified name. A library may pass `name="acme.solve"` when a serialized
program needs an identity independent of Python refactoring. Advect owns graph
schema revisions; authors do not declare versions or compatibility ranges. The
loading environment is responsible for providing the matching implementation;
Advect does not infer semantic compatibility after application code changes.

Rule bodies use traceable primitives when nested differentiation is promised.
Advect structurally validates that a JVP is linear—not affine or nonlinear—in
its tangent inputs before synthesizing a transpose. It never probes a JVP on
basis vectors to guess a pullback.

The decorated implementation defines the call signature, including ordinary
Python defaults. Concrete calls use those defaults directly. If a default
participates in tracing as static metadata, the normal closed artifact codec
validates and snapshots its actual value at that boundary. There is one
implementation, so backend portability comes from the operations used in that
function rather than a string-keyed provider table.

When an exact adjoint needs invocation-local implementation state, declare it on the
same primitive:

```python
@ad.primitive(name="acme.remote_solve", residual=True)
def remote_solve(a, b):
    output, handle = submit_solve(a, b)
    return ad.PrimitiveResult(output, handle, release=close_handle)

@remote_solve.def_transpose
def remote_solve_transpose(cotangent, primals, output, handle):
    return pullback_solve(handle, cotangent)
```

Only `output` is public. Advect pairs `handle` with that exact invocation and
keeps it alive across repeated applications of a reusable `LinearMap` until
the map is closed. Opaque residual primitives deliberately reject nested
differentiation.

Outside differentiation there is no transpose consumer, so Advect releases the
residual before the primitive call returns. The same is true for plain staged
replay. If a staged program is called under a dynamic transform, replay
preserves the primitive as one atomic node and transfers the residual to the
enclosing dynamic tape.

Primitive authors can run the optional authoring checks from
`advect.testing`:

```python
from advect.testing import check_primitive

check_primitive(
    solve,
    primals=(a, b),
    check=("abstract", "jvp", "transpose", "complex", "nested", "stage"),
)
```

Failures name the primitive, the missing rule, and the contract or identity
that failed.

## Pytrees and arguments

Tuples, lists, dictionaries, and custom nodes are pytrees in dynamic
transforms. A custom type may be registered explicitly through `advect.pytree`,
or a model base class may define inherited `__advect_tree_flatten__` and
`__advect_tree_unflatten__` hooks. The flatten hook returns a tuple of dynamic
children plus equality-safe static metadata; the classmethod reconstructs the
concrete subclass from that metadata and the transformed children. An exact
registration takes precedence over inherited hooks.

Durable staging accepts the built-in containers and `Static`; a custom node
needs an explicit stable serialization codec before it can be staged.
`grad` and `value_and_grad` accept both positional `argnums` and keyword
`argnames`; `jacobian` and `vjp_program` use the same selection model. The
remaining linear-map and higher-order transforms select positional inputs with
`argnums`. Selected structures are preserved in results. Primitive and stage
boundaries declare static arguments explicitly; Advect never treats a value as
static simply because it could not trace it.

## Durable artifacts

A staged graph records independent versions for:

- the graph file format;
- the core primitive opset;
- the semantic profile;
- the compiler and fixed optimizer;
- Advect's graph schemas for every recorded operation.
- the required Array API revision.

Loading rejects unknown Advect-owned versions before execution. Custom
implementations are linked explicitly by primitive name. Runtime buffers,
callbacks, residuals, and resources are not serialized. `program.graph`
provides immutable structural inspection but does not expose mutable constant
payloads or raw native serialization; `program.to_dict()` returns detached
public artifact data.

## Safety and diagnostics

Tracer payloads are private. Conversion through `__array__`, array-interface,
DLPack, or an equivalent raw-buffer escape raises rather than silently
detaching. A tracer used after its trace closes raises `EscapedTracerError`.

Expected errors include a filtered user location and a concrete rewrite. Full
per-node source maps are debug-only; lightweight locations for views, mutation,
pending updates, constants, and trace boundaries are part of the shipping
configuration and its latency benchmark.

Staged provider exceptions retain their original type and traceback and append
a bounded local graph slice. `StagedProgram` and `GraphStore` have useful,
bounded text representations; `program.graph` remains the expert structural
API rather than a required explanation step.

## Current boundary

The current public surface has no workflow runtime, object store, public
buffer-donation control, SciPy/xarray compatibility beyond the named surfaces, general
`vmap`, or dense complex Hessian.

Manual dynamic `checkpoint`, the built-in Array API compatibility bridge,
direct named basic-slice mutation, and internal staged temporary donation are
implemented. Single-device CuPy has bounded
three-profile donation and scientific-transform evidence. Its contracts and
remaining boundaries are recorded in
[Runtime Extension Boundaries](decisions/2026-07-24-runtime-extension-boundaries.md).

Workflow persistence and generic object storage remain independent layers.
Dynamic tapes are not converted to durable graphs and optimized before
backward. Users cannot select or apply raw compiler passes, and Advect does not
promise complete NumPy view-replay or overlap semantics. Compatibility aliases
for the former graph-first API remain intentionally absent.
