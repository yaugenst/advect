# Core Requirements

These requirements are normative for the numerical-transformation package.
They supersede contracts from the former graph-first product surface.

## R1: Small semantic core

- Advect's differentiable runtime consists of primitives, tracers, a native
  ephemeral SSA owner, real-linear maps, and pytrees.
- One required Rust `RawArena` defines compact SSA structure. It lives in the
  Python-independent `advect-runtime` crate. A PyO3 `DynamicTape` owns one
  concrete invocation; runtime-owned `GraphBuilder` and immutable
  `GraphStore` own abstract construction and durable staging.
- Python owns mutable tracer wrappers, provider and derivative callbacks,
  pytrees, and API orchestration. It owns no graph or tape data structure.
- All differentiable numerical leaves use the array tracer. Transform
  boundaries lift selected real Python scalars to zero-dimensional `float64`
  provider arrays and unlift their derivative results; there is no parallel
  Python-scalar tracer or derivative engine.
- Dynamic calls do not require a durable graph artifact.
- Abstract staging records directly into `GraphBuilder` and produces the
  native immutable `GraphStore`.
- Runtime code must not maintain overlapping derivative engines or compatibility
  paths.

## R2: Primitive contract

Every differentiable operation has one stable primitive identity. Advect owns
the graph schema revision attached to that identity. A custom primitive
declares:

- one concrete implementation;
- abstract output evaluation when it supports staging;
- static and nondifferentiable arguments;
- a traceable JVP rule that receives the already-computed primal output when
  the primitive supports forward mode or structural transposition;
- an explicit transpose when it cannot be derived safely, including
  transpose-only reverse mode without a JVP;
- whether its implementation returns an invocation-local residual.

Transforms use the installed rules directly. Missing-rule errors name the
primitive and required rule. Structural transposition validates a JVP's
real-linearity when it is used, and higher-order differentiation requires rule
bodies that themselves trace. A transpose-only primitive has no forward-mode
capability, but a traceable non-residual transpose may compose under reverse
mode. Residual primitives require an explicit transpose and are first-order
boundaries. Author checks report contract failures but do not mutate runtime
capability state.

A residual-capable implementation returns
`PrimitiveResult(output, residual, release=None)`. Advect passes the exact
same-invocation output and residual to its transpose, releases owned residuals
deterministically, and never serializes them. During differentiation, ownership
transfers into the invocation-local `DynamicTape`. An ordinary primitive call,
including plain staged replay, releases the unused residual before returning;
staged replay under an enclosing dynamic transform records one atomic custom
node and transfers its residual to that outer tape. Residuals never enter
`RawArena`, `GraphStore`, node attributes, staged constants, or output pytrees.
Opaque residual primitives are first-order-only.
Because `GraphStore` has no residual table, they are also barriers to staged
derivative compilation.

## R3: Concrete dynamic tracing

- `grad`, `value_and_grad`, `vjp`, `jvp`, `jacobian`, `hvp`, `hessian`, and
  `hessian_diag` applied to ordinary Python callables trace with concrete
  values on every call.
- A selected real Python `int` or `float` primal is normalized to a
  zero-dimensional `float64` provider array before tracing. Corresponding
  rank-zero derivative results are returned as Python scalars. Booleans are
  nondifferentiable, and complex Python scalar primals require an explicit
  array until a separate scalar convention is admitted.
- Python control flow and data-dependent shapes use the current values.
- `DynamicTape` owns one `RawArena`; node identity is an arena position and
  operation identity is a dense arena-local ID backed by a stable
  name/schema-version table.
- Concrete values, literals, residuals, operand positions, and derivative
  metadata are invocation-local `DynamicTape` side tables rather than SSA
  fields.
- Rust owns forward and reverse arena traversal, derivative slots, retention,
  and release. Python-authored derivative callbacks remain reentrant.
- Built-in operation definitions already contain their derivative bindings;
  each concrete trace freezes the bindings for the operations it recorded.
- In a nested transform, an outer tracer used as an unselected argument is an
  explicit passive input to the inner tape. Its value remains available to
  derivative rules, but activity does not propagate into that input and it is
  absent from the selected gradient result.
- Every frozen reverse binding carries explicit output, all-or-no-primal, and
  residual requirements. A consuming reverse prunes zero-use payloads before
  its first callback and releases retained provider values and residuals at
  their last reverse use. Reusable `LinearMap` application is non-consuming.
- Dynamic tracing does not compute durable hashes, fingerprints,
  canonicalization, serialization payloads, or optimization plans and never
  runs staged optimization passes.
- A dynamic trace is thread-affine and closes after its transform completes.

## R4: Abstract staging

- `stage(f, *examples, kw_specs=...)` infers, or
  `stage(f, specs=..., kw_specs=...)` declares, exactly one explicit positional
  and keyword shape/dtype/device signature. Example data is not available to
  the abstract trace.
- Abstract tracing records directly into the native `GraphBuilder`.
- Value-dependent truth, iteration, Python branching, or result shape fails
  unless represented by a supported staged primitive.
- Construction compiles immediately and returns one `StagedProgram` containing
  one immutable `GraphStore`. Omitting both examples and specifications is an
  error rather than an implicit JIT-cache boundary.
- Finishing a staged trace validates and canonicalizes it, then always runs
  the fixed `DCE -> simplify -> CSE` pipeline before producing `GraphStore`.
  There is no user-selectable optimizer or placeholder fusion pass.
- Custom and remote primitives are optimization barriers.
- The compiled signature owns one immutable native `GraphExecutionPlan`.
  The plan resolves structure and evaluator bindings once and is reused across
  calls. Each call allocates only its dense value table and temporary operand
  tuples; native bulk traversal does not reconstruct Python graph nodes.
- A staged program exposes its signature, graph, compilation time, constant
  manifest, pre/post node counts, and per-pass rewrite reports directly.
- Multi-signature caching and `cache=` modes are not `StagedProgram`
  responsibilities or public arguments.
- `grad(program)`, `value_and_grad(program)`, and `vjp_program(program)`
  compile that program's one signature, pass the derivative through the same
  fixed optimizer, and return another ordinary serializable `StagedProgram`.
- A staged VJP takes the primal program's call arguments plus one reserved
  keyword-only `cotangent` pytree. Its structure, shape, dtype, device, and
  weakness come from the primal output specifications.
- A warm staged derivative call executes only its prebound
  `GraphExecutionPlan`; it does not create a `DynamicTape` or run a reverse
  sweep.

## R5: Functionalized source mutation

- Source mutation is an opt-in frontend capability. The initial complete tier
  belongs to the NumPy frontend; generic Array API providers may reject it.
- An owned mutable tracer wrapper points to the current immutable SSA value.
- In-place operators, supported setitem, and supported `out=` calls emit pure
  nodes and advance that pointer during concrete tracing and abstract staging.
- All Python references to the same wrapper observe the update.
- Input wrappers reject mutation immediately; `copy()` creates an owned wrapper.
- Mutation never creates an impure IR node and is never validated during
  backward by a tensor version counter.

## R6: View epochs and indexed augmented assignment

- These alias and pending-update rules define the NumPy mutation tier.
- Aliasing views record their root wrapper and creation epoch.
- Advancing a root wrapper invalidates older views; using one raises
  `StaleViewError` with creation, mutation, and use locations when available.
- Alias classification comes from the frontend semantic profile. It may be
  conservative and need not prove region overlap.
- The trace context permits one pending indexed update for Python's
  getitem/in-place/setitem augmented-assignment protocol.
- A direct basic view of an owned root applies its update immediately, advances
  the root epoch, refreshes itself, and leaves a completed acknowledgement.
- A matching generated setitem consumes that acknowledgement using returned-view
  identity, destination-wrapper identity, epoch, and structural index
  validation. A named-view statement lets it expire harmlessly.
- Dynamic basic-slice `+=` may fuse its getitem/add/setitem protocol into one
  pure additive `index_update`; this is dead-value elimination, not an IR
  mutation effect.
- Mutation through a nested, transformed, broadcast, or advanced-index view is
  unsupported.

## R7: Physical storage is not semantics

- Logical updates are pure; buffer donation is not part of observable
  semantics.
- Staged execution may overwrite an internally owned last-use temporary only
  after proving writable compatible storage and the absence of live aliases.
- Inputs, constants, graph outputs, dynamic values, and failed proofs use fresh
  storage.
- The Python-independent execution plan retains its graph store and owns use
  counts, conservative alias-root-set accounting, and donation eligibility. A
  linked host declares ownership and aliases and may accept or decline the
  selected donor.
- Tracer payloads are private. `__array__`, array-interface, and DLPack export
  raise instead of silently detaching.
- `is_traced(value)` inspects tracer identity without reading its payload.
  `stop_gradient(value)` is the one explicit dynamic detachment boundary: it
  preserves pytree structure, replaces traced leaves with defensive concrete
  copies, and rejects abstract staging.

## R8: Real-linear autodiff

- `linearize` returns a primal result and a reusable `LinearMap`.
- `vjp` returns a primal result and a one-shot `Pullback` that releases its
  retained trace after application.
- JVP applies the map; VJP applies its real adjoint.
- Reverse mode is the adjoint of linearization, not a separately authored
  numerical interpretation.
- Tangent dependence is validated structurally for real-linearity.
- Higher-order differentiation composes transforms and never chooses a mode by
  catching runtime failures.
- Multi-seed application is internal to `LinearMap`; it does not commit Advect to
  a general batching transform. Dynamic application uses bounded groups of at
  most 16 seeds. Each group shares one native arena traversal and one snapshot
  of each active node, while primitive derivative callbacks still receive one
  ordinary tangent or cotangent at a time.
- `jacobian` accepts positional and named selected arguments and arbitrary
  input and output pytrees. Each derivative block preserves
  `output_leaf.shape + input_leaf.shape`; the result is an output pytree whose
  leaves contain the selected-input gradient pytree. It uses forward mode when
  the selected input has fewer scalar coordinates than the output and every
  active operation has a JVP. A transpose-only operation forces reverse mode
  regardless of dimensions. Equal-size maps use forward mode when the trace
  contains a JVP-first operation whose transpose is synthesized and reverse
  mode otherwise; larger inputs use reverse mode. Each block is represented in
  its input tangent-space dtype, so mode selection cannot change the result
  dtype. The choice is structural and never an exception-driven fallback.
  Reverse assembly allocates the
  standard-basis rows once per output leaf on that leaf's provider and device,
  sends node-ID cotangent tables directly to the native sweep, and reconstructs
  pytrees only for the completed derivative blocks.

## R9: Complex convention

For a real scalar loss over complex inputs, Advect returns
`dL/dx + 1j*dL/dy == 2*dL/dconj(z)`. Therefore:

- `grad(abs(z)**2) == 2*z`;
- `dL == real(vdot(grad(L), dz))`;
- transpose identities use the real inner product;
- non-holomorphic primitives may depend on both a tangent and its conjugate;
- `grad` rejects complex outputs;
- complex-output differentiation uses `linearize`, `jvp`, or `vjp`;
- dense complex Hessians are rejected until an explicit block result exists.

There is no `holomorphic=True` promise in the initial API.

## R10: Array API frontend

- A backend-neutral traced array implements `__array_namespace__`.
- The returned namespace emits canonical `array.*` and explicit extension
  primitives.
- Advect supports ordered Array API 2022.12, 2023.12, and 2024.12 profiles. The
  oldest revision is a complete frozen base and later revisions are explicit
  additions and contract overrides. Canonical operations and derivative rules
  are shared across profiles.
- At the start of a concrete transform, discovery requests supported revisions
  newest first from every array input and selects the newest revision all
  inputs from one provider can serve. Unversioned, incompatible, or mixed
  providers are rejected before tracing. A provider may report a newer
  revision after accepting the explicit request; Advect retains that metadata
  while enforcing the selected profile.
- `stage(..., array_api_version=...)` accepts an explicit supported target.
  Example-based staging otherwise negotiates from its providers, while
  specification-only staging defaults to 2024.12. The abstract namespace
  exposes only that target, and the durable graph records it as
  `required_array_api_version` for runtime validation and derived programs.
- Concrete promotion, dtype, device, and shape behavior comes from a provider
  that implements the selected version; Advect records and validates the
  resulting specifications instead of maintaining a second array type system.
  Staged calls validate shape, dtype, weakness, declared device, and the
  required revision.
- Python weak-scalar cases that matter to scientific workloads are differential
  tested against that provider, including float32/complex64 preservation.
- An operation outside the generated frontend bindings, registered abstract
  schemas, or explicit compile-time metadata surface fails at the trace boundary.
  Data-dependent-shape operations require an explicit dynamic rule and fail
  during staging unless they expose static output semantics.
- Each applicable official Array API subset passes through concrete,
  staged, and serialized-and-restored trace-and-execute round trips. The
  selected operation and revision list is recorded with the evidence; this
  does not imply that Advect is an Array API implementation.
- Every function classified as staged has a deterministic case that executes
  concretely, as a staged program, and after staged-program serialization on
  `array-api-strict`. A separately identified portable subset runs through the
  same lifetimes on every provider named by its checked evidence.
- An executable support inventory materializes each supported revision's
  official function surface and joins it to the live
  binder, abstract-rule, and derivative registries. Every official function is
  classified as staged, dynamic-only, compile-time metadata, provider
  passthrough, a missing binder, or an unsupported result structure. The
  checked JSON report is generated evidence and is never a runtime capability
  table.
- The internal `advect.autodiff` module attaches canonical array-family rules
  directly after frontend ops exist. Registration is guarded by registry
  identity and revision so lazily introduced ops are discovered without
  rescanning the registry on every call.
- Conforming runtime namespaces, including NumPy, resolve directly through
  `__array_namespace__`; no backend-specific derivative adapter is required.

## R11: NumPy frontend

- NumPy remains a first-class frontend through `__array_ufunc__` and
  `__array_function__`.
- Durable execution accepts NumPy 2.0 through 2.5 under `advect-array-1`. The
  Array API target is 2022.12 for NumPy 2.0, 2023.12 for NumPy 2.1-2.2, and
  2024.12 for NumPy 2.3-2.5.
- Live NumPy handlers remain the lowering authority. Data-only executable
  invocation cases own advertised dynamic, staged, and serialized lifetimes;
  registration or abstract-rule presence alone cannot promote a mode. The
  cases do not duplicate upstream signatures or primitive semantics. NumPy-only
  calls remain part of the richer NumPy frontend rather than being forced
  through `array-api-compat`.
- NumPy protocol dispatch is a concrete `advect.numpy` implementation, not a
  configurable backend adapter. It and the provider-neutral Array API frontend
  converge only at canonical operation emission and the complete `OpDef`
  authorities below it.
- Python weak-scalar behavior is pinned; common operations must not
  accidentally promote float32/complex64 workloads to float64/complex128.
- Ordinary NumPy is the default user namespace. Creation from direct tracers
  or rectangular nested tracer sequences uses NumPy's standard `like=`
  dispatch, for example `numpy.array(values, like=x)`. Advect handles
  `array`, `asarray`, and `asanyarray` through that explicit anchor without
  patching those NumPy array constructors. The frontend's only process-visible
  patch is the scoped ambient-RNG tripwire during abstract staging.
- A failed ndarray coercion points to the `like=` rewrite. If coercion happened
  inside another library, the diagnostic identifies that call as
  tracer-incompatible rather than silently detaching.
- `advect.array` and `advect.asarray` remain provider-neutral explicit
  constructors. The secondary `advect.numpy` namespace overrides only the three
  traced constructors and returns every other attribute directly from the
  installed NumPy module.
- Traced `.item()` remains differentiable by returning a rank-zero tracer
  during tracing. The transform boundary, rather than the array method,
  materializes any final Python scalar.
- Every registered single-output array function whose selected NumPy-minor
  signature accepts `out=` uses the same owned-wrapper functionalization path.
  NumPy itself validates the callable-specific casting and shape contract
  against a private destination before Advect commits the pure SSA replacement.
- NumPy ufunc calls preserve the standard dtype, casting, order, signature,
  subclass, and masked-output controls that their admitted form declares.
  Supported ufunc methods are separate contracts: `add`/`multiply`
  `reduce` and `accumulate`, plus binary single-output `outer`. Other methods
  reject explicitly rather than inheriting support from `__call__`.
- Unsupported coercion or third-party calls fail with a custom-primitive
  suggestion.
- Basic indexing, functional `out=`, and the mutation tier follow R5-R7.
- NumPy derivatives use the canonical `advect.autodiff` array-family rules; there
  is no separate NumPy derivative distribution or `advect.derivatives` plugin
  boundary.

## R12: Constants and effects

- Concrete operands encountered during staging become ordinary constant nodes.
- Identity-based constant deduplication retains each source object for the
  compile lifetime, so Python object-id reuse cannot alias distinct constants.
- Every staged artifact reports constant origin, location, shape, dtype, bytes,
  and digest.
- Constant capture requires no wrapper and remains visible in the manifest.
- Capture immediately detaches numeric data into the closed artifact value
  model. Runtime materialization is cached per namespace and device.
- Live construction, graph storage, and Python/native transfer use canonical
  raw little-endian bytes. Canonical artifact JSON encodes those bytes as
  lowercase hexadecimal only at the serialization boundary.
- Known ambient RNG/stateful provider calls fail while staging.
- Staged randomness uses explicit state or keys.
- Concrete capture must be attributable and inspectable; Advect does not claim to
  detect every possible Python side effect.

## R13: Diagnostics and tracer lifetime

- Errors contain an operation, user-code location where cheaply available, and
  a concrete rewrite.
- Shipping mode always captures lightweight locations for mutation hazards,
  views, pending updates, constants, stage effects, and trace boundaries.
- Per-node source mapping is debug-only.
- One public, thread-local `debug()` scope enables per-node locations and
  bounded tracer representations; `numerics=True` stops at the first
  non-finite primal, JVP, or VJP value during a dynamic transform.
- Staged execution errors append a bounded parent/failing-node graph slice
  without replacing the provider exception.
- Whole-function gradient checking uses an epsilon sweep and the reverse
  gradient. It is distinct from atomic primitive conformance checking.
- Every tracer carries a trace generation. Use after close or in another trace
  raises `EscapedTracerError`.
- Shipping diagnostics are included in latency benchmarks.

## R14: Pytrees and call structure

- Tuple, list, dict, and custom pytree nodes are supported by dynamic
  transforms. Custom nodes may use exact process registration or inherited
  `__advect_tree_flatten__` and `__advect_tree_unflatten__` hooks; exact
  registration wins. Durable staging initially serializes only the built-in
  container nodes and `Static`; a custom node requires an explicit stable codec
  before it can cross that boundary.
- Gradient results preserve selected input structure.
- Dense Jacobians preserve both output and selected-input structure rather
  than flattening either side into a package-specific matrix convention.
- Positional and named argument selection is explicit and deterministic.
- Static arguments are declared by the primitive or stage boundary; they are
  never inferred from accidental inability to trace a value.
- Static metadata is accepted only through the closed durable codec and is
  snapshotted when the staged call signature is compiled.

## R15: Durable programs

- The Python envelope has identity `advect.ssa-program`, envelope version 2, and
  contains exactly one program. Its nested native graph header is the sole
  source for graph-format, core-opset, semantic-profile, compiler, and optimizer
  versions. Operation schema revisions live on graph nodes and are owned by
  Advect.
- The envelope serializes the one exact positional/keyword input signature and
  exact output `ArraySpec` leaves. Output specifications are authoritative when
  another staged transform introduces a cotangent input.
- Compiler and optimizer versions participate in artifact provenance.
- Loaders reject unknown versions before execution.
- The Python-independent `advect-runtime::GraphStore` is the sole durable
  topology and canonical-metadata authority. `advect-runtime` also owns graph
  artifact version 2 validation and serialization, fixed cleanup, and the
  host-independent execution schedule.
- The graph header is the sole authority for the required Array API revision;
  `StagedProgram.array_api_version` exposes it to Python.
- Runtime values, buffers, callbacks, residuals, and resources are not
  serialized. Staged values are call-local; primitive residuals follow R2 and
  are retained only by an enclosing dynamic tape.
- Loaded artifacts are already optimized and are not silently reoptimized.
- Custom primitive implementations must be explicitly linked by name before
  execution.
- Portable execution uses flat inputs and outputs through a host contract for
  operation linking, constant materialization, evaluation, validation of every
  input, constant, and evaluated output leaf, ownership, aliases,
  repeated-output handle retention, and donation. Python implements that
  contract in the `advect-native` adapter.
- Every staged custom node records and validates its output pytree before any
  output leaf is selected.
- Loading validates and links the singular program as one transaction.

## R16: Author validation

`advect.testing` supplies optional primitive-author checks for:

- concrete/abstract output parity;
- finite-difference JVP and VJP agreement;
- real-adjoint dot-product identities;
- complex real-linearity;
- nested differentiation of traceable rules;
- staged round-trip execution and serialization.

## R17: Performance and memory

- The blocking latency gate compares one exact candidate Advect wheel with one
  exact reference Advect wheel. Both wheel digests and source revisions are
  recorded.
- Reference and candidate run in separate environments and worker processes on
  the same host. Their order alternates by warmed replicate. Dependencies,
  native build profile, diagnostic mode, and correctness results must agree
  before timings are accepted.
- Acceptance compares every named workload phase independently. Dynamic
  evidence covers one-shot gradients, linearization plus required release, and
  retained reverse application. One representative staged stencil covers
  primal and derivative compilation, first and warm execution, derivative
  serialization, reload, and restored execution.
- Per-phase thresholds are derived from paired replicate noise, with explicit
  configurable minimum and stability ceiling. A geometric mean cannot hide a
  workload or lifecycle regression.
- HIPS Autograd and other ecosystem implementations are informative historical
  comparisons. Same-process measurements in that report are called warmed
  replicates and never determine acceptance.
- Isolated-process memory reports distinguish peak RSS, provider-live bytes,
  provider or pool reservation, native structural bytes, reverse-entry bytes,
  and post-close provider-owned bytes. Timing runs keep profiling disabled.
- Memory acceptance requires a named exact profile. Each measured runtime or
  control has a small correctness preflight in a separate worker, and only the
  lifetime invariants explicitly named by that profile are gated. A free-form
  workload selection is diagnostic-only and cannot pass acceptance.
- The CPU runtime profile includes a representative stencil, checkpoint
  control pair, residual release, reusable linear map, and captured staged
  constant. The CuPy donation profile separately gates its exact donation and
  forced-fresh pair on qualified hardware.
- Staged compilation reports node and rewrite counts for the fixed pass
  sequence. Static execution must not regress without an explicit, measured
  tradeoff.
- Reproducible programs live in source control. Generated benchmark and
  qualification JSON is uploaded as CI or release evidence rather than
  versioned as runtime source.

## Implemented extension contracts

These contracts govern bounded extensions built on the shared native SSA
runtime. Their rationale and scope are defined by
[Runtime Extension Boundaries](decisions/2026-07-24-runtime-extension-boundaries.md).

### E1: Manual rematerialization

- `checkpoint(f)` is a callable-level autodiff transform that trades
  recomputation for saved-forward memory.
- A dynamic checkpoint region retains explicit inputs, output structure, and an
  invocation-local replay recipe instead of its interior primal values.
- Replay recipes, callables, and live resources never enter `RawArena`,
  `GraphStore`, or staged serialization.
- The initial transform belongs to concrete dynamic tracing. Abstract staging
  rejects its boundary rather than silently discarding the memory contract.
- Regions are pure and deterministic from explicit inputs. Randomness uses
  explicit state or keys.
- Residual-bearing primitives are replay barriers in the initial
  implementation.
- Automatic checkpoint placement and memory-budget planning are deferred until
  manual placement is implemented and measured.

### E2: Direct named basic-slice mutation

- A named basic-slice view rooted directly in an owned NumPy tracer may replay
  mutation as an `index_update` on the root's current SSA value.
- A successful replay advances the root epoch and refreshes the mutated view;
  sibling views from the previous epoch become stale.
- Input roots, advanced indices, broadcast or overlapping views,
  layout-dependent reshape/ravel views, and arbitrary composed view chains
  remain unsupported.
- The extension does not perform region-overlap analysis and does not add view
  or mutation nodes to the IR.

### E3: Internal staged buffer donation

- Donation is a provider-specific execution optimization over logical SSA, not
  an observable program operation.
- The staged executor may reuse a buffer only after proving ownership,
  writability, last use, absence of live aliases, internal provenance, and
  storage compatibility.
- Failure to prove reuse falls back to fresh allocation without changing
  results or supported alias behavior.
- Dynamic donation and destructive donation of caller-owned inputs remain
  deferred.

### E4: Compatibility providers

- A private built-in bridge ships in the `advect` distribution;
  `array-api-compat` is a base dependency, and importing `advect` configures its
  fixed fallback after direct object-protocol discovery.
- Provider resolution lives outside the stdlib-only `advect.core` Python module
  and feeds the upstream namespace into the existing generic Array API frontend.
- The bridge uses `array-api-compat`; CuPy is its first GPU acceptance provider.
  Resolution remains provider-neutral, but qualification is an explicit
  per-provider decision.
- The initial CuPy contract is backend-neutral
  `__array_namespace__`-style user code on one process and one device, not
  arbitrary functions written against `cupy.*`.
- The CuPy staged-donation and scientific-program gates run separately for
  2022.12, 2023.12, and 2024.12. A provider pass requires retained reports and
  digests; the existing source-only historical record is not verification.
- One backend-neutral complex scientific program covers dynamic value/gradient,
  JVP, VJP, staged primal/gradient, serialized-gradient, weak-scalar,
  device-preserving-constant, FFT/solve, and functionalized-mutation checks.
  NumPy and `array-api-strict` run automatically for every profile; CuPy uses
  the synchronized manual GPU gate.
- Providers with an independent autodiff system are not qualified as Advect
  array providers. Their concrete composition path is E9; incidental fallback
  compatibility is not a support promise.
- This qualification does not admit arbitrary `cupy.*` calls, mixed providers,
  multiple devices, or extension functions outside the explicit staged table.

### E5: Matrix-free implicit roots

- `implicit_root` differentiates a converged root from
  `residual(solution, parameters) == 0` without tracing solver iterations.
- State and parameter pytrees retain their public structure. The residual and
  state structures and leaf shapes must match, and the initial guess is
  nondifferentiable.
- JVP solves the state-Jacobian system; transpose solves its real adjoint and
  applies the parameter derivative. One joint reusable matrix-free `LinearMap`
  supplies both derivatives without constructing dense Jacobians.
- A successful callback return certifies convergence. Uncertain or failed
  solves raise `ImplicitSolveError`.
- The transform is dynamic. Opaque callbacks reject during abstract staging
  and never enter `RawArena`, `GraphStore`, or an artifact.

### E6: Bounded optional SciPy frontend

- The built-in `advect.scipy` module contributes an explicit SciPy 1.18 surface
  when the `scipy` extra is installed. `special` contains `gammaln`, `digamma`,
  `polygamma`, `erf`, `erfc`, `erfcx`, `erfinv`, `expit`, `log_expit`, `ndtr`,
  `log_ndtr`, `ndtri`, `logsumexp`, `softmax`, and `log_softmax`.
  `ndimage` contains Gaussian and uniform filters, convolution and correlation,
  Laplace/Sobel/Prewitt filters, minimum/maximum and rank filters, greyscale
  morphology. The module also supplies the NumPy/scalar `root_solver` and
  `gmres_solver` callback factories.
- Special functions are stable custom primitives with concrete, abstract,
  traceable derivative, staging, and serialization contracts. Each exposed
  function implements its complete SciPy 1.18 call contract for the admitted
  NumPy provider: unary ufunc controls and functionalized `out=`,
  array-valued `polygamma` orders, and weighted or signed `logsumexp`.
  Artifact loading requires importing `advect.scipy` to link the primitive
  schemas.
- `ndimage` forward evaluation follows SciPy exactly, including boundary-mode
  aliases, `axes`, `origin`, `radius`, footprints, structures, output arrays,
  and output dtypes. Linear filters use one traceable boundary-aware stencil
  contract, including exact non-self-adjoint boundary transposes; convolution
  and correlation differentiate their weights. Order-statistic and greyscale
  morphology rules share tangents equally among equal winning window slots,
  including duplicated boundary slots. Greyscale structures and constant
  boundary values are differentiable operands.
- `root_solver` and `gmres_solver` are concrete NumPy callbacks for
  `implicit_root`; they accept NumPy arrays, NumPy scalars, and Python numeric
  scalars, preserve shape and scalar category, and raise on nonconvergence
  rather than returning partial iterates. They are a first-order dynamic
  boundary, not staged operations.
- Complex linear solves use a doubled real representation so real-linear
  operators remain admissible.
- Base Advect does not import or depend on SciPy, and this explicit surface does
  not imply a mirrored `scipy` namespace.

### E7: Dynamic xarray pytrees

- With the `xarray` extra installed, explicitly importing the built-in
  `advect.xarray` module registers `DataArray` and `Dataset` as custom dynamic
  pytrees.
- Floating- and complex-valued data buffers are differentiable children.
  Integer, boolean, string, and object variables reject at registration.
  Dimensions, coordinates, names, variable order, and attributes are
  deterministic static metadata.
- xarray owns alignment and named-axis semantics; Advect differentiates the
  supported array operations emitted against each data leaf.
- The integration is not an array backend. Passing xarray custom pytree nodes
  through durable staging rejects with a rewrite to stage the raw array
  kernel.
- Data-dependent coordinates, MultiIndex, Dask, sparse storage, and broad
  groupby, rolling, interpolation, or resampling semantics remain outside the
  admitted surface.

### E8: Runtime-derived extension support

- One machine-readable catalog lists supported Array API, NumPy, and SciPy
  extension functions directly from their live bindings and public exports.
- Every direct lowering joins the canonical primitive's JVP, VJP, abstract,
  and evaluator capabilities. Composite frontends remain labelled composite.
- NumPy ufunc calls and supported ufunc methods are separate entries. Support
  for one form does not imply support for another.
- NumPy rows identify reuse of an Array API primitive without claiming that
  the two frontends share signatures or calling conventions.
- Runtime validation owns parameter restrictions. The catalog does not
  duplicate upstream signatures, parameter roles, completeness flags, or
  restriction notes.
- The official Array API suite, executable operation cases, primitive law
  tests, and provider qualification remain independent evidence gates.
- Runtime and qualification dependencies admit named upstream minor profiles.
  A manually installed provider such as CuPy records an exact evidence version.
- The catalog reports runtime-derived frontend support. Dependency bounds and
  exact qualification versions remain separate package and evidence records.

### E9: Host autodiff VJP bridges

- The built-in `advect.interop.torch`, `advect.interop.jax`, and
  `advect.interop.autograd` modules adapt a NumPy-backed Advect callable into
  one first-order reverse-mode operation owned by the host framework.
- The modules import explicitly behind independent dependency extras. Importing
  `advect` or `advect.interop` imports no host framework and registers no array
  provider.
- Positional tuple, list, and dictionary pytrees containing standard NumPy
  floating or complex values are differentiable; a custom container must be
  recognized by Advect and its host. Static configuration is closed over by the
  callable. Keyword differentiation, forward mode, higher derivatives, JAX
  batching, and durable staging remain outside the initial contract.
- PyTorch retains and consumes the exact forward Advect pullback. JAX executes
  concrete eager calls directly; JIT compilation and abstract shape evaluation
  supply an explicit result shape/dtype pytree and use pure host callbacks. Both
  JAX reverse paths replay the callable during backward. With `has_aux=True`,
  the JAX callable returns `(value, aux)` and only `value` participates in that
  replayed VJP. HIPS Autograd retains the exact forward pullback in its custom
  primitive.
- PyTorch shares Advect's real-adjoint complex cotangent representation. JAX
  and HIPS Autograd conjugate both the incoming output cotangent and returned
  input gradient at their boundaries.

## Stable exclusions

The core does not own workflow orchestration, workflow restart, or a generic
key-to-bytes object store. Dynamic differentiation does not convert each tape
to `GraphStore`, run the staged optimizer before backward, or expose selectable
graph passes. The fixed staged optimizer is compiler policy, not a public
`Graph -> Graph` transformation API.

Advect does not promise complete NumPy mutation-through-view or memory-overlap
semantics. Mutation remains tracer-level functionalization rather than an IR
effect, and buffer donation remains physical execution rather than logical
semantics. Compatibility aliases for the former graph-first API remain
excluded.

SciPy/xarray compatibility beyond the explicit E6-E7 surfaces, host-framework
behavior beyond E9, a general `vmap`, dense complex Hessians, artifact blob
externalization, automatic rematerialization placement, explicit input
donation, and dedicated provider-module frontends are deferred rather than
current requirements.

Advect has no kernel compiler, code generation, fusion, backend-specific
lowering, automatic checkpoint planner, public compiler-plugin API, or Rust
implementation of NumPy or the Array API.
