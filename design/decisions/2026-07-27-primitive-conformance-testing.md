# ADR: Primitive conformance testing

**Date:** 2026-07-27
**Status:** Accepted (amended 2026-08-01)

## Context

Advect had enough example tests to demonstrate individual APIs, but no
mechanical argument that every registered primitive was correctly reachable,
differentiated, and composed. Two silent defects showed why that distinction
matters:

- `numpy.linspace` dropped a positional `num`. The traced forward value was
  wrong while its derivative was perfectly consistent with that wrong value.
- `numpy.interp` had a correct local JVP but transposed to zero after its
  tangent became detached from the trace.

Neither a rule test nor a public `grad` test catches both classes. A rule test
localises wrong mathematics but cannot see frontend argument binding. A
transform test exercises the whole path but can fail several operations away
from the defective rule. Randomly composed programs cover a third class:
interactions which are absent when each operation is tested alone.

The earlier property suite on `main` contained useful domain knowledge, but it
was coupled to the old package layout and rule API. Restoring it wholesale
would preserve two architectures. The conformance suite instead keeps the
small domain vocabulary and rebuilds the checks around the current registry,
frontends, JVP-first rules, structural transposition, and staged graph store.

## Decision

### A primitive may have several invocation contracts

The unit of executable evidence is an `InvocationCase`, not one universal
example or an otherwise inert grouping object. A flat declaration table may
contain several cases for the same canonical registry operation. Separate cases
are required when behavior passes through a materially different boundary:

- NumPy and Array API frontends;
- positional and keyword call forms;
- static attributes which select different semantics;
- shape or dtype families which exercise different rule branches.

One invocation may own several named `InputVariant` values. A variant changes
argument shapes, dtypes, numerical reference, or tolerance without copying the
frontend callable and semantic declaration. Pointwise families cover scalar,
vector, matrix, tensor, broadcast, float32/64, and complex64/128 forms as
applicable. Reductions cover multiple ranks; contractions cover vector,
matrix, batched, and broadcast signatures; linear algebra covers vector and
matrix right-hand sides, rectangular matrices, batches, and complex dtypes.
Operation-specific variants exclude shapes that the provider itself rejects,
such as concatenating zero-dimensional arrays.

The declaration remains compact: a callable, argument domains, static
arguments, applicable laws, tolerances, and a reason for any non-default
contract. Cases remain together under family headings. Splitting the table
into many modules would add navigation and aggregation machinery without
improving the contract.

### Three boundaries answer different questions

| Boundary | Input | Main question | Typical failure |
| --- | --- | --- | --- |
| Registered rule | exact emitted operands and attributes | Is the local mathematics correct? | wrong JVP formula |
| Public transform | frontend callable | Is tracing, binding, transposition, and result reconstruction correct? | dropped `linspace` argument |
| Composition | generated program | Do individually valid rules remain valid together? | nested tracing or mutation interaction |

The rule tier does not recreate frontend binding metadata. It temporarily
wraps the target registry JVP, calls the real frontend invocation, and captures
the exact answer, operands, tangents, and decoded attributes received by that
rule. The captured rule is then checked directly. This both localises failures
and proves that the declared invocation actually emitted the named operation.

An operation with no frontend path gets a small `RawRuleCase`. This is an
explicit statement that the mathematics exists below an absent binding, not a
pretend public invocation.

Composition uses a deliberately small grammar of smooth, bounded operations.
The grammar is curated separately from registry completeness: including every
operation would mostly generate invalid or numerically useless programs.
Focused programs additionally cover broadcast-plus-reduction chains,
multi-argument linear algebra, mutation, and complex FFTs because these
interactions are too consequential to leave to random grammar frequency.

### Laws apply at named boundaries

Not every law is meaningful at every boundary.

| Law | Rule | Transform | Staged | Composition |
| --- | :---: | :---: | :---: | :---: |
| provider primal agreement | ✓ | ✓ | ✓ |  |
| numerical JVP | ✓ | ✓ |  |  |
| JVP/VJP real-adjoint identity | explicit VJPs | ✓ |  |  |
| output dtype |  | ✓ | ✓ |  |
| cotangent structure and dtype |  | ✓ |  |  |
| no input mutation |  | ✓ | ✓ |  |
| declared local dependence |  | ✓ |  |  |
| staged primal and derivative serialize/load round trip |  |  | ✓ |  |
| program directional derivative |  |  |  | ✓ |

`PRIMAL` compares the traced result to provider execution and catches wrong
frontend binding even when all derivative machinery agrees. `FINITE_DIFFERENCE`
anchors a JVP outside Advect. `ADJOINT` checks

```text
Re <v, J u> = Re <J* v, u>
```

at derivative tolerance rather than finite-difference tolerance. It covers
both explicit VJPs and JVPs transposed structurally by the public reverse path.

`DTYPE`, `STRUCTURE`, and `NO_INPUT_MUTATION` are independent contracts rather
than side effects of an all-close assertion. Staging is exhaustively
classified: every invocation either runs a serialized primal and a compiled,
serialized VJP program or its exact invocation ID belongs to the closed
`DYNAMIC_ONLY_STAGING_INVOCATIONS` set. Classification is deliberately keyed by
frontend and call form rather than canonical operation: another invocation
which lowers to the same operation may have a different staging contract. The
coverage gate fails if the declared and observed invocation sets diverge. This
makes dynamic-only support visible without pretending every dynamic rule has
an abstract lowering. `SECOND_ORDER` compares a Hessian-vector product with both
the dense Hessian contraction and an independent directional finite difference
of the nested gradient. It remains opt-in: a first-order operation is not
silently advertised as higher-order merely because its first derivative exists.

### Dependence is an explicit domain promise

Differentiability does not imply a nonzero derivative at a particular point.
For example, one argument of `maximum` may be locally inactive, and a
cotangent seed may lie in a Jacobian's null space. Requiring every gradient to
be nonzero under one all-ones seed is therefore not a valid general law.

`DEPENDENCE` is absent from the default laws. An invocation opts in with exact
argument indices only when its domain guarantees locally nonzero activity.
The check uses several independent directional probes. Interpolation cases,
for example, guarantee at least one interior query and separated function
values. The promise then catches a detached argument without misclassifying a
legitimate zero derivative.

### Hypothesis owns values and shrinking

Domains return real Hypothesis strategies. The suite does not draw an integer
seed and hand it to `numpy.random`, because that turns a minimal failing array
back into an opaque seed.

Domains construct valid values directly:

- positive and nonzero values stay away from singularities;
- selection operands are separated elementwise rather than filtered after a
  tie;
- interpolation points include a guaranteed cell interior;
- matrix domains remain full-rank and avoid provider gauge boundaries even
  after shrinking;
- eigendecomposition domains have separated eigenvalues.

Finite-difference oracles promote float32 and complex64 values to float64 and
complex128 before perturbation. The actual transform still runs at the declared
low precision, and separate dtype and cotangent-structure laws require exact
metadata preservation. This removes cancellation noise from the oracle without
letting a widened derivative path satisfy the dtype contract. Decomposition
oracles also align only mathematically arbitrary permutations and vector
phases; QR and general-eigenvector domains stay away from provider gauge
boundaries instead of hiding them with tolerance.

There is no `assume()` in the domain layer. Hypothesis's normal health checks
therefore remain meaningful and a failing example shrinks to the actual
smallest valid input.

The selected Hypothesis profile owns depth. Presubmit runs a small number of
examples in every law-by-invocation cell; the `thorough` profile increases that
depth and the composition search without changing which laws exist.
`--hypothesis-show-statistics` is useful observability, but examples per second
is not treated as a coverage metric.

CI runs both the complete ordinary suite and the conformance directory under
the `thorough` profile. The shallow local profile keeps iteration fast; it is
not the evidence used to merge changes to primitive mathematics.

### Complex-step is opt-in

Complex-step differentiation is extremely accurate for a real-domain
holomorphic call whose real output admits the usual imaginary-part formula.
It is invalid for `abs`, `real`, `conj`, selection, clipping, interpolation,
rounding, and general real-linear complex functions.

Central differences are therefore the default. A case selects complex-step
only when it states the stronger analytic contract. Complex-step cases use
`rtol=1e-11` and `atol=1e-12`; they do not inherit the looser
central-difference tolerance. Complex programs are separately checked under
Advect's real-inner-product convention.

### Registry coverage is three independent gates

A single “accounted for” bucket can hide unrelated omissions. Coverage is
split into:

1. **Semantic classification.** Every deterministic product operation belongs
   to exactly one of: frontend invocation, structural operation, unbound rule,
   honest derivative gap, or registry-declared non-differentiable operation.
2. **Rule coverage.** Every differentiable invocation has a JVP; every emitted
   JVP is tested directly; explicit VJPs get a direct adjoint test; installed
   rules without a frontend have raw cases.
3. **Differentiability classification.** A non-differentiable operation has a
   nonempty registry reason and no derivative rule. A known gap has neither a
   rule nor a non-differentiable classification.

The deterministic inventory is built in an isolated `OpRegistry`, including
NumPy's declared operation table. Bundled custom primitives are then added by
their bounded declarations. Arbitrary custom primitives registered by other
tests cannot make coverage pass or fail based on collection order.
The portable Array API derivative gate is joined mechanically to the
executable provider-qualification catalog: every portable case with
differentiable operands and a registered JVP must have an Array API
conformance invocation. Constant-only `linspace` and non-differentiable
`equal` therefore remain in provider qualification without pretending to be
derivative entry points.

### Abstract domain tables rely on the same gate

Abstract evaluation is divided into ordinary domain tables rather than one
closed dispatcher. The split remains exhaustive because conformance joins
every staged array invocation to exactly one native abstract rule or declared
composite lowering. It also rejects abstract rules that name unregistered
operations. The table merger rejects duplicate operation and result-kind
registrations before tracing begins.

This is the compensating control for a modular dispatcher. Plain Python tables
remain the implementation mechanism; the conformance inventory, not generated
code or a second schema framework, is the source of truth.

### The harness remains test-internal

Third-party authors use `advect.testing.check_primitive`; Hypothesis is not
imported by the runtime core. The checker reports failures without installing
capability state. Both surfaces use the same real-adjoint convention. The
repository harness may use richer provider domains, frontend capture, and
registry-wide coverage without freezing those details as public API.

## Consequences

The default conformance run contains more named tests than the former seeded
loop, but it removes the duplicate deterministic pass and remains small enough
for ordinary CI. Failures now report a minimal input and one of four locations:
registered JVP, registered VJP, public transform law, or composed program.

The first direct search found useful defects immediately:

- independent “distinct” domains could still tie elementwise across the two
  inputs of `maximum` and `minimum`;
- the old rounding domain included the `rint` kink at `0.5`;
- diagonal matrix shrink targets crossed NumPy's QR/eigenvector gauge
  boundaries;
- the declared Array API “partition” example actually emitted `sort`;
- NumPy `sort` and `partition` could run a JVP but failed when structural
  transposition called an unbound `take_along_axis`;
- Array API reverse rules called provider-specific array methods;
- reduction rules passed `dtype=None` to strict namespaces and used provider
  reductions for Python-only shape arithmetic;
- explicitly typed staged scalar constants rounded through weak float32 before
  being cast to float64;
- batched `linalg.solve` confused NumPy's matrix right-hand side with a vector
  right-hand side;
- complex `slogdet` treated its unit-phase output as locally constant;
- structural transposition could not represent a repeated output node and
  would overwrite rather than add its cotangents;
- staged decomposition replay omitted valid `UPLO`, SVD, and argsort metadata;
- untyped zero and one branches in extrema and NaN reductions widened float32
  and complex64 cotangents during structural transposition;
- the original clipping domain could approach a fixed bound despite claiming
  to stay on a smooth branch.

The domains and declarations were corrected. NumPy `take` and
`take_along_axis` bindings were added, closing the reverse path for NumPy sort
and partition instead of weakening their laws. The backend-neutral rules now
use namespace operations, optional keywords are omitted rather than passed as
`None`, reduction shape arithmetic stays in Python, and typed scalar constants
are recorded directly in their requested dtype.

Multi-argument finite-difference and adjoint laws probe every partial
independently before probing a combined direction. The matrix also includes
float32 reductions, complex sign and contractions, unequal complex signal
operands, batched solve signatures, wide QR, decomposition phase gauges, a
non-normal complex eigensystem, FFT resize and normalization forms, and selected
second-order contracts. Composite NumPy handlers have separate public-path
tests because their correctness is not implied by the primitives they emit.

The suite does not claim exhaustive frontend cross-product coverage. It makes
multiple invocations representable and covers the currently consequential
dual paths; adding a supported frontend path requires adding its invocation.
All 33 currently qualified portable Array API derivative operations have an
Array API invocation and staged round-trip law. Bundled custom primitives also
opt into the staged law explicitly.

Fault-injection mutation testing is not part of this decision. This is distinct
from Advect's supported functionalized array mutation, which remains in the
composition and public behavior suites. A global mutation-testing kill-rate is
a separate cost and policy choice, not evidence that the conformance domains
or boundaries are correct.

## Adding an operation

1. Define the operation's abstract semantics and JVP-first derivative contract.
2. Add an `InvocationCase` for every materially supported call path.
3. Choose a domain that is smooth and well conditioned after shrinking.
4. Opt into `DEPENDENCE`, complex-step, or second order only when the case can
   state the stronger contract; add a genuine abstract-lowering gap to the
   closed dynamic-only staging set.
5. If no frontend reaches an installed rule, add a `RawRuleCase`.
6. If the operation is non-differentiable, put the reason in the registry.
7. Run:

```bash
uv run pytest packages/advect/tests/advect_conformance_tests
uv run pytest packages/advect/tests/advect_conformance_tests \
  --hypothesis-profile=thorough --hypothesis-show-statistics
```
