# ADR: Versioned Upstream Callable Contracts

**Date:** 2026-07-29
**Status:** Superseded by [Runtime-Derived Extension Catalogs](2026-07-31-runtime-derived-extension-catalogs.md)
**Implementation status:** Removed

## Context

Advect reaches numerical operations through three different kinds of public
surface:

- NumPy protocol dispatch;
- the Python Array API namespace;
- a small, explicit `advect.scipy` namespace.

Primitive conformance establishes that a declared invocation has the right
primal, JVP, adjoint, dtype, structure, and staging behavior. It does not
establish that every legal parameter of the corresponding upstream callable is
supported. A callable can therefore have an excellent derivative rule while
still rejecting a valid `out=`, alternate output form, array-valued operand, or
static option.

Upstream signatures also evolve. An unbounded `numpy>=2,<3` or
`scipy>=1.16` dependency makes an exact compatibility claim impossible: a new
minor release may add a parameter or result form that Advect has never traced.

## Decision

### One versioned manifest is the compatibility claim

Advect ships one machine-readable support manifest. Each upstream profile names:

- the exact upstream API profile;
- every admitted upstream callable form reached through that profile;
- its upstream signature;
- whether dynamic, staged, and serialized execution are qualified;
- every parameter's role: differentiable, static, nondifferentiable,
  mutation destination, or unsupported;
- separately qualified callable forms, including user-visible ndarray methods,
  NumPy ufunc `__call__`, and each ufunc method.

Unknown parameters are unsupported. No test count, registry count, handler
registration, or documentation table independently constitutes a compatibility
claim.

Advect-native conveniences such as `advect.array` and `advect.asarray` are documented
and tested as Advect APIs; they are not upstream compatibility claims and
therefore do not appear as NumPy callable rows.

A callable is complete only when every legal upstream parameter and output
form is supported or is explicitly nondifferentiable by the upstream
semantics. Partial callables may remain reachable while the current
implementation is completed, but the manifest marks them incomplete and the
documentation does not present them as complete.

Execution modes are whole-callable qualifications, not reachability hints.
An incomplete row therefore has no qualified modes; its note may still explain
which narrower variants are currently reachable.

### Breadth is explicit; depth is mandatory

Namespaces may remain intentionally small. `advect.scipy.special`, for example,
does not need to mirror all of `scipy.special`. Once Advect exposes a callable
under an upstream name, however, the admitted upstream profile defines the
whole callable contract.

For NumPy ufuncs, the callable and its methods are separate surface entries.
Supporting `np.add(x, y)` does not silently claim `np.add.reduce` or
`np.add.at`. Each form earns its own complete parameter contract.

### Qualification is generated from the manifest

The manifest drives three complementary gates:

1. signature parity against the admitted upstream version;
2. property-based primal and derivative cases for every differentiable
   parameter and materially different static option;
3. dynamic, staged, and serialized execution in every mode the entry claims.

The official Array API suite runs every admitted array-returning form that the
suite invokes through Advect. Compile-time metadata forms have dedicated
dynamic, staged, and serialized cases because they do not return an array that
the suite bridge can reconstruct. A small scientific smoke workload remains
useful for provider qualification, but it is not a substitute for callable
conformance.

For the Array API profile, the packaged data-only evidence catalog is the
input to both the manifest and the generated tests. A callable earns
`complete=true` only when the catalog contains a baseline invocation, a live
binding for every differentiable and nondifferentiable parameter, default and
explicit forms for every optional static parameter, an explicit form for every
required static parameter, and every claimed lifetime. Removing any one of
those records demotes the callable and clears its qualified modes. The same
records drive primal round trips and per-parameter property tests of the
JVP/VJP adjoint identity, so a registration or abstract rule cannot promote a
callable by itself.

### Upstream versions are bounded

The first profiles are:

- NumPy 2.3;
- SciPy 1.18;
- Array API 2024.12 through `array-api-strict` 2.4.1;
- `array-api-compat` 1.15;
- xarray 2026.7 for the bounded pytree-container integration.

Runtime and qualification dependency ranges admit only the named minor series.
CuPy is installed separately according to the local CUDA environment, so its
qualification records an exact evidence version rather than an Advect dependency
range. Updating a range, provider evidence version, signature snapshot, or
Array API revision is one intentional compatibility change accompanied by
regenerated evidence.

CuPy remains the designated provider for the portable Array API program. Its
historical run is source-only until retained reports and digests verify a
current provider pass. The manifest does not claim arbitrary `cupy.*`
interception.

## Consequences

The support count becomes less flattering but more useful. A user can inspect
exactly which call forms are portable, stageable, and differentiable without
reading handlers or inferring capability from a passing smoke test.

The manifest also turns compatibility work into a finite queue. Adding a new
callable means classifying its whole signature and supplying its generated
cases. Expanding an upstream version means reviewing a deterministic diff
rather than discovering signature drift in user code.

The cost is additional qualification time, especially for staged property
tests. Those tests are shardable by manifest entry and may use a small
deterministic example budget in the ordinary suite, with a deeper scheduled
profile. The contract itself remains singular.
