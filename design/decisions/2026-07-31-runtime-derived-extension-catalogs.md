# ADR: Runtime-Derived Extension Catalogs

**Date:** 2026-07-31
**Status:** Accepted
**Implementation status:** Implemented

## Context

Advect previously described upstream compatibility with a parameter-level
manifest. NumPy, SciPy, and Array API rows carried manually maintained
signatures, parameter roles, lifetime claims, and whole-callable completeness
flags. Executable examples were then interpreted as proof of those claims.

That model duplicated runtime authority and did not scale. Its metadata grew
with every callable parameter while still leaving dtype, shape, broadcasting,
parameter interactions, error behavior, and numerical domains combinatorial.
The resulting `complete` label sounded stronger than the evidence warranted.

Advect already has more direct authorities:

- canonical primitive definitions and their derivative rules;
- abstract-rule and evaluator registries;
- Array API namespace bindings;
- NumPy protocol handlers and supported ufunc methods;
- public `advect.scipy` exports and registered SciPy primitives.

## Decision

Advect exposes one runtime-derived support catalog with separate Array API,
NumPy, and SciPy extension sections plus a shared primitive capability matrix.
Live bindings discover candidate callables; discovery is not itself a support
claim.

A frontend function appears only under the exact extension spelling Advect
advertises. Each row reports its canonical primitive or composite lowering,
separate dynamic, staged, and serialized support, and directly joined primitive
capabilities. A mode is supported only when the callable's declared contract
works end to end in that mode. Partial implementations remain internal or are
reported as restricted; a representative trace cannot promote a callable.
Composite frontends remain labelled `composite`; the catalog does not infer
their derivative contract from one operand or example call.

NumPy rows identify Array API reuse when their canonical lowering is also
present in the generated Array API bindings. This means the two frontends share
primitive semantics and derivative rules. It does not imply that NumPy inherits
the Array API calling convention.

Array API accounting distinguishes the direct generated binding table from the
complete public catalog, which also contains composite and compile-time metadata
functions. Evidence reports name the inventory they count instead of using
"bindings" for both.

SciPy-compatible functions are discovered from public `advect.scipy` exports.
Their actual Advect entry points are shown because direct `scipy.*` calls are
not transparently intercepted. Solver callback factories are listed separately
as dynamic adapters.

The catalog contains no parallel hand-maintained signature model. Runtime
normalization and validation define the declared frontend contract, while
executable conformance covers every differentiable operand and important value
regime before a mode is advertised. Behavioral and provider qualification
tests remain evidence for that one public model rather than generating another
manifest.

For NumPy, the compact installed declaration owns the public callable, lifetime,
and derivative claim consumed by the catalog. Test-only executable cases own
concrete proof specimens. Exact bidirectional tests require every declaration
to have cases, every case to have a declaration, declared modes to equal the
intersection of material case modes, and derivative roles to agree. This
deliberate overlap keeps policy independent from its evidence; it is not a
second runtime implementation.

Removing or changing a required case fails CI until a maintainer explicitly
changes the public declaration. Test deletion does not silently demote working
support. The runtime never imports test modules, test cases are not shipped in
the wheel, and no generated runtime manifest or support-schema language is
required.

Value-dependent result dtypes are a staging boundary. The
`numpy.lib.scimath` functions are dynamic-only because the same real
`ArraySpec` can produce either a real or complex result depending on runtime
values. Unqualified subpackage aliases, such as
`numpy.polynomial.polynomial.polyval`, remain absent even when a related
top-level NumPy function has a handler.

Upstream dependency versions remain bounded independently of the catalog.

## Consequences

Users get explicit supported-function lists organized by the extension they
call. Primitive capabilities remain distinct from frontend contract support,
and NumPy's reuse of Array API semantics is explicit.

Adding a primitive updates the capability matrix automatically. Adding a
frontend handler makes a callable discoverable, but it does not justify modes
that have not passed the extension's end-to-end contract. Genuine aliases and
frontend-only ndarray methods declare their target on the executable handler
itself. Semantic staging exclusions remain explicit.

The callable-level NumPy row is conservative across material invocation
variants. Documentation may explain a small number of restricted staged forms,
but the catalog does not grow variant predicates or nested contracts until a
real machine consumer requires structured variant data.

The current restricted forms are deliberately kept as prose beside the
generated table: `numpy.round` stages and serializes when `decimals` is omitted
or zero, while `numpy.linalg.eig` and `numpy.linalg.eigvals` stage and serialize
for complex input dtypes. Nonzero `round` decimals and real-input eigensystems
remain dynamic-only. The callable rows therefore stay conservative without
introducing a parameter-level contract language.

`numpy.linspace` is a different boundary and remains dynamic-only when its
endpoints are traced. NumPy accepts array-valued endpoints, while the shared
Array API primitive treats its scalar endpoints as static attributes; neither
the baseline nor the current implementation can serialize traced NumPy
endpoints. Dynamic differentiation, including `retstep=True`, remains
supported. The former staged catalog flags and the private stage classifier
that repeated them were removed as false claims, not as runtime functionality.

The catalog remains smaller than NumPy, SciPy, or the Array API by design. The
rows it does publish make a stronger claim than reachability: users can call
that declared extension contract normally in every mode marked supported.
