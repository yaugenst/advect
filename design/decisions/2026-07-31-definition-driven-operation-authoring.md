# ADR: Definition-Driven Operation Authoring

**Date:** 2026-07-31
**Status:** Accepted (amended 2026-08-01)
**Implementation status:** Implemented

## Context

Advect previously assembled built-in operations in stages. The Array API
binding table created placeholder `OpDef` records, NumPy carried a second
193-name operation inventory and another output-arity map, and autodiff scanned
the registry after tracing to attach derivative rules. Test-only registry
replacement then required production relinking paths in NumPy, SciPy, autodiff,
and the support catalog.

That architecture made frontend metadata look authoritative even though a
frontend call and an IR operation are different things. It also made adding an
operation require coordinated edits to tables that did not contain independent
semantics.

## Decision

Built-in operation definitions are complete when the process registry is first
created. One `OpDef` owns the stable name, fixed output arity, abstract operand
and attribute schema, abstract evaluator, JVP, explicit VJP retention contract,
and non-differentiability contract. Domain modules keep the formulas and shape
logic close to their implementations; bootstrap mechanically joins those
semantics into the operation records once.

The only separate arity metadata is the small set of operations that return
more than one value. Single output is the primitive default.

The Array API binding surface is generated from the pinned upstream signatures
and primitive schemas. Handwritten data is limited to actual differences:

- upstream functions Advect intentionally does not expose;
- public names that lower to a differently named primitive;
- calls such as `clip`, `full`, and `pinv` whose live operands do not follow the
  conventional leading-operand layout.

NumPy handlers are the authority for NumPy lowering. Conventional same-name
calls need no declaration. Alias handlers and ndarray methods carry their
non-conventional primitive target on the executable handler itself, and the
support catalog reads that metadata. It has no parallel alias or method table.

JVP functions use a naming convention and are discovered from their domain
modules. Their registries emit canonical `array.*`, `array_ext.*`, or
`advect.*` identities without a NumPy-name intermediate or a bootstrap
canonicalization callback. Conventional JVP suffixes resolve against the
canonical abstract semantics; the small set of genuine aliases and
dynamic-only shape identities remains explicit. VJP and
non-differentiability records author canonical identities directly.

Abstract semantics seed staged operation definitions. A canonical JVP or
non-differentiability record also contributes the `OpDef` fragment for an
operation that is intentionally dynamic-only. An independent coverage test
joins derivative identities against abstract semantics, executable conformance
cases, raw-rule cases, frontend lowering metadata, exceptional concrete
evaluators, and fixed output arities. It rejects an extra or misspelled rule ID
without requiring a second runtime inventory of every dynamic-only operation.
VJP attachment uses exact existing identities and cannot create an operation.

Shared-formula aliases remain explicit because they are genuine
many-name-to-one exceptions. Exceptional concrete evaluators and attribute
decoders use decorators at their definitions instead of registration batches at
the bottom of a file.

An explicit VJP earns its place through required real-linear semantics or
measured performance and is authoritative by being present in the complete
operation definition. Advect does not maintain a parallel derivative-policy
membership table or registry query surface. Durable rationale belongs beside
the exceptional rule or in this decision record, where it cannot become a
second capability authority.

The global built-in registry is stable process state. Tests construct local
registries when they need isolation; they do not replace the global registry.
Consequently frontends do not create built-in operation records, autodiff does
not rescan registry revisions, and bundled primitives do not relink themselves
after a reset.

## Primitive authoring boundary

The public `@primitive` API is intentionally smaller than built-in frontend
authoring. Decorating the concrete implementation creates the callable
primitive handle and its complete registry record in one step. Advect derives
the call contract and default operation identity from that function;
`def_abstract`, `def_jvp`, and `def_transpose` attach the remaining semantics to
the same `OpDef`.

There is one concrete implementation. Primitive authors do not instantiate an
empty object, rediscover a handle through a public string lookup, register
string-keyed providers, or declare schema versions. An explicit `name=` remains
available for package authors who need serialized programs to survive a Python
module rename. Graph schema revisions remain an Advect-owned field on `OpDef`
and graph nodes; changing one is a library format decision, not part of the user
authoring contract. It describes Advect's graph encoding, not the semantic
version of the decorated function. Advect does not infer compatibility between
two implementations registered under the same name; durable artifacts require
the matching application code.

Ordinary Python defaults belong to the implementation signature. Advect does
not inspect hypothetical default values when the primitive is defined. If an
actual value becomes static graph metadata, the closed artifact codec validates
and snapshots it at that authoritative boundary.

The staged envelope moves to version 2 and drops both its duplicate
custom-schema manifest and its duplicate Python version block. Each graph node
already carries the authoritative Advect-owned schema revision, while the
native graph header owns the graph, opset, semantic-profile, compiler, and
optimizer versions. The loader accepts the current envelope and exact
registered revision; there is no version-range negotiation, migration registry,
or compatibility callback system.

A built-in array operation may be exposed through the Array API, NumPy
functions, ndarray methods, ufunc methods, aliases, and composites. Those
frontends must normalize their own calling conventions, but they do not own or
duplicate primitive semantics. Both authoring paths end in the same registry
record; the extra built-in code exists only at the genuine foreign-frontend
boundary.

## Consequences

Adding a conventional built-in operation means adding its semantic rule
functions. The registry, same-name frontend binding, derivative capability, and
support catalog then follow mechanically. An exceptional frontend spelling is
declared once beside its executable lowering.

The operation model has fewer extension points: there is no frontend operation
inventory, lazy derivative-registration phase, registry-reset recovery path,
catalog-only binding map, custom provider map, public primitive lookup,
capability-state object, user schema manifest, or duplicate program-version
record. Invalid combinations such as a registered frontend placeholder without
its known derivative contract, or a custom primitive without an implementation,
cannot occur during normal package initialization.

Frontend validation remains separate from primitive semantics. A primitive
cannot by itself describe sequence expansion, NumPy ufunc methods, composite
functions, or every public alias; those differences stay in small executable
bindings rather than being forced into `OpDef`.
