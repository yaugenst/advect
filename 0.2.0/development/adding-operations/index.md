# Adding Operations

Classify the change before writing code. A public custom primitive, a built-in canonical operation, a composite frontend form, and a solver adapter have different authorities and therefore different evidence.

| Change                                    | Semantic authority                                         | Owning evidence                                                          |
| ----------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| Public custom primitive                   | The decorated authoring handle                             | `advect.testing` checks plus the package author's tests                  |
| Built-in canonical operation              | The record assembled in `advect/_builtin_ops.py`           | Primitive conformance suite                                              |
| Composite NumPy form                      | NumPy dispatch and support declaration                     | NumPy public support evidence; conformance only for any new canonical op |
| Direct SciPy form                         | Public wrapper plus its bundled `custom.scipy.*` primitive | SciPy parity tests and primitive conformance                             |
| Composite SciPy form                      | Public SciPy composition                                   | SciPy parity, derivative, and lifetime tests                             |
| SciPy solver callback                     | Concrete dynamic adapter                                   | SciPy solver and `implicit_root` integration tests                       |
| Python staged-program envelope change     | `advect.core` staging boundary                             | Core durability tests and Python static checks                           |
| Runtime graph artifact change             | `advect-runtime`, translated by `advect-native`            | Runtime Rust tests plus adapter evidence when its contract changes       |
| Native translation or dynamic-tape change | `advect-native`                                            | Native contract tests and a wheel build                                  |

## Public custom primitives

Use `@advect.primitive` when application or library code needs one closed operation that Advect cannot see through. The [tutorial](https://yaugenst.github.io/advect/0.2.0/tutorials/primitives/index.md) and [API reference](https://yaugenst.github.io/advect/0.2.0/api/primitives/index.md) own the rule signatures and common JVP-first workflow; this page adds the repository-facing boundaries and evidence.

The implementation uses fixed named parameters. Positional-or-keyword and keyword-only parameters are valid; positional-only parameters, `*args`, and `**kwargs` are not. `static_argnames` turns complete named arguments into operation attributes; staged calls require values that the artifact can serialize. `nondiff_argnames` keeps arguments dynamic but removes their tangent and transpose contributions. One argument cannot be both.

Prefer a traceable real-linear JVP so forward mode, structural transposition, and higher-order paths can reuse it. Add an explicit transpose only when the real adjoint cannot be derived or a measured hot path justifies it. A transpose-only primitive supports reverse mode but not forward mode.

Use `residual=True` only when reverse mode needs exact opaque data from the forward invocation. Return `PrimitiveResult(output, residual, release=...)` and provide an explicit transpose. Advect retains the residual until the owning pullback or linear map releases it, calling `release` exactly once. This is a first-order boundary: the primal may stage, but staged derivatives, higher-order derivatives, and `checkpoint` cannot retain the residual.

Use `variable_output_arity=True` only when a concrete invocation determines the number of output leaves. The ordinary primitive call records that realized pytree for dynamic differentiation. Variable-arity primitives do not stage; keep the default fixed contract whenever an abstract output rule can describe the operation.

Give the primitive an explicit stable name only when saved programs need an identity independent of the Python module path. The name is a link key, not semantic versioning; loading requires the matching implementation to be registered under that name.

Run `check_primitive` for every materially different shape, dtype, pytree, static-argument, and complex case. Its default abstract/JVP/transpose tuple is a first-order smoke check. Add `nested` and `stage` only when the primitive claims those paths, then run `check_gradient` on a representative composition. These author checks do not create a built-in support claim or require an internal `InvocationCase`.

## Built-in canonical operations

A built-in operation is a stable `array.*`, `array_ext.*`, or explicit `advect.*` semantic record, not a Python public callable. The record is assembled in `advect/_builtin_ops.py` from authorities that remain separate:

- abstract operand and result semantics live in `core/_abstract_domains/<family>.py`;
- JVP-first formulas live under `autodiff/rules/array_family/jvp/`;
- the smaller set of explicit real-adjoint formulas lives under `autodiff/rules/array_family/vjp/`;
- NumPy reachability belongs to its protocol path: ufunc admission in `numpy/_supported_ufuncs.py`, array-function handlers under `numpy/_array_function/`, canonical naming in `numpy/_op_bindings.py`, and exceptional serialized replay in `numpy/_eval.py`;
- Array API binding and provider execution belong in `core/_array_api/frontend.py`; NumPy and Array API reachability must be added independently; and
- exceptional fixed output counts live in `_output_arities()` in `advect/_builtin_ops.py`. Every unlisted operation has one output.

For a multi-output op, add its fixed arity there and make each frontend agree with that result structure. Do not create one registry operation per result: `advect.getoutput` is the structural projection from the canonical parent.

Complete all applicable owners as one coherent change. Define abstract semantics for each staged form, install the JVP, and add an explicit VJP only when structural transposition cannot express the correct real adjoint or a measured hot path justifies it. Never silently substitute a numerical pullback.

Add an `InvocationCase` in `packages/advect/tests/advect_conformance_tests/_builtin_cases.py`, naming the canonical registry operation. Add separate cases when frontends, positional or keyword signatures, input roles, or static attributes are materially different.

Choose a Hypothesis domain on which every generated example is smooth and well conditioned. Encode constraints in the domain rather than widening tolerances around kinks, singularities, ties, or unstable decompositions. If a law truly does not apply, narrow `laws` and give the case a `reason`.

Then classify the operation and each invocation precisely:

| Situation                                                                                  | Required declaration                                                                                               |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| A differentiable frontend path exists                                                      | An `InvocationCase` for every material frontend, signature, input-role, or static-attribute form                   |
| That exact differentiable invocation cannot abstractly stage                               | Keep its `InvocationCase`, and add its generated `op[frontend]#n` identifier to `DYNAMIC_ONLY_STAGING_INVOCATIONS` |
| A non-differentiable form has an abstract staged contract outside the derivative inventory | A focused case in `STAGED_ONLY_INVOCATIONS`, with only `Law.STAGED` and a reason                                   |
| The operation has no differentiable user input                                             | A reason in `core/_primitive_classification.py:STRUCTURAL_OPS`                                                     |
| A derivative rule intentionally has no supported differentiable frontend path              | A reason in `test_registry_coverage.py:_UNBOUND_OPS` and a `RawRuleCase`                                           |
| The operation is mathematically non-differentiable                                         | A nonempty registry `non_differentiable_reason` and no JVP or VJP                                                  |

The two invocation-lifetime collections live beside `InvocationCase`s in `_builtin_cases.py`. Built-in non-differentiability declarations are assembled from `autodiff/rules/array_family/vjp/registry.py:non_differentiable_items()`.

`InvocationCase` IDs are invocation-specific: the first case is `op[frontend]`, and later cases for that op append `#1`, `#2`, and so on. `DYNAMIC_ONLY_STAGING_INVOCATIONS` therefore records an abstract-lowering gap for one call form, not for the canonical operation as a whole. Do not put a differentiable dynamic-only form in `STAGED_ONLY_INVOCATIONS`.

The registry coverage gate requires every product operation to belong to exactly one semantic bucket: frontend invocation, structural, unbound, or registry-declared non-differentiable. A structural or unbound operation that has an installed derivative rule also needs a focused `RawRuleCase` in `_raw_rule_cases.py`. Raw operands exercise a rule that cannot be reached from a public differentiable call; they are never a shortcut for an available frontend invocation.

Run the ordinary and deep-search conformance gates:

```bash
uv run pytest packages/advect/tests/advect_conformance_tests
uv run pytest packages/advect/tests/advect_conformance_tests \
  --hypothesis-profile=thorough
```

A traced-operation or derivative change also requires the full Python suite:

```bash
uv run pytest
```

## NumPy forms

NumPy owns exact public spelling, signature, protocol dispatch, aliases, mutation behavior, and lifetime claims. A composite under `advect/numpy/_array_function/` should be ordinary traceable composition over existing canonical operations. It does not earn a duplicate registry entry or bespoke derivative rule.

Dynamic and abstract dispatch are related but distinct implementations. For a new array-valued or data-dependent array-function form, put its concrete traced handler in the owning `_array_function/` family, expose it through that family's `register_*_handlers()`, and let `families.register_family_handlers()` and `registry._register_all_handlers()` aggregate it. A trace-independent metadata form instead belongs in `registry._STATIC_ARRAY_FUNCTIONS` and resolves from concrete or abstract metadata without a handler or graph node. Do not create another handler registry or derive a support claim from either dispatch table.

A dynamic handler does not make the form stageable. The abstract path in `numpy/_abstract_calls.py` must either:

- spell the operation as an explicit composition of already staged canonical operations, when the public form is a genuine composition; or
- bind one canonical operation whose schema under `core/_abstract_domains/` owns the abstract result; or
- resolve trace-independent shape, dtype, or scalar metadata directly from the abstract inputs without emitting a graph node.

Keep those three choices honest. A dynamic composition must not record a frontend-only pseudo-primitive merely to make staging convenient, and a direct canonical operation must not acquire a second abstract formula in the NumPy frontend. Metadata resolution must not perform value-dependent computation or masquerade as an array-valued result.

For every new or materially different public form:

1. Declare its spelling, conservative lifetimes, and derivative availability in the NumPy support contract.
1. Add a test-only `NumpySupportCase` in `advect_numpy_tests/_support_cases.py` or `advect_numpy_tests/_support_case_families.py`.
1. Set `derivative_argnums` to the exact independently active input-role groups and every meaningful combined group. These integers index `NumpySupportCase.inputs`, which are the arguments passed to the test's internal `call(*inputs)` wrapper; they are not positions in NumPy's public signature. An `Input(i)` may place the same test input inside a nested positional argument or keyword value.
1. Exercise every claimed `dynamic`, `staged`, and `serialized` lifetime. Unsupported lifetimes must remain undeclared.
1. For a non-differentiable form, still prove primal values, dtype, no input mutation, and every declared staged or serialized mode.
1. Add an `InvocationCase` as well if the form introduces or newly exposes a differentiable canonical operation; primitive conformance owns its formula.

Run the executable declaration/case coverage gate:

```bash
uv run pytest \
  packages/advect/tests/advect_numpy_tests/test_support_evidence.py
```

Changes sensitive to NumPy-minor behavior also require the repository's NumPy 2.0–2.5 compatibility matrix. That matrix is owned by the `numpy-compatibility` job in `.github/workflows/ci.yml`; the locked local environment proves only its installed NumPy minor. Do not report range-wide qualification until that CI matrix passes.

## Array API forms and providers

Array API work belongs under `core/_array_api/`, with revision profiles, signatures, result rules, provider negotiation, evidence, and support kept as separate responsibilities. Extend the binding and executable evidence for every live or static parameter and every claimed lifetime on each admitted revision.

The frozen official spelling and signature belong in `profiles.py`; binding to a canonical op or a declared composite belongs in `frontend.py`; executable parameter and lifetime cases belong in `evidence.py`; and `support.py` derives the fail-closed claim from those sources. Do not hand-author a second support inventory.

Run the provider-neutral gradient and core staging tests that cover the changed form, then execute every affected revision explicitly:

```bash
uv run pytest \
  packages/advect/tests/advect_grad_tests/test_array_api_operation_qualification.py

for version in 2022.12 2023.12 2024.12; do
  uv run python -m scripts.qualify_array_api_operations \
    --array-api-version "$version" \
    --providers array-api-strict \
    --subset all
  uv run python -m scripts.qualify_array_api_operations \
    --array-api-version "$version" \
    --providers array-api-strict,numpy \
    --subset portable
done
```

Run only the revisions the profile change can affect while iterating; run all three before review for a shared canonical operation or binding. A provider change also requires:

```bash
uv run pytest packages/advect/tests/advect_array_api_compat_tests
```

Run CuPy qualification only on a CUDA host and record that hardware gap explicitly when it is unavailable. Correctness evidence does not create a performance claim.

## SciPy forms

A SciPy-compatible function wrapper owns value, dtype, and signature parity with the admitted SciPy 1.18 release line. Solver factories are Advect adapters instead and own the callback contract described below. Additional function evidence depends on how the wrapper is implemented.

The support catalog walks functions exported through the public SciPy modules' `__all__`; there is no separate SciPy support declaration. Export every new direct function, composite, or adapter through the intended public module. The catalog then discovers a matching `custom.scipy.*` primitive, an explicit composite marker, or a dynamic-only adapter.

### Direct primitive

A shipped direct SciPy form is a public wrapper around a private `@advect.primitive` handle. The decorator name `scipy.<module>.<function>` registers `custom.scipy.<module>.<function>`; that exact relationship lets the support catalog join the public wrapper to its primitive. It is not a built-in array-family operation: do not add it to `advect/_builtin_ops.py`, the array-family JVP/VJP registries, or the NumPy or Array API support inventories.

Implement concrete SciPy execution, `def_abstract`, a JVP, and any necessary explicit real-adjoint transpose on the private primitive. Keep the public SciPy-compatible signature and concrete fast path on the wrapper. Because these primitives ship as Advect product behavior, add an `InvocationCase` to the SciPy group in `_builtin_cases.py` naming its `custom.scipy.*` ID, even though third-party custom primitives do not enter the repository inventory. The owning `advect_scipy_tests` test must independently prove upstream value, dtype, and signature parity; every differentiable input alone and in meaningful combinations; and each dynamic, staged, and serialized lifetime published by the catalog. A serialized artifact also depends on importing `advect.scipy` before loading so the custom operation is linked.

### Traceable composite

A composite built only from existing operations stays owned by its public SciPy test and is marked `__advect_lowering__ = "composite"` for support discovery. Check upstream parity, all independently active and combined input roles, and every claimed lifetime. The marker is an unconditional dynamic-plus-staged-plus-serialized catalog claim, so do not set it until all three paths are executable. Do not decorate the composition as a new primitive and do not add it to primitive conformance; add an `InvocationCase` only for a direct bundled primitive it newly introduces.

### Solver callback

`root_solver` and `gmres_solver` are concrete, first-order dynamic callbacks for `implicit_root`. Test shape and scalar-category preservation, upstream solver behavior, convergence failures, and integration with the implicit derivative boundary. Do not claim staging or serialization for opaque solver iterations; stage explicit iterations or define a closed primitive when a staged program is required.

Run the public SciPy suite after any of these changes:

```bash
uv run pytest packages/advect/tests/advect_scipy_tests
```

A direct bundled primitive also changes the shipped derivative inventory, so run:

```bash
uv run pytest packages/advect/tests/advect_conformance_tests
uv run pytest packages/advect/tests/advect_conformance_tests \
  --hypothesis-profile=thorough
uv run pytest
```

After changing the exported public surface, regenerate the compatibility pages as described below, then run:

```bash
uv run pytest packages/advect/tests/advect_core_tests/test_support_catalog.py
uv run mkdocs build --strict
```

## Staged-program, runtime-graph, and native changes

Classify a durability change by the malformed or changing field before choosing the implementation boundary.

### Python `StagedProgram` envelope

Python core owns the outer format and version, call and output pytrees, leaf specifications, the constant manifest, the optimization report, and validation that those records agree with one another and with the enclosed graph. Keep pytree and Python-call structure out of the Rust graph format. Test envelope shape, version handling, cross-record mismatches, and public round trips in `advect_core_tests`, especially `test_stage_durability.py`.

For an envelope-only change, run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyrefly check
uv run pytest packages/advect/tests/advect_core_tests/test_stage_durability.py
uv run pytest packages/advect/tests/advect_core_tests
```

Run the full Python suite as well when public invocation, staging, or transform behavior changes. An envelope-only codec or validation change does not require a native wheel build when the enclosed graph and PyO3 contract are unchanged.

### Canonical runtime graph artifact

`advect-runtime` owns closed graph data, the graph format and version, `GraphStore` validation before ownership is accepted, canonical serialization, optimization, topology, execution planning, and staged value lifetimes. Test a graph-local invariant at this boundary, including canonical round trips and pure-Rust fixture execution where relevant. Do not mirror its node model or graph-local validators in Python.

For an internal runtime change whose adapter contract is unchanged, run:

```bash
cargo fmt --all --check
cargo clippy --locked -p advect-runtime --all-targets --all-features -- -D warnings
cargo test --locked -p advect-runtime --all-targets
cargo deny --all-features check -W unmaintained
```

If the accepted graph format, validation result, or Python-visible graph contract changes, also run the workspace and native boundary gates below.

### Native translation and dynamic tape

`advect-native` owns PyO3 conversion, exception mapping, callbacks, thin graph handles, the Python host adapter, and the invocation-local dynamic tape. It does not own the outer envelope or graph policy. A translation change must exercise the corresponding Python-to-Rust round trip; a tape regression stays with the dynamic native tests and must not be expressed as a graph-artifact rule.

Run the workspace and adapter gates for a native change or adapter-visible runtime change:

```bash
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets
cargo deny --all-features check -W unmaintained
uv run pytest packages/advect/tests/advect_native_tests
uv build --package advect --wheel --out-dir dist/wheelhouse --clear
```

If Python runtime behavior changed, also run the owning core or integration tests and the full Python suite. Add format compatibility only for an explicit supported-version requirement; do not create an unowned fallback between the two formats.

## Publish the support claim

After changing a public NumPy, Array API, or SciPy surface, regenerate the catalog pages from the live declarations:

```bash
uv run python -m scripts.report_extension_support \
  --format markdown \
  --output docs/compatibility
```

Never edit the generated compatibility inventories by hand. The [testing guide](https://yaugenst.github.io/advect/0.2.0/development/testing/index.md) lists the complete gates and suite ownership. Update the hand-written xarray compatibility contract when its container or metadata boundary changes.
