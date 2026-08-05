# Extension Support and Upstream Compatibility

Advect reports support from the runtime structures that implement it. The
public catalog separates the Array API, NumPy, and SciPy extensions and joins
their frontend bindings to the canonical primitive registry.

```python
import advect as ad

catalog = ad.support_catalog()
for name, extension in catalog["extensions"].items():
    print(name, len(extension["functions"]))
```

`support_catalog()` returns a JSON-serializable dictionary. The generated
[compatibility pages](../../docs/compatibility/index.md) present the same
data for users.

## What a listed function means

The catalog is organized by the extension users actually call. Live frontend
structures discover candidate implementations, while executable contracts
decide what is published:

- Array API functions come from the generated namespace bindings and compile-time
  metadata handlers.
- NumPy functions come from compact declarations in
  `advect.numpy._support_contract`, joined to live array-function handlers,
  ufunc calls and methods, and ndarray methods. Test-only executable cases in
  `advect_numpy_tests._support_cases` cover declarations in both directions.
- SciPy functions come from the public `advect.scipy` exports and their
  registered primitives. Solver callback factories are shown separately as
  adapters.

For each exact spelling shown, the catalog reports dynamic, staged, and
serialized support. A mode marked `yes` is an end-to-end support claim for the
callable's declared contract in that mode, not evidence that one representative
call happened to reach a handler. A composite frontend remains labelled
`composite`; the primitive columns do not invent a derivative contract for it.

Partial paths are kept internal or explicitly restricted. Additional aliases
and subpackage spellings are not inferred from a related registered function.
In particular, `numpy.polynomial.polynomial.polyval` is not advertised by the
separate implementation of `numpy.polyval`.

All nine `numpy.lib.scimath` functions are supported dynamically. With the same
real input shape and dtype, their output can be real for one value and complex
for another, so `ArraySpec` cannot express a sound staged result dtype. Their
public declarations and executable cases therefore claim only the dynamic
lifetime.

Three callable rows are conservative across materially different forms:

- `numpy.round` supports dynamic, staged, and serialized execution when
  `decimals` is omitted or zero; nonzero `decimals` is dynamic-only.
- `numpy.linalg.eig` supports all three lifetimes for complex input dtypes and
  dynamic execution for real input dtypes.
- `numpy.linalg.eigvals` has the same complex-versus-real lifetime boundary.

These restrictions are explanatory notes, not a parameter-level support
schema. The callable declaration remains the fail-closed public catalog policy,
and independent executable cases prove each material form.

`numpy.linspace` is dynamic-only when `start` or `stop` is traced. Its dynamic
path differentiates both endpoints and preserves the two-output
`retstep=True` form. NumPy permits array-valued endpoints, while the shared
Array API primitive has static scalar endpoint attributes, so a traced NumPy
call cannot become a durable graph node. The baseline catalog's staged and
serialized flags were therefore an overclaim: the baseline already failed to
stage these forms. A constant-only `numpy.linspace` call may execute eagerly
and be captured as a graph constant, which is not staged `numpy.linspace`
support.

## NumPy preservation snapshot

The completion audit compared exact baseline and candidate wheels under the
same installed NumPy minor. Row totals follow the upstream surface available in
that minor; they are not one cross-version invariant.

| NumPy | Baseline rows | Candidate rows | Array-function handlers | Ufunc calls |
|---|---:|---:|---:|---:|
| 2.0.2 | 381 | 390 | 251 | 82 |
| 2.1.3 | 384 | 393 | 254 | 82 |
| 2.2.6 | 395 | 395 | 254 | 84 |
| 2.3.5 | 395 | 395 | 254 | 84 |
| 2.4.6 | 393 | 393 | 252 | 84 |

No callable, handler, or ufunc call was removed in a same-minor comparison.
NumPy 2.1 adds `cumulative_prod`, `cumulative_sum`, and `unstack`; NumPy 2.2
adds `matvec` and `vecmat`; NumPy 2.4 removes `in1d` and `trapz`. The candidate
restores the nine `numpy.lib.scimath` rows missing from the baseline catalog on
NumPy 2.0 and 2.1.

The audit also preserves 44 root exports, 154 direct generated Array API
bindings, 169 public Array API rows, and 224 built-in registry IDs. The full
scientific catalog currently reports 258 primitive rows after optional SciPy
operations are loaded; that is a different inventory from the built-in
registry and is reported separately.

## Adding a supported form

The foreign callable and the canonical primitive are separate contracts. Use
the lane that matches the change:

| Change | Implementation authority | Required executable evidence |
|---|---|---|
| Canonical differentiable operation | Complete `OpDef` and its JVP; an explicit VJP only when required | Conformance invocation for every material frontend/signature form, or a raw-rule case when no frontend reaches it |
| NumPy function, ufunc form, or ndarray method | NumPy handler or method implementation plus compact support declaration | Test-only `NumpySupportCase` for every materially different static form and every claimed lifetime; retained contract on NumPy 2.0-2.4 |
| Array API operation or signature | Generated binding plus canonical operation | Data-only Array API operation case covering each live/static parameter and claimed lifetime on admitted revisions/providers |
| Non-differentiable form | Complete registry reason and frontend implementation | Primal, dtype, input-immutability, and staged/serialized execution when claimed |

Removing a NumPy declaration removes its public mode claim. A coverage test
requires every declaration to have an executable case and every case to name a
declaration; registration or an abstract rule cannot promote a callable. Every
marked staged mode executes both the compiled program and its
serialized/restored artifact.
The locked development environment uses NumPy 2.4 and runs the full suite; the
NumPy 2.0-2.4 matrix repeats the focused executable support contract on every
retained minor.

Benchmarking is selective. A new spelling or ordinary formula needs no
benchmark by default. Add a reference-versus-candidate Advect workload only
when the change introduces a new mechanism, touches a measured hot path, or
makes a performance claim; ecosystem comparisons are informative rather than
acceptance gates.

## Intentional compatibility removal

`advect.numpy.register_backend()` was a public-looking test reset helper, not a
user-facing backend extension contract. It repopulated mutable internal
registries after tests cleared them. NumPy registration still happens on import,
but the reset function is intentionally absent now that frontend hooks are
single-assignment and have no unregister or reset lifecycle. No supported NumPy
callable or constructor behavior depends on calling it.

## Shared Array API semantics

NumPy rows marked `array_api` lower to a canonical operation also used by an
Array API binding. For example, `numpy.sin` and the Array API `sin` function
both lower to `array.sin` and therefore share its evaluator and derivative
rules.

This is primitive inheritance, not signature inheritance. NumPy's frontend
still owns NumPy keyword spelling, mutation controls, and calling conventions.

## Version and provider bounds

| Component | Bound | Boundary |
|---|---|---|
| NumPy | `>=2.0,<2.5` | Required first-class protocol frontend |
| Python Array API | 2022.12, 2023.12, 2024.12 | Provider-neutral namespace profiles |
| SciPy | `>=1.18,<1.19` | Optional explicit `advect.scipy` extension |
| `array-api-compat` | `>=1.11.2,<2` | Required provider-discovery fallback |
| xarray | `>=2026.7,<2026.8` | Dynamic pytree containers |
| `array-api-strict` | 2.4.1 | Reference Array API qualification provider |
| CuPy | separately installed | Manual single-device GPU qualification |

The generic Array API profiles form one ordered contract. Advect materializes
2022.12 as the frozen base, applies the 2023.12 additions and changed
signatures, then applies the 2024.12 delta. Dynamic transforms request versions
newest first and choose the newest revision every input can serve. A provider
that accepts a request may report a newer namespace; Advect preserves that
metadata rather than rewriting it to the requested revision.

NumPy is not routed through `array-api-compat`. Its installed minor selects the
default portable target:

| NumPy minor | Default Array API target |
|---|---|
| 2.0 | 2022.12 |
| 2.1-2.2 | 2023.12 |
| 2.3-2.4 | 2024.12 |

The target is a compatibility default, not a reduction of NumPy's richer API
to the standard. NumPy-only call forms retain NumPy's own frontend semantics.
Cross-minor and cross-provider portability comes from staging against an
explicit Array API revision.

Dependency bounds live in `pyproject.toml`. Provider qualification remains a
test and evidence concern: the official Array API suite and the portable
scientific workload exercise concrete calls across program lifetimes. Those
tests do not maintain a second user-facing support catalog.

`array-api-compat` supplies the fixed fallback for providers that need one. The
lower-bound and latest 1.x lanes must accept explicit requests for all three
Advect revisions. Historical source notes cover manual CuPy/CUDA runs for
2022.12, 2023.12, and 2024.12, but no retained artifact and digest currently
verify those provider passes.

Regenerate the checked-in catalog with:

```bash
uv run python -m scripts.report_extension_support \
  --format markdown \
  --output docs/compatibility
```
