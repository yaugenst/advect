# AGENTS

Keep Advect small. Follow the architecture docs, prefer deletion to compatibility
layers, and run the direct checks that cover the code you changed.
This repository intentionally has no contributor harness; the commands
below are the canonical local and CI gates.

## Read First

- `design/requirements.md`
- `design/data-structures.md`
- `design/target-api.md`
- `design/implementation/roadmap.md`
- `design/decisions/`

## Setup

- Python >= 3.12: `uv sync --all-groups`
- Locked environment: `uv lock --check`
- Rust >= 1.94: `rustup toolchain install stable --component clippy rustfmt`

## Direct Checks

- Format: `uv run ruff format --check .`
- Lint: `uv run ruff check .`
- Types:
  `uv run pyrefly check --config pyproject.toml --preset strict packages/advect/src`
- Python tests: `uv run pytest`
- Thorough primitive conformance:
  `uv run pytest packages/advect/tests/advect_conformance_tests --hypothesis-profile=thorough`
- Rust format: `cargo fmt --all --check`
- Rust lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Rust tests: `cargo test --workspace --all-targets`
- Docs: `uv run mkdocs build --strict`
- Browser wheel (needed once before the docs build; the mkdocs pre-build
  hook stages it plus the playground adapter and example scripts into
  docs/assets): `mkdir -p dist && uvx --from pyodide-build==0.36.0 pyodide build . --outdir dist/pyodide`
- Compatibility pages: `docs/compatibility/{index,numpy,array-api,cupy,scipy}.md`
  are generated from the live support catalog — regenerate with
  `uv run python -m scripts.report_extension_support --format markdown
  --output docs/compatibility`; a test pins them to the catalog
- Runnable docs snippets: mark a fence with ` ```{.python .run} ` to give it
  a [ run ] chip. The page is one Python session: running a block first
  executes any not-yet-run marked blocks above it, so tutorial snippets may
  build on earlier ones. A marked block must run cleanly in Pyodide given
  numpy + advect and everything marked above it. Execute them all with
  `uv run python scripts/run_doc_snippets.py docs` (CI runs this natively
  and in the Pyodide venv)
- Native wheel:
  `uv build --package advect --wheel --out-dir dist/wheelhouse --clear`
- Non-gating dynamic ecosystem comparison:
  `uv run --with autograd python -m scripts.bench_autodiff_runtime`
- Blocking performance evidence compares exact reference and candidate wheels:
  `uv run python -m scripts.bench_advect_regression --help`
- Memory acceptance uses one exact profile:
  `uv run python -m scripts.bench_runtime_memory --profile cpu-runtime --acceptance`

## Repository Rules

- Python source lives in `packages/advect/src`; do not use relative imports.
- Keep `advect.core` stdlib-only. The required Rust extension owns
  the staged compute graph; third-party array dependencies belong in frontends.
- NumPy protocol dispatch belongs under `advect.numpy`. `advect.core` keeps
  only stateless helpers used by a real core or Array API consumer; the two
  frontends meet at canonical operation emission.
- Runtime derivative IDs are `array.*`, `array_ext.*`, and explicit `advect.*`
  internals.
- Prefer JVP-first derivative rules. Never silently substitute numeric
  pullbacks.
- Do not add compatibility aliases, deprecated dual paths, or dead code.

## Validation

- Runtime Python changes: format, lint, strict types, and relevant tests.
- Traced-op or derivative changes: run the full Python test suite.
- Rust/native changes: Rust format, lint, tests, relevant Python tests, and a
  native wheel build.
- Documentation changes: strict docs build.
- CI changes: validate workflow syntax and run the corresponding direct checks.

## Test Suite Ownership

Put each contract at its owning boundary; do not repeat the same happy path in
several suites.

| Suite | Owns |
| --- | --- |
| `advect_conformance_tests` | Canonical operation laws, invocation variants, and raw rules with no frontend path |
| `advect_core_tests` | Trace lifetimes, registry, pytrees, staging, serialization, and provider-neutral core behavior |
| `advect_numpy_tests` | Exact NumPy signatures, protocols, aliases, `out=`/mutation, and executable support claims |
| `advect_grad_tests` | Cross-frontend differentiation, Array API revisions, and provider-neutral integration |
| `advect_scipy_tests` | Public SciPy function/solver parity, derivatives, and every claimed staged lifetime |
| `advect_array_api_compat_tests` | Compatibility-provider behavior and real-provider qualification such as CuPy |
| `advect_interop_tests` | Optional JAX, PyTorch, and HIPS Autograd host-transform bridges |
| `advect_xarray_tests` | xarray pytree/alignment behavior and its explicit staging boundary |
| `advect_native_tests` | Python-to-Rust graph, artifact, ownership, and execution contracts |

A new SciPy entry point needs upstream value/dtype/signature coverage, its
derivative contract, and stage/serialize/load coverage when the support catalog
claims those modes. Add a canonical conformance case when it introduces a
primitive; a composition stays owned by its public SciPy test. Regenerate the
compatibility pages after changing the public surface.

A provider change needs the provider-neutral contract tests and the matching
compatibility suite. Run the Array API revision qualification for every affected
revision; run CuPy qualification only on a CUDA host and record that hardware
gap explicitly when unavailable. Provider correctness does not create a
performance claim.

Changes to xarray or a host-autodiff bridge belong in their optional suite and
must exercise both the supported dynamic path and the documented staging or
nested-transform boundary. Run the `host-autodiff-interop` dependency lane for
JAX, PyTorch, or HIPS changes.

## Branches

- Work on a feature branch.
- Rebase on current `origin/main` before merge; keep history linear.
- Merge pull requests with rebase.

## Adding a Primitive

See the [conformance testing
ADR](design/decisions/2026-07-27-primitive-conformance-testing.md)
for the tiers, law battery, and authoring workflow.

Every registered operation must be accounted for by the conformance suite;
`advect_conformance_tests/test_registry_coverage.py` fails until it is.

1. Complete the canonical operation record: binding and tracing handler,
   abstract semantics, and JVP. Add an explicit VJP only when structural
   transposition cannot express the correct real adjoint or measurement
   justifies a direct implementation.
2. Add an `InvocationCase` to `packages/advect/tests/advect_conformance_tests/
   _builtin_cases.py`, naming the canonical registry op. Add distinct cases
   for materially different frontends, signatures, or static attributes.
3. Choose the domain that keeps the primitive smooth and well conditioned on
   every value it can draw. Do not widen tolerances to hide a kink.
4. If a law genuinely does not apply, narrow `laws` and record `reason`.
5. If the op is not differentiable at all, mark it in the registry instead;
   the gate reads `non_differentiable_reason` directly.
6. If no supported frontend reaches a registered rule, add a focused
   `RawRuleCase` in `_raw_rule_cases.py`.

Run `uv run pytest packages/advect/tests/advect_conformance_tests`.
Deep search: `uv run pytest --hypothesis-profile=thorough`.

## Adding Supported Frontend Forms

- NumPy: declare the public spelling, conservative lifetimes, and derivative
  availability in `advect.numpy._support_contract`. Add a test-only
  `NumpySupportCase` in `advect_numpy_tests._support_cases` for every material
  signature/static form. Qualification executes every declared lifetime and
  checks declaration/case coverage in both directions. `derivative_argnums`
  owns the exact independently active and combined input-role groups; primitive
  conformance owns their formulas, so add a conformance invocation for each
  differentiable operation.
- Array API: extend the generated binding and `_array_api_evidence` cases for
  every live/static parameter and claimed lifetime on the admitted revisions.
- Non-differentiable forms still require executable primal, dtype,
  no-input-mutation, and staged/serialized evidence for every marked mode.
- Use Hypothesis for a shrinkable numerical domain or combinatorial invariant,
  not a two-value closed matrix. CI does not retain the local `.hypothesis`
  database; promote discoveries to `@example` or focused regression tests.
- Benchmark only a new mechanism, a changed measured hot path, or a performance
  claim. Ordinary aliases and signatures do not need a new workload.

Run the NumPy contract with
`uv run pytest packages/advect/tests/advect_numpy_tests/test_support_evidence.py`.
Regenerate compatibility pages after changing it.

## Adding a Benchmark

An ordinary new function earns conformance evidence, not a permanent
microbenchmark. Add or extend one representative performance scenario only for
a new runtime/lifetime mechanism, a claimed-faster bespoke rule, a known hot
path, or a new provider execution class with a performance requirement.

Keep the permanent reference-versus-candidate gate singular. Add a phase only
for a new runtime boundary that no existing phase measures, and add a workload
only for a concrete performance requirement the representative stencil cannot
express. Run the existing gate for a refactor with no new performance claim.
See `design/implementation/performance.md`.
