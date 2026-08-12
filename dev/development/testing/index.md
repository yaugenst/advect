# Testing

Put each contract in the suite that owns the boundary. A focused integration test may cross boundaries, but do not repeat the same happy path throughout the tree.

## Suite ownership

| Suite                           | Owns                                                                                                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `advect_conformance_tests`      | Canonical operation laws, material invocation variants, registry accounting, and raw rules with no frontend path                                                                |
| `advect_core_tests`             | Trace lifetimes, registry behavior, pytrees, staging, the outer `StagedProgram` envelope and its cross-record validation, support reporting, and provider-neutral core behavior |
| `advect_numpy_tests`            | Exact NumPy signatures, protocols, aliases, `out=` and mutation, and executable public support claims                                                                           |
| `advect_grad_tests`             | Cross-frontend differentiation, transform APIs, Array API revisions, and provider-neutral integration                                                                           |
| `advect_scipy_tests`            | Public SciPy parity, derivatives, solver adapters, and every claimed dynamic, staged, and serialized lifetime                                                                   |
| `advect_array_api_compat_tests` | Compatibility-provider behavior and real-provider qualification such as CuPy                                                                                                    |
| `advect_interop_tests`          | Optional JAX, PyTorch, and HIPS Autograd host-transform bridges                                                                                                                 |
| `advect_xarray_tests`           | xarray pytree and alignment behavior, plus its explicit staging boundary                                                                                                        |
| `advect_native_tests`           | PyO3 translation, the invocation-local dynamic tape, Python/runtime graph integration, ownership, and execution contracts                                                       |

The [operation-authoring guide](https://yaugenst.github.io/advect/dev/development/adding-operations/index.md) explains the exact `InvocationCase`, `RawRuleCase`, `NumpySupportCase`, derivative-role, and lifetime evidence required for a new public form.

## Python gates

Set up the locked development environment with Python 3.12 through 3.14:

```bash
uv sync --all-groups --locked
uv run pre-commit install --install-hooks
```

Hook installation is per clone. Before review, verify the complete repository hygiene configuration independently of the files staged in the current commit:

```bash
uv run pre-commit run --all-files
```

The installed `pre-commit` hook may fix files and require restaging. The `commit-msg` hook enforces Conventional Commits. Avoid `--no-verify`, which bypasses all checks.

The canonical static and test gates are:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyrefly check
uv run pytest
uv run pytest packages/advect/tests/advect_conformance_tests \
  --hypothesis-profile=thorough
```

Pyrefly checks the runtime package and the release/browser-documentation scripts listed in `pyproject.toml`. Dynamic provider-qualification and benchmark scripts, plus tests, remain outside this production typing gate; their necessary boundary suppressions stay local.

Run the owning suite while iterating. Run the full Python suite for changes to traced operations, derivative rules, shared transform behavior, or public support declarations.

Hypothesis should own shrinkable numerical domains and combinatorial invariants. CI does not retain the local `.hypothesis` database, so promote a discovered failure to `@example` or a focused regression. Do not replace an open numerical domain with a two-value matrix or loosen tolerances to hide a nonsmooth case.

## Rust and native gates

Use Rust 1.94 or newer with Clippy and rustfmt:

```bash
rustup toolchain install stable --component clippy rustfmt
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets
cargo deny --all-features check -W unmaintained
uv run pytest packages/advect/tests/advect_native_tests
uv build --package advect --wheel --out-dir dist/wheelhouse --clear
```

CI runs the locked workspace Clippy and test gates on both Rust 1.94 and the current stable toolchain.

For an internal `advect-runtime` change whose adapter contract is unchanged, the focused Clippy and test forms (`-p advect-runtime`) plus format and `cargo deny` are sufficient while iterating. Run the workspace forms above for a native change. A runtime change also needs the Python/native boundary tests and wheel build when it alters the accepted graph format, validation result, ownership, execution behavior, or another adapter-visible contract.

An outer `StagedProgram` envelope change instead runs Python static checks and `advect_core_tests`, with `test_stage_durability.py` as the focused regression owner. It does not require a wheel build when the enclosed runtime graph and PyO3 contract are unchanged. The [operation-authoring guide](https://yaugenst.github.io/advect/dev/development/adding-operations/#staged-program-runtime-graph-and-native-changes) lists the proportional command sets.

## Documentation gates

```bash
uv run mkdocs build --strict
uv run python scripts/run_doc_snippets.py docs
```

The native snippet runner is necessary but not sufficient for browser support. When runnable snippets or the playground boundary changes, build the Pyodide wheel and run the browser lane described in the [documentation guide](https://yaugenst.github.io/advect/dev/development/documentation/index.md).

## Specialized evidence

- NumPy surface changes run `uv run pytest packages/advect/tests/advect_numpy_tests/test_support_evidence.py` locally. NumPy 2.0–2.5 range evidence is owned by the `numpy-compatibility` matrix in `.github/workflows/ci.yml`; one locally installed minor is not range qualification.
- Array API operation changes run `uv run pytest packages/advect/tests/advect_grad_tests/test_array_api_operation_qualification.py` and `uv run python -m scripts.qualify_array_api_operations` for every affected revision, as shown in the [operation-authoring guide](https://yaugenst.github.io/advect/dev/development/adding-operations/#array-api-forms-and-providers). Provider changes also run `uv run pytest packages/advect/tests/advect_array_api_compat_tests`. CuPy is qualified only on a CUDA host; record it as unverified when unavailable.
- JAX, PyTorch, or HIPS changes run `uv run --no-sync pytest packages/advect/tests/advect_interop_tests` in an environment containing all qualified host dependencies.
- xarray changes run `uv run pytest packages/advect/tests/advect_xarray_tests`.
- Blocking performance evidence compares exact reference and candidate wheels through `uv run python -m scripts.bench_advect_regression --help`.
- Runtime memory acceptance uses `uv run python -m scripts.bench_runtime_memory --profile cpu-runtime --acceptance`.

Provider correctness, one local dependency version, or a non-gating benchmark must not be reported as a broader compatibility or performance claim.
