# Contributing to Advect

Advect is deliberately small: changes should preserve one core tracing model,
one canonical operation registry, and one source of truth for each public
support claim. A focused issue or draft pull request is the best place to align
on a user-visible change before implementing a broad extension.

## Set up the repository

Advect requires Python 3.12 or newer and Rust 1.94 or newer.

```bash
uv sync --all-groups
uv lock --check
rustup toolchain install stable --component clippy rustfmt
```

The repository has no wrapper around its validation commands. The complete
local gates are:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyrefly check --config pyproject.toml --preset strict packages/advect/src
uv run pytest
uv run pytest packages/advect/tests/advect_conformance_tests --hypothesis-profile=thorough
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets
cargo deny check -W unmaintained
uv run mkdocs build --strict
```

Run the checks that own the changed contract while iterating, then run every
required gate before requesting review. A GPU or browser qualification that is
not available locally should be named as an unverified lane rather than
inferred from CPU or native results.

## Put evidence at its owning boundary

- Core tracing, pytrees, staging, serialization, and registry behavior belongs
  in `advect_core_tests`.
- Canonical operation laws and raw derivative rules belong in
  `advect_conformance_tests`.
- NumPy signatures, protocols, aliases, mutation, and executable support claims
  belong in `advect_numpy_tests`.
- Array API provider behavior and CuPy qualification belong in
  `advect_array_api_compat_tests`.
- Public SciPy, xarray, and host-autodiff behavior belongs in its corresponding
  optional suite.
- Python-to-Rust graph, artifact, ownership, and execution contracts belong in
  `advect_native_tests`.

Do not repeat one happy path across several suites. The full ownership table
and the primitive-authoring workflow are in
[`AGENTS.md`](https://github.com/yaugenst/advect/blob/main/AGENTS.md).

## Change a public operation

A registered primitive needs its binding or tracing handler, abstract
semantics, and JVP rule. Add an explicit VJP only when structural transposition
cannot express the correct real adjoint or measurement justifies a direct
implementation. Every registered operation must also have a conformance case;
the registry-coverage test enforces that accounting.

A new NumPy spelling or materially different signature needs one executable
support case for each claimed lifetime. A new SciPy entry point needs upstream
value, dtype, and signature parity; a derivative contract; and staged
serialize/load coverage whenever the support catalog claims those modes.
Regenerate the compatibility pages after any public-surface change:

```bash
uv run python -m scripts.report_extension_support \
  --format markdown \
  --output docs/compatibility
```

## Submit the change

Work on a feature branch and keep the pull request to one coherent problem.
Explain the user-visible contract, the evidence run, and any hardware or remote
CI lane that remains outstanding. Do not add compatibility aliases, duplicated
registries, or speculative fallbacks to make a narrow change appear broader.

Contributions accepted into this repository are distributed under its MIT
license. Report suspected vulnerabilities through the repository's private
security-advisory form rather than a public issue.
