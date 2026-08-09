## Owns

- The required PyO3 extension, invocation-local dynamic tape, Python conversion
  and exception mapping, derivative callbacks, thin runtime graph handles, and
  `PythonHost` adapter over `advect-runtime`.

## Must not own

- The outer `StagedProgram` envelope, a second durable graph, graph serializer
  or validator, optimizer, staged execution loop, or numerical array library.
- Host-independent graph policy that belongs in `advect-runtime`.
- A separately installable public Python or Rust API.

## Read

These are routes, not a checklist. Read only what the change needs:

- [Crate README](README.md) for crate concepts and layout.
- [Codebase map](../../docs/development/codebase.md) for an ownership question.
- [Native workflow](../../docs/development/adding-operations.md#native-translation-and-dynamic-tape)
  for an implementation or adapter-contract change.
- [Runtime extension decision](../../design/decisions/2026-07-24-runtime-extension-boundaries.md)
  only when changing the accepted boundary or its rationale.

## Verify

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo deny --all-features check -W unmaintained`
- `uv run pytest packages/advect/tests/advect_native_tests`
- `uv build --package advect --wheel --out-dir dist/wheelhouse --clear`
