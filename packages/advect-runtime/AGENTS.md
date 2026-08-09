# advect-runtime

## Owns

- The Python-independent staged SSA graph, closed attributes and constants,
  `GraphStore` validation, the canonical graph artifact format, optimization,
  topology, execution planning, and host-independent value lifetimes.
- The host contract consumed by native and fixture adapters.

## Must not own

- Python, PyO3, the outer `StagedProgram` envelope, numerical array kernels,
  provider dispatch, or the invocation-local dynamic tape and callbacks.
- Code generation, fusion, backend lowering, or another artifact format without
  an accepted requirement.
- Python-adapter policy that belongs in `advect-native`.

## Read

These are routes, not a checklist. Read only what the change needs:

- [Crate README](README.md) for crate concepts and layout.
- [Codebase map](../../docs/development/codebase.md) for an ownership question.
- [Runtime graph workflow](../../docs/development/adding-operations.md#canonical-runtime-graph-artifact)
  for an implementation or validation change.
- [Lifetime-aware runtime decision](../../design/decisions/2026-07-24-lifetime-aware-portable-rust-runtime.md)
  only when changing the accepted boundary or its rationale.

## Verify

- `cargo fmt --all --check`
- `cargo clippy -p advect-runtime --all-targets --all-features -- -D warnings`
- `cargo test -p advect-runtime --all-targets`
- `cargo deny --all-features check -W unmaintained`
- If the accepted graph format, validation result, or another adapter-visible
  contract changes, also run the native boundary tests and wheel build in the
  [testing guide](../../docs/development/testing.md).
