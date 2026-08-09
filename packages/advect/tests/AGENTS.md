# Tests

## Owns

- Executable evidence for public and internal contracts, placed in the suite
  that owns the boundary.
- Test-only conformance domains, invocation cases, raw-rule cases, frontend
  support cases, fixtures, and promoted regressions.

## Must not own

- Runtime support declarations, production dispatch metadata, or helper logic
  required for the package to function.
- Repeated happy paths that prove the same contract in several suites.
- Broad tolerance increases or closed example matrices that conceal an
  ill-chosen numerical domain.

## Read

- [Suite ownership](../../../docs/development/testing.md#suite-ownership)
- [Adding operations](../../../docs/development/adding-operations.md)
- [Primitive conformance decision](../../../design/decisions/2026-07-27-primitive-conformance-testing.md)
- The nearest source `AGENTS.md` for the behavior under test

## Verify

- Run `uv run pytest` on the owning suite or test file while iterating.
- Run the full Python suite for shared trace, derivative, registry, or public
  support changes.
- Run the thorough conformance profile for a new or changed canonical operation.
- Run format and lint from the [testing guide](../../../docs/development/testing.md).
