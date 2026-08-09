# Core

## Owns

- Standard-library-only registry, tracing context, abstract values, pytrees,
  diagnostics, staging orchestration, and native-extension protocols.
- Provider-neutral Array API behavior under `_array_api/`: frontend calls,
  revision profiles, signatures, results, providers, evidence, and support.
- Canonical operation records and the versioned outer `StagedProgram` envelope:
  Python-specific codecs and metadata plus validation across pytrees,
  call/output specifications, the constant manifest, optimization report, and
  the runtime-owned graph.

## Must not own

- NumPy protocol dispatch, SciPy behavior, optional-framework imports, or any
  third-party array dependency.
- A frontend-specific callable inventory or duplicate support registry.
- A node-by-node graph model, graph-local serializer or validator, optimizer,
  or execution plan parallel to `advect-runtime`.

## Read

These are routes, not a checklist. Read only what the change needs:

- [Codebase map](../../../../../docs/development/codebase.md) for an ownership question.
- [Adding operations](../../../../../docs/development/adding-operations.md) for
  an operation or staged-program implementation.
- [Target API](../../../../../design/target-api.md) only for a public-contract change.
- The relevant record under [design decisions](../../../../../design/decisions/README.md)
  only for a boundary or rationale change.

## Verify

- `uv run pyrefly check --config pyproject.toml --preset strict packages/advect/src`
- `uv run pytest packages/advect/tests/advect_core_tests`
- For an Array API integration change, also run the affected
  `advect_grad_tests` and `advect_array_api_compat_tests`.
- For a canonical operation or derivative change, run `uv run pytest` and the
  thorough conformance command in the [testing guide](../../../../../docs/development/testing.md).
