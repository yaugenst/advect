## Owns

- Qualification, support reporting, evidence assembly, benchmarks, release
  checks, smoke tests, documentation hooks, and documentation snippet tooling.
- Thin command entry points over reusable private script support.

## Must not own

- Product runtime behavior, canonical operation semantics, or a second support
  catalog.
- Hand-maintained facts that can be derived from package declarations or test
  results.
- New standalone Python automation without inline script metadata and a
  documented `uv run --script` invocation.

## Read

- [Testing and specialized evidence](../docs/development/testing.md)
- [Documentation tooling](../docs/development/documentation.md)
- [Performance contract](../design/implementation/performance.md) for benchmark work
- [Array provider qualification](../design/implementation/array-provider-qualification.md)
  for provider tooling

## Verify

- `uv run ruff format --check .`
- `uv run ruff check .`
- Run the command's `--help`, focused test, or smoke path with `uv run`.
- For generated docs, run the generator followed by `uv run mkdocs build --strict`;
  for benchmark changes, run the owning benchmark-contract tests.
