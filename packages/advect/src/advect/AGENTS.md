# Python Package

## Owns

- Public package assembly, explicit `array`/`asarray` construction, pytree
  utilities, support reporting, and author-facing numerical checks.
- Package-wide import behavior and the optional `interop` and `xarray`
  boundaries when no more specific guidance applies.
- Stable public exports; private implementation remains free to move.

## Must not own

- A second copy of canonical operation semantics, frontend support metadata, or
  the durable graph model.
- Eager imports of optional SciPy, xarray, JAX, PyTorch, or HIPS Autograd
  dependencies from the base `advect` import.
- Compatibility aliases, deprecated dual paths, or public exports without an
  owning API-reference page.

## Read

- [Codebase map](../../../../docs/development/codebase.md)
- [Adding operations](../../../../docs/development/adding-operations.md)
- [Public API reference](../../../../docs/api/index.md)
- The nearest child `AGENTS.md` for core, autodiff, NumPy, or SciPy work

## Verify

- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run pyrefly check --config pyproject.toml --preset strict packages/advect/src`
- Run the owning suite from the [testing guide](../../../../docs/development/testing.md);
  run `uv run pytest` when package assembly or multiple integrations change.
