# SciPy Frontend

## Owns

- The bounded optional SciPy namespace: public parity wrappers, direct
  `custom.scipy.*` primitives behind those wrappers, traceable composites,
  ndimage implementation under `_ndimage/`, and concrete solver callbacks.
- Explicit NumPy-backed value, dtype, signature, derivative, and lifetime
  contracts for every claimed public form.
- Public `__all__` exports from which the SciPy support catalog is discovered.

## Must not own

- Generic NumPy protocol behavior, provider-neutral array semantics, or a
  second solver abstraction outside `implicit_root`.
- Built-in array-family records for direct SciPy forms; bundled SciPy
  primitives use the ordinary custom-primitive mechanism.
- Staged or serialized claims for opaque `root_solver` or `gmres_solver`
  iterations.
- A hand-edited SciPy compatibility inventory.

## Read

- [SciPy operation workflows](../../../../../docs/development/adding-operations.md#scipy-forms)
- [Scientific extension decision](../../../../../design/decisions/2026-07-26-scientific-extension-contracts.md)
- [SciPy public reference](../../../../../docs/api/scipy/index.md)
- [Test ownership](../../../../../docs/development/testing.md)

## Verify

- `uv run pytest packages/advect/tests/advect_scipy_tests`
- Add and run an in-tree `InvocationCase` for a direct bundled `custom.scipy.*`
  primitive; run the full Python suite for a traced-operation or derivative
  change.
- Prove every support-catalog lifetime claimed by the changed public form.
- For an exported-surface change, regenerate compatibility pages and run
  `advect_core_tests/test_support_catalog.py`, as described in the
  [documentation guide](../../../../../docs/development/documentation.md).
