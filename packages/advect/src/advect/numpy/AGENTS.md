# NumPy Frontend

## Owns

- Exact NumPy signatures, ufunc and array-function protocols, traced-array
  behavior, functionalized mutation, evaluation, and NumPy support claims.
- Traceable array-function compositions grouped by responsibility under
  `_array_function/`.
- Conservative dynamic, staged, serialized, and derivative declarations for
  every supported NumPy form.
- Separate concrete handler aggregation from abstract lowering: a dynamic
  handler alone does not establish staged or serialized support.

## Must not own

- Provider-neutral canonical semantics or a NumPy-specific derivative registry.
- Array API revision negotiation, SciPy compatibility, or hand-maintained
  copies of the generated compatibility inventory.
- A support claim without a matching executable `NumpySupportCase`.
- A `derivative_argnums` list copied from public-signature positions; those
  indices refer to the case's independent `inputs` tuple.

## Read

- [NumPy operation workflow](../../../../../docs/development/adding-operations.md#numpy-forms)
- [Test ownership](../../../../../docs/development/testing.md)
- [Upstream compatibility](../../../../../design/implementation/upstream-compatibility.md)
- [NumPy public reference](../../../../../docs/api/numpy.md)

## Verify

- `uv run pytest packages/advect/tests/advect_numpy_tests`
- `uv run pytest packages/advect/tests/advect_numpy_tests/test_support_evidence.py`
- Run the full Python suite for a public support declaration. Also run
  conformance if a canonical operation or derivative changes; require the CI
  NumPy 2.0-2.4 matrix for a range claim.
- Regenerate compatibility pages as described in the
  [documentation guide](../../../../../docs/development/documentation.md).
