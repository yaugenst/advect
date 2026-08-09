# Autodiff

## Owns

- Differentiation APIs and input selection across dynamic calls and staged
  programs, including staged `grad`, `value_and_grad`, and `vjp_program`.
- Concrete trace lifetimes, pullbacks, higher-order transforms, checkpointing,
  and implicit roots.
- Canonical derivative rules, with array-family JVPs and the necessary explicit
  real-adjoint rules separated under `rules/array_family/{jvp,vjp}/`.

## Must not own

- NumPy or SciPy signatures and protocol dispatch, Array API revision policy,
  or provider-specific support declarations.
- Durable graph serialization or staged runtime scheduling.
- Numerical pullbacks used as silent substitutes for missing analytic rules.

## Read

- [Adding operations](../../../../../docs/development/adding-operations.md)
- [Testing](../../../../../docs/development/testing.md)
- [Primitive conformance decision](../../../../../design/decisions/2026-07-27-primitive-conformance-testing.md)
- The relevant transform semantics in [target API](../../../../../design/target-api.md)

## Verify

- `uv run pytest packages/advect/tests/advect_grad_tests`
- `uv run pytest packages/advect/tests/advect_conformance_tests`
- For a derivative or traced-operation change, run `uv run pytest` and
  `uv run pytest packages/advect/tests/advect_conformance_tests --hypothesis-profile=thorough`.
- Run format, lint, and strict types from the [testing guide](../../../../../docs/development/testing.md).
