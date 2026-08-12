# Testing Utilities

[`check_gradient`](https://yaugenst.github.io/advect/dev/api/testing/#advect.testing.check_gradient) checks one representative composition against directional finite differences and reverse-mode consistency. [`check_primitive`](https://yaugenst.github.io/advect/dev/api/testing/#advect.testing.check_primitive) validates the capabilities requested for one representative custom-primitive call.

They complement application tests; neither utility can prove that the implemented function is the mathematics you intended.

See [Troubleshooting](https://yaugenst.github.io/advect/dev/tutorials/debugging/#check-a-suspicious-gradient) for a composed gradient check and [Custom primitives](https://yaugenst.github.io/advect/dev/tutorials/primitives/#check-the-promised-capabilities) for the extension-author workflow.

## testing

Numerical validation for composed functions and custom primitives.

### check_gradient

```python
check_gradient(
    function: Callable[..., Any],
    primal: Any,
    *,
    tangent: Any | None = None,
    epsilons: Sequence[float] = (
        0.01,
        0.001,
        0.0001,
        1e-05,
    ),
    atol: float = 1e-05,
    rtol: float = 0.0001,
) -> None
```

Check a unary composed function against directional differences.

The check compares Advect's whole-function JVP with a central finite- difference sweep, then checks the reverse gradient against the same directional derivative. It checks consistency with the function that actually ran, not whether that function encodes the intended mathematics.

Parameters:

- **`function`** (`Callable[..., Any]`) – Unary function with a real scalar output. This may be a composition of built-in operations and public custom primitives.
- **`primal`** (`Any`) – Representative input value or pytree at which to check function.
- **`tangent`** (`Any | None`, default: `None` ) – Direction pytree matching primal. When omitted, every numeric leaf receives an all-ones direction.
- **`epsilons`** (`Sequence[float]`, default: `(0.01, 0.001, 0.0001, 1e-05)` ) – Non-empty sequence of finite positive central-difference steps. The JVP comparison passes when at least one step agrees within tolerance.
- **`atol`** (`float`, default: `1e-05` ) – Absolute tolerance for the finite-difference and real-adjoint checks.
- **`rtol`** (`float`, default: `0.0001` ) – Relative tolerance for the finite-difference and real-adjoint checks.

Returns:

- `None` – The check returns only when both comparisons pass.

Raises:

- `ValueError` – If epsilons is empty or contains a non-finite or non-positive step, or if the tangent, input, or scalar-output structure is invalid.
- `TypeError` – If a selected input or tangent leaf is unsupported by the active array provider.
- `NoJVPError` – If an operation on the checked path has no forward-mode rule.
- `NoVJPError` – If an operation on the checked path has no reverse-mode rule.
- `AssertionError` – If the JVP disagrees with every finite-difference step or the reverse gradient violates the JVP's real-adjoint identity. The error names any custom primitives observed on the failing path.

Notes

This is a representative author check, not exhaustive conformance evidence. Run it on a composed public path in addition to testing each custom primitive with `check_primitive`.

### check_primitive

```python
check_primitive(
    primitive: Primitive[Any, Any],
    *,
    primals: tuple[Any, ...],
    static: Mapping[str, Any] | None = None,
    tangents: tuple[Any, ...] | None = None,
    cotangent: Any | None = None,
    check: tuple[str, ...] = (
        "abstract",
        "jvp",
        "transpose",
    ),
    epsilon: float = 0.0001,
    atol: float = 1e-05,
    rtol: float = 0.0001,
) -> None
```

Run selected author checks for one representative primitive invocation.

The default `("abstract", "jvp", "transpose")` is a first-order smoke check. It does not stage the primitive or check input preservation. Authors of a serializable non-residual primitive should normally run `("abstract", "jvp", "transpose", "nested", "stage")` for every materially different shape, dtype, and static-argument form. Add `"complex"` in a separate call whose primals are complex when the primitive supports complex values. Residual primitives are first-order boundaries and therefore omit `"nested"`. A transpose-only primitive may request just `"transpose"`; the check then compares its explicit rule with a central finite difference. The `"jvp"`, `"complex"`, and `"nested"` checks require a JVP.

The stage check executes both the compiled and serialized program, compares output structure, shape, and dtype exactly, and verifies that inputs remain unchanged. Repository-wide support still requires the conformance inventory; this helper intentionally does not import Hypothesis or claim exhaustive coverage from one sample.

Parameters:

- **`primitive`** (`Primitive[Any, Any]`) – Authoring handle returned by advect.primitive.
- **`primals`** (`tuple[Any, ...]`) – One representative value for each non-static implementation argument, in implementation-parameter order. Nested pytrees are preserved.
- **`static`** (`Mapping[str, Any] | None`, default: `None` ) – Values for declared static arguments that do not use their implementation defaults.
- **`tangents`** (`tuple[Any, ...] | None`, default: `None` ) – Optional tangent pytree matching primals. Ones are used by default; leaves of nondifferentiable arguments are ignored.
- **`cotangent`** (`Any | None`, default: `None` ) – Optional pytree matching the primitive output. Ones are used by default.
- **`check`** (`tuple[str, ...]`, default: `('abstract', 'jvp', 'transpose')` ) – Any of "abstract", "jvp", "transpose", "complex", "nested", and "stage". Only "stage" compiles and restores a program and checks that those executions preserve their inputs.
- **`epsilon`** (`float`, default: `0.0001` ) – Central finite-difference step.
- **`atol`** (`float`, default: `1e-05` ) – Absolute numerical tolerance.
- **`rtol`** (`float`, default: `0.0001` ) – Relative numerical tolerance.

Raises:

- `ValueError` – If the requested checks or representative inputs are invalid.
- `MissingPrimitiveRuleError` – If a requested capability has no required author rule.
- `AssertionError` – If an installed rule, staged execution, or numerical identity fails its check.
