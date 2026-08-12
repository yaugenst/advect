# Errors

When Advect cannot differentiate a call safely, it raises an error at the offending line instead of returning a plausible but wrong derivative. All exceptions derive from [`AdvectError`](https://yaugenst.github.io/advect/0.1.1/api/errors/#advect.AdvectError). The [troubleshooting guide](https://yaugenst.github.io/advect/0.1.1/tutorials/debugging/index.md) shows how to use [`debug`](https://yaugenst.github.io/advect/0.1.1/api/errors/#advect.debug) and act on the common tracing, numerical, and staging failures.

## Diagnostic scope

## debug

```python
debug(*, numerics: bool = False) -> Generator[None]
```

Enable scoped trace diagnostics.

Debug mode records per-operation user locations and gives live tracers a bounded concrete-value summary. `numerics=True` additionally raises at the first non-finite primal, JVP, or VJP value found by a dynamic transform. State is thread-local and restored exactly when the scope exits.

## AdvectError

Bases: `Exception`

Base class for all Advect-specific errors.

All custom Advect exceptions inherit from this class, enabling users to catch all Advect-related errors with a single except clause.

## TracingError

Bases: `AdvectError`

Error raised during tracing for unsupported operations.

This exception is raised when:

- Operations are performed on TracedArray outside a trace context
- An unsupported ufunc or ufunc method is called
- TracedArrays from different trace recorders are mixed in an operation
- A TracedArray is converted to ndarray via `np.asarray()`
- A TracedArray from a closed trace is used in a different trace

`TracingError` is for semantic errors while recording a dynamic or staged transform.

## EscapedTracerError

Bases: `TracingError`

A tracer was read, converted, or mutated after its trace closed.

## MutationError

Bases: `AdvectError`

Source mutation could not be represented as an unambiguous SSA update.

## StaleViewError

Bases: `MutationError`

A view was used after its root tracer advanced to a new SSA value.

## NumericsError

```python
NumericsError(
    *,
    phase: str,
    op: str,
    summary: str,
    source_location: str | None = None,
)
```

Bases: `AdvectError`

A dynamic trace first produced a NaN or infinity in debug mode.

## NoJVPError

```python
NoJVPError(
    message: str,
    *,
    op: str | None = None,
    source_location: str | None = None,
)
```

Bases: `AdvectError`

Error raised when forward-mode autodiff encounters an op without a JVP rule.

Parameters:

- **`message`** (`str`) – Human-readable error message.
- **`op`** (`str | None`, default: `None` ) – Name of the operation missing a JVP rule.
- **`source_location`** (`str | None`, default: `None` ) – Source location where the operation was traced (if available).

## NoVJPError

```python
NoVJPError(
    message: str,
    *,
    op: str | None = None,
    source_location: str | None = None,
    non_differentiable: bool = False,
    grad_reason: str | None = None,
)
```

Bases: `AdvectError`

Error raised when reverse mode cannot transpose an operation.

For a custom primitive, the error points to the public `@primitive` authoring surface. Built-in derivative rules remain an Advect implementation detail and do not expose the internal registry as a user extension API.

Parameters:

- **`message`** (`str`) – Human-readable error message.
- **`op`** (`str | None`, default: `None` ) – Name of the operation missing a VJP rule.
- **`source_location`** (`str | None`, default: `None` ) – Source location where the operation was traced (if available).
- **`non_differentiable`** (`bool`, default: `False` ) – Whether the operation is explicitly marked as non-differentiable.
- **`grad_reason`** (`str | None`, default: `None` ) – Human-readable explanation for non-differentiable classification.

Examples:

```pycon
>>> error = NoVJPError(
...     "No VJP rule for operation",
...     op="custom.my_op",
...     source_location="model.py:42 in forward()",
... )
>>> error.op
'custom.my_op'
>>> "model.py:42" in str(error)
True
```

## MissingPrimitiveRuleError

Bases: `TracingError`

Raised when a primitive lacks a rule required by a transform.

## ImplicitSolveError

Bases: `AdvectError`

A nonlinear or linear solve did not produce its promised solution.
