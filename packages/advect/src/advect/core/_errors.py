"""Error classes for Advect.

This module defines exception classes used throughout Advect to provide
clear error messages for unsupported operations during tracing.
"""

from __future__ import annotations


def _array_conversion_error() -> str:
    return (
        "NumPy attempted to convert a live Advect value into an ndarray. "
        "When constructing an array, pass a traced dispatch anchor, for example "
        "np.array(values, like=x). This preserves differentiation and selects x's "
        "array provider. If the conversion happened inside another library, that "
        "operation is not trace-compatible; wrap it with @advect.primitive or move it "
        "outside the transformed function."
    )


def _debug_retry_hint() -> str:
    return (
        "\n  Debug: rerun the transform call inside `with advect.debug():` "
        "to capture the user operation."
    )


class AdvectError(Exception):
    """Base class for all Advect-specific errors.

    All custom Advect exceptions inherit from this class, enabling users
    to catch all Advect-related errors with a single except clause.
    """


class TracingError(AdvectError):
    """Error raised during tracing for unsupported operations.

    This exception is raised when:
    - Operations are performed on TracedArray outside a trace context
    - An unsupported ufunc or ufunc method is called
    - TracedArrays from different trace recorders are mixed in an operation
    - A TracedArray is converted to ndarray via ``np.asarray()``
    - A TracedArray from a closed trace is used in a different trace

    ``TracingError`` is for semantic errors while recording a dynamic or
    staged transform.
    """


class EscapedTracerError(TracingError):
    """A tracer was read, converted, or mutated after its trace closed."""


class MutationError(AdvectError):
    """Source mutation could not be represented as an unambiguous SSA update."""


class StaleViewError(MutationError):
    """A view was used after its root tracer advanced to a new SSA value."""


class NumericsError(AdvectError):
    """A dynamic trace first produced a NaN or infinity in debug mode."""

    def __init__(
        self,
        *,
        phase: str,
        op: str,
        summary: str,
        source_location: str | None = None,
    ) -> None:
        self.phase = phase
        self.op = op
        self.summary = summary
        self.source_location = source_location

        parts = [f"Non-finite value first detected during {phase}."]
        parts.append(f"\n  Operation: {op}")
        if source_location:
            parts.append(f"\n  Location: {source_location}")
        parts.append(f"\n  Value: {summary}")
        super().__init__("".join(parts))


class HigherOrderNotSupportedError(AdvectError):
    """Error raised when a higher-order autodiff path is unsupported."""

    def __init__(
        self,
        message: str,
        *,
        op: str | None = None,
        source_location: str | None = None,
    ) -> None:
        self.op = op
        self.source_location = source_location

        parts = [message]
        if op:
            parts.append(f"\n  Operation: {op}")
        if source_location:
            parts.append(f"\n  Location: {source_location}")
        elif op:
            parts.append(_debug_retry_hint())
        super().__init__("".join(parts))


class TraceLevelError(AdvectError):
    """Error raised when traced values are used across incompatible trace levels."""

    def __init__(
        self,
        message: str,
        *,
        value_level: int | None = None,
        active_level: int | None = None,
    ) -> None:
        self.value_level = value_level
        self.active_level = active_level

        parts = [message]
        if value_level is not None:
            parts.append(f"\n  Value trace level: {value_level}")
        if active_level is not None:
            parts.append(f"\n  Active trace level: {active_level}")
        super().__init__("".join(parts))


class NoVJPError(AdvectError):
    """Error raised when reverse mode cannot transpose an operation.

    For a custom primitive, the error points to the public ``@primitive``
    authoring surface. Built-in derivative rules remain an Advect implementation
    detail and do not expose the internal registry as a user extension API.

    Parameters
    ----------
    message
        Human-readable error message.
    op
        Name of the operation missing a VJP rule.
    source_location
        Source location where the operation was traced (if available).
    non_differentiable
        Whether the operation is explicitly marked as non-differentiable.
    grad_reason
        Human-readable explanation for non-differentiable classification.

    Examples
    --------
    >>> error = NoVJPError(
    ...     "No VJP rule for operation",
    ...     op="custom.my_op",
    ...     source_location="model.py:42 in forward()",
    ... )
    >>> error.op
    'custom.my_op'
    >>> "model.py:42" in str(error)
    True
    """

    def __init__(
        self,
        message: str,
        *,
        op: str | None = None,
        source_location: str | None = None,
        non_differentiable: bool = False,
        grad_reason: str | None = None,
    ) -> None:
        self.op = op
        self.source_location = source_location
        self.non_differentiable = non_differentiable
        self.grad_reason = grad_reason

        parts = [message]
        if op:
            parts.append(f"\n  Operation: {op}")
        if source_location:
            parts.append(f"\n  Location: {source_location}")
        elif op:
            parts.append(_debug_retry_hint())
        if grad_reason:
            parts.append(f"\n  Reason: {grad_reason}")

        if not non_differentiable and op is not None and op.startswith("custom."):
            parts.append("""

  In the module that defines this primitive, attach a transpose rule to the
  handle returned by @advect.primitive:

    @primitive_handle.def_transpose
    def transpose(cotangent, primals, output, **static_attrs):
        # output is the exact primitive result from this invocation.
        # Return one cotangent per differentiable input.
        return (...,)

  Alternatively, define @primitive.def_jvp and validate structural transposition
  with check_primitive from advect.testing.""")
        super().__init__("".join(parts))


class NoJVPError(AdvectError):
    """Error raised when forward-mode autodiff encounters an op without a JVP rule.

    Parameters
    ----------
    message
        Human-readable error message.
    op
        Name of the operation missing a JVP rule.
    source_location
        Source location where the operation was traced (if available).
    """

    def __init__(
        self,
        message: str,
        *,
        op: str | None = None,
        source_location: str | None = None,
    ) -> None:
        self.op = op
        self.source_location = source_location

        parts = [message]
        if op:
            parts.append(f"\n  Operation: {op}")
        if source_location:
            parts.append(f"\n  Location: {source_location}")
        elif op:
            parts.append(_debug_retry_hint())

        if op is not None and op.startswith("custom."):
            parts.append("""

  In the module that defines this primitive, attach a JVP rule to the handle
  returned by @advect.primitive:

    @primitive_handle.def_jvp
    def jvp(output, primals, tangents, **static_attrs):
        # output is the exact primitive result from this invocation.
        # Tangents align with primals; None denotes a symbolic zero.
        return ...""")

        super().__init__("".join(parts))
