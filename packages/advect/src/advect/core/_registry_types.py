"""Canonical operation records stored by Advect's process registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import inspect
    from collections.abc import Callable

    from advect.core._abstract_model import AbstractRule, ResultEvaluator


@dataclass(frozen=True, slots=True)
class OpDef:
    """All process-local semantics attached to one stable operation ID.

    Built-ins and user-authored primitives share this record. The callable
    returned by :func:`advect.primitive` is only a handle for the corresponding
    ``custom.*`` record; it does not retain a second copy of these fields.
    """

    name: str
    num_outputs: int = 1
    output_arity_known: bool = True
    vjp: Callable[..., tuple[Any, ...]] | None = None
    vjp_needs_inputs: bool = True
    vjp_needs_output: bool = True
    jvp: Callable[..., Any] | None = None
    non_differentiable_reason: str | None = None

    # Built-in array semantics. The schema classifies operands and attributes;
    # the evaluator computes payload-free result specifications.
    abstract_schema: AbstractRule | None = field(default=None, repr=False)
    abstract_evaluator: ResultEvaluator | None = field(default=None, repr=False)

    # Graph schema revisions are owned by Advect, including for custom ops.
    schema_version: int = 1
    static_argnames: tuple[str, ...] = ()
    nondiff_argnames: tuple[str, ...] = ()
    has_residual: bool = False
    variable_output_arity: bool = False
    implementation: Callable[..., Any] | None = field(default=None, repr=False)
    signature: inspect.Signature | None = field(default=None, repr=False)
    abstract_rule: Callable[..., Any] | None = field(default=None, repr=False)
