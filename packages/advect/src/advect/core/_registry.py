"""Process-local operation and derivative-rule registry."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from advect.core._registry_types import OpDef


def _validate_opdef(op_def: OpDef) -> OpDef:
    """Validate invariants of one canonical operation record."""
    reason = op_def.non_differentiable_reason
    if reason is not None and not reason:
        msg = f"Op '{op_def.name}' has an empty non-differentiable reason"
        raise ValueError(msg)
    if op_def.vjp is not None and reason is not None:
        msg = f"Op '{op_def.name}' has a VJP rule but is marked non-differentiable"
        raise ValueError(msg)
    if type(op_def.vjp_needs_inputs) is not bool or type(op_def.vjp_needs_output) is not bool:
        msg = f"Op '{op_def.name}' has non-boolean VJP retention metadata"
        raise TypeError(msg)
    if op_def.schema_version < 1:
        msg = f"Op '{op_def.name}' has invalid schema version {op_def.schema_version}"
        raise ValueError(msg)
    if type(op_def.has_residual) is not bool:
        msg = f"Op '{op_def.name}' has a non-boolean residual capability"
        raise TypeError(msg)
    if op_def.name.startswith("custom.") and (
        op_def.implementation is None or op_def.signature is None
    ):
        msg = f"Custom op '{op_def.name}' must have one implementation and signature"
        raise ValueError(msg)
    if op_def.has_residual and not op_def.name.startswith("custom."):
        msg = f"Built-in op '{op_def.name}' cannot declare an opaque residual"
        raise ValueError(msg)
    return op_def


class OpRegistry:
    """Registry of operation definitions.

    The singleton starts with Advect's built-ins. Frontends and custom
    primitives register additional definitions programmatically.
    """

    __slots__ = ("_ops", "_revision")

    def __init__(self) -> None:
        self._ops: dict[str, OpDef] = {}
        self._revision: int = 0

    def _bump_revision(self) -> None:
        self._revision += 1

    def get_revision(self) -> int:
        """Return a revision that changes on committed registry mutations."""
        return self._revision

    def definitions(self) -> tuple[OpDef, ...]:
        """Return an immutable, name-sorted snapshot of registered operations."""
        return tuple(self._ops[name] for name in sorted(self._ops))

    @contextmanager
    def transaction(self) -> Generator[None]:
        """Roll back all registry state when the enclosed operation fails.

        Transactions are intended for atomic consumers such as graph
        deserialization, where validating one node may register runtime metadata
        before a later node or graph-level invariant fails.
        """
        ops_before = self._ops.copy()
        revision_before = self._revision
        committed = False
        try:
            yield
            committed = True
        finally:
            if not committed:
                self._ops = ops_before
                self._revision = revision_before

    def register(self, op_def: OpDef) -> None:
        """Register an operation definition.

        Parameters
        ----------
        op_def
            The operation definition to register.

        Raises
        ------
        ValueError
            If an operation with the same name is already registered.
        """
        if op_def.name in self._ops:
            msg = f"Op '{op_def.name}' is already registered"
            raise ValueError(msg)
        self._ops[op_def.name] = _validate_opdef(op_def)
        self._bump_revision()

    def update(self, name: str, **changes: Any) -> OpDef:  # noqa: ANN401
        """Atomically replace fields on one canonical operation record."""
        if name not in self._ops:
            msg = f"Op '{name}' not found in registry"
            raise KeyError(msg)
        if "name" in changes:
            msg = "Operation identity cannot be changed"
            raise ValueError(msg)
        old_def = self._ops[name]
        new_def = _validate_opdef(replace(old_def, **changes))
        if new_def == old_def:
            return old_def
        self._ops[name] = new_def
        self._bump_revision()
        return new_def

    def get(self, name: str) -> OpDef:
        """Get an operation definition by name.

        Parameters
        ----------
        name
            The namespaced operation name.

        Returns
        -------
        OpDef
            The operation definition.

        Raises
        ------
        KeyError
            If the operation is not found in the registry.
        """
        if name not in self._ops:
            msg = f"Op '{name}' not found in registry"
            raise KeyError(msg)
        return self._ops[name]

    def _get_canonical(self, name: str) -> OpDef:
        """Return an op whose caller already holds a canonical runtime ID."""
        return self._ops[name]

    def get_optional(self, name: str) -> OpDef | None:
        """Return an operation definition when registered."""
        return self._ops.get(name)

    def has(self, name: str) -> bool:
        """Check if an operation is registered.

        Parameters
        ----------
        name
            The namespaced operation name.

        Returns
        -------
        bool
            True if the operation is registered.
        """
        return name in self._ops

    def update_num_outputs(self, name: str, *, num_outputs: int) -> None:
        """Update the output arity for an existing operation.

        This replaces the stored :class:`~advect.OpDef` while preserving all other
        fields (docstring, VJP rule, requirements, etc.).

        Parameters
        ----------
        name
            The namespaced operation name.
        num_outputs
            New output arity. Must be >= 1.

        Raises
        ------
        KeyError
            If the operation is not found.
        ValueError
            If num_outputs is invalid.
        """
        if num_outputs < 1:
            msg = f"Op '{name}': num_outputs must be >= 1 (got {num_outputs})"
            raise ValueError(msg)
        if name not in self._ops:
            msg = f"Op '{name}' not found in registry"
            raise KeyError(msg)

        self.update(name, num_outputs=num_outputs)

    def has_vjp(self, name: str) -> bool:
        """Check if an operation has a registered VJP rule.

        Parameters
        ----------
        name
            The namespaced operation name.

        Returns
        -------
        bool
            True if the operation exists and has a VJP rule.
        """
        if name not in self._ops:
            return False
        return self._ops[name].vjp is not None

    def has_jvp(self, name: str) -> bool:
        """Check if an operation has a registered JVP rule.

        Parameters
        ----------
        name
            The namespaced operation name.

        Returns
        -------
        bool
            True if the operation exists and has a JVP rule.
        """
        if name not in self._ops:
            return False
        return self._ops[name].jvp is not None

    def register_vjp(
        self,
        name: str,
        vjp: Callable[..., tuple[Any, ...]],
        *,
        needs_inputs: bool = True,
        needs_output: bool = True,
    ) -> None:
        """Register or update the VJP rule for an operation.

        This method updates an existing OpDef with VJP information. The operation
        must already be registered.

        Parameters
        ----------
        name
            The namespaced operation name.
        vjp
            The VJP rule function.
        needs_inputs
            Whether reverse execution must retain and pass the primal inputs.
        needs_output
            Whether reverse execution must retain and pass the primal output.

        Raises
        ------
        KeyError
            If the operation is not found.
        """
        if name not in self._ops:
            msg = f"Op '{name}' not found in registry. Register the op first."
            raise KeyError(msg)

        old_def = self._ops[name]
        if (
            old_def.vjp is vjp
            and old_def.vjp_needs_inputs is needs_inputs
            and old_def.vjp_needs_output is needs_output
            and old_def.non_differentiable_reason is None
        ):
            return
        self.update(
            name,
            vjp=vjp,
            vjp_needs_inputs=needs_inputs,
            vjp_needs_output=needs_output,
            non_differentiable_reason=None,
        )

    def register_jvp(
        self,
        name: str,
        jvp: Callable[..., Any],
    ) -> None:
        """Register or update the JVP rule for an operation.

        Parameters
        ----------
        name
            The namespaced operation name.
        jvp
            The JVP rule function.

        Raises
        ------
        KeyError
            If the operation is not found.
        """
        if name not in self._ops:
            msg = f"Op '{name}' not found in registry. Register the op first."
            raise KeyError(msg)

        self.update(name, jvp=jvp)


@lru_cache(maxsize=1)
def get_registry() -> OpRegistry:
    """Get the global operation registry, initializing if needed.

    The registry is lazily initialized with complete built-in Advect operation
    definitions on first access. User primitives may add definitions later.

    Returns
    -------
    OpRegistry
        The global operation registry singleton.

    """
    registry = OpRegistry()
    _register_builtin_ops(registry)
    return registry


def _register_builtin_ops(registry: OpRegistry) -> None:
    """Register the complete semantics shipped with Advect."""
    from advect._builtin_ops import builtin_operation_definitions  # noqa: PLC0415

    for definition in builtin_operation_definitions():
        registry.register(definition)
