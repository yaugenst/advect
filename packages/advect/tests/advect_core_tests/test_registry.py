"""Tests for the operation registry."""

from __future__ import annotations

from typing import Any, cast

import pytest

from advect.core._array_api import _FUNCTION_SPECS
from advect.core._registry import OpRegistry, get_registry
from advect.core._registry_types import OpDef


class TestOpDef:
    """Tests for OpDef dataclass."""

    def test_create_minimal(self):
        """Test creating an OpDef with minimal parameters."""
        op = OpDef(name="test.op")
        assert op.name == "test.op"
        assert op.num_outputs == 1
        assert op.non_differentiable_reason is None
        assert op.schema_version == 1
        assert op.implementation is None

    def test_create_full(self):
        """Test creating an OpDef with all parameters."""
        op = OpDef(
            name="flex.flow360.solve",
            num_outputs=2,
        )
        assert op.name == "flex.flow360.solve"
        assert op.num_outputs == 2

    def test_frozen(self):
        """Test that OpDef is immutable."""
        op = OpDef(name="test.op")
        with pytest.raises(AttributeError):
            cast("Any", op).name = "other.op"


def _abort_registry_transaction(registry: OpRegistry) -> None:
    message = "abort transaction"
    with registry.transaction():
        registry.update_num_outputs("test.transaction.original", num_outputs=2)
        registry.register(OpDef(name="test.transaction.transient"))
        raise RuntimeError(message)


class TestOpRegistry:
    """Tests for OpRegistry class."""

    def test_custom_operations_must_be_complete(self):
        """Custom records cannot exist without their implementation contract."""
        registry = OpRegistry()

        with pytest.raises(ValueError, match="must have one implementation and signature"):
            registry.register(OpDef(name="custom.incomplete"))

    def test_register_and_get(self):
        """Test registering and retrieving an operation."""
        registry = OpRegistry()
        op = OpDef(name="test.op")
        registry.register(op)

        retrieved = registry.get("test.op")
        assert retrieved is op

    def test_revision_increments_on_register(self):
        """Register increments the registry revision."""
        registry = OpRegistry()
        revision_before = registry.get_revision()
        registry.register(OpDef(name="test.rev.register"))
        assert registry.get_revision() == revision_before + 1

    def test_transaction_commits_successful_mutations(self):
        """A successful registry transaction keeps its mutations."""
        registry = OpRegistry()
        revision_before = registry.get_revision()

        with registry.transaction():
            registry.register(OpDef(name="test.transaction.commit"))

        assert registry.has("test.transaction.commit")
        assert registry.get_revision() == revision_before + 1

    def test_transaction_rolls_back_all_registry_state(self):
        """A failed transaction restores operations and revision exactly."""
        registry = OpRegistry()
        original = OpDef(name="test.transaction.original")
        registry.register(original)
        revision_before = registry.get_revision()

        with pytest.raises(RuntimeError, match="abort transaction"):
            _abort_registry_transaction(registry)

        assert registry.get("test.transaction.original") is original
        assert not registry.has("test.transaction.transient")
        assert registry.get_revision() == revision_before

    def test_register_duplicate_raises(self):
        """Test that registering a duplicate op raises ValueError."""
        registry = OpRegistry()
        op1 = OpDef(name="test.op")
        op2 = OpDef(name="test.op", num_outputs=2)

        registry.register(op1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(op2)

    def test_get_missing_raises(self):
        """Test that getting a missing op raises KeyError."""
        registry = OpRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent.op")

    def test_get_optional_returns_registered_or_none(self):
        """Optional lookup avoids a separate has/get registry round trip."""
        registry = OpRegistry()
        op = OpDef(name="test.op")
        registry.register(op)

        assert registry.get_optional("test.op") is op
        assert registry.get_optional("other.op") is None

    def test_has(self):
        """Test the has() method."""
        registry = OpRegistry()
        registry.register(OpDef(name="test.op"))

        assert registry.has("test.op") is True
        assert registry.has("other.op") is False

    def test_register_op_with_vjp_keeps_one_derivative_record(self):
        """A VJP and its residual requirement live on the operation record."""

        def vjp(_ans, *, g, **_attrs):
            return (g,)

        registry = OpRegistry()
        registry.register(
            OpDef(
                name="test.vjp",
                vjp=vjp,
                vjp_needs_inputs=False,
                vjp_needs_output=False,
            )
        )
        op_def = registry.get("test.vjp")
        assert op_def.vjp is vjp
        assert op_def.vjp_needs_inputs is False
        assert op_def.vjp_needs_output is False
        assert op_def.non_differentiable_reason is None

    def test_register_jvp_and_get(self):
        """register_jvp stores and exposes JVP rules."""
        registry = OpRegistry()
        registry.register(OpDef(name="test.jvp"))

        def _jvp(_ans, _x, *, tangents, **_attrs):
            return tangents[0]

        registry.register_jvp("test.jvp", _jvp)

        assert registry.has_jvp("test.jvp") is True
        assert registry.get("test.jvp").jvp is _jvp

    def test_register_jvp_noop_does_not_increment_revision(self):
        """Re-registering the exact same JVP payload is a no-op."""
        registry = OpRegistry()
        registry.register(OpDef(name="test.jvp.noop"))

        def _jvp(_ans, _x, *, tangents, **_attrs):
            return tangents[0]

        registry.register_jvp("test.jvp.noop", _jvp)
        revision_after_first = registry.get_revision()
        registry.register_jvp("test.jvp.noop", _jvp)
        assert registry.get_revision() == revision_after_first

    def test_register_jvp_missing_op_raises(self):
        """register_jvp requires pre-registered operation names."""
        registry = OpRegistry()
        with pytest.raises(KeyError, match="Register the op first"):
            registry.register_jvp(
                "test.unknown.jvp",
                lambda _ans, _x, *, tangents, **_attrs: tangents[0],
            )

    def test_register_vjp_overrides_non_differentiable_status(self):
        """register_vjp marks op as HAS_VJP and clears non-diff reason."""
        registry = OpRegistry()
        registry.register(OpDef(name="test.promote", non_differentiable_reason="temporary"))
        registry.register_vjp(
            "test.promote",
            lambda _ans, *, g, **_attrs: (g,),
            needs_inputs=False,
        )

        op_def = registry.get("test.promote")
        assert op_def.vjp is not None
        assert op_def.non_differentiable_reason is None

    def test_update_num_outputs_preserves_grad_metadata(self):
        """update_num_outputs preserves grad contract metadata."""
        registry = OpRegistry()
        registry.register(
            OpDef(name="test.shape", num_outputs=1, non_differentiable_reason="contract")
        )

        registry.update_num_outputs("test.shape", num_outputs=3)
        op_def = registry.get("test.shape")

        assert op_def.num_outputs == 3
        assert op_def.non_differentiable_reason == "contract"

    def test_update_num_outputs_noop_does_not_increment_revision(self):
        """update_num_outputs no-op keeps revision stable."""
        registry = OpRegistry()
        registry.register(OpDef(name="test.shape.noop", num_outputs=2))
        revision_before = registry.get_revision()
        registry.update_num_outputs("test.shape.noop", num_outputs=2)
        assert registry.get_revision() == revision_before


class TestGetRegistry:
    """Tests for the get_registry singleton function."""

    def test_returns_same_instance(self):
        """Test that get_registry returns the same instance."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_has_builtin_ops(self):
        """Test that the registry is initialized with built-in ops."""
        registry = get_registry()

        # Advect internal ops
        assert registry.has("advect.input")
        assert registry.has("advect.const")
        assert registry.has("advect.getitem")
        assert registry.has("advect.index_update")
        assert not registry.has("advect.setitem")

        # The core frontend has a declared catalog; tracing never synthesizes
        # empty registry entries from the operations it happens to observe.
        for op_name in {spec.op for spec in _FUNCTION_SPECS.values()}:
            assert registry.has(op_name), f"Missing Array API op: {op_name}"
            assert registry.get(op_name).abstract_schema is not None

        assert registry.get("array_ext.linalg.eigh").num_outputs == 2
        assert registry.get("array_ext.linalg.svd").num_outputs == 3
