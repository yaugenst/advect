"""Tests for the operation registry."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import hypothesis.strategies as st
import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from advect.core._array_api.frontend import _FUNCTION_SPECS
from advect.core._registry import OpRegistry, get_registry
from advect.core._registry_types import OpDef

if TYPE_CHECKING:
    from collections.abc import Callable


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

    def test_frozen(self):
        """Test that OpDef is immutable."""
        op = OpDef(name="test.op")
        with pytest.raises(AttributeError):
            cast("Any", op).name = "other.op"


def _abort_registry_transaction(registry: OpRegistry, pause: Callable[[], object]) -> None:
    message = "abort transaction"
    with registry.transaction():
        registry.update("test.transaction.original", num_outputs=2)
        registry.register(OpDef(name="test.transaction.transient"))
        pause()
        raise RuntimeError(message)


class TestOpRegistry:
    """Tests for OpRegistry class."""

    def test_custom_operations_must_be_complete(self):
        """Custom records cannot exist without their implementation contract."""
        registry = OpRegistry()

        with pytest.raises(ValueError, match="must have one implementation and signature"):
            registry.register(OpDef(name="custom.incomplete"))

    def test_failed_transaction_keeps_concurrent_registrations(self):
        """Another thread's registration survives a transaction rollback."""
        registry = OpRegistry()
        registry.register(OpDef(name="test.transaction.original"))
        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def pause() -> None:
            entered.set()
            release.wait(timeout=10)

        def failing_transaction() -> None:
            try:
                _abort_registry_transaction(registry, pause)
            except RuntimeError as error:
                errors.append(error)

        transaction = threading.Thread(target=failing_transaction)
        transaction.start()
        assert entered.wait(timeout=10)
        concurrent = threading.Thread(
            target=registry.register, args=(OpDef(name="test.transaction.concurrent"),)
        )
        concurrent.start()
        try:
            concurrent.join(timeout=0.2)
            # The transaction holds the registry lock, so the writer must wait.
            assert concurrent.is_alive()
            assert not registry.has("test.transaction.concurrent")
        finally:
            release.set()
            transaction.join(timeout=10)
            concurrent.join(timeout=10)

        assert len(errors) == 1
        assert registry.has("test.transaction.concurrent")
        assert not registry.has("test.transaction.transient")

    @pytest.mark.parametrize(
        ("num_outputs", "error", "message"),
        [
            (0, ValueError, "num_outputs must be >= 1"),
            (-3, ValueError, "num_outputs must be >= 1"),
            (True, TypeError, "non-integer output arity"),
        ],
    )
    def test_output_arity_must_be_a_positive_integer(self, num_outputs, error, message):
        """Register and update reject an output arity no operation can have."""
        registry = OpRegistry()
        with pytest.raises(error, match=message):
            registry.register(OpDef(name="test.arity", num_outputs=num_outputs))
        registry.register(OpDef(name="test.arity"))
        revision = registry.get_revision()
        with pytest.raises(error, match=message):
            registry.update("test.arity", num_outputs=num_outputs)
        assert registry.get("test.arity").num_outputs == 1
        assert registry.get_revision() == revision


_MACHINE_NAMES = ("test.machine.a", "test.machine.b", "test.machine.c", "test.machine.d")


def _rule_a(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
    return ()


def _rule_b(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
    return ()


def _rule_c(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
    return ()


_NAME = st.sampled_from(_MACHINE_NAMES)
_RULE = st.sampled_from((_rule_a, _rule_b, _rule_c))
_OPTIONAL_RULE = st.one_of(st.none(), _RULE)
_FIELDS = st.fixed_dictionaries(
    {},
    optional={
        "num_outputs": st.sampled_from((-1, 0, 1, 2, 3, True)),
        "vjp": _OPTIONAL_RULE,
        "vjp_needs_inputs": st.booleans(),
        "vjp_needs_output": st.booleans(),
        "jvp": _OPTIONAL_RULE,
        "non_differentiable_reason": st.sampled_from((None, "", "discrete")),
    },
)


def _expected_error(op_def: OpDef) -> type[Exception] | None:
    """Model the record validation for the drawn fields, in checking order."""
    reason = op_def.non_differentiable_reason
    if reason == "" or (op_def.vjp is not None and reason is not None):
        return ValueError
    if type(op_def.num_outputs) is not int:
        return TypeError
    if op_def.num_outputs < 1:
        return ValueError
    return None


class _AbortTransactionError(Exception):
    pass


class RegistryMachine(RuleBasedStateMachine):
    """Every registry transition against a dictionary model and a revision count.

    An effective mutation advances the revision by exactly one; a no-op or a
    rejected write changes neither state nor revision, and a failed
    transaction restores the model under a revision past every one it saw.
    """

    def __init__(self) -> None:
        super().__init__()
        self.registry = OpRegistry()
        self.model: dict[str, OpDef] = {}
        self.revision = self.registry.get_revision()

    def _register(self, name: str, fields: dict[str, Any]) -> None:
        op_def = OpDef(name=name, **fields)
        if name in self.model:
            with pytest.raises(ValueError, match="already registered"):
                self.registry.register(op_def)
            return
        error = _expected_error(op_def)
        if error is not None:
            with pytest.raises(error):
                self.registry.register(op_def)
            return
        self.registry.register(op_def)
        assert self.registry.get(name) is op_def
        self.model[name] = op_def
        self.revision += 1

    def _update(self, name: str, changes: dict[str, Any], write: Callable[[], object]) -> None:
        old = self.model.get(name)
        if old is None:
            with pytest.raises(KeyError, match="not found in registry"):
                write()
            return
        new = replace(old, **changes)
        error = _expected_error(new)
        if error is not None:
            with pytest.raises(error):
                write()
            return
        write()
        stored = self.registry.get(name)
        if new == old:
            assert stored is old
            return
        assert stored == new
        self.model[name] = stored
        self.revision += 1

    @rule(name=_NAME, fields=_FIELDS)
    def register(self, *, name: str, fields: dict[str, Any]) -> None:
        """Register a drawn record: new and valid, invalid, or a duplicate."""
        self._register(name, fields)

    @rule(name=_NAME, changes=_FIELDS)
    def update(self, *, name: str, changes: dict[str, Any]) -> None:
        """Replace drawn fields: effective, no-op, invalid, or unknown op."""
        self._update(name, changes, lambda: self.registry.update(name, **changes))

    @rule(name=_NAME, jvp=_OPTIONAL_RULE)
    def register_jvp(self, *, name: str, jvp: Callable[..., Any] | None) -> None:
        """Install a JVP rule, which is an update of one field."""
        self._update(name, {"jvp": jvp}, lambda: self.registry.register_jvp(name, cast("Any", jvp)))

    @rule(name=_NAME, vjp=_RULE, needs_inputs=st.booleans(), needs_output=st.booleans())
    def register_vjp(
        self,
        *,
        name: str,
        vjp: Callable[..., tuple[Any, ...]],
        needs_inputs: bool,
        needs_output: bool,
    ) -> None:
        """Install a VJP rule, which also clears any non-differentiable reason."""
        changes = {
            "vjp": vjp,
            "vjp_needs_inputs": needs_inputs,
            "vjp_needs_output": needs_output,
            "non_differentiable_reason": None,
        }
        self._update(
            name,
            changes,
            lambda: self.registry.register_vjp(
                name, vjp, needs_inputs=needs_inputs, needs_output=needs_output
            ),
        )

    @rule(name=_NAME, new_name=_NAME)
    def rename(self, *, name: str, new_name: str) -> None:
        """Reject an update that would change an operation's identity."""
        with pytest.raises(TypeError, match="multiple values for argument 'name'"):
            self.registry.update(name, name=new_name)

    @rule(
        steps=st.lists(st.tuples(st.booleans(), _NAME, _FIELDS), max_size=3),
        commit=st.booleans(),
    )
    def transaction(self, *, steps: list[tuple[bool, str, dict[str, Any]]], commit: bool) -> None:
        """Apply writes in a transaction that either commits or rolls back."""
        model_before, revision_before = dict(self.model), self.revision
        with contextlib.suppress(_AbortTransactionError), self.registry.transaction():
            for is_register, name, fields in steps:
                if is_register:
                    self._register(name, fields)
                else:
                    self._update(name, fields, lambda: self.registry.update(name, **fields))  # noqa: B023
            if not commit:
                raise _AbortTransactionError
        if not commit:
            self.model = model_before
            if self.revision != revision_before:
                self.revision += 1

    @invariant()
    def registry_matches_model(self) -> None:
        """Every read agrees with the model after each step."""
        assert self.registry.get_revision() == self.revision
        assert self.registry.definitions() == tuple(self.model[name] for name in sorted(self.model))
        for name in _MACHINE_NAMES:
            expected = self.model.get(name)
            assert self.registry.get_optional(name) is expected
            assert self.registry.has(name) is (expected is not None)
            assert self.registry.has_jvp(name) is (
                expected is not None and expected.jvp is not None
            )
            assert self.registry.has_vjp(name) is (
                expected is not None and expected.vjp is not None
            )
            if expected is None:
                with pytest.raises(KeyError, match="not found in registry"):
                    self.registry.get(name)


TestRegistryMachine = RegistryMachine.TestCase
TestRegistryMachine.settings = settings(stateful_step_count=20, deadline=None)


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
