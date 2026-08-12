"""Additional contracts at Advect's provider-neutral runtime boundaries."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import advect.core._eval_dispatch as dispatch
import advect.core._primitive_call as primitive_call
from advect.core._basic_index import decode_basic_index, decode_index, encode_basic_index
from advect.core._errors import (
    HigherOrderNotSupportedError,
    NoJVPError,
    NoVJPError,
    TraceLevelError,
    TracingError,
    _array_conversion_error,
)
from advect.core._eval_dispatch import (
    _can_donate_array,
    _decode_attrs_for_vjp,
    bind_native_node_evaluator,
    bind_node_evaluator,
    evaluate_node_value,
)
from advect.core._graph_attrs import _PRIMITIVE_CALL_KEY
from advect.core._portable_constant import (
    _constant_payload,
    portable_constant_from_native,
    snapshot_constant_parts,
    validate_constant,
)
from advect.core._primitive_call import (
    _attach_residual,
    _decode_primitive_call_meta,
    _encode_bool_mask,
    _encode_primitive_call_meta,
    _flatten_input_gradients,
    _flatten_primitive_output,
    _normalize_output_pytree,
    _output_shape_and_dtype,
    _PrimitiveCallMeta,
    _reconstruct_primitive_call,
    _reconstruct_primitive_output,
    _record_primitive_output_count,
    _split_primitive_attrs,
    _trace_call_arguments,
    _wrap_traced_output,
    trace_primitive_call,
)
from advect.core._pytree import tree_flatten
from advect.core._registry import OpRegistry
from advect.core._registry_types import OpDef
from advect.core._residual import _PrimitiveExecution, _ResidualSlot


def _primitive_meta() -> _PrimitiveCallMeta:
    _call_leaves, call_treedef = tree_flatten(((1,), {"tag": "static"}))
    _output_leaves, output_treedef = tree_flatten({"left": 1, "right": 2})
    return _PrimitiveCallMeta(
        call_treedef=call_treedef,
        input_leaf_mask=(True, False),
        static_leaves=("static",),
        output_treedef=output_treedef,
        nondiff_input_mask=(False,),
    )


def _empty_vjp(*_args: object, **_kwargs: object) -> tuple[object, ...]:
    return ()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"type": "int", "value": True}, "integer index"),
        ({"type": "slice", "start": 1.5, "stop": None, "step": None}, "slice bounds"),
        ({"type": "newaxis", "extra": None}, "new-axis"),
        ({"type": "ellipsis", "extra": None}, "ellipsis index"),
        ({"type": "unknown"}, "Unknown serialized index"),
        ({"type": "array", "dtype": 1, "shape": [1], "values": [0]}, "dtype"),
        ({"type": "array", "dtype": "int64", "shape": "1", "values": [0]}, "shape"),
        (
            {"type": "array", "dtype": "int64", "shape": [-1], "values": []},
            "dimensions",
        ),
        (
            {"type": "array", "dtype": "int64", "shape": [1], "values": [0]},
            "not supported at this boundary",
        ),
    ],
)
def test_index_decoder_rejects_malformed_wire_components(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        decode_index(payload)


def test_basic_index_boundary_rejects_invalid_public_and_wire_forms() -> None:
    with pytest.raises(TracingError, match="Basic indexing supports only"):
        encode_basic_index((object(),))
    with pytest.raises(TypeError, match="at most one ellipsis"):
        decode_index([{"type": "ellipsis"}, {"type": "ellipsis"}])
    with pytest.raises(TypeError, match="Invalid serialized index component"):
        decode_index("invalid")
    with pytest.raises(TypeError, match="metadata must be a sequence"):
        decode_basic_index({"type": "int", "value": 1})
    with pytest.raises(TypeError, match="Invalid serialized slice index"):
        decode_index({"type": "slice", "start": None, "stop": None})
    with pytest.raises(TypeError, match="Invalid serialized array index"):
        decode_index({"type": "array"})

    assert decode_index(None) is None
    assert decode_index(3) == 3


def test_constant_payload_validation_rejects_corruption_at_each_contract_layer() -> None:
    constant = snapshot_constant_parts([1, 2], shape=(2,), dtype="int32")
    payload = _constant_payload(constant)

    corruptions = (
        (None, TypeError, "mapping"),
        ({**payload, "extra": None}, ValueError, "invalid fields"),
        ({**payload, "format": "other"}, ValueError, "Unknown staged constant format"),
        ({**payload, "version": 1}, ValueError, "Unsupported staged constant version"),
        ({**payload, "layout": "F"}, ValueError, "Unsupported staged constant layout"),
        ({**payload, "byte_order": "big"}, ValueError, "Unsupported staged constant byte order"),
        ({**payload, "kind": "opaque"}, ValueError, "Unknown staged constant kind"),
        ({**payload, "dtype": 1}, TypeError, "dtype must be a string"),
        ({**payload, "dtype": "int"}, ValueError, "canonical portable name"),
        ({**payload, "shape": [True]}, TypeError, "non-negative integers"),
        ({**payload, "data": 1}, TypeError, "hexadecimal string"),
        ({**payload, "data": "AA"}, ValueError, "lowercase hexadecimal"),
        ({**payload, "data": "0"}, ValueError, "complete bytes"),
        ({**payload, "digest": "0" * 64}, ValueError, "digest does not match"),
    )
    for candidate, error, message in corruptions:
        with pytest.raises(error, match=message):
            validate_constant(candidate)


def test_constant_payload_validation_checks_graph_manifest() -> None:
    constant = snapshot_constant_parts([1, 2], shape=(2,), dtype="int32")
    payload = _constant_payload(constant)

    with pytest.raises(ValueError, match="shape does not match"):
        validate_constant(payload, shape=(1, 2))
    with pytest.raises(ValueError, match="dtype does not match"):
        validate_constant(payload, dtype="float32")
    with pytest.raises(ValueError, match="byte count does not match"):
        validate_constant(payload, byte_count=4)


def test_native_constant_boundary_rejects_invalid_store_parts() -> None:
    constant = snapshot_constant_parts([1, 2], shape=(2,), dtype="int32")

    with pytest.raises(ValueError, match="Unknown staged constant kind"):
        portable_constant_from_native("opaque", "int32", [2], constant.data, constant.digest)
    with pytest.raises(ValueError, match="scalar constant must have rank zero"):
        portable_constant_from_native("scalar", "int32", [2], constant.data, constant.digest)
    with pytest.raises(ValueError, match="require 8 bytes"):
        portable_constant_from_native("array", "int32", [2], b"", constant.digest)
    with pytest.raises(ValueError, match="digest does not match"):
        portable_constant_from_native("array", "int32", [2], constant.data, "0" * 64)


def test_constant_snapshot_rejects_provider_shape_and_element_mismatches() -> None:
    with pytest.raises(ValueError, match="scalar constant must have rank zero"):
        snapshot_constant_parts(1, shape=(1,), dtype="int64")
    with pytest.raises(TypeError, match=r"Could not snapshot.*element"):
        snapshot_constant_parts(object(), shape=(1,), dtype="int32")
    with pytest.raises(TypeError, match="Could not encode"):
        snapshot_constant_parts([10**100], shape=(1,), dtype="int8")


def test_constant_snapshot_accepts_provider_tobytes_without_an_order_keyword() -> None:
    class Array:
        dtype = type("DType", (), {"byteorder": "<"})()

        @staticmethod
        def tobytes() -> bytes:
            return b"\x01\x00\x00\x00"

    assert snapshot_constant_parts(Array(), shape=(1,), dtype="int32").data == b"\x01\x00\x00\x00"


def test_constant_payload_rejects_invalid_scalar_and_boolean_encodings() -> None:
    scalar = snapshot_constant_parts(1, shape=(), dtype="int64")
    with pytest.raises(ValueError, match="scalar constant must have rank zero"):
        validate_constant({**_constant_payload(scalar), "shape": [1]})

    boolean = snapshot_constant_parts([True], shape=(1,), dtype="bool")
    with pytest.raises(ValueError, match="bytes must be zero or one"):
        validate_constant({**_constant_payload(boolean), "data": "02"})


def test_primitive_call_metadata_round_trips_and_reconstructs_public_trees() -> None:
    meta = _primitive_meta()
    encoded = _encode_primitive_call_meta(meta)
    assert isinstance(encoded, dict)

    assert _decode_primitive_call_meta(encoded) == meta
    assert _reconstruct_primitive_call(meta, (3,)) == ((3,), {"tag": "static"})
    assert _reconstruct_primitive_output(meta, (4, 5), label="output") == {
        "left": 4,
        "right": 5,
    }
    assert _flatten_primitive_output(meta, {"left": 4, "right": 5}, label="output") == (
        4,
        5,
    )
    split_meta, attrs = _split_primitive_attrs({_PRIMITIVE_CALL_KEY: meta, "scale": 2})
    assert split_meta is meta
    assert attrs == {"scale": 2}


def test_primitive_call_metadata_rejects_malformed_runtime_and_wire_values() -> None:
    meta = _primitive_meta()
    encoded = _encode_primitive_call_meta(meta)
    assert isinstance(encoded, dict)

    with pytest.raises(TypeError, match="invalid runtime value"):
        _encode_primitive_call_meta(None)
    with pytest.raises(ValueError, match="input mask"):
        _encode_primitive_call_meta(
            _PrimitiveCallMeta(meta.call_treedef, (True,), meta.static_leaves, meta.output_treedef)
        )
    with pytest.raises(ValueError, match="static leaves"):
        _encode_primitive_call_meta(
            _PrimitiveCallMeta(meta.call_treedef, (True, False), (), meta.output_treedef)
        )
    with pytest.raises(TypeError, match="nondifferentiable mask"):
        _encode_primitive_call_meta(
            _PrimitiveCallMeta(
                meta.call_treedef,
                meta.input_leaf_mask,
                meta.static_leaves,
                meta.output_treedef,
                (False, True),
            )
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        _decode_primitive_call_meta(None)
    with pytest.raises(ValueError, match="invalid fields"):
        _decode_primitive_call_meta({**encoded, "extra": None})
    with pytest.raises(TypeError, match="static leaves must be a list"):
        _decode_primitive_call_meta({**encoded, "static_leaves": ()})
    with pytest.raises(TypeError, match="list of exact booleans"):
        _decode_primitive_call_meta({**encoded, "input_leaf_mask": [1, False]})
    with pytest.raises(TypeError, match="exact booleans"):
        _encode_bool_mask((True, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="missing its internal"):
        _split_primitive_attrs({})

    assert _PrimitiveCallMeta(
        meta.call_treedef,
        meta.input_leaf_mask,
        meta.static_leaves,
        meta.output_treedef,
    ).nondiff_mask(1) == (False,)


def test_primitive_output_and_transpose_contracts_reject_structure_drift() -> None:
    meta = _primitive_meta()

    with pytest.raises(TypeError, match="expected 2 leaves"):
        _reconstruct_primitive_output(meta, 1, label="output")
    with pytest.raises(ValueError, match="must match"):
        _flatten_primitive_output(meta, (1, 2), label="output")
    with pytest.raises(TypeError, match="must return a tuple"):
        _flatten_input_gradients([1], expected_input_count=1)
    with pytest.raises(ValueError, match="flat tuple"):
        _flatten_input_gradients(((1,),), expected_input_count=1)
    with pytest.raises(TypeError, match="return at least one"):
        _normalize_output_pytree({}, namespace=None)
    with pytest.raises(TypeError, match=r"\['bad'\].*object"):
        _normalize_output_pytree({"bad": object(), "also_bad": "text"}, namespace=None)

    leaves, treedef = _normalize_output_pytree(3.0, namespace=None)
    assert leaves == [3.0]
    assert treedef.num_leaves == 1
    assert _output_shape_and_dtype(3.0) == ((), "float64")


def test_primitive_call_partition_rejects_untraceable_dynamic_keywords() -> None:
    result = _trace_call_arguments(
        object(),  # type: ignore[arg-type]
        op_name="custom.test.partition",
        args=(),
        kwargs={"config": object()},
        nondiff_argnames=frozenset(),
        dynamic_argnames=frozenset(),
    )
    assert result[8] == {"config": result[5][0]}
    assert result[9] is None

    with pytest.raises(TypeError, match="argument 'config' is not traceable"):
        _trace_call_arguments(
            object(),  # type: ignore[arg-type]
            op_name="custom.test.partition",
            args=(),
            kwargs={"config": object()},
            nondiff_argnames=frozenset(),
            dynamic_argnames=frozenset({"config"}),
        )


def test_primitive_output_arity_cannot_drift_after_becoming_multi_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = OpRegistry()
    registry.register(OpDef(name="test.arity", num_outputs=2))
    monkeypatch.setattr(primitive_call, "get_registry", lambda: registry)

    with pytest.raises(ValueError, match="changed its output count from 2 to 3"):
        _record_primitive_output_count("test.arity", 3)


def test_primitive_residual_is_released_when_tape_attachment_fails() -> None:
    released: list[object] = []
    payload = object()
    execution = _PrimitiveExecution(payload, _ResidualSlot(payload, released.append))

    class FailingRecorder:
        @staticmethod
        def record_residual(_node_id: int, _residual: object) -> None:
            raise RuntimeError("recording failed")

    with pytest.raises(RuntimeError, match="recording failed"):
        _attach_residual(FailingRecorder(), 1, execution)  # type: ignore[arg-type]

    assert released == [payload]


def test_primitive_trace_reserves_its_internal_metadata_keyword() -> None:
    with pytest.raises(TypeError, match="reserved for Advect internals"):
        trace_primitive_call(
            lambda: _PrimitiveExecution(None, None),
            abstract_function=None,
            op_name="custom.test.reserved",
            schema_version=1,
            recorder=object(),  # type: ignore[arg-type]
            args=(),
            kwargs={_PRIMITIVE_CALL_KEY: None},
            node_attrs={},
            nondiff_argnames=frozenset(),
            dynamic_argnames=frozenset(),
            has_residual=False,
        )


def test_primitive_traced_outputs_require_a_supported_array_provider() -> None:
    with pytest.raises(TypeError, match="scalar without an array provider"):
        _wrap_traced_output(1.0, node_id=1, recorder=object(), namespace=None)  # type: ignore[arg-type]

    class Array:
        shape = (1,)
        dtype = "float64"
        ndim = 1
        size = 1

    with pytest.raises(RuntimeError, match="provider does not support Advect tracing"):
        _wrap_traced_output(Array(), node_id=1, recorder=object(), namespace=None)  # type: ignore[arg-type]


def test_structural_evaluator_validates_attrs_and_input_arity() -> None:
    invalid_bindings = (
        ("advect.getoutput", {"index": "0", "num_outputs": 1}, TypeError),
        ("advect.getoutput", {"index": 0, "num_outputs": "1"}, TypeError),
        ("advect.getoutput", {"index": 1, "num_outputs": 1}, IndexError),
        ("advect.index_update", {"index": list[object](), "mode": "remove"}, ValueError),
    )
    for op, attrs, error in invalid_bindings:
        with pytest.raises(error):
            bind_node_evaluator(op, attrs)

    checks = (
        (bind_node_evaluator("advect.getoutput", {"index": 0, "num_outputs": 1}), (), ValueError),
        (
            bind_node_evaluator("advect.getoutput", {"index": 0, "num_outputs": 1}),
            (1,),
            TypeError,
        ),
        (
            bind_node_evaluator("advect.getoutput", {"index": 0, "num_outputs": 2}),
            ((1,),),
            ValueError,
        ),
        (bind_node_evaluator("advect.getitem", {"index": []}), (), ValueError),
        (bind_node_evaluator("advect.copy", {}), (), ValueError),
        (bind_node_evaluator("advect.copy", {}), (object(),), TypeError),
        (bind_node_evaluator("advect.index_update", {"index": []}), (object(),), ValueError),
    )
    for evaluator, inputs, error in checks:
        with pytest.raises(error):
            evaluator(inputs, None, None)


def test_native_structural_evaluator_exposes_ownership_and_donation_contracts() -> None:
    copy = bind_native_node_evaluator("advect.copy", {})
    update = bind_native_node_evaluator("advect.index_update", {"index": [], "mode": "set"})
    getitem = bind_native_node_evaluator("advect.getitem", {"index": []})

    assert copy.__dict__["__advect_owned_output__"]
    assert update.__dict__["__advect_owned_output__"]
    assert update.__dict__["__advect_donation_positions__"] == (0,)
    assert getitem.__dict__["__advect_alias_positions__"] == (0,)


def test_index_update_adds_into_a_copy_unless_donation_is_explicit() -> None:
    source = np.array([1, 2])
    evaluator = bind_node_evaluator(
        "advect.index_update",
        {"index": [{"type": "int", "value": 0}], "mode": "add"},
    )

    np.testing.assert_array_equal(evaluator((source, 3), None, None), [4, 2])
    np.testing.assert_array_equal(source, [1, 2])


def test_array_donation_rejects_values_without_owned_writable_storage() -> None:
    assert not _can_donate_array(object())

    class Array:
        flags = type("Flags", (), {"owndata": False, "writeable": True})()
        base: object | None = None

    assert not _can_donate_array(Array())


def test_array_evaluator_validates_namespace_and_operation_metadata() -> None:
    class Namespace:
        __name__ = "test_namespace"
        asarray = 1

    with pytest.raises(RuntimeError, match="without an array namespace"):
        bind_node_evaluator("array.sin", {})((1.0,), None, None)
    with pytest.raises(TypeError, match="not callable"):
        bind_node_evaluator("array.asarray", {})((1,), Namespace(), None)

    class DevicesInfo:
        @staticmethod
        def devices() -> int:
            return 1

    class DevicesNamespace:
        __name__ = "test_namespace"

        @staticmethod
        def __array_namespace_info__() -> DevicesInfo:
            return DevicesInfo()

        @staticmethod
        def zeros(_shape: tuple[int, ...], **_kwargs: object) -> object:
            return object()

    with pytest.raises(TypeError, match=r"devices\(\) must return an iterable"):
        bind_node_evaluator(
            "array.zeros",
            {"shape": (1,), "_advect_device": "cpu"},
        )((), DevicesNamespace(), None)

    with pytest.raises(ValueError, match="unavailable at execution"):
        bind_node_evaluator(
            "array.zeros",
            {"shape": (1,), "_advect_device": "cuda"},
        )((), np, None)
    with pytest.raises(ValueError, match="pinv tolerance metadata"):
        bind_node_evaluator(
            "array.linalg.pinv",
            {"_advect_pinv_tolerance": "invalid"},
        )((np.eye(2),), np, None)
    with pytest.raises(NotImplementedError, match="descending sort"):
        bind_node_evaluator("array.sort", {"descending": True})((np.arange(2),), np, None)
    with pytest.raises(TypeError, match="astype requires a dtype"):
        bind_node_evaluator("array.astype", {})((np.arange(2),), np, None)

    np.testing.assert_array_equal(
        bind_node_evaluator("array.fft.fftfreq", {"n": 4, "dtype": "float32"})((), np, None),
        np.fft.fftfreq(4).astype("float32"),
    )
    assert bind_node_evaluator("array.negative", {})((2,), None, None) == -2

    selected_device = object()

    class AvailableDeviceInfo:
        @staticmethod
        def devices() -> tuple[object, ...]:
            return (selected_device,)

    class AvailableDeviceNamespace:
        __name__ = "test_namespace"

        @staticmethod
        def __array_namespace_info__() -> AvailableDeviceInfo:
            return AvailableDeviceInfo()

        @staticmethod
        def zeros(_shape: tuple[int, ...], *, device: object) -> object:
            return device

    assert (
        bind_node_evaluator(
            "array.zeros",
            {"shape": (1,), "_advect_device": str(selected_device)},
        )((), AvailableDeviceNamespace(), None)
        is selected_device
    )


def test_array_evaluator_resolves_a_namespace_from_runtime_inputs() -> None:
    class Namespace:
        __name__ = "test_namespace"

        @staticmethod
        def negative(value: object) -> tuple[str, object]:
            return "negative", value

    namespace = Namespace()

    class Array:
        shape = (1,)
        dtype = "float64"
        ndim = 1
        size = 1

        @staticmethod
        def __array_namespace__(*, api_version: str | None = None) -> Namespace:
            assert api_version is not None
            return namespace

    value = Array()
    assert bind_node_evaluator("array.negative", {})((value,), None, None) == ("negative", value)


def test_evaluate_node_value_routes_core_ops_without_a_backend() -> None:
    assert (
        evaluate_node_value(
            "advect.getoutput",
            ((1, 2),),
            {"index": 1, "num_outputs": 2},
        )
        == 2
    )
    assert evaluate_node_value("array.negative", (2,), {}) == -2


def test_vjp_attribute_decoder_uses_backend_and_operation_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decode_attrs(_op: object, attrs: dict[str, object]) -> dict[str, object]:
        return {"decoded": attrs["value"]}

    def get_hook(name: str) -> object:
        if name in {"array.decode_attrs", "fake.decode_attrs"}:
            return decode_attrs
        return None

    monkeypatch.setattr(dispatch, "get_hook", get_hook)

    assert _decode_attrs_for_vjp("array.sin", {}) == {}
    assert _decode_attrs_for_vjp("array.sin", {"_advect_backend": "fake"}) == {}
    assert _decode_attrs_for_vjp(
        "array.sin",
        {"_advect_backend": "fake", "value": 3},
    ) == {"decoded": 3}
    assert _decode_attrs_for_vjp("plain", {"value": 3}) == {"value": 3}
    assert _decode_attrs_for_vjp("array.sin", {"value": 4}) == {"decoded": 4}


def test_evaluator_dispatch_uses_bound_dynamic_and_direct_backend_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decode_attrs(_op: object, attrs: dict[str, object]) -> dict[str, object]:
        return {"amount": attrs["amount"]}

    def evaluate(
        _op: object,
        inputs: tuple[object, ...],
        attrs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        return inputs, attrs

    def get_hook(name: str) -> object:
        return {
            "fake.decode_attrs": decode_attrs,
            "fake.evaluate_op": evaluate,
        }.get(name)

    monkeypatch.setattr(dispatch, "get_hook", get_hook)
    attrs = {"_advect_backend": "fake", "amount": 3}

    evaluator = bind_node_evaluator("fake.operation", attrs)
    assert evaluator((2,), None, None) == ((2,), {"amount": 3})
    assert evaluate_node_value("fake.operation", (4,), attrs) == ((4,), {"amount": 3})

    def fallback_evaluate(
        op: str,
        inputs: tuple[object, ...],
        attrs: dict[str, object],
    ) -> tuple[str, tuple[object, ...], dict[str, object]]:
        return op, inputs, attrs

    def fallback_decoder(_op: str, attrs: dict[str, object]) -> dict[str, object]:
        return {"decoded": attrs["value"]}

    def resolve_hooks(
        _op: str,
        _inputs: tuple[object, ...],
    ) -> tuple[object, object]:
        return fallback_evaluate, fallback_decoder

    monkeypatch.setattr(
        dispatch,
        "resolve_backend_hooks",
        resolve_hooks,
    )
    dynamic = bind_node_evaluator("legacy.operation", {"value": 5})
    assert dynamic((6,), None, None) == ("legacy.operation", (6,), {"decoded": 5})


@pytest.mark.parametrize(
    "op_def",
    [
        OpDef(name="test.empty_reason", non_differentiable_reason=""),
        OpDef(
            name="test.conflicting_rule",
            vjp=_empty_vjp,
            non_differentiable_reason="reason",
        ),
        OpDef(name="test.bad_retention", vjp_needs_inputs=1),  # type: ignore[arg-type]
        OpDef(name="test.bad_schema", schema_version=0),
        OpDef(name="test.bad_residual", has_residual=1),  # type: ignore[arg-type]
        OpDef(name="test.builtin_residual", has_residual=True),
    ],
)
def test_registry_rejects_incoherent_operation_records(op_def: OpDef) -> None:
    with pytest.raises((TypeError, ValueError)):
        OpRegistry().register(op_def)


def test_registry_mutation_contracts_are_atomic_and_revisioned() -> None:
    registry = OpRegistry()
    registry.register(OpDef(name="test.operation"))
    original = registry.get("test.operation")
    revision = registry.get_revision()

    with pytest.raises(KeyError, match="not found"):
        registry.update("missing", num_outputs=2)
    with pytest.raises(ValueError, match=">= 1"):
        registry.update_num_outputs("test.operation", num_outputs=0)
    with pytest.raises(KeyError, match="not found"):
        registry.update_num_outputs("missing", num_outputs=2)
    with pytest.raises(KeyError, match="Register the op first"):
        registry.register_vjp("missing", _empty_vjp)

    assert registry.get("test.operation") is original
    assert registry.get_revision() == revision
    assert registry.has_vjp("missing") is False
    assert registry.has_jvp("missing") is False


def test_registry_repeated_vjp_registration_is_a_noop() -> None:
    registry = OpRegistry()

    def rule(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        return ()

    registry.register(OpDef(name="test.operation"))
    registry.register_vjp("test.operation", rule, needs_inputs=False, needs_output=False)
    revision = registry.get_revision()

    registry.register_vjp("test.operation", rule, needs_inputs=False, needs_output=False)

    assert registry.get_revision() == revision
    assert registry.definitions() == (registry.get("test.operation"),)


def test_error_messages_preserve_diagnostic_context() -> None:
    assert "np.array(values, like=x)" in _array_conversion_error()

    higher_order = HigherOrderNotSupportedError("unsupported", op="array.sin")
    located = HigherOrderNotSupportedError(
        "unsupported",
        op="array.sin",
        source_location="model.py:3",
    )
    trace_level = TraceLevelError("wrong level", value_level=1, active_level=2)
    no_vjp = NoVJPError(
        "missing transpose",
        op="array.sort",
        source_location="model.py:4",
        grad_reason="sorting is discrete",
    )
    no_jvp = NoJVPError("missing JVP", op="array.sort", source_location="model.py:5")

    assert "with advect.debug()" in str(higher_order)
    assert "model.py:3" in str(located)
    assert "with advect.debug()" not in str(located)
    assert "Value trace level: 1" in str(trace_level)
    assert "Active trace level: 2" in str(trace_level)
    assert "model.py:4" in str(no_vjp)
    assert "sorting is discrete" in str(no_vjp)
    assert "model.py:5" in str(no_jvp)


def test_custom_registry_record_can_satisfy_the_complete_primitive_contract() -> None:
    registry = OpRegistry()

    def implementation(value: object) -> object:
        return value

    registry.register(
        OpDef(
            name="custom.test.complete",
            implementation=implementation,
            signature=inspect.signature(implementation),
            has_residual=True,
        )
    )

    assert registry.get("custom.test.complete").implementation is implementation
