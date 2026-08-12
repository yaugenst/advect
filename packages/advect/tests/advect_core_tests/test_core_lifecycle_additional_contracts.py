"""Additional contracts for core authoring, lifecycle, and diagnostics boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad
from advect.core._array_protocol_helpers import (
    materialize_weak_scalar_operands,
    weak_scalar_runtime_value,
)
from advect.core._backend_hooks import resolve_backend_hooks
from advect.core._backends import register_hook
from advect.core._context import (
    _get_active_recorder,
    _get_active_trace_level,
    _get_numerics_context,
    _numerics_context,
    _peek_pending_update,
    _set_active_recorder,
    _set_pending_update,
    _take_pending_update,
    _trace_use_status,
    is_tracing,
)
from advect.core._diagnostics import (
    check_tape_numerics,
    raise_if_nonfinite,
    summarize_value,
)
from advect.core._protocols import _snapshot_traced
from advect_core_tests._backend_state import isolated_backend_state


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"name": ""}, ValueError, "non-empty"),
        ({"name": "advect.reserved"}, ValueError, "reserved"),
        (
            {"name": "tests.lifecycle.empty_static", "static_argnames": ("",)},
            TypeError,
            "non-empty strings",
        ),
        (
            {"name": "tests.lifecycle.duplicate_static", "static_argnames": ("x", "x")},
            ValueError,
            "duplicates",
        ),
        (
            {
                "name": "tests.lifecycle.overlap",
                "static_argnames": ("x",),
                "nondiff_argnames": ("x",),
            },
            ValueError,
            "both static and nondifferentiable",
        ),
        ({"name": "tests.lifecycle.nonbool_residual", "residual": 1}, TypeError, "boolean"),
    ],
)
def test_primitive_rejects_invalid_declarations(
    options: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    def implementation(x: object) -> object:
        return x

    with pytest.raises(error, match=match):
        ad.primitive(implementation, **options)


def test_primitive_rejects_uninspectable_and_variadic_implementations() -> None:
    with pytest.raises(TypeError, match="Cannot inspect"):
        ad.primitive(max, name="tests.lifecycle.uninspectable")

    def variadic(*values: object) -> tuple[object, ...]:
        return values

    with pytest.raises(TypeError, match="fixed parameters"):
        ad.primitive(variadic, name="tests.lifecycle.variadic")


def test_primitive_identity_call_and_rule_validation() -> None:
    @ad.primitive(name="tests.lifecycle.rules")
    def primitive(x: object, scale: int = 1) -> object:
        return x * scale

    with pytest.raises(ValueError, match="already registered"):
        ad.primitive(lambda x: x, name=primitive.name)
    with pytest.raises(TypeError, match="Invalid call"):
        primitive()

    def invalid_abstract() -> ad.ArraySpec:
        return ad.ArraySpec((), "float64")

    with pytest.raises(TypeError, match="abstract rule must accept"):
        primitive.def_abstract(invalid_abstract)

    def abstract(x: ad.AbstractValue, scale: int = 1) -> ad.ArraySpec:
        del scale
        return x.spec

    assert primitive.def_abstract(abstract) is abstract
    with pytest.raises(ValueError, match="already has abstract"):
        primitive.def_abstract(abstract)

    def invalid_jvp(output: object, primals: tuple[object, ...]) -> object:
        return output, primals

    with pytest.raises(TypeError, match="JVP rule must accept"):
        primitive.def_jvp(invalid_jvp)

    def jvp(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> object:
        del primals, tangents
        return output

    assert primitive.def_jvp(jvp) is jvp
    with pytest.raises(ValueError, match="already has a JVP"):
        primitive.def_jvp(jvp)

    def invalid_transpose(cotangent: object, primals: tuple[object, ...]) -> object:
        return cotangent, primals

    with pytest.raises(TypeError, match="transpose rule must accept"):
        primitive.def_transpose(invalid_transpose)

    def transpose(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
    ) -> tuple[object]:
        del primals, output
        return (cotangent,)

    assert primitive.def_transpose(transpose) is transpose
    with pytest.raises(ValueError, match="already has a transpose"):
        primitive.def_transpose(transpose)


def test_selective_transpose_index_argument_is_keyword_only() -> None:
    @ad.primitive(name="tests.lifecycle.selective_transpose_signature")
    def primitive(x: object) -> object:
        return x

    def transpose(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[object]:
        del primals, output, active_input_indices
        return (cotangent,)

    with pytest.raises(TypeError, match="active_input_indices must be keyword-only"):
        primitive.def_transpose(transpose)


def test_trace_frame_pending_update_lifecycle_and_errors() -> None:
    recorder = object()
    pending = object()

    assert _get_active_recorder() is None
    assert _get_active_trace_level() is None
    assert _take_pending_update(recorder) is None
    with pytest.raises(RuntimeError, match="active trace frame"):
        _set_pending_update(recorder, pending)

    _set_active_recorder(recorder)
    try:
        assert _trace_use_status(object(), take_pending=False) == (True, False, None)
        _set_pending_update(recorder, pending)
        assert _peek_pending_update(recorder) is pending
        with pytest.raises(RuntimeError, match="already pending"):
            _set_pending_update(recorder, object())
        assert _take_pending_update(recorder) is pending
        assert _take_pending_update(recorder) is None
    finally:
        _set_active_recorder(None)

    _set_active_recorder(None)
    assert not is_tracing()


def test_unconsumed_pending_update_uses_the_functional_update_error() -> None:
    recorder = object()
    _set_active_recorder(recorder)
    _set_pending_update(recorder, object())

    with pytest.raises(ad.TracingError, match="explicit functional update"):
        _set_active_recorder(None)
    assert not is_tracing()


def test_nested_numerics_context_preserves_the_outer_derivative_phase() -> None:
    with ad.debug(numerics=True), _numerics_context("JVP propagation", "model.py:4"):
        assert _get_numerics_context() == ("JVP propagation", "model.py:4")
        with _numerics_context("VJP propagation", "rule.py:8"):
            assert _get_numerics_context() == ("JVP propagation", "model.py:4")
    assert _get_numerics_context() == ("primal evaluation", None)


def test_backend_hooks_resolve_from_input_namespace_and_report_absence() -> None:
    class Namespace:
        __name__ = "lifecycle_backend"

    namespace = Namespace()

    class PluginArray:
        def __array_namespace__(self, *, api_version: str | None = None) -> object:
            del api_version
            return namespace

    def evaluate(op: str, inputs: tuple[object, ...], attrs: dict[str, object]) -> object:
        return op, inputs, attrs

    def decode(attrs: object) -> object:
        return attrs

    with isolated_backend_state():
        register_hook("lifecycle_backend.evaluate_op", evaluate)
        register_hook("lifecycle_backend.decode_attrs", decode)
        assert resolve_backend_hooks("unqualified", (PluginArray(),)) == (evaluate, decode)

        with pytest.raises(RuntimeError, match="No backend evaluator"):
            resolve_backend_hooks("missing.operation", (object(),))


def test_pytree_protocol_validation_and_nested_static_tracers() -> None:
    class NonCallableProtocol:
        __advect_tree_flatten__ = 1
        __advect_tree_unflatten__ = 2

    class InvalidResultProtocol:
        def __advect_tree_flatten__(self) -> list[object]:
            return [(), None]

        @classmethod
        def __advect_tree_unflatten__(
            cls,
            aux_data: object,
            children: tuple[object, ...],
        ) -> InvalidResultProtocol:
            del aux_data, children
            return cls()

    class InvalidChildrenProtocol(InvalidResultProtocol):
        def __advect_tree_flatten__(self) -> tuple[list[object], None]:
            return [], None

    with pytest.raises(TypeError, match=r"hooks.*callable"):
        ad.pytree.tree_flatten(NonCallableProtocol())
    with pytest.raises(TypeError, match="must return"):
        ad.pytree.tree_flatten(InvalidResultProtocol())
    with pytest.raises(TypeError, match="children must be a tuple"):
        ad.pytree.tree_flatten(InvalidChildrenProtocol())

    class DictSubclass(dict[str, object]):
        pass

    class ListSubclass(list[object]):
        pass

    def reject_static_tracer(value: object) -> object:
        ad.pytree.static(DictSubclass(items=ListSubclass([value])))
        return value

    with pytest.raises(TypeError, match="Static pytree metadata"):
        ad.jvp(reject_static_tracer)(np.array(1.0), tangents=np.array(1.0))

    cycle: list[object] = []
    cycle.append(cycle)
    assert ad.pytree.static(cycle).value is cycle


class _ScalarWithoutItem:
    shape = ()

    def __init__(self, value: object, dtype: str) -> None:
        self.value = value
        self.dtype = dtype

    def __bool__(self) -> bool:
        return bool(self.value)

    def __complex__(self) -> complex:
        return complex(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_ScalarWithoutItem(value=True, dtype="bool"), True),
        (_ScalarWithoutItem(1 + 2j, "complex64"), 1 + 2j),
        (_ScalarWithoutItem(1.5, "float32"), 1.5),
        (_ScalarWithoutItem(4, "int16"), 4),
    ],
)
def test_weak_scalar_runtime_fallbacks(value: object, expected: object) -> None:
    class WeakTracer:
        _advect_weak = True

    assert weak_scalar_runtime_value(WeakTracer(), value) == expected


def test_weak_scalar_materialization_validates_provider_namespace() -> None:
    array = np.ones(2, dtype=np.float32)
    with pytest.raises(TypeError, match=r"does not provide asarray\(\)"):
        materialize_weak_scalar_operands("array.add", (array, 1), namespace=object())

    class NamespaceWithoutDtypes:
        @staticmethod
        def asarray(value: object, *, dtype: object) -> object:
            return value, dtype

    with pytest.raises(TypeError, match="does not provide dtype"):
        materialize_weak_scalar_operands(
            "array.add",
            (array, True),
            namespace=NamespaceWithoutDtypes(),
        )


def test_tracer_snapshot_protocol_reports_missing_implementation() -> None:
    with pytest.raises(TypeError, match="does not implement"):
        _snapshot_traced(object())


def test_diagnostics_are_bounded_and_never_mask_values() -> None:
    class BrokenValue:
        dtype = "opaque"
        shape = (object(),)

        def __repr__(self) -> str:
            msg = "render failed"
            raise RuntimeError(msg)

    class Tracer:
        def _advect_snapshot(self) -> tuple[int, object]:
            return 0, object()

    assert summarize_value(1 + 2j) == "shape=(), dtype=complex, values=(1+2j)"
    assert "values=<BrokenValue>" in summarize_value(BrokenValue())
    assert "finite=" not in summarize_value(Tracer())

    with pytest.raises(ad.NumericsError, match=r"leaf\['payload'\]\[0\]"):
        raise_if_nonfinite(
            {"payload": [np.array([np.nan])]},
            phase="primal evaluation",
            op="tests.lifecycle.nonfinite",
            source_location=None,
        )


def test_nested_tape_numerics_are_owned_by_the_outer_trace() -> None:
    class NestedTape:
        @staticmethod
        def runtime_trace_identity() -> tuple[int, int]:
            return 1, 2

        @staticmethod
        def _diagnostic_snapshot() -> object:
            raise AssertionError("nested tape should not be inspected")

    check_tape_numerics(NestedTape())  # type: ignore[arg-type]


def test_primitive_result_rejects_a_noncallable_release() -> None:
    with pytest.raises(TypeError, match="release must be callable"):
        ad.PrimitiveResult(output=1, residual=object(), release=1)  # type: ignore[arg-type]


def test_weak_array_specs_are_rank_zero_at_live_and_durable_boundaries() -> None:
    with pytest.raises(ValueError, match="rank-zero ArraySpec"):
        ad.ArraySpec((2,), "float32", weak=True)

    program = ad.stage(lambda value: value, specs=(ad.ArraySpec((2,), "float32"),))
    payload: Any = program.to_dict()
    payload["program"]["call_specs"][0]["weak"] = True

    with pytest.raises(ValueError, match="rank-zero ArraySpec"):
        ad.StagedProgram.from_dict(payload)
