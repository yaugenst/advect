"""Durability and input-classification contracts for abstract staging."""

from __future__ import annotations

import enum
import functools
import json
from typing import TYPE_CHECKING, Any, cast

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable


@ad.primitive(name="tests.durability_pair")
def _pair(x: Any) -> tuple[Any, Any]:
    return x, x


@_pair.def_abstract
def _pair_abstract(x: ad.AbstractValue) -> tuple[ad.ArraySpec, ad.ArraySpec]:
    return x.spec, x.spec


_KERNEL = np.array([1.0, 2.0], dtype=np.float32)


@functools.cache
def _artifact_text() -> str:
    """Serialize %0 input, %1 custom pair, %2 getoutput, %3 constant, %4 add."""
    program = ad.stage(lambda x: _pair(x)[1] + _KERNEL, specs=(ad.ArraySpec((2,), "float32"),))
    return json.dumps(program.to_dict())


def _dead_node(node_id: int) -> dict[str, object]:
    """Return an ``array.negative`` of input %0 that no output uses."""
    return {
        "id": node_id,
        "op": "array.negative",
        "schema_version": 1,
        "inputs": [0],
        "attrs": {},
        "shape": [2],
        "dtype": "float32",
        "num_outputs": 1,
        "output_shapes": None,
        "output_dtypes": None,
        "name": None,
        "source_location": None,
    }


def _edit(path: str, change: Callable[[Any], object]) -> Callable[[Any], object]:
    """Replace the value at a ``/``-separated artifact path with ``change(value)``."""
    *parents, last = path.split("/")

    def mutate(payload: Any) -> object:
        target = payload["program"]
        for key in parents:
            target = target[int(key) if isinstance(target, list) else key]
        index = int(last) if isinstance(target, list) else last
        target[index] = change(target[index] if isinstance(target, list) else target.get(index))
        return payload

    return mutate


def _set(path: str, value: object) -> Callable[[Any], object]:
    return _edit(path, lambda _old: value)


_CUSTOM_CALL = "graph/nodes/1/attrs/__advect_primitive_call__/value"
_REPORT = "optimization"
_CORRUPTIONS = [
    # outer envelope
    (lambda _payload: None, TypeError, "must be a mapping", "not-a-mapping"),
    (
        lambda _payload: {"format": "advect.ssa-program", "version": 3},
        ValueError,
        "invalid fields",
        "missing-program",
    ),
    (
        lambda payload: {**payload, "format": "future.program"},
        ValueError,
        "Unknown staged program format",
        "unknown-format",
    ),
    (
        lambda payload: {**payload, "version": "2"},
        TypeError,
        "version must be an integer",
        "noninteger-version",
    ),
    (
        lambda payload: {**payload, "version": 2},
        ValueError,
        "Unsupported staged program format version 2",
        "old-version",
    ),
    (
        lambda payload: {**payload, "version": 999},
        ValueError,
        "Unsupported staged program format version 999",
        "future-version",
    ),
    # artifact structure
    (
        lambda payload: {**payload, "program": []},
        TypeError,
        "artifact must be a mapping",
        "artifact-type",
    ),
    (_set("extra", value=True), ValueError, "artifact has invalid fields", "artifact-fields"),
    (_set("call_specs", {}), TypeError, "call_specs must be a list", "call-specs-type"),
    (_set("output_specs", {}), TypeError, "output_specs must be a list", "output-specs-type"),
    (_set("constants", {}), TypeError, "constants must be a list", "constants-type"),
    (
        _set("output_specs/0", {"kind": "static", "value": {"kind": "scalar", "value": 1}}),
        TypeError,
        "output specs must all be array specs",
        "static-output",
    ),
    # call and output specs
    (_set("call_specs", []), ValueError, "call specs do not match their pytree", "call-tree"),
    (_set("output_specs", []), ValueError, "output specs do not match their pytree", "output-tree"),
    (
        _set("call_specs/0/shape", [3]),
        ValueError,
        "graph inputs do not match its call specs",
        "call-shape",
    ),
    (
        _set("call_specs/0/dtype", "float64"),
        ValueError,
        "graph inputs do not match its call specs",
        "call-dtype",
    ),
    (_set("call_specs/0/weak", value=True), ValueError, "rank-zero ArraySpec", "weak-rank"),
    (
        _set("output_specs/0/shape", [3]),
        ValueError,
        "output specs do not match graph outputs",
        "output-shape",
    ),
    (_set("call_specs/0", None), TypeError, "spec must be a mapping", "spec-type"),
    (
        _set("call_specs/0", {"kind": "array", "shape": [2]}),
        ValueError,
        "array spec has invalid fields",
        "spec-fields",
    ),
    (_set("call_specs/0/shape", "2"), TypeError, "shape must be a list of integers", "spec-shape"),
    (_set("call_specs/0/dtype", 32), TypeError, "dtype must be a string", "spec-dtype"),
    (_set("call_specs/0/device", 0), TypeError, "device must be a string or None", "spec-device"),
    (_set("call_specs/0/weak", 0), TypeError, "weak flag must be a bool", "spec-weak"),
    (
        _set(
            "call_specs/0",
            {"kind": "static", "value": {"kind": "scalar", "value": 1}, "extra": True},
        ),
        ValueError,
        "static spec has invalid fields",
        "static-spec-fields",
    ),
    (
        _set("call_specs/0", {"kind": "future"}),
        ValueError,
        "Unknown staged call spec kind",
        "spec-kind",
    ),
    # constant manifest
    (_set("constants", []), ValueError, "does not match graph constants", "missing-manifest"),
    (
        _edit("constants", lambda records: [*records, records[0]]),
        ValueError,
        "repeats value",
        "duplicate-record",
    ),
    (
        _edit("constants/0/bytes", lambda count: count + 1),
        ValueError,
        "metadata does not match its payload",
        "byte-count",
    ),
    (
        _set("constants/0/digest", "0" * 64),
        ValueError,
        "digest does not match its payload",
        "manifest-digest",
    ),
    (_set("constants/0/shape", [1, 2]), ValueError, "shape/dtype", "manifest-shape"),
    (
        _set("graph/constants/3/data", "0000104100000040"),
        ValueError,
        "constant digest",
        "payload-data",
    ),
    (_set("constants/0", None), TypeError, "record must be a mapping", "record-type"),
    (
        _set("constants/0/extra", value=True),
        ValueError,
        "record has invalid fields",
        "record-fields",
    ),
    (_set("constants/0/value_id", -1), TypeError, "value_id must be", "record-value-id"),
    (_set("constants/0/origin", "future"), ValueError, "origin must be", "record-origin"),
    (_set("constants/0/location", 1), TypeError, "location must be", "record-location"),
    (_set("constants/0/shape", [-1]), TypeError, "shape must be", "record-shape"),
    (_set("constants/0/dtype", ""), TypeError, "dtype must be", "record-dtype"),
    (_set("constants/0/bytes", -1), TypeError, "bytes must be", "record-bytes"),
    (_set("constants/0/digest", "ABC"), TypeError, "digest must be", "record-digest"),
    (_set("constants/0/name", 1), TypeError, "name must be", "record-name"),
    # optimization report
    (_set(_REPORT, []), TypeError, "report must be a mapping", "report-type"),
    (_set(f"{_REPORT}/extra", 0), ValueError, "report has invalid fields", "report-fields"),
    (
        _set(f"{_REPORT}/nodes_before", -1),
        TypeError,
        "must be a non-negative integer",
        "report-count",
    ),
    (_set(f"{_REPORT}/passes", {}), TypeError, "passes must be a list", "passes-type"),
    (_set(f"{_REPORT}/passes/0", []), TypeError, "pass must be a mapping", "pass-type"),
    (_set(f"{_REPORT}/passes/0/extra", 0), ValueError, "pass has invalid fields", "pass-fields"),
    (_set(f"{_REPORT}/passes/0/name", 1), TypeError, "pass name must be a string", "pass-name"),
    (
        _set(f"{_REPORT}/passes/0/rewritten_nodes", -1),
        TypeError,
        "must be a non-negative integer",
        "pass-count",
    ),
    (
        _edit(f"{_REPORT}/passes", lambda passes: passes[::-1]),
        ValueError,
        "pass sequence is invalid",
        "pass-order",
    ),
    (
        _edit(f"{_REPORT}/rewritten_nodes", lambda count: count + 1),
        ValueError,
        "aggregate counts are inconsistent",
        "aggregate-count",
    ),
    (
        _edit(f"{_REPORT}/passes/0/removed_nodes", lambda count: count + 1),
        ValueError,
        "removed-node count is inconsistent",
        "removed-count",
    ),
    # graph records and linkage
    (_set("graph", []), TypeError, "graph payload must be a mapping", "graph-type"),
    (_set("graph/compiler_version", 999), ValueError, "compiler version", "compiler-version"),
    (_set("graph/optimizer_version", 999), ValueError, "optimizer version", "optimizer-version"),
    (
        _set("graph/outputs", []),
        ValueError,
        "graph output count does not match its output pytree",
        "graph-outputs",
    ),
    (
        _edit("graph/nodes", lambda nodes: [*nodes, _dead_node(len(nodes))]),
        ValueError,
        "graph node count does not match its optimization report",
        "graph-nodes",
    ),
    (_set("graph/nodes", {}), TypeError, "graph nodes must be a list", "nodes-type"),
    (_set("graph/nodes/0", []), TypeError, "graph node must be a mapping", "node-type"),
    (_set("graph/nodes/4/op", None), TypeError, "node op must be a string", "op-type"),
    (
        _set("graph/nodes/4/op", "array.future_add"),
        ValueError,
        "is not registered",
        "unknown-operation",
    ),
    (
        _set("graph/nodes/4/op", "custom.tests.missing"),
        ValueError,
        "requires unlinked primitive 'tests.missing'",
        "unlinked-custom",
    ),
    (_set("graph/nodes/4/schema_version", 0), TypeError, "schema_version must be", "schema-type"),
    (_set("graph/nodes/4/schema_version", 999), ValueError, "linked schema is 1", "schema-version"),
    (_set("graph/nodes/4/num_outputs", 0), TypeError, "num_outputs must be", "outputs-type"),
    (_set("graph/nodes/4/num_outputs", 2), ValueError, "expects num_outputs=1", "output-arity"),
    (
        _set("graph/nodes/2/attrs/index/value", 2),
        ValueError,
        "index 2 out of range",
        "getoutput-index",
    ),
    # An extra operand would reach a NumPy ufunc's out= and overwrite that value.
    (
        _edit("graph/nodes/4/inputs", lambda inputs: [*inputs, 0]),
        ValueError,
        r"%4 passes 3 operands to 'array.add', which takes 2",
        "extra-operand",
    ),
    (_set("graph/nodes/4/inputs", [3]), ValueError, "which takes 2", "missing-operand"),
    (_set("graph/nodes/2/inputs", [1, 1]), ValueError, "which takes 1", "structural-operand"),
    (
        _set("graph/nodes/4/op", "array_ext.fmax"),
        ValueError,
        "cannot appear in a staged program",
        "unstageable-operation",
    ),
    # custom primitive call contracts
    (_set("graph/nodes/1/attrs", {}), ValueError, "invalid call contract", "custom-call-metadata"),
    (
        _edit(_CUSTOM_CALL, lambda call: {**call, "output_treedef": call["call_treedef"]}),
        ValueError,
        "output structure does not match its arity",
        "custom-output-structure",
    ),
    (
        _set("graph/nodes/1/inputs", []),
        ValueError,
        "input count does not match",
        "custom-missing-input",
    ),
    (
        _edit("graph/nodes/1/inputs", lambda inputs: [*inputs, inputs[0]]),
        ValueError,
        "input count does not match",
        "custom-extra-input",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        pytest.param(mutate, error, match, id=case_id)
        for mutate, error, match, case_id in _CORRUPTIONS
    ],
)
def test_loaded_program_rejects_corrupted_payload(
    mutate: Callable[[Any], object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        ad.StagedProgram.from_dict(mutate(json.loads(_artifact_text())))


def test_loaded_stage_is_not_reoptimized() -> None:
    payload = json.loads(_artifact_text())
    graph = payload["program"]["graph"]
    dead_id = len(graph["nodes"])
    graph["nodes"].append(_dead_node(dead_id))
    payload["program"]["optimization"] = {
        "nodes_before": dead_id + 1,
        "nodes_after": dead_id + 1,
        "rewritten_nodes": 0,
        "passes": [
            {
                "name": name,
                "nodes_before": dead_id + 1,
                "nodes_after": dead_id + 1,
                "removed_nodes": 0,
                "rewritten_nodes": 0,
            }
            for name in ("dce", "simplify", "cse")
        ],
    }

    restored = ad.StagedProgram.from_dict(payload)

    assert restored.graph.node_count == dead_id + 1
    assert restored.graph.get_node(dead_id).op == "array.negative"
    value = np.array([2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(restored(value), value + _KERNEL)


def test_captured_array_constant_reports_a_typed_manifest() -> None:
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: x * kernel, specs=(ad.ArraySpec((2, 2), "float32"),)),
    )

    (record,) = program.constants
    assert (record.origin, record.shape, record.dtype) == ("closure", (2, 2), "float32")
    assert record.bytes == kernel.nbytes
    assert len(record.digest) == 64
    payload = cast("dict[str, Any]", program.to_dict())
    constant = next(iter(payload["program"]["graph"]["constants"].values()))
    assert constant["format"] == "advect.numeric-constant"
    assert (constant["dtype"], constant["shape"]) == ("float32", [2, 2])
    assert constant["digest"] == payload["program"]["constants"][0]["digest"] == record.digest

    result = ad.StagedProgram.from_dict(payload)(np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(result, kernel)
    assert result.dtype == np.float32


def test_captured_array_api_constant_is_detached_and_round_trips() -> None:
    kernel = strict.asarray([1.0, 2.0], dtype=strict.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x + kernel,
            specs=(ad.ArraySpec((2,), "float32"),),
        ),
    )

    kernel[0] = 99.0
    value = strict.asarray([3.0, 4.0], dtype=strict.float32)
    expected = strict.asarray([4.0, 6.0], dtype=strict.float32)
    live_result = program(value)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    restored_result = restored(value)

    assert bool(strict.all(live_result == expected))
    assert bool(strict.all(restored_result == expected))
    assert restored_result.dtype == strict.float32
    record = restored.constants[0]
    assert record.dtype == "float32"
    assert record.bytes == 2 * 4


def test_loaded_array_api_constant_materializes_once_per_runtime_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = strict.asarray([1.0, 2.0], dtype=strict.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x + kernel,
            specs=(ad.ArraySpec((2,), "float32"),),
        ),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    value = strict.asarray([3.0, 4.0], dtype=strict.float32)
    original_asarray = strict.asarray
    materializations = 0

    def counting_asarray(*args: Any, **kwargs: Any) -> Any:
        nonlocal materializations
        materializations += 1
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(strict, "asarray", counting_asarray)
    first = restored(value)
    second = restored(value)

    assert bool(strict.all(first == second))
    assert materializations == 1


def test_trace_bound_constant_materializations_are_not_cached() -> None:
    kernel = np.array([0.5, -1.0, 2.0])
    program = cast("ad.StagedProgram", ad.stage(lambda x: np.sum(x * kernel), np.zeros(3)))
    value = np.array([1.0, 2.0, 3.0])

    for _ in range(3):
        ad.grad(program)
        ad.stage(lambda x: 2 * program(x), value)
        dynamic = ad.grad(lambda x: program(x))  # noqa: PLW0108 - explicit trace boundary
        np.testing.assert_allclose(dynamic(value), kernel)
    np.testing.assert_allclose(program(value), np.sum(value * kernel))

    cache = cast("Any", program)._execution_state.materialized_constants
    assert [(entry.namespace, entry.device) for entry in cache] == [(np, "cpu")]


def test_repeated_staged_calls_share_their_constants_in_one_trace() -> None:
    kernel = np.array([0.5, -1.0, 2.0])
    program = cast("ad.StagedProgram", ad.stage(lambda x: x * kernel, np.zeros(3)))
    value = np.array([1.0, 2.0, 3.0])

    twice = cast("ad.StagedProgram", ad.stage(lambda x: program(x) + program(x), value))
    gradient = cast(
        "ad.StagedProgram",
        ad.stage(ad.grad(lambda x: np.sum(program(x) * program(x))), value),
    )

    assert len(twice.constants) == 1
    assert len([record for record in gradient.constants if record.shape == (3,)]) == 1
    np.testing.assert_allclose(twice(value), 2.0 * value * kernel)
    np.testing.assert_allclose(gradient(value), 2.0 * value * kernel**2)


def test_returned_constants_cannot_change_later_calls() -> None:
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]])
    program = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: (x + 1.0, kernel, kernel.T), np.zeros(2)),
    )

    _, constant, view = program(np.zeros(2))
    for returned in (constant, view):
        with pytest.raises(ValueError, match="read-only"):
            returned[0, 0] = 100.0

    _, constant, view = program(np.zeros(2))
    np.testing.assert_array_equal(constant, kernel)
    np.testing.assert_array_equal(view, kernel.T)


def test_returned_constants_of_a_writable_provider_cannot_change_later_calls() -> None:
    kernel = np.array([1.0, 2.0, 3.0])

    def outputs(x: Any) -> tuple[Any, Any, Any]:
        constant = x.__array_namespace__().asarray(kernel)
        return x @ constant, constant, constant[1:]

    program = ad.stage(outputs, specs=(ad.ArraySpec((3,), "float64"),))
    value = strict.ones(3, dtype=strict.float64)

    for _ in range(2):
        total, constant, view = program(value)
        assert float(total) == 6.0
        np.testing.assert_array_equal(np.asarray(constant), kernel)
        np.testing.assert_array_equal(np.asarray(view), kernel[1:])
        constant[...] = 0.0
        view[...] = -1.0
        # A traced call's primal outputs are provider arrays too.
        ad.vjp(program)(value)[0][1][...] = 0.0


def test_public_staged_inspection_cannot_mutate_the_store() -> None:
    kernel = np.array([1.0, 2.0], dtype=np.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x + kernel,
            specs=(ad.ArraySpec((2,), "float32"),),
        ),
    )
    graph = program.graph
    assert not hasattr(graph, "get_constant")
    assert not hasattr(graph, "to_dict")

    kernel[:] = -100.0
    constant_ids = graph.constant_ids()
    constant_ids.clear()
    assert len(graph.constant_ids()) == 1

    add_id = next(
        node_id for node_id in graph.node_ids() if graph.get_node(node_id).op == "array.add"
    )
    inspected_attrs = graph.get_node(add_id).attrs
    inspected_attrs["forged"] = True
    assert "forged" not in graph.get_node(add_id).attrs

    payload = cast("dict[str, Any]", program.to_dict())
    constant = next(iter(payload["program"]["graph"]["constants"].values()))
    constant["data"] = ""
    payload["program"]["graph"]["nodes"][add_id]["attrs"]["forged"] = True

    value = np.array([3.0, 4.0], dtype=np.float32)
    np.testing.assert_array_equal(program(value), value + np.array([1.0, 2.0]))
    assert "forged" not in graph.get_node(add_id).attrs


def test_python_numeric_scalars_are_weak_dynamic_rank_zero_inputs() -> None:
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, scale: x * scale,
            specs=(
                ad.ArraySpec((2,), "float32"),
                ad.ArraySpec((), "float64", weak=True),
            ),
        ),
    )
    value = np.array([1.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(program(value, 2.0), np.array([2.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(program(value, 3.0), np.array([3.0, 6.0], dtype=np.float32))

    serialized = cast("dict[str, Any]", program.to_dict())
    scalar_spec = serialized["program"]["call_specs"][1]
    assert scalar_spec == {
        "kind": "array",
        "shape": [],
        "dtype": "float64",
        "device": None,
        "weak": True,
    }
    assert serialized["program"]["output_specs"] == [
        {"kind": "array", "shape": [2], "dtype": "float32", "device": None, "weak": False}
    ]


def _double(value: Any) -> Any:
    return value * 2


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.dtype(">f4"), strict.float32, "float32"],
    ids=["numpy-type", "big-endian", "array-api", "name"],
)
def test_equivalent_dtype_spellings_stage_one_canonical_artifact(dtype: object) -> None:
    program = cast("ad.StagedProgram", ad.stage(_double, specs=(ad.ArraySpec((2,), dtype),)))
    reference = cast(
        "ad.StagedProgram",
        ad.stage(_double, specs=(ad.ArraySpec((2,), "float32"),)),
    )

    assert program.to_dict() == reference.to_dict()
    assert program.signature == ((ad.ArraySpec((2,), "float32"),), {})
    value = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(ad.StagedProgram.from_dict(program.to_dict())(value), value * 2)


def test_loaded_dtype_spellings_are_stored_canonically() -> None:
    program = cast("ad.StagedProgram", ad.stage(_double, specs=(ad.ArraySpec((2,), "float32"),)))
    payload = cast("dict[str, Any]", program.to_dict())
    payload["program"]["call_specs"][0]["dtype"] = "<class 'numpy.float32'>"
    payload["program"]["output_specs"][0]["dtype"] = "array_api_strict.float32"

    assert ad.StagedProgram.from_dict(payload).to_dict() == program.to_dict()


def test_non_native_byte_order_capture_views_round_trip() -> None:
    captured = np.arange(6.0).reshape(2, 3).astype(">f8")
    program = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: (x + 1, captured.T, captured[0]), np.zeros((2, 3))),
    )
    payload = cast("dict[str, Any]", program.to_dict())

    assert [spec["dtype"] for spec in payload["program"]["output_specs"]] == ["float64"] * 3
    _shifted, transposed, row = ad.StagedProgram.from_dict(payload)(np.zeros((2, 3)))
    np.testing.assert_array_equal(transposed, captured.T)
    np.testing.assert_array_equal(row, captured[0])


@pytest.mark.parametrize("staging", ["example", "spec"])
def test_staged_calls_accept_either_byte_order(staging: str) -> None:
    swapped = np.array([1.0, -2.0, 3.0], dtype=">f8")
    native = swapped.astype(np.float64)

    def loss(x: Any) -> Any:
        return np.sum(x * x)

    program = cast(
        "ad.StagedProgram",
        ad.stage(loss, swapped)
        if staging == "example"
        else ad.stage(loss, specs=(ad.ArraySpec((3,), np.dtype(">f8")),)),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    for staged in (program, restored):
        for value in (swapped, native):
            assert staged(value) == pytest.approx(14.0)

    def through_dynamic_trace(x: Any) -> Any:
        return program(x)

    np.testing.assert_array_equal(ad.grad(program)(swapped), 2.0 * native)
    np.testing.assert_array_equal(ad.grad(through_dynamic_trace)(swapped), 2.0 * native)


class _Count(enum.IntEnum):
    TWO = 2


class _Real(float):
    pass


class _Phase(complex):
    pass


@pytest.mark.parametrize(
    ("example", "value", "builtin"),
    [(2, _Count.TWO, 2), (2.0, _Real(1.5), 1.5), (2j, _Phase(1j), 1j)],
    ids=["int", "float", "complex"],
)
def test_staged_call_accepts_builtin_scalar_subclasses_as_weak_scalars(
    example: object, value: object, builtin: complex
) -> None:
    # NumPy promotes a built-in subclass as a strong type, so a float32 operand
    # shows whether the call boundary kept the declared weak category.
    array = np.ones(3, dtype=np.complex64 if isinstance(builtin, complex) else np.float32)
    program = cast("ad.StagedProgram", ad.stage(lambda x, scale: x * scale, array, example))

    result = program(array, value)
    expected = program(array, builtin)
    assert result.dtype == expected.dtype == array.dtype
    np.testing.assert_array_equal(result, expected)

    doubled = cast("ad.StagedProgram", ad.stage(lambda scale: 2 * scale, example))(value)
    assert type(doubled) is type(builtin)
    assert doubled == 2 * builtin


def test_static_specs_use_serialized_value_identity() -> None:
    value = np.array([1.0, 2.0], dtype=np.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, config: x * config["scale"],
            specs=(
                ad.ArraySpec((2,), "float32"),
                ad.StaticSpec({"scale": 2, "mode": "forward"}),
            ),
        ),
    )

    reordered = {"mode": "forward", "scale": 2}
    np.testing.assert_array_equal(program(value, reordered), value * 2)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_array_equal(restored(value, reordered), value * 2)
    with pytest.raises(TypeError, match="changed value"):
        program(value, {"scale": 3, "mode": "forward"})
    with pytest.raises(TypeError, match="changed value"):
        restored(value, {"scale": 3, "mode": "forward"})


@pytest.mark.parametrize(
    ("compiled", "called"),
    [(1, 1.0), (1, True), (1.0, 1), (0.0, -0.0), (1, np.int64(1)), (True, np.True_)],
    ids=["int-float", "int-bool", "float-int", "signed-zero", "int-numpy", "bool-numpy"],
)
def test_static_pytree_values_and_keys_use_serialized_identity(
    compiled: object,
    called: object,
) -> None:
    value = np.arange(3.0)
    by_spec = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, _factor: x, specs=(ad.ArraySpec((3,), "float64"), ad.StaticSpec(compiled))
        ),
    )
    by_node = cast(
        "ad.StagedProgram",
        ad.stage(lambda x, factor: x * factor.value, value, ad.pytree.static(compiled)),
    )
    by_key = cast(
        "ad.StagedProgram",
        ad.stage(lambda table: table[compiled] - table["other"], {compiled: value, "other": value}),
    )

    np.testing.assert_array_equal(by_spec(value, compiled), value)
    np.testing.assert_array_equal(by_node(value, ad.pytree.static(compiled)), value * compiled)
    np.testing.assert_array_equal(by_key({"other": value, compiled: value}), np.zeros(3))
    with pytest.raises(TypeError, match="changed value"):
        by_spec(value, called)
    with pytest.raises(TypeError, match="differs from the declared specs"):
        by_node(value, ad.pytree.static(called))
    with pytest.raises(TypeError, match="differs from the declared specs"):
        by_key({called: value, "other": value})


def test_static_specs_reject_repr_only_identity() -> None:
    class SameRepr:
        __hash__ = None

        def __repr__(self) -> str:
            return "same"

    with pytest.raises(TypeError, match="not JSON serializable"):
        ad.stage(
            lambda x, _config: x,
            specs=(ad.ArraySpec((2,), "float32"), ad.StaticSpec(SameRepr())),
        )


def test_static_specs_snapshot_mutable_values_before_tracing_and_storage() -> None:
    config = {"scale": 2}
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, settings: x * settings["scale"],
            specs=(ad.ArraySpec((2,), "float32"), ad.StaticSpec(config)),
        ),
    )
    config["scale"] = 3
    value = np.array([1.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(program(value, {"scale": 2}), value * 2)
    with pytest.raises(TypeError, match="changed value"):
        program(value, config)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_array_equal(restored(value, {"scale": 2}), value * 2)
    with pytest.raises(TypeError, match="changed value"):
        restored(value, config)


def test_static_pytree_aux_data_is_snapshotted_before_tracing_and_storage() -> None:
    config = {"scale": 2}
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, settings: x * settings.value["scale"],
            specs=(
                ad.ArraySpec((2,), "float32"),
                ad.pytree.static(config),
            ),
        ),
    )
    config["scale"] = 3
    value = np.array([1.0, 2.0], dtype=np.float32)
    original_static = ad.pytree.static({"scale": 2})

    np.testing.assert_array_equal(program(value, original_static), value * 2)
    with pytest.raises(TypeError, match="declared specs"):
        program(value, ad.pytree.static(config))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_array_equal(restored(value, original_static), value * 2)


def test_created_constant_identity_dedup_retains_objects_until_compile_finishes() -> None:
    constant_count = 1_000

    def add_temporaries(x: object) -> object:
        result = x
        for index in range(constant_count):
            result = result + np.asarray(index, dtype=np.float64)
        return result

    program = cast(
        "ad.StagedProgram",
        ad.stage(add_temporaries, specs=(ad.ArraySpec((), "float64"),)),
    )
    assert len(program.constants) == constant_count
    np.testing.assert_array_equal(
        program(np.asarray(0.0)),
        np.asarray(sum(range(constant_count)), dtype=np.float64),
    )


def test_static_specs_reject_nested_builtin_subclasses() -> None:
    class Factor(int):
        pass

    with pytest.raises(TypeError, match="not JSON serializable"):
        ad.stage(
            lambda x, config: x * config["scale"],
            specs=(
                ad.ArraySpec((2,), "float32"),
                ad.StaticSpec({"scale": Factor(2)}),
            ),
        )


@pytest.mark.parametrize(
    ("value", "dtype"),
    [(True, "bool"), (2, "int64"), (1.25, "float64"), (1.0 + 2.0j, "complex128")],
)
def test_stage_infers_and_replays_python_scalar_categories(value: complex, dtype: str) -> None:
    program = cast("ad.StagedProgram", ad.stage(lambda item: (item, 2 * item), value))
    restored = ad.StagedProgram.from_dict(program.to_dict())

    assert program.signature == ((ad.ArraySpec((), dtype, weak=True),), {})
    assert cast("dict[str, Any]", program.to_dict())["program"]["call_specs"] == [
        {"kind": "array", "shape": [], "dtype": dtype, "device": None, "weak": True}
    ]
    # Scalar-only programs replay without an array provider, in Python's own types.
    for staged in (program, restored):
        same, doubled = staged(value)
        assert (type(same), same) == (type(value), value)
        assert (type(doubled), doubled) == (type(2 * value), 2 * value)


@pytest.mark.parametrize(
    ("shape", "weak", "match"),
    [
        ((-1,), False, "non-negative"),
        ((1, -2), False, "non-negative"),
        # The weak-rank corruption above checks the same rule on a loaded artifact.
        ((2,), True, "rank-zero ArraySpec"),
    ],
    ids=["negative", "negative-inner", "weak-vector"],
)
def test_array_spec_rejects_negative_dimensions_and_weak_arrays(
    shape: tuple[int, ...], *, weak: bool, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ad.ArraySpec(shape, "float32", weak=weak)


def test_stage_rejects_non_array_examples_and_specs() -> None:
    with pytest.raises(TypeError, match="declare non-array inputs with StaticSpec"):
        ad.stage(lambda item: item, "dynamic")
    with pytest.raises(TypeError, match="specs must contain ArraySpec or StaticSpec"):
        ad.stage(lambda item: item, specs=(object(),))


def test_staged_call_rejects_changed_nested_call_structure() -> None:
    program = ad.stage(
        lambda items: items["value"],
        specs=({"value": ad.ArraySpec((1,), "float32")},),
    )

    with pytest.raises(TypeError, match="pytree differs from the declared specs"):
        program({"other": np.ones(1, dtype=np.float32)})


@pytest.mark.parametrize(
    ("dtype", "value"),
    [("bool", 1), ("complex128", True), ("float64", 1.0j), ("int64", 1.0)],
)
def test_staged_call_rejects_changed_weak_scalar_category(dtype: str, value: object) -> None:
    program = ad.stage(lambda item: item, specs=(ad.ArraySpec((), dtype, weak=True),))

    with pytest.raises(ValueError, match="Weak staged argument"):
        program(value)
