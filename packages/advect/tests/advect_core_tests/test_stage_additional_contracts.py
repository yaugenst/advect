"""Additional public durability contracts for staged programs."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest

import advect as ad


def _payload() -> dict[str, Any]:
    kernel = np.array([1.0, 2.0], dtype=np.float32)
    program = ad.stage(
        lambda x: x + kernel,
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    return cast("dict[str, Any]", json.loads(json.dumps(program.to_dict())))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("shape", [3], id="shape"),
        pytest.param("dtype", "float64", id="dtype"),
    ],
)
def test_loaded_program_rejects_call_specs_inconsistent_with_graph_inputs(
    field: str,
    value: object,
) -> None:
    payload = _payload()
    payload["program"]["call_specs"][0][field] = value

    with pytest.raises(ValueError, match="graph inputs do not match its call specs"):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("payload", "error", "match"),
    [
        pytest.param(None, TypeError, "must be a mapping", id="not-a-mapping"),
        pytest.param(
            {"format": "advect.ssa-program", "version": 2},
            ValueError,
            "invalid fields",
            id="missing-program",
        ),
        pytest.param(
            {"format": "future.program", "version": 2, "program": {}},
            ValueError,
            "Unknown staged program format",
            id="unknown-format",
        ),
        pytest.param(
            {"format": "advect.ssa-program", "version": "2", "program": {}},
            TypeError,
            "version must be an integer",
            id="noninteger-version",
        ),
    ],
)
def test_loaded_program_rejects_invalid_outer_envelopes(
    payload: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        pytest.param("op", "array.future_add", "is not registered", id="unknown-operation"),
        pytest.param("schema_version", 999, "linked schema is 1", id="schema-version"),
        pytest.param("num_outputs", 2, "expects num_outputs=1", id="output-arity"),
    ],
)
def test_loaded_program_rejects_incompatible_linked_operations(
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _payload()
    operation = payload["program"]["graph"]["nodes"][-1]
    operation[field] = value

    with pytest.raises(ValueError, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        pytest.param("pass-order", "pass sequence is invalid", id="pass-order"),
        pytest.param("aggregate", "aggregate counts are inconsistent", id="aggregate-count"),
        pytest.param("removed", "removed-node count is inconsistent", id="removed-count"),
    ],
)
def test_loaded_program_rejects_corrupted_optimization_report(
    corruption: str,
    match: str,
) -> None:
    payload = _payload()
    optimization = payload["program"]["optimization"]
    if corruption == "pass-order":
        optimization["passes"][0], optimization["passes"][1] = (
            optimization["passes"][1],
            optimization["passes"][0],
        )
    elif corruption == "aggregate":
        optimization["rewritten_nodes"] += 1
    else:
        optimization["passes"][0]["removed_nodes"] += 1

    with pytest.raises(ValueError, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        pytest.param("call-spec-count", "call specs do not match their pytree", id="call-tree"),
        pytest.param(
            "output-spec-count",
            "output specs do not match their pytree",
            id="output-tree",
        ),
        pytest.param(
            "graph-output-count",
            "graph output count does not match its output pytree",
            id="graph-outputs",
        ),
        pytest.param(
            "graph-node-count",
            "graph node count does not match its optimization report",
            id="graph-nodes",
        ),
    ],
)
def test_loaded_program_rejects_inconsistent_artifact_records(
    corruption: str,
    match: str,
) -> None:
    payload = _payload()
    artifact = payload["program"]
    if corruption == "call-spec-count":
        artifact["call_specs"].clear()
    elif corruption == "output-spec-count":
        artifact["output_specs"].clear()
    elif corruption == "graph-output-count":
        artifact["graph"]["outputs"].clear()
    else:
        graph = artifact["graph"]
        graph["nodes"].append(
            {
                "id": len(graph["nodes"]),
                "op": "array.negative",
                "schema_version": 1,
                "inputs": [graph["inputs"][0]],
                "attrs": {},
                "shape": [2],
                "dtype": "float32",
                "num_outputs": 1,
                "output_shapes": None,
                "output_dtypes": None,
                "name": None,
                "source_location": None,
            }
        )

    with pytest.raises(ValueError, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        pytest.param("duplicate", "repeats value", id="duplicate-record"),
        pytest.param("bytes", "metadata does not match its payload", id="byte-count"),
        pytest.param("digest", "digest does not match its payload", id="digest"),
    ],
)
def test_loaded_program_rejects_inconsistent_constant_manifest(
    corruption: str,
    match: str,
) -> None:
    payload = _payload()
    constants = payload["program"]["constants"]
    if corruption == "duplicate":
        constants.append(deepcopy(constants[0]))
    elif corruption == "bytes":
        constants[0]["bytes"] += 1
    else:
        constants[0]["digest"] = "0" * 64

    with pytest.raises(ValueError, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("corruption", "error", "match"),
    [
        pytest.param("report-type", TypeError, "report must be a mapping", id="report-type"),
        pytest.param("report-fields", ValueError, "report has invalid fields", id="report-fields"),
        pytest.param(
            "report-count",
            TypeError,
            "must be a non-negative integer",
            id="report-count",
        ),
        pytest.param("passes-type", TypeError, "passes must be a list", id="passes-type"),
        pytest.param("pass-type", TypeError, "pass must be a mapping", id="pass-type"),
        pytest.param("pass-fields", ValueError, "pass has invalid fields", id="pass-fields"),
        pytest.param("pass-name", TypeError, "pass name must be a string", id="pass-name"),
        pytest.param(
            "pass-count",
            TypeError,
            "must be a non-negative integer",
            id="pass-count",
        ),
    ],
)
def test_loaded_program_rejects_structurally_invalid_optimization_report(
    corruption: str,
    error: type[Exception],
    match: str,
) -> None:
    payload = _payload()
    artifact = payload["program"]
    if corruption == "report-type":
        artifact["optimization"] = []
    else:
        report = artifact["optimization"]
        if corruption == "report-fields":
            report["extra"] = 0
        elif corruption == "report-count":
            report["nodes_before"] = -1
        elif corruption == "passes-type":
            report["passes"] = {}
        elif corruption == "pass-type":
            report["passes"][0] = []
        elif corruption == "pass-fields":
            report["passes"][0]["extra"] = 0
        elif corruption == "pass-name":
            report["passes"][0]["name"] = 1
        else:
            report["passes"][0]["rewritten_nodes"] = -1

    with pytest.raises(error, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("spec", "error", "match"),
    [
        pytest.param(None, TypeError, "spec must be a mapping", id="not-a-mapping"),
        pytest.param(
            {"kind": "array", "shape": [2]},
            ValueError,
            "array spec has invalid fields",
            id="array-fields",
        ),
        pytest.param(
            {"kind": "array", "shape": "2", "dtype": "float32", "device": None, "weak": False},
            TypeError,
            "shape must be a list of integers",
            id="shape",
        ),
        pytest.param(
            {"kind": "array", "shape": [2], "dtype": 32, "device": None, "weak": False},
            TypeError,
            "dtype must be a string",
            id="dtype",
        ),
        pytest.param(
            {"kind": "array", "shape": [2], "dtype": "float32", "device": 0, "weak": False},
            TypeError,
            "device must be a string or None",
            id="device",
        ),
        pytest.param(
            {"kind": "array", "shape": [2], "dtype": "float32", "device": None, "weak": 0},
            TypeError,
            "weak flag must be a bool",
            id="weak",
        ),
        pytest.param(
            {"kind": "static", "value": {"kind": "scalar", "value": 1}, "extra": True},
            ValueError,
            "static spec has invalid fields",
            id="static-fields",
        ),
        pytest.param(
            {"kind": "future"},
            ValueError,
            "Unknown staged call spec kind",
            id="unknown-kind",
        ),
    ],
)
def test_loaded_program_rejects_malformed_call_specs(
    spec: object,
    error: type[Exception],
    match: str,
) -> None:
    payload = _payload()
    payload["program"]["call_specs"][0] = spec

    with pytest.raises(error, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("corruption", "field", "value", "error", "match"),
    [
        pytest.param(
            "record-type",
            None,
            None,
            TypeError,
            "record must be a mapping",
            id="record-type",
        ),
        pytest.param(
            "record-fields",
            None,
            None,
            ValueError,
            "record has invalid fields",
            id="record-fields",
        ),
        pytest.param("field", "value_id", -1, TypeError, "value_id must be", id="value-id"),
        pytest.param("field", "origin", "future", ValueError, "origin must be", id="origin"),
        pytest.param("field", "location", 1, TypeError, "location must be", id="location"),
        pytest.param("field", "shape", [-1], TypeError, "shape must be", id="shape"),
        pytest.param("field", "dtype", "", TypeError, "dtype must be", id="dtype"),
        pytest.param("field", "bytes", -1, TypeError, "bytes must be", id="bytes"),
        pytest.param("field", "digest", "ABC", TypeError, "digest must be", id="digest"),
        pytest.param("field", "name", 1, TypeError, "name must be", id="name"),
    ],
)
def test_loaded_program_rejects_malformed_constant_records(
    corruption: str,
    field: str | None,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    payload = _payload()
    constants = payload["program"]["constants"]
    if corruption == "record-type":
        constants[0] = None
    elif corruption == "record-fields":
        constants[0]["extra"] = True
    else:
        assert field is not None
        constants[0][field] = value

    with pytest.raises(error, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("corruption", "error", "match"),
    [
        pytest.param("artifact-type", TypeError, "artifact must be a mapping", id="artifact-type"),
        pytest.param("artifact-fields", ValueError, "artifact has invalid fields", id="fields"),
        pytest.param("call-specs", TypeError, "call_specs must be a list", id="call-specs"),
        pytest.param("output-specs", TypeError, "output_specs must be a list", id="output-specs"),
        pytest.param("constants", TypeError, "constants must be a list", id="constants"),
        pytest.param(
            "static-output",
            TypeError,
            "output specs must all be array specs",
            id="static-output",
        ),
    ],
)
def test_loaded_program_rejects_malformed_artifact_structure(
    corruption: str,
    error: type[Exception],
    match: str,
) -> None:
    payload = _payload()
    if corruption == "artifact-type":
        payload["program"] = []
    else:
        artifact = payload["program"]
        if corruption == "artifact-fields":
            artifact["extra"] = True
        elif corruption == "static-output":
            artifact["output_specs"][0] = {
                "kind": "static",
                "value": {"kind": "scalar", "value": 1},
            }
        else:
            artifact[corruption.replace("-", "_")] = {}

    with pytest.raises(error, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("corruption", "error", "match"),
    [
        pytest.param("graph-type", TypeError, "graph payload must be a mapping", id="graph-type"),
        pytest.param("nodes-type", TypeError, "graph nodes must be a list", id="nodes-type"),
        pytest.param("node-type", TypeError, "graph node must be a mapping", id="node-type"),
        pytest.param("op", TypeError, "node op must be a string", id="op"),
        pytest.param("schema_version", TypeError, "schema_version must be", id="schema"),
        pytest.param("num_outputs", TypeError, "num_outputs must be", id="outputs"),
        pytest.param(
            "unlinked-custom",
            ValueError,
            "requires unlinked primitive 'tests.missing'",
            id="unlinked-custom",
        ),
    ],
)
def test_loaded_program_rejects_malformed_graph_linkage(
    corruption: str,
    error: type[Exception],
    match: str,
) -> None:
    payload = _payload()
    artifact = payload["program"]
    if corruption == "graph-type":
        artifact["graph"] = []
    else:
        graph = artifact["graph"]
        if corruption == "nodes-type":
            graph["nodes"] = {}
        elif corruption == "node-type":
            graph["nodes"][0] = []
        else:
            operation = graph["nodes"][-1]
            field = "op" if corruption == "unlinked-custom" else corruption
            operation[field] = {
                "op": None,
                "schema_version": 0,
                "num_outputs": 0,
                "unlinked-custom": "custom.tests.missing",
            }[corruption]

    with pytest.raises(error, match=match):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        pytest.param(True, "bool", id="bool"),
        pytest.param(2, "int64", id="int"),
        pytest.param(1.0 + 2.0j, "complex128", id="complex"),
    ],
)
def test_stage_infers_and_replays_python_scalar_categories(value: object, dtype: str) -> None:
    program = ad.stage(lambda item: item, value)

    assert program.signature == ((ad.ArraySpec((), dtype, weak=True),), {})
    assert program(value) == value


def test_stage_rejects_unclassified_examples_and_specs() -> None:
    with pytest.raises(TypeError, match="declare non-array inputs with StaticSpec"):
        ad.stage(lambda item: item, "dynamic")

    with pytest.raises(TypeError, match="specs must contain ArraySpec or StaticSpec"):
        ad.stage(lambda item: item, specs=(object(),))


@pytest.mark.parametrize(
    ("dtype", "value", "match"),
    [
        pytest.param("bool", 1, "expected dtype=bool", id="bool"),
        pytest.param("complex128", True, "got bool", id="complex"),
        pytest.param("float64", 1.0j, "expected a real scalar", id="float"),
        pytest.param("int64", 1.0, "expected an integer", id="int"),
    ],
)
def test_staged_call_rejects_changed_weak_scalar_category(
    dtype: str,
    value: object,
    match: str,
) -> None:
    program = ad.stage(
        lambda item: item,
        specs=(ad.ArraySpec((), dtype, weak=True),),
    )

    with pytest.raises(ValueError, match=match):
        program(value)


def test_staged_call_rejects_changed_nested_call_structure() -> None:
    program = ad.stage(
        lambda items: items["value"],
        specs=({"value": ad.ArraySpec((1,), "float32")},),
    )
    value = np.ones(1, dtype=np.float32)

    with pytest.raises(TypeError, match="pytree differs from the declared specs"):
        program([value])
    with pytest.raises(TypeError, match="pytree differs from the declared specs"):
        program({"other": value})


def test_loaded_custom_primitive_rejects_output_structure_arity_mismatch() -> None:
    @ad.primitive(name="tests.additional_stage_pair")
    def pair(value: object) -> tuple[object, object]:
        return value, value

    @pair.def_abstract
    def pair_abstract(value: ad.AbstractValue) -> tuple[ad.ArraySpec, ad.ArraySpec]:
        return value.spec, value.spec

    program = ad.stage(
        lambda value: pair(value),  # noqa: PLW0108 - explicit trace boundary
        specs=(ad.ArraySpec((1,), "float32"),),
    )
    payload = cast("dict[str, Any]", deepcopy(program.to_dict()))
    node = next(item for item in payload["program"]["graph"]["nodes"] if item["op"] == pair.op_name)
    call_meta = node["attrs"]["__advect_primitive_call__"]["value"]
    call_meta["output_treedef"] = deepcopy(call_meta["call_treedef"])

    with pytest.raises(ValueError, match="output structure does not match its arity"):
        ad.StagedProgram.from_dict(payload)


def test_staged_call_allows_multiple_devices_when_no_constants_are_materialized() -> None:
    class Namespace:
        __name__ = "tests_multi_device"
        __array_api_version__ = "2024.12"

        @staticmethod
        def __array_namespace_info__() -> object:
            return object()

        @staticmethod
        def asarray(value: object) -> object:
            return value

    namespace = Namespace()

    class Array:
        shape = (1,)
        dtype = "float64"

        def __init__(self, device: str) -> None:
            self.device = device

        def __array_namespace__(self, *, api_version: str | None = None) -> object:
            assert api_version == "2024.12"
            return namespace

    program = ad.stage(
        lambda left, right: (left, right),
        specs=(ad.ArraySpec((1,), "float64"), ad.ArraySpec((1,), "float64")),
    )
    left = Array("cpu:0")
    right = Array("cpu:1")

    assert program(left, right) == (left, right)

    constant = np.ones(1)
    with_constant = ad.stage(
        lambda first, second: (first, second, constant),
        specs=(ad.ArraySpec((1,), "float64"), ad.ArraySpec((1,), "float64")),
    )
    with pytest.raises(TypeError, match="cannot materialize constants across multiple devices"):
        with_constant(left, right)


@pytest.mark.parametrize(
    ("kind", "error", "match"),
    [
        pytest.param(
            "missing",
            ad.MissingPrimitiveRuleError,
            "missing 'abstract'",
            id="missing",
        ),
        pytest.param("empty", TypeError, "returned no values", id="empty"),
        pytest.param("invalid-leaf", TypeError, "must return ArraySpec", id="invalid-leaf"),
    ],
)
def test_stage_rejects_invalid_primitive_abstract_contracts(
    kind: str,
    error: type[Exception],
    match: str,
) -> None:
    @ad.primitive(name=f"tests.additional_stage_{kind}")
    def identity(value: object) -> object:
        return value

    if kind != "missing":

        @identity.def_abstract
        def identity_abstract(value: ad.AbstractValue) -> object:
            del value
            return () if kind == "empty" else object()

    with pytest.raises(error, match=match):
        ad.stage(identity, specs=(ad.ArraySpec((1,), "float32"),))


def test_stage_rejects_untraceable_dynamic_primitive_arguments() -> None:
    @ad.primitive(name="tests.additional_stage_dynamic_config")
    def identity(value: object, config: object) -> object:
        del config
        return value

    with pytest.raises(TypeError, match="argument 'config' is not traceable"):
        ad.stage(
            lambda value: identity(value, config=object()),
            specs=(ad.ArraySpec((1,), "float32"),),
        )


def test_stage_folds_concrete_primitive_calls_into_captured_constants() -> None:
    @ad.primitive(name="tests.additional_stage_concrete_primitive")
    def double(value: int) -> int:
        return 2 * value

    program = ad.stage(
        lambda value: value + double(2),
        specs=(ad.ArraySpec((1,), "int64"),),
    )

    np.testing.assert_array_equal(program(np.array([3], dtype=np.int64)), np.array([7]))


def test_stage_accepts_bound_methods_with_captured_array_state() -> None:
    class Model:
        def __init__(self) -> None:
            self.offset = np.array([1.0, 2.0], dtype=np.float32)

        def apply(self, value: object) -> object:
            return value + self.offset

    model = Model()
    program = ad.stage(model.apply, specs=(ad.ArraySpec((2,), "float32"),))
    model.offset[:] = 0

    np.testing.assert_array_equal(
        program(np.array([3.0, 4.0], dtype=np.float32)),
        np.array([4.0, 6.0], dtype=np.float32),
    )


def test_stage_accepts_functions_with_empty_closure_cells() -> None:
    def make_function() -> Any:
        captured = object()

        def identity(value: object) -> object:
            return captured if value is None else value

        assert identity.__closure__ is not None
        del identity.__closure__[0].cell_contents
        return identity

    program = ad.stage(
        make_function(),
        specs=(ad.ArraySpec((1,), "float32"),),
    )

    value = np.array([2.0], dtype=np.float32)
    np.testing.assert_array_equal(program(value), value)
