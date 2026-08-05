from __future__ import annotations

from copy import deepcopy

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad
from advect.core._registry import get_registry


def test_stage_traces_abstract_array_api_and_executes_without_retracing() -> None:
    calls = 0

    def energy(x: object) -> object:
        nonlocal calls
        calls += 1
        xp = x.__array_namespace__()
        centered = x - xp.mean(x)
        return xp.sum(centered * centered)

    program = ad.stage(energy, specs=(ad.ArraySpec((3,), "float32"),))
    assert calls == 1
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(program(x), 2.0)
    np.testing.assert_allclose(program(2 * x), 8.0)
    assert calls == 1
    assert program.graph.node_count > 0
    assert program.compile_seconds > 0


def test_stage_infers_positional_signature_from_example_arguments() -> None:
    example = np.arange(3, dtype=np.float32)

    program = ad.stage(
        lambda x, scale, direction: x * scale if direction == "forward" else -x,
        example,
        2.0,
        ad.StaticSpec("forward"),
    )

    assert program.signature == (
        (
            ad.ArraySpec((3,), "float32", device="cpu"),
            ad.ArraySpec((), "float64", weak=True),
            ad.StaticSpec("forward"),
        ),
        {},
    )
    np.testing.assert_array_equal(
        program(
            np.array([2.0, 3.0, 4.0], dtype=np.float32),
            3.0,
            "forward",
        ),
        np.array([6.0, 9.0, 12.0], dtype=np.float32),
    )


def test_stage_requires_an_explicit_signature() -> None:
    with pytest.raises(
        TypeError,
        match=r"requires example arguments or specs=.*single-signature",
    ):
        ad.stage(lambda x: x)

    with pytest.raises(TypeError, match=r"example arguments or specs=.*not both"):
        ad.stage(
            lambda x: x,
            np.ones(2, dtype=np.float32),
            specs=(ad.ArraySpec((2,), "float32"),),
        )


def test_staged_program_rejects_a_different_signature_without_retracing() -> None:
    program = ad.stage(lambda x: x + 1, specs=(ad.ArraySpec((2,), "float32"),))

    with pytest.raises(ValueError, match=r"expected shape=\(2,\).*got shape=\(3,\)"):
        program(np.ones(3, dtype=np.float32))


def test_staged_program_exposes_one_detached_positional_and_keyword_signature() -> None:
    positional_spec = ad.ArraySpec((2,), np.dtype("float32"))
    keyword_spec = ad.ArraySpec((), "float64", weak=True)
    program = ad.stage(
        lambda value, *, scale: value * scale,
        specs=(positional_spec,),
        kw_specs={"scale": keyword_spec},
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())

    expected = (
        (ad.ArraySpec((2,), "float32"),),
        {"scale": keyword_spec},
    )
    assert program.signature == expected
    assert restored.signature == expected


def test_stage_preserves_repeated_output_leaves() -> None:
    program = ad.stage(
        lambda x: (x, x),
        specs=(ad.ArraySpec((3,), "float32"),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    value = np.arange(3, dtype=np.float32)

    left, right = restored(value)

    assert restored.graph.outputs == [0, 0]
    assert left is right
    np.testing.assert_array_equal(left, value)


def test_stage_preserves_zero_leaf_output_through_serialization() -> None:
    program = ad.stage(
        lambda _x: (),
        specs=(ad.ArraySpec((3,), "float32"),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    value = np.arange(3, dtype=np.float32)

    assert program.graph.outputs == []
    assert program(value) == ()
    assert restored(value) == ()
    assert restored.to_dict() == program.to_dict()


def test_stage_reports_captured_constants() -> None:
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    program = ad.stage(
        lambda x: np.sum(x * kernel),
        specs=(ad.ArraySpec((3,), "float32"),),
    )
    records = program.constants
    assert len(records) == 1
    assert sum(record.bytes for record in records) == kernel.nbytes
    assert records[0].origin == "closure"
    assert records[0].shape == (3,)
    assert records[0].bytes == kernel.nbytes
    assert len(records[0].digest) == 64


def test_stage_optimizes_once_and_reports_remapped_graph() -> None:
    def redundant(x: object) -> object:
        left = x + 1
        right = x + 1
        _unused = x * 2
        return left + right

    program = ad.stage(
        redundant,
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    report = program.optimization

    assert report.nodes_after == program.graph.node_count
    assert report == ad.OptimizationReport(
        nodes_before=7,
        nodes_after=4,
        rewritten_nodes=3,
        passes=(
            ad.OptimizationPass("dce", 7, 5, 2, 2),
            ad.OptimizationPass("simplify", 5, 5, 0, 0),
            ad.OptimizationPass("cse", 5, 4, 1, 1),
        ),
    )
    assert len(program.constants) == 1
    assert program.constants[0].value_id == 1

    restored = ad.StagedProgram.from_dict(program.to_dict())
    assert restored.optimization == report
    value = np.array([2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(restored(value), 2 * (value + 1))


def test_stage_exposes_the_pre_optimization_trace() -> None:
    def redundant(x: object) -> object:
        left = x + 1
        right = x + 1
        _unused = x * 2
        return left + right

    program = ad.stage(redundant, specs=(ad.ArraySpec((2,), "float32"),))
    trace = program.trace

    assert trace is not None
    assert len(trace.nodes) == program.optimization.nodes_before
    assert len(trace.old_to_new) == len(trace.nodes)
    assert trace.nodes[0] == ad.TracedNode(id=0, op="advect.input", inputs=(), name="arg0")
    assert trace.old_to_new == (0, 1, 2, 2, None, None, 3)
    assert [node.op for node in trace.nodes] == [
        "advect.input",
        "advect.const",
        "array.add",
        "array.add",
        "advect.const",
        "array.multiply",
        "array.add",
    ]
    survivors = {target for target in trace.old_to_new if target is not None}
    assert len(survivors) == program.optimization.nodes_after
    # captured constants are reported in tape numbering, including dropped ones
    assert [record.value_id for record in trace.constants] == [1, 4]
    # the trace is an in-process staging byproduct; loaded artifacts have none
    assert ad.StagedProgram.from_dict(program.to_dict()).trace is None


def test_staged_transforms_expose_their_own_trace() -> None:
    program = ad.stage(lambda x: x * x, specs=(ad.ArraySpec((2,), "float32"),))
    pullback = ad.vjp_program(program)
    assert pullback.trace is not None
    assert len(pullback.trace.nodes) == pullback.optimization.nodes_before


def test_loaded_stage_is_not_reoptimized() -> None:
    program = ad.stage(
        lambda x: x + 1,
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    payload = deepcopy(program.to_dict())
    artifact = payload["program"]
    graph = artifact["graph"]
    dead_id = len(graph["nodes"])
    graph["nodes"].append(
        {
            "id": dead_id,
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
    artifact["optimization"] = {
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
    np.testing.assert_array_equal(
        restored(np.array([2.0, 3.0], dtype=np.float32)),
        np.array([3.0, 4.0], dtype=np.float32),
    )


def test_custom_primitives_are_optimization_barriers() -> None:
    calls = 0

    @ad.primitive(name="tests.stage_optimizer_barrier")
    def primitive(x: object) -> object:
        nonlocal calls
        calls += 1
        return x

    @primitive.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    def repeated(x: object) -> object:
        left = primitive(x)
        right = primitive(x)
        primitive(x)
        return left + right

    program = ad.stage(
        repeated,
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    custom_nodes = [
        program.graph.get_node(node_id)
        for node_id in program.graph.node_ids()
        if program.graph.get_node(node_id).op == primitive.op_name
    ]

    assert len(custom_nodes) == 3
    assert program.optimization.nodes_before == program.optimization.nodes_after == 5
    value = np.array([2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(program(value), 2 * value)
    assert calls == 3


def test_stage_rejects_data_dependent_control_flow() -> None:
    with pytest.raises(ad.TracingError, match="control flow"):
        ad.stage(
            lambda x: x if x.sum() > 0 else -x,
            np.ones(2, dtype=np.float32),
        )


def test_stage_uses_custom_primitive_abstract_rule() -> None:
    @ad.primitive(name="tests.stage_double")
    def double(x: object) -> object:
        return x * 2

    @double.def_abstract
    def double_abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    program = ad.stage(
        lambda x: double(x),
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    result = program(np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(result, np.array([4.0, 6.0], dtype=np.float32))
    custom_node = next(
        node
        for node in program.to_dict()["program"]["graph"]["nodes"]
        if node["op"] == double.op_name
    )
    assert custom_node["schema_version"] == 1
    assert set(custom_node["attrs"]) == {"__advect_primitive_call__"}

    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_array_equal(
        restored(np.array([3.0, 4.0], dtype=np.float32)),
        np.array([6.0, 8.0], dtype=np.float32),
    )


def test_staged_primitive_preserves_nested_and_keyword_only_call_structure() -> None:
    @ad.primitive(
        name="tests.stage_nested_call",
        static_argnames=("scale",),
        nondiff_argnames=("bias",),
    )
    def combine(
        values: dict[str, tuple[object, ...]],
        *,
        bias: object,
        scale: float,
    ) -> dict[str, object]:
        left, right = values["operands"]
        return {
            "sum": scale * (left + right) + bias,
            "difference": left - right,
        }

    @combine.def_abstract
    def combine_abstract(
        values: dict[str, tuple[ad.AbstractValue, ...]],
        *,
        bias: ad.AbstractValue,
        scale: float,
    ) -> dict[str, ad.ArraySpec]:
        del bias, scale
        left, _right = values["operands"]
        return {"sum": left.spec, "difference": left.spec}

    program = ad.stage(
        lambda left, right, bias: combine(
            {"operands": (left, right)},
            bias=bias,
            scale=2.0,
        ),
        specs=(
            ad.ArraySpec((2,), "float32"),
            ad.ArraySpec((2,), "float32"),
            ad.ArraySpec((2,), "float32"),
        ),
    )
    payload = program.to_dict()
    custom_node = next(
        node for node in payload["program"]["graph"]["nodes"] if node["op"] == combine.op_name
    )
    assert "__advect_primitive_call__" in custom_node["attrs"]
    assert "_advect_primitive_output_treedef" not in custom_node["attrs"]

    restored = ad.StagedProgram.from_dict(payload)
    left = np.array([2.0, 3.0], dtype=np.float32)
    right = np.array([0.5, 1.0], dtype=np.float32)
    bias = np.array([10.0, 20.0], dtype=np.float32)
    result = restored(left, right, bias)

    np.testing.assert_array_equal(result["sum"], 2 * (left + right) + bias)
    np.testing.assert_array_equal(result["difference"], left - right)


def test_staged_multi_output_primitive_round_trips_flat_node_outputs() -> None:
    @ad.primitive(name="tests.stage_pair")
    def pair(x: object) -> dict[str, object]:
        return {"double": x * 2, "square": x * x}

    @pair.def_abstract
    def pair_abstract(x: ad.AbstractValue) -> dict[str, ad.ArraySpec]:
        return {"double": x.spec, "square": x.spec}

    program = ad.stage(
        lambda x: pair(x),
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    get_registry().update_num_outputs(pair.op_name, num_outputs=1)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    value = np.array([2.0, 3.0], dtype=np.float32)
    result = restored(value)

    np.testing.assert_array_equal(result["double"], np.array([4.0, 6.0]))
    np.testing.assert_array_equal(result["square"], np.array([4.0, 9.0]))


def test_staged_multi_output_primitive_validates_every_result() -> None:
    @ad.primitive(name="tests.stage_bad_pair")
    def pair(x: object) -> tuple[object, object]:
        return x, x[:1]

    @pair.def_abstract
    def pair_abstract(x: ad.AbstractValue) -> tuple[ad.ArraySpec, ad.ArraySpec]:
        return x.spec, x.spec

    program = ad.stage(
        lambda x: pair(x),
        specs=(ad.ArraySpec((2,), "float32"),),
    )

    with pytest.raises(ValueError, match=r"produced shape=\(1,\), dtype=float32"):
        program(np.array([2.0, 3.0], dtype=np.float32))


def test_failed_staged_load_rolls_back_custom_output_arity() -> None:
    @ad.primitive(name="tests.stage_pair_rollback")
    def pair(x: object) -> tuple[object, object]:
        return x, x

    @pair.def_abstract
    def pair_abstract(x: ad.AbstractValue) -> tuple[ad.ArraySpec, ad.ArraySpec]:
        return x.spec, x.spec

    program = ad.stage(lambda x: pair(x), specs=(ad.ArraySpec((2,), "float32"),))
    payload = deepcopy(program.to_dict())
    get_registry().update_num_outputs(pair.op_name, num_outputs=1)
    payload["program"]["graph"]["version"] = "invalid"

    with pytest.raises(ValueError, match="Unsupported graph version"):
        ad.StagedProgram.from_dict(payload)
    assert get_registry().get(pair.op_name).num_outputs == 1


def test_custom_primitive_output_order_must_match_abstract_structure() -> None:
    @ad.primitive(name="tests.stage_output_structure")
    def pair(x: object) -> dict[str, object]:
        return {"right": x + 20, "left": x + 10}

    @pair.def_abstract
    def pair_abstract(x: ad.AbstractValue) -> dict[str, ad.ArraySpec]:
        return {"left": x.spec, "right": x.spec}

    program = ad.stage(
        lambda x: pair(x)["left"],
        specs=(ad.ArraySpec((1,), "float32"),),
    )
    with pytest.raises(ValueError, match="different structure"):
        program(np.asarray([1.0], dtype=np.float32))


def test_staged_load_rejects_custom_nodes_missing_call_metadata() -> None:
    @ad.primitive(name="tests.stage_missing_link")
    def identity(x: object) -> object:
        return x

    @identity.def_abstract
    def identity_abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    program = ad.stage(lambda x: identity(x), specs=(ad.ArraySpec((1,), "float32"),))
    payload = deepcopy(program.to_dict())
    artifact = payload["program"]
    custom_node = next(
        node for node in artifact["graph"]["nodes"] if node["op"] == identity.op_name
    )
    del custom_node["attrs"]["__advect_primitive_call__"]

    with pytest.raises(ValueError, match="invalid call contract"):
        ad.StagedProgram.from_dict(payload)


def test_staged_program_envelope_contains_exactly_one_program() -> None:
    program = ad.stage(lambda x: x + 1, specs=(ad.ArraySpec((2,), "float32"),))

    payload = program.to_dict()

    assert set(payload) == {"format", "version", "program"}
    assert payload["format"] == "advect.ssa-program"
    assert payload["version"] == 2
    assert isinstance(payload["program"], dict)
    assert "version" not in payload["program"]
    assert payload["program"]["graph"]["semantic_profile"] == "advect-array-1"
    assert payload["program"]["graph"]["semantic_profile_version"] == 1
    assert payload["program"]["graph"]["required_array_api_version"] == "2024.12"


def test_abstract_tracer_cannot_escape() -> None:
    leaked: list[object] = []

    def leak(x: object) -> object:
        leaked.append(x)
        return x + 1

    ad.stage(leak, specs=(ad.ArraySpec((1,), "float32"),))
    with pytest.raises(ad.TracingError, match="escaped"):
        _ = leaked[0].shape


def test_staged_program_round_trips_without_python_function() -> None:
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    program = ad.stage(
        lambda x: np.sum(x * kernel),
        specs=(ad.ArraySpec((3,), "float32"),),
    )

    payload = program.to_dict()
    restored = ad.StagedProgram.from_dict(payload)

    x = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    np.testing.assert_allclose(restored(x), 13.0)
    assert restored.constants == program.constants
    assert restored.to_dict() == payload


def test_dynamic_grad_composes_with_loaded_staged_program() -> None:
    program = ad.stage(
        lambda x: np.sum(x * x),
        specs=(ad.ArraySpec((3,), "float32"),),
    )
    restored = ad.StagedProgram.from_dict(program.to_dict())
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    np.testing.assert_allclose(ad.grad(restored)(x), 2 * x)


def test_stage_round_trip_executes_strict_array_api_shape_and_complex_ops() -> None:
    def transform(x: object) -> object:
        xp = x.__array_namespace__()
        matrix = xp.reshape(1j * x, (2, 2))
        return xp.sum(xp.real(xp.conj(xp.permute_dims(matrix, (1, 0)))))

    program = ad.stage(transform, specs=(ad.ArraySpec((4,), "float32"),))
    restored = ad.StagedProgram.from_dict(program.to_dict())
    value = strict.arange(4, dtype=strict.float32)
    result = restored(value)

    assert result.dtype == strict.float32
    assert float(result) == 0.0


@pytest.mark.parametrize("version", [1, 999])
def test_staged_program_rejects_other_versions_before_loading_graph(version: int) -> None:
    program = ad.stage(
        lambda x: x + 1,
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    payload = deepcopy(program.to_dict())
    payload["version"] = version

    with pytest.raises(ValueError, match=f"Unsupported staged program format version {version}"):
        ad.StagedProgram.from_dict(payload)


@pytest.mark.parametrize("field", ["compiler_version", "optimizer_version"])
def test_staged_program_rejects_unknown_compiler_versions(field: str) -> None:
    program = ad.stage(
        lambda x: x + 1,
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    payload = deepcopy(program.to_dict())
    payload["program"]["graph"][field] = 999

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        ad.StagedProgram.from_dict(payload)
