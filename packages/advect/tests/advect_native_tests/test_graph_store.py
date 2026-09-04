"""Tests for native graph building, validation, and durable storage."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from advect import _native_core as advect_native
from advect.core._portable_constant import snapshot_constant_parts

if TYPE_CHECKING:
    from advect.core._portable_constant import _PortableConstant


def _append_constant(
    builder: advect_native.GraphBuilder,
    constant: _PortableConstant,
) -> int:
    node_id, digest = builder.append_constant(
        constant.data,
        list(constant.shape),
        constant.dtype,
        kind=constant.kind,
    )
    assert digest == constant.digest
    return node_id


def _finish(
    builder: advect_native.GraphBuilder,
) -> tuple[advect_native.GraphStore, list[int | None], dict[str, object]]:
    store, old_to_new, report, trace = builder.finish()
    # every finish exposes the raw tape alongside the dense remap
    assert len(trace) == len(old_to_new)
    assert all(isinstance(row[0], int) and isinstance(row[1], str) for row in trace)
    return store, old_to_new, report


def test_builder_finalizes_dense_topology_once() -> None:
    builder = advect_native.GraphBuilder()
    left = builder.append_input_node([2], "float64", name="left")
    right = builder.append_input_node([2], "float64", name="right")
    output = builder.append_node("array.add", [left, right], {}, [2], "float64")
    builder.append_output(output)

    store, old_to_new, report = _finish(builder)
    assert store.inputs == [left, right]
    assert store.node_ids() == [0, 1, 2]
    assert old_to_new == [0, 1, 2]
    assert report["nodes_before"] == report["nodes_after"] == 3
    with pytest.raises(RuntimeError, match="already finished"):
        builder.finish()


def test_finish_transfers_nary_arena_without_changing_nodes() -> None:
    builder = advect_native.GraphBuilder()
    inputs = [
        builder.append_input_node([2], "float64", name=f"input_{index}") for index in range(3)
    ]
    output = builder.append_node("array.stack", inputs, {"axis": 0}, [3, 2], "float64")
    builder.append_output(output)

    store, old_to_new, report = _finish(builder)

    assert store.get_node(output).inputs == inputs
    assert store.get_node(output).attrs == {"axis": 0}
    assert old_to_new == [0, 1, 2, 3]
    assert report["rewritten_nodes"] == 0


def test_finish_runs_fixed_optimizer_and_reports_dense_remap() -> None:
    builder = advect_native.GraphBuilder()
    input_id = builder.append_input_node([2], "float32")
    first_sin = builder.append_node(
        "array.sin",
        [input_id],
        {},
        [2],
        "float32",
        schema_version=7,
    )
    duplicate_sin = builder.append_node(
        "array.sin",
        [input_id],
        {},
        [2],
        "float32",
        schema_version=7,
    )
    dead_cos = builder.append_node("array.cos", [input_id], {}, [2], "float32")
    output_id = builder.append_node(
        "array.add",
        [first_sin, duplicate_sin],
        {},
        [2],
        "float32",
    )
    builder.append_output(output_id)

    store, old_to_new, report = _finish(builder)

    assert old_to_new == [0, 1, 1, None, 2]
    assert old_to_new[dead_cos] is None
    assert old_to_new[duplicate_sin] == old_to_new[first_sin]
    assert store.node_ids() == [0, 1, 2]
    assert store.outputs == [2]
    assert store.get_node(1).schema_version == 7
    assert store.get_node(2).inputs == [1, 1]
    assert report == {
        "nodes_before": 5,
        "nodes_after": 3,
        "rewritten_nodes": 2,
        "passes": [
            {
                "name": "dce",
                "nodes_before": 5,
                "nodes_after": 4,
                "removed_nodes": 1,
                "rewritten_nodes": 1,
            },
            {
                "name": "simplify",
                "nodes_before": 4,
                "nodes_after": 4,
                "removed_nodes": 0,
                "rewritten_nodes": 0,
            },
            {
                "name": "cse",
                "nodes_before": 4,
                "nodes_after": 3,
                "removed_nodes": 1,
                "rewritten_nodes": 1,
            },
        ],
    }


def test_builder_assigns_dense_ids_and_rejects_forward_references() -> None:
    builder = advect_native.GraphBuilder()
    first = builder.append_input_node([2], "float32")
    second = builder.append_node("array.sin", [first], {}, [2], "float32")
    assert (first, second) == (0, 1)
    with pytest.raises(ValueError, match="only earlier nodes"):
        builder.append_node("array.sin", [9], {}, [2], "float32")


def test_node_metadata_and_attrs_are_owned_snapshots() -> None:
    attrs = {
        "axis": (0,),
        "nested": [{"values": [1, 2]}],
        "negative_zero": -0.0,
        "nan": float("nan"),
    }
    builder = advect_native.GraphBuilder()
    node_id = builder.append_node("array.sum", [], attrs, [1], "float64", name="total")
    builder.append_output(node_id)
    attrs["nested"][0]["values"].append(3)
    store, _, _ = _finish(builder)

    first = store.get_node(node_id)
    assert first.name == "total"
    assert first.attrs["axis"] == (0,)
    assert first.attrs["nested"] == [{"values": [1, 2]}]
    assert math.copysign(1.0, first.attrs["negative_zero"]) == -1.0
    assert math.isnan(first.attrs["nan"])
    first.attrs["nested"][0]["values"].append(99)
    assert store.get_node(node_id).attrs["nested"] == [{"values": [1, 2]}]


@pytest.mark.parametrize("unsupported", [object(), {1, 2}, 1 + 2j])
def test_attrs_reject_unsupported_objects(unsupported: object) -> None:
    with pytest.raises(TypeError, match="unsupported graph attribute value"):
        advect_native.GraphBuilder().append_node(
            "advect.input", [], {"value": unsupported}, [1], "float64"
        )


def test_dtype_descriptors_are_backend_neutral_strings() -> None:
    builder = advect_native.GraphBuilder()
    native_id = builder.append_input_node([1], np.dtype("float64"))
    endian_id = builder.append_input_node([1], np.dtype(">f8"))
    store, _, _ = _finish(builder)

    assert store.get_node(native_id).dtype == "float64"
    assert store.get_node(endian_id).dtype == ">f8"
    with pytest.raises(TypeError, match="unsupported dtype object"):
        advect_native.GraphBuilder().append_input_node([1], object())


def test_constants_are_portable_immutable_payloads() -> None:
    builder = advect_native.GraphBuilder()
    input_id = builder.append_input_node([], "float64")
    constant = snapshot_constant_parts(1.0, shape=(), dtype="float64")
    constant_id = _append_constant(builder, constant)
    output_id = builder.append_node("array.add", [input_id, constant_id], {}, [], "float64")
    builder.append_output(output_id)
    store, _, _ = _finish(builder)

    kind, dtype, shape, detached, digest = store._constant_parts(constant_id)
    assert (kind, dtype, shape, detached.hex(), digest) == (
        constant.kind,
        constant.dtype,
        list(constant.shape),
        constant.data.hex(),
        constant.digest,
    )
    assert isinstance(detached, bytes)
    assert store._constant_parts(constant_id)[3] == detached


def test_multi_output_metadata_is_validated_and_exposed() -> None:
    builder = advect_native.GraphBuilder()
    with pytest.raises(ValueError, match="output_shapes/output_dtypes are missing"):
        builder.append_node("array.modf", [], {}, [2], "float64", num_outputs=2)

    node_id = builder.append_node(
        "array.modf",
        [],
        {},
        [2],
        "float64",
        num_outputs=2,
        output_shapes=[[2], [2]],
        output_dtypes=["float64", "float64"],
    )
    builder.append_output(node_id)
    store, old_to_new, _ = _finish(builder)
    mapped_node_id = old_to_new[node_id]
    assert mapped_node_id is not None
    node = store.get_node(mapped_node_id)
    assert node.num_outputs == 2
    assert node.output_shapes == [[2], [2]]
    assert node.output_dtypes == ["float64", "float64"]


def test_graph_serialization_preserves_canonical_payload() -> None:
    builder = advect_native.GraphBuilder()
    input_id = builder.append_input_node([2], "float32", name="x")
    constant_id = _append_constant(
        builder,
        snapshot_constant_parts(2.0, shape=(), dtype="float32"),
    )
    output_id = builder.append_node(
        "array.multiply",
        [input_id, constant_id],
        {"scale": 2},
        [2],
        "float32",
        source_location="model.py:10",
    )
    builder.append_output(output_id)

    store, old_to_new, _ = _finish(builder)
    mapped_input_id = old_to_new[input_id]
    mapped_constant_id = old_to_new[constant_id]
    mapped_output_id = old_to_new[output_id]
    assert mapped_input_id is not None
    assert mapped_constant_id is not None
    assert mapped_output_id is not None
    encoded = store._to_json()
    payload = json.loads(encoded)
    assert payload["format"] == "advect.graph"
    assert payload["version"] == "2.0"
    assert payload["core_opset"] == 1
    assert payload["semantic_profile"] == "advect-array-1"
    assert payload["semantic_profile_version"] == 1
    assert payload["required_array_api_version"] == "2024.12"
    assert payload["optimizer_version"] == 2
    assert payload["inputs"] == [mapped_input_id]
    assert payload["outputs"] == [mapped_output_id]
    assert payload["constants"] == {
        str(mapped_constant_id): {
            "format": "advect.numeric-constant",
            "version": 2,
            "kind": "scalar",
            "dtype": "float32",
            "shape": [],
            "layout": "C",
            "byte_order": "little",
            "data": "00000040",
            "digest": "5deac6129385a16d0f217f7045512e8fd2d4875f16e9fe434c9022d711f967e4",
        }
    }
    assert payload["nodes"][mapped_output_id] == {
        "id": mapped_output_id,
        "op": "array.multiply",
        "schema_version": 1,
        "inputs": [mapped_input_id, mapped_constant_id],
        "attrs": {"scale": {"kind": "integer", "value": 2}},
        "shape": [2],
        "dtype": "float32",
        "num_outputs": 1,
        "output_shapes": None,
        "output_dtypes": None,
        "name": None,
        "source_location": "model.py:10",
    }
    restored = advect_native.deserialize_graph_json(encoded)
    assert restored._to_json() == encoded


def test_graph_schema_versions_survive_native_round_trip() -> None:
    builder = advect_native.GraphBuilder()
    input_id = builder.append_input_node([2], "float32")
    output_id = builder.append_node(
        "custom.versioned",
        [input_id],
        {},
        [2],
        "float32",
        schema_version=7,
    )
    builder.append_output(output_id)
    store, old_to_new, _ = _finish(builder)
    mapped_output_id = old_to_new[output_id]
    assert mapped_output_id is not None
    encoded = store._to_json()
    payload = json.loads(encoded)

    assert store.get_node(mapped_output_id).schema_version == 7
    assert payload["nodes"][mapped_output_id]["schema_version"] == 7
    restored = advect_native.deserialize_graph_json(encoded)
    assert restored.get_node(mapped_output_id).schema_version == 7
    assert restored._to_json() == encoded


def test_graph_schema_version_is_required_and_consistent_per_op() -> None:
    builder = advect_native.GraphBuilder()
    builder.append_node("custom.versioned", [], {}, [], "float32", schema_version=2)
    with pytest.raises(ValueError, match="already schema version 2"):
        builder.append_node("custom.versioned", [], {}, [], "float32", schema_version=3)

    store, _, _ = _finish(builder)
    payload = json.loads(store._to_json())
    del payload["nodes"][0]["schema_version"]
    with pytest.raises(ValueError, match=r"missing field.*schema_version"):
        advect_native.deserialize_graph_json(json.dumps(payload))


def test_native_staged_execution_binds_once_and_supports_constants_and_outputs() -> None:
    builder = advect_native.GraphBuilder()
    input_id = builder.append_input_node([2], "float32")
    constant_id = _append_constant(
        builder,
        snapshot_constant_parts(
            np.array([1.0, 2.0], dtype=np.float32),
            shape=(2,),
            dtype="float32",
        ),
    )
    pair_id = builder.append_node(
        "custom.pair",
        [input_id, constant_id],
        {"offset": 1},
        [2],
        "float32",
        num_outputs=2,
        output_shapes=[[2], [2]],
        output_dtypes=["float32", "float32"],
    )
    left_id = builder.append_node(
        "advect.getoutput",
        [pair_id],
        {"index": 0, "num_outputs": 2},
        [2],
        "float32",
    )
    right_id = builder.append_node(
        "advect.getoutput",
        [pair_id],
        {"index": 1, "num_outputs": 2},
        [2],
        "float32",
    )
    builder.append_output(left_id)
    builder.append_output(right_id)
    store, _, _ = _finish(builder)

    bound_ops: list[str] = []

    def bind(op: str, attrs: dict[str, object]) -> object:
        bound_ops.append(op)
        if op == "custom.pair":
            assert attrs == {"offset": 1}
            return lambda values, _context, _donation: (
                values[0] + values[1],
                values[0] * values[1],
            )
        index = int(attrs["index"])
        return lambda values, _context, _donation: values[0][index]

    plan = advect_native.build_graph_execution_plan(store, bind)
    assert bound_ops == ["custom.pair", "advect.getoutput", "advect.getoutput"]
    constant = np.array([1.0, 2.0], dtype=np.float32)
    first = advect_native.execute_graph(
        plan,
        [np.array([3.0, 4.0], dtype=np.float32)],
        [constant],
    )
    second = advect_native.execute_graph(
        plan,
        [np.array([5.0, 6.0], dtype=np.float32)],
        [constant],
    )

    np.testing.assert_array_equal(first[0], np.array([4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(first[1], np.array([3.0, 8.0], dtype=np.float32))
    np.testing.assert_array_equal(second[0], np.array([6.0, 8.0], dtype=np.float32))
    assert bound_ops == ["custom.pair", "advect.getoutput", "advect.getoutput"]


@pytest.mark.parametrize(
    ("result_shape", "result_dtype"),
    [((1,), "float32"), ((2,), "float64"), ((2,), ">f4")],
)
def test_native_staged_execution_validates_results_and_annotates_callback_errors(
    result_shape: tuple[int, ...], result_dtype: str
) -> None:
    builder = advect_native.GraphBuilder()
    input_id = builder.append_input_node([2], "float32")
    output_id = builder.append_node("custom.bad", [input_id], {}, [2], "float32")
    builder.append_output(output_id)
    store, _, _ = _finish(builder)

    invalid_result_plan = advect_native.build_graph_execution_plan(
        store,
        lambda _op, _attrs: (
            lambda _values, _context, _donation: np.ones(result_shape, dtype=result_dtype)
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"declared shape=\(2,\), dtype=float32; produced shape=",
    ):
        advect_native.execute_graph(
            invalid_result_plan,
            [np.ones(2, dtype=np.float32)],
            [],
        )

    def raise_from_callback(
        _values: object,
        _context: object,
        _donation: object,
    ) -> object:
        message = "provider failed"
        raise RuntimeError(message)

    failing_plan = advect_native.build_graph_execution_plan(
        store,
        lambda _op, _attrs: raise_from_callback,
    )
    with pytest.raises(RuntimeError, match="provider failed") as caught:
        advect_native.execute_graph(
            failing_plan,
            [np.ones(2, dtype=np.float32)],
            [],
        )
    assert caught.value.__notes__ == ["while executing staged operation 'custom.bad' at node %1"]


def test_native_staged_execution_plan_is_reentrant() -> None:
    builder = advect_native.GraphBuilder()
    input_id = builder.append_input_node([2], "float32")
    output_id = builder.append_node("custom.reentrant", [input_id], {}, [2], "float32")
    builder.append_output(output_id)
    store, _, _ = _finish(builder)
    plan_box: dict[str, object] = {}

    def evaluate(
        values: tuple[object, ...],
        context: object,
        _donation: object,
    ) -> object:
        value = values[0]
        if context == "outer":
            inner = advect_native.execute_graph(
                plan_box["plan"],
                [value],
                [],
                "inner",
            )
            return inner[0] + 1
        return value * 2

    plan = advect_native.build_graph_execution_plan(
        store,
        lambda _op, _attrs: evaluate,
    )
    plan_box["plan"] = plan
    value = np.array([2.0, 3.0], dtype=np.float32)

    result = advect_native.execute_graph(plan, [value], [], "outer")

    np.testing.assert_array_equal(result[0], 2 * value + 1)


def test_native_staged_execution_donates_only_owned_unaliased_last_uses() -> None:
    donations: list[int | None] = []

    def bind(op: str, _attrs: dict[str, object]) -> object:
        if op == "custom.owned":

            def own(
                values: tuple[object, ...],
                _context: object,
                _donation: int | None,
            ) -> object:
                return values[0].copy()

            own.__advect_owned_output__ = True
            return own

        if op == "custom.alias":

            def alias(
                values: tuple[object, ...],
                _context: object,
                _donation: int | None,
            ) -> object:
                return values[0][:]

            alias.__advect_alias_positions__ = (0,)
            return alias

        if op == "custom.update":

            def update(
                values: tuple[object, ...],
                _context: object,
                donation: int | None,
            ) -> object:
                donations.append(donation)
                result = values[0] if donation == 0 else values[0].copy()
                result += 1
                return result

            update.__advect_donation_positions__ = (0,)
            update.__advect_owned_output__ = True
            return update

        def consume(
            values: tuple[object, ...],
            _context: object,
            _donation: int | None,
        ) -> object:
            return values[0] + 0

        return consume

    eligible_builder = advect_native.GraphBuilder()
    eligible_input = eligible_builder.append_input_node([3], "float32")
    owned = eligible_builder.append_node("custom.owned", [eligible_input], {}, [3], "float32")
    updated = eligible_builder.append_node("custom.update", [owned], {}, [3], "float32")
    eligible_builder.append_output(updated)
    eligible_store, _, _ = _finish(eligible_builder)
    eligible_plan = advect_native.build_graph_execution_plan(eligible_store, bind)

    original = np.arange(3, dtype=np.float32)
    eligible_result = advect_native.execute_graph(eligible_plan, [original], [])

    assert donations == [0]
    np.testing.assert_array_equal(original, np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(eligible_result[0], original + 1)

    donations.clear()
    input_builder = advect_native.GraphBuilder()
    direct_input = input_builder.append_input_node([3], "float32")
    direct_update = input_builder.append_node("custom.update", [direct_input], {}, [3], "float32")
    input_builder.append_output(direct_update)
    input_store, _, _ = _finish(input_builder)
    input_plan = advect_native.build_graph_execution_plan(input_store, bind)

    input_result = advect_native.execute_graph(input_plan, [original], [])

    assert donations == [None]
    np.testing.assert_array_equal(original, np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(input_result[0], original + 1)

    donations.clear()
    alias_builder = advect_native.GraphBuilder()
    alias_input = alias_builder.append_input_node([3], "float32")
    alias_owned = alias_builder.append_node("custom.owned", [alias_input], {}, [3], "float32")
    live_alias = alias_builder.append_node("custom.alias", [alias_owned], {}, [3], "float32")
    alias_update = alias_builder.append_node("custom.update", [alias_owned], {}, [3], "float32")
    alias_output = alias_builder.append_node("custom.consume", [live_alias], {}, [3], "float32")
    alias_builder.append_output(alias_update)
    alias_builder.append_output(alias_output)
    alias_store, _, _ = _finish(alias_builder)
    alias_plan = advect_native.build_graph_execution_plan(alias_store, bind)

    update_result, preserved_alias = advect_native.execute_graph(alias_plan, [original], [])

    assert donations == [None]
    np.testing.assert_array_equal(update_result, original + 1)
    np.testing.assert_array_equal(preserved_alias, original)
