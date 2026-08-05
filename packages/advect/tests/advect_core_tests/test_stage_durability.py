"""Durability and input-classification contracts for abstract staging."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad


def test_captured_array_constant_round_trip_preserves_typed_manifest() -> None:
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x * kernel,
            specs=(ad.ArraySpec((2, 2), "float32"),),
        ),
    )

    payload = cast("dict[str, Any]", program.to_dict())
    artifact = payload["program"]
    constant = next(iter(artifact["graph"]["constants"].values()))
    record = artifact["constants"][0]

    assert constant["format"] == "advect.numeric-constant"
    assert constant["dtype"] == record["dtype"] == "float32"
    assert constant["shape"] == record["shape"] == [2, 2]
    assert constant["digest"] == record["digest"]

    restored = ad.StagedProgram.from_dict(payload)
    result = restored(np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(result, kernel)
    assert result.dtype == np.float32
    assert restored.constants == program.constants


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


def test_captured_constant_payload_and_manifest_validate_each_other() -> None:
    kernel = np.array([1.0, 2.0], dtype=np.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x + kernel,
            specs=(ad.ArraySpec((2,), "float32"),),
        ),
    )

    corrupt_payload = cast("dict[str, Any]", deepcopy(program.to_dict()))
    constant = next(iter(corrupt_payload["program"]["graph"]["constants"].values()))
    constant["data"] = "0000104100000040"
    with pytest.raises(ValueError, match="constant digest"):
        ad.StagedProgram.from_dict(corrupt_payload)

    corrupt_manifest = cast("dict[str, Any]", deepcopy(program.to_dict()))
    corrupt_manifest["program"]["constants"][0]["shape"] = [1, 2]
    with pytest.raises(ValueError, match="shape/dtype"):
        ad.StagedProgram.from_dict(corrupt_manifest)

    missing_manifest = cast("dict[str, Any]", deepcopy(program.to_dict()))
    empty_constants: list[object] = []
    missing_manifest["program"]["constants"] = empty_constants
    with pytest.raises(ValueError, match="does not match graph constants"):
        ad.StagedProgram.from_dict(missing_manifest)


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


def test_standalone_python_complex_scalars_stage_as_weak_complex128() -> None:
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda value: value + value,
            specs=(ad.ArraySpec((), "complex128", weak=True),),
        ),
    )

    assert program(1.0 + 2.0j) == 2.0 + 4.0j
    serialized = cast("dict[str, Any]", program.to_dict())
    scalar_spec = serialized["program"]["call_specs"][0]
    assert scalar_spec == {
        "kind": "array",
        "shape": [],
        "dtype": "complex128",
        "device": None,
        "weak": True,
    }


def test_scalar_only_staged_arithmetic_replays_without_an_array_provider() -> None:
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda value: value + value,
            specs=(ad.ArraySpec((), "float64", weak=True),),
        ),
    )

    assert program(1.25) == 2.5
    restored = ad.StagedProgram.from_dict(program.to_dict())
    assert restored(2.0) == 4.0


def test_serialized_output_specs_are_validated_against_the_graph() -> None:
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda value: value + 1,
            specs=(ad.ArraySpec((2,), "float32"),),
        ),
    )
    payload = cast("dict[str, Any]", deepcopy(program.to_dict()))
    assert payload["program"]["output_specs"] == [
        {
            "kind": "array",
            "shape": [2],
            "dtype": "float32",
            "device": None,
            "weak": False,
        }
    ]

    payload["program"]["output_specs"][0]["shape"] = [3]
    with pytest.raises(ValueError, match="output specs do not match graph outputs"):
        ad.StagedProgram.from_dict(payload)


def test_non_array_dynamic_leaves_require_explicit_static_spec() -> None:
    value = np.array([1.0, 2.0], dtype=np.float32)
    explicit = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, direction: x if direction == "forward" else -x,
            specs=(ad.ArraySpec((2,), "float32"), ad.StaticSpec("forward")),
        ),
    )
    np.testing.assert_array_equal(explicit(value, "forward"), value)
    with pytest.raises(TypeError, match="changed value"):
        explicit(value, "backward")


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
