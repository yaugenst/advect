"""Tests for staged-program tracing, execution, and immutable metadata."""

from __future__ import annotations

import json
import operator
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

import advect as ad
from advect.core._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FunctionType


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


@pytest.mark.parametrize(
    "capture",
    [
        lambda x, kernel: x * kernel,
        lambda x, kernel: np.multiply(x, kernel),  # noqa: PLW0108 - explicit call site
        lambda x, kernel: np.convolve(x, kernel, mode="same"),
    ],
    ids=["operator", "ufunc", "array-function"],
)
def test_captured_constant_location_names_the_user_call_site(capture: FunctionType) -> None:
    kernel = np.array([1.0, 2.0, 1.0])
    program = ad.stage(lambda x: capture(x, kernel), specs=(ad.ArraySpec((3,), "float64"),))

    code = capture.__code__
    (record,) = program.constants
    assert record.location == f"{code.co_filename}:{code.co_firstlineno} in {code.co_name}()"


def test_stage_snapshots_bound_method_state() -> None:
    class Model:
        offset = np.array([1.0, 2.0], dtype=np.float32)

        def apply(self, value: object) -> object:
            return value + self.offset

    model = Model()
    program = ad.stage(model.apply, specs=(ad.ArraySpec((2,), "float32"),))
    model.offset[:] = 0

    np.testing.assert_array_equal(
        program(np.array([3.0, 4.0], dtype=np.float32)),
        [4.0, 6.0],
    )


def _redundant(x: object) -> object:
    left = x + 1
    right = x + 1
    _unused = x * 2
    return left + right


def test_stage_optimizes_once_and_reports_remapped_graph() -> None:
    program = ad.stage(_redundant, specs=(ad.ArraySpec((2,), "float32"),))
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
    program = ad.stage(_redundant, specs=(ad.ArraySpec((2,), "float32"),))
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


@pytest.mark.parametrize("provider", [np.asarray, strict.asarray], ids=["numpy", "strict"])
def test_staged_gradient_replays_created_arrays_beside_traced_values(
    provider: Callable[[Any], Any],
) -> None:
    """A created array meets the gradient's tracer when grad replays the program."""
    value = provider(np.arange(9.0).reshape(3, 3))

    def loss(matrix: Any) -> Any:
        xp = matrix.__array_namespace__()
        identity = xp.eye(3, dtype=matrix.dtype)
        return xp.sum(identity * matrix * matrix) + xp.sum(
            xp.where(identity > matrix, matrix, 2 * matrix)
        )

    gradient = ad.grad(ad.stage(loss, value))(value)

    expected = [[1.0, 2.0, 2.0], [2.0, 10.0, 2.0], [2.0, 2.0, 18.0]]
    np.testing.assert_array_equal(np.asarray(gradient), expected)


@pytest.mark.parametrize("provider", [np.asarray, strict.asarray], ids=["numpy", "strict"])
@pytest.mark.parametrize("combine", [operator.mul, operator.eq], ids=["mul", "eq"])
def test_stage_rejects_an_enclosing_tracer_as_a_captured_constant(
    provider: Callable[[Any], Any],
    combine: Callable[[Any, Any], Any],
) -> None:
    """Only a tracer of the stage's own values takes over its abstract operators."""
    value = provider(np.array([1.0, 2.0]))

    def outer(captured: Any) -> Any:
        return ad.stage(lambda argument: combine(argument, captured), value)

    with pytest.raises(TypeError, match="constant element from"):
        ad.jvp(outer)(value, tangents=value)


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
        lambda x: double(x),  # noqa: PLW0108 - explicit trace boundary
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
        lambda x: pair(x),  # noqa: PLW0108 - explicit trace boundary
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    get_registry().update(pair.op_name, num_outputs=1, output_arity_known=False)
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
        lambda x: pair(x),  # noqa: PLW0108 - explicit trace boundary
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

    program = ad.stage(
        lambda x: pair(x),  # noqa: PLW0108 - explicit trace boundary
        specs=(ad.ArraySpec((2,), "float32"),),
    )
    payload = deepcopy(program.to_dict())
    get_registry().update(pair.op_name, num_outputs=1, output_arity_known=False)
    payload["program"]["graph"]["version"] = "invalid"

    with pytest.raises(ValueError, match="Unsupported graph version"):
        ad.StagedProgram.from_dict(payload)
    assert get_registry().get(pair.op_name).num_outputs == 1
    assert not get_registry().get(pair.op_name).output_arity_known


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


def test_staged_program_envelope_contains_exactly_one_program() -> None:
    program = ad.stage(lambda x: x + 1, specs=(ad.ArraySpec((2,), "float32"),))

    payload = program.to_dict()

    assert set(payload) == {"format", "version", "program"}
    assert payload["format"] == "advect.ssa-program"
    assert payload["version"] == 3
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
    ("kind", "result", "error", "match"),
    [
        ("missing", None, ad.MissingPrimitiveRuleError, "missing 'abstract'"),
        ("empty", (), TypeError, "returned no values"),
        ("invalid", object(), TypeError, "must return ArraySpec"),
    ],
)
def test_stage_rejects_invalid_primitive_abstract_contracts(
    kind: str,
    result: object,
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
            return result

    with pytest.raises(error, match=match):
        ad.stage(identity, specs=(ad.ArraySpec((1,), "float32"),))


_UNARY = {
    "sin": np.sin,
    "tanh": np.tanh,
    "negative": np.negative,
    "square": np.square,
    "flip": lambda value: np.flip(value, axis=-1),
    "roll": lambda value: np.roll(value, 1, axis=-1),
    "cumsum": lambda value: np.cumsum(value, axis=-1),
}
_BINARY = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply}
_OPERAND = st.one_of(st.sampled_from(["scale", "static", "constant"]), st.integers(0, 7))
_STEPS = st.lists(
    st.one_of(
        st.tuples(st.sampled_from(sorted(_UNARY)), st.none()),
        st.tuples(st.sampled_from([*sorted(_BINARY), "where", "update"]), _OPERAND),
    ),
    min_size=1,
    max_size=8,
)
_ELEMENTS = st.floats(-2.0, 2.0)


def _combine(op: str, value: Any, other: Any) -> Any:
    if op == "where":
        return np.where(value > 0, value, other)
    if op == "update":
        value = value.copy()
        value[..., :1] = other[..., :1] if getattr(other, "shape", ()) else other
        return value
    return _BINARY[op](value, other)


def _random_program(steps: list[tuple[str, object]], constant: Any) -> Callable[..., Any]:
    """Chain the steps over a fan-out history of earlier values and three operand sources."""

    def program(x: Any, scale: Any, static: int) -> Any:
        values = [x]
        for op, source in steps:
            if source is None:
                values.append(_UNARY[op](values[-1]))
                continue
            sources = {"scale": scale, "static": static, "constant": constant}
            other = sources[source] if isinstance(source, str) else values[source % len(values)]
            values.append(_combine(op, values[-1], other))
        return values[-1]

    return program


def _json_round_trip(program: ad.StagedProgram) -> ad.StagedProgram:
    payload = json.loads(json.dumps(program.to_dict()))
    restored = ad.StagedProgram.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.signature == program.signature
    assert restored.constants == program.constants
    assert restored.optimization == program.optimization
    return restored


@given(
    data=st.data(),
    shape=hnp.array_shapes(min_dims=1, max_dims=2, max_side=3),
    byte_orders=st.tuples(st.sampled_from("<>"), st.sampled_from("<>")),
    steps=_STEPS,
    scale=_ELEMENTS,
    static=st.integers(-2, 2),
)
@settings(deadline=None)
def test_staged_json_round_trip_is_canonical_and_differentiable(
    data: st.DataObject,
    shape: tuple[int, ...],
    byte_orders: tuple[str, str],
    steps: list[tuple[str, object]],
    scale: float,
    static: int,
) -> None:
    """Random programs over either byte order load, rerun and differentiate identically."""
    x, constant = (
        data.draw(hnp.arrays(np.dtype(f"{order}f8"), shape, elements=_ELEMENTS))
        for order in byte_orders
    )
    function = _random_program(steps, constant)

    def loss(x: Any, scale: Any, static: int) -> Any:
        return np.sum(np.sin(function(x, scale, static)))

    args = (x, scale, static)
    program = ad.stage(function, x, scale, ad.StaticSpec(static))
    np.testing.assert_allclose(program(*args), function(*args), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(_json_round_trip(program)(*args), program(*args))
    assert ad.stage(function, x, scale, ad.StaticSpec(static)).to_dict() == program.to_dict()

    staged_loss = ad.stage(loss, x, scale, ad.StaticSpec(static))
    gradient = ad.grad(staged_loss, argnums=(0, 1))
    expected = ad.grad(loss, argnums=(0, 1))(*args)
    for result in (
        gradient(*args),
        _json_round_trip(gradient)(*args),
        ad.grad(_json_round_trip(staged_loss), argnums=(0, 1))(*args),
        # a dynamic trace through the staged call
        ad.grad(lambda *values: staged_loss(*values), argnums=(0, 1))(*args),  # noqa: PLW0108
    ):
        for actual, reference in zip(result, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)
