"""Focused contracts for the unified primitive-authoring surface."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

import array_api_strict as strict
import numpy as np
import pytest
from numpy.testing import assert_allclose

import advect as ad
from advect.core import ArraySpec, TracingError
from advect.core._primitive import MissingPrimitiveRuleError
from advect.core._registry import get_registry
from advect.testing import check_primitive

if TYPE_CHECKING:
    from collections.abc import Callable


def test_primitive_infers_its_name_and_preserves_its_signature() -> None:
    @ad.primitive
    def square(x: float, scale: float = 1.0) -> float:
        return x * x * scale

    assert square.name == f"{square.__module__}.{square.__qualname__}"
    assert str(inspect.signature(square)) == "(x: 'float', scale: 'float' = 1.0) -> 'float'"
    assert square(3.0, scale=2.0) == 18.0
    assert not hasattr(square, "schema_version")

    with pytest.raises(TypeError, match="schema_version"):
        ad.primitive(schema_version=1)  # type: ignore[call-overload]


def test_primitive_handle_writes_one_canonical_operation_record() -> None:
    @ad.primitive(
        name="tests.unified.canonical_record",
        static_argnames=("scale",),
    )
    def primitive(x: np.ndarray, scale: float) -> np.ndarray:
        return x * scale

    @primitive.def_abstract
    def abstract(x: object, scale: float) -> object:
        del scale
        return x.spec  # type: ignore[attr-defined]

    definition = get_registry().get(primitive.op_name)
    assert definition.schema_version == 1
    assert definition.static_argnames == ("scale",)
    assert definition.implementation is primitive.__wrapped__
    assert definition.abstract_rule is abstract
    assert not hasattr(primitive, "def_impl")
    assert_allclose(primitive(np.array([2.0]), 3.0), np.array([6.0]))


def test_primitive_supports_weak_scalar_transforms_and_staging() -> None:
    @ad.primitive(name="tests.unified.weak_scalar")
    def primitive(x: float) -> float:
        return x * x

    @primitive.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    @primitive.def_jvp
    def jvp_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> object:
        del output
        tangent = tangents[0]
        assert tangent is not None
        return 2 * cast("Any", primals[0]) * cast("Any", tangent)

    @primitive.def_transpose
    def transpose_rule(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
    ) -> tuple[object]:
        del output
        return (2 * cast("Any", primals[0]) * cast("Any", cotangent),)

    gradient = ad.grad(primitive)(3.0)
    value, tangent = ad.jvp(primitive)(3.0, tangents=2.0)
    vjp_value, pullback = ad.vjp(primitive)(3.0)
    vjp_gradient = pullback(1.0)
    second = ad.grad(ad.grad(primitive))(3.0)

    program = ad.stage(
        primitive,
        specs=(ad.ArraySpec((), "float64", weak=True),),
    )
    staged_gradient = ad.grad(program)
    staged_results = (
        program(3.0),
        ad.StagedProgram.from_dict(program.to_dict())(3.0),
        staged_gradient(3.0),
        ad.StagedProgram.from_dict(staged_gradient.to_dict())(3.0),
    )

    assert all(
        type(result) is float
        for result in (
            gradient,
            value,
            tangent,
            vjp_value,
            vjp_gradient,
            second,
            *staged_results,
        )
    )
    assert (gradient, value, tangent, vjp_value, vjp_gradient, second) == pytest.approx(
        (6.0, 9.0, 12.0, 9.0, 6.0, 2.0)
    )
    assert staged_results == pytest.approx((9.0, 9.0, 6.0, 6.0))


def test_static_and_nondiff_arguments_have_one_call_contract() -> None:
    seen_tangents: list[tuple[object | None, ...]] = []

    @ad.primitive(
        name="tests.unified.static_nondiff",
        static_argnames=("scale",),
        nondiff_argnames=("tag",),
    )
    def primitive(
        x: np.ndarray,
        scale: float,
        tag: np.ndarray,
    ) -> np.ndarray:
        return x * scale + tag

    @primitive.def_abstract
    def abstract(x: object, scale: float, tag: object) -> ArraySpec:
        del scale, tag
        return x.spec  # type: ignore[attr-defined]

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
        scale: float,
    ) -> np.ndarray:
        del output, primals
        seen_tangents.append(tangents)
        tangent = tangents[0]
        assert tangent is not None
        assert tangents[1] is None
        return tangent * scale

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del output, primals
        return cotangent * scale, cotangent * 999

    x = np.array([1.0, 2.0])
    tag = np.array([10.0, 20.0])
    expected = x * 2.0 + tag
    assert_allclose(primitive(x, 2.0, tag), expected)
    assert_allclose(primitive(x=x, tag=tag, scale=2.0), expected)

    value, tangent = ad.jvp(
        lambda left, label: primitive(left, 2.0, label),
        argnums=(0, 1),
    )(
        x,
        tag,
        tangents=(np.ones_like(x), np.ones_like(tag)),
    )
    assert_allclose(value, expected)
    assert_allclose(tangent, np.full_like(x, 2.0))
    assert seen_tangents[-1][1] is None

    dx, dtag = ad.grad(
        lambda left, label: np.sum(primitive(left, 2.0, label)),
        argnums=(0, 1),
    )(x, tag)
    assert_allclose(dx, np.full_like(x, 2.0))
    assert_allclose(dtag, np.zeros_like(tag))

    staged = ad.stage(
        lambda left, label: primitive(left, 2.0, label),
        specs=(ArraySpec(x.shape, x.dtype), ArraySpec(tag.shape, tag.dtype)),
    )
    staged_call = cast("Callable[..., Any]", staged)
    assert_allclose(staged_call(x, tag), expected)

    check_primitive(
        primitive,
        primals=(x, tag),
        static={"scale": 2.0},
        check=("nested",),
    )


def test_primitive_transpose_can_skip_inactive_input_contributions() -> None:
    active_calls: list[tuple[int, ...] | None] = []

    @ad.primitive(name="tests.unified.selective_transpose")
    def primitive(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left * right

    @primitive.def_abstract
    def abstract(left: ad.AbstractValue, right: ad.AbstractValue) -> ad.ArraySpec:
        del right
        return left.spec

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output
        left, right = primals
        left_tangent, right_tangent = tangents
        return (0 if left_tangent is None else left_tangent * right) + (
            0 if right_tangent is None else left * right_tangent
        )

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        *,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        del output
        active_calls.append(active_input_indices)
        active = {0, 1} if active_input_indices is None else set(active_input_indices)
        left, right = primals
        return (
            cotangent * right if 0 in active else None,
            cotangent * left if 1 in active else None,
        )

    left = np.array([1.0, 2.0])
    right = np.array([3.0, 4.0])
    left_gradient = ad.grad(lambda value: np.sum(primitive(value, right)))(left)
    both_gradients = ad.grad(
        lambda first, second: np.sum(primitive(first, second)),
        argnums=(0, 1),
    )(left, right)

    assert_allclose(left_gradient, right)
    assert_allclose(both_gradients[0], right)
    assert_allclose(both_gradients[1], left)
    assert active_calls == [(0,), (0, 1)]


def test_loaded_staged_primitive_keeps_nested_call_atomic_under_grad() -> None:
    implementation_inputs: list[tuple[type[object], ...]] = []
    transpose_calls = 0

    @ad.primitive(
        name="tests.unified.staged_nested_atomic",
        nondiff_argnames=("offset",),
    )
    def primitive(
        pair: tuple[np.ndarray, np.ndarray],
        *,
        offset: np.ndarray,
    ) -> np.ndarray:
        implementation_inputs.append((type(pair[0]), type(pair[1]), type(offset)))
        return pair[0] * pair[1] + offset

    @primitive.def_abstract
    def abstract(
        pair: tuple[ad.AbstractValue, ad.AbstractValue],
        *,
        offset: ad.AbstractValue,
    ) -> ad.ArraySpec:
        del offset
        return pair[0].spec

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del output
        nonlocal transpose_calls
        transpose_calls += 1
        left, right, offset = primals
        return cotangent * right, cotangent * left, np.zeros_like(offset)

    staged = ad.stage(
        lambda left, right, offset: primitive((left, right), offset=offset),
        specs=(
            ArraySpec((2,), "float64"),
            ArraySpec((2,), "float64"),
            ArraySpec((2,), "float64"),
        ),
    )
    restored = ad.StagedProgram.from_dict(staged.to_dict())
    left = np.array([2.0, 3.0])
    right = np.array([4.0, 5.0])
    offset = np.array([10.0, 20.0])

    dleft, dright, doffset = ad.grad(
        lambda x, y, bias: np.sum(restored(x, y, bias)),
        argnums=(0, 1, 2),
    )(left, right, offset)

    assert_allclose(dleft, right)
    assert_allclose(dright, left)
    assert_allclose(doffset, np.zeros_like(offset))
    assert implementation_inputs == [(np.ndarray, np.ndarray, np.ndarray)]
    assert transpose_calls == 1


def test_declared_static_argument_rejects_a_tracer() -> None:
    @ad.primitive(
        name="tests.unified.static_tracer",
        static_argnames=("scale",),
    )
    def primitive(x: np.ndarray, scale: object) -> np.ndarray:
        return x * scale

    x = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match=r"declared static.*received a traced value"):
        ad.grad(lambda value: np.sum(primitive(value, value)))(x)


def test_implementation_signature_and_declared_names_are_validated() -> None:
    with pytest.raises(ValueError, match=r"declares unknown argument.*config"):
        ad.primitive(
            lambda x: x,
            name="tests.unified.unknown_declared_name",
            static_argnames=("config",),
        )

    class ReprOnlyDefault:
        def __repr__(self) -> str:
            return "same-default"

    default = ReprOnlyDefault()

    def implementation(x: object, config: object = default) -> object:
        del config
        return x

    primitive = ad.primitive(
        implementation,
        name="tests.unified.python_implementation_defaults",
    )
    assert primitive("value") == "value"


def test_jvp_only_primitive_transposes_structurally_and_nests() -> None:
    @ad.primitive(name="tests.unified.jvp_only")
    def primitive(x: np.ndarray) -> np.ndarray:
        return x * x

    @primitive.def_abstract
    def abstract(x: object) -> object:
        return x.spec  # type: ignore[attr-defined]

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output
        tangent = tangents[0]
        assert tangent is not None
        return 2 * primals[0] * tangent

    x = np.array([1.0, 2.0, -3.0])
    gradient = ad.grad(lambda value: np.sum(primitive(value)))
    assert_allclose(gradient(x), 2 * x)

    check_primitive(
        primitive,
        primals=(x,),
        check=("abstract", "jvp", "transpose", "nested", "stage"),
    )
    assert_allclose(
        ad.grad(lambda value: np.sum(gradient(value)))(x),
        np.full_like(x, 2.0),
    )


def test_check_primitive_accepts_a_transpose_only_residual_boundary() -> None:
    released: list[object] = []
    transposed: list[object] = []

    @ad.primitive(
        name="tests.unified.transpose_only_residual_check",
        nondiff_argnames=("offset",),
        residual=True,
    )
    def primitive(x: np.ndarray, offset: np.ndarray) -> ad.PrimitiveResult[np.ndarray]:
        residual = 2 * x.copy()
        return ad.PrimitiveResult(x * x + offset, residual, release=released.append)

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        residual: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        del primals, output
        transposed.append(residual)
        return cotangent * cast("np.ndarray", residual), 999 * cotangent

    x = np.array([0.5, 1.5])
    offset = np.array([4.0, -2.0])
    direction = np.array([0.25, -0.75])
    cotangent = np.array([1.5, -0.5])

    check_primitive(
        primitive,
        primals=(x, offset),
        tangents=(direction, np.full_like(offset, 123.0)),
        cotangent=cotangent,
        check=("transpose",),
    )

    assert len(released) == 4
    assert len(transposed) == 1
    assert any(value is transposed[0] for value in released)


def test_check_primitive_rejects_a_wrong_transpose_without_a_jvp() -> None:
    released: list[object] = []

    @ad.primitive(name="tests.unified.wrong_transpose_only_check", residual=True)
    def primitive(x: np.ndarray) -> ad.PrimitiveResult[np.ndarray]:
        residual = 2 * x.copy()
        return ad.PrimitiveResult(x * x, residual, release=released.append)

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
        residual: object,
    ) -> tuple[np.ndarray]:
        del primals, output, residual
        return (np.zeros_like(cotangent),)

    with pytest.raises(AssertionError, match="transpose violates the real-adjoint identity"):
        check_primitive(
            primitive,
            primals=(np.array([0.5, 1.5]),),
            check=("transpose",),
        )

    assert len(released) == 4


def test_nested_transforms_keep_opaque_implementation_calls_atomic() -> None:
    implementation_calls: list[np.ndarray] = []

    @ad.primitive(name="tests.unified.opaque_implementation_nested")
    def primitive(x: np.ndarray) -> np.ndarray:
        if callable(getattr(x, "_advect_snapshot", None)):
            msg = "opaque implementation received a tracer"
            raise TypeError(msg)
        implementation_calls.append(x)
        return np.exp(x)

    @primitive.def_abstract
    def abstract(x: object) -> object:
        return x.spec  # type: ignore[attr-defined]

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del primals
        tangent = tangents[0]
        assert tangent is not None
        return output * tangent

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        del primals
        return (output * cotangent,)

    x = np.array([0.25, -0.5, 1.0])
    direction = np.array([0.3, -0.2, 0.4])

    def first_directional(value: np.ndarray) -> np.ndarray:
        return ad.jvp(primitive)(value, tangents=np.ones_like(x))[1]

    value, tangent = ad.jvp(first_directional)(x, tangents=direction)
    assert_allclose(value, np.exp(x))
    assert_allclose(tangent, np.exp(x) * direction)
    assert len(implementation_calls) == 1

    implementation_calls.clear()
    first_gradient = ad.grad(lambda value: np.sum(primitive(value)))
    second_gradient = ad.grad(lambda value: np.sum(first_gradient(value)))
    assert_allclose(second_gradient(x), np.exp(x))
    assert len(implementation_calls) == 1


def test_implementation_calls_to_other_primitives_remain_inside_the_atomic_boundary() -> None:
    implementation_calls: list[str] = []

    @ad.primitive(name="tests.unified.implementation_composition_inner")
    def inner(x: np.ndarray) -> np.ndarray:
        implementation_calls.append("inner")
        return x * x

    @ad.primitive(name="tests.unified.implementation_composition_outer")
    def outer(x: np.ndarray) -> np.ndarray:
        implementation_calls.append("outer")
        return inner(x) + 1

    @outer.def_abstract
    def outer_abstract(x: object) -> object:
        return x.spec  # type: ignore[attr-defined]

    @outer.def_jvp
    def outer_jvp(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output
        tangent = tangents[0]
        assert tangent is not None
        return 2 * primals[0] * tangent

    x = np.array([0.25, -0.5, 1.0])
    _value, reusable = ad.linearize(outer, x)
    try:
        op_names = cast("Any", reusable)._trace.tape.op_names
        assert outer.op_name in op_names
        assert inner.op_name not in op_names
    finally:
        reusable.close()

    implementation_calls.clear()
    first_gradient = ad.grad(lambda value: np.sum(outer(value)))
    second_gradient = ad.grad(lambda value: np.sum(first_gradient(value)))
    assert_allclose(second_gradient(x), np.full_like(x, 2.0))
    assert implementation_calls == ["outer", "inner"]


def test_implementation_cannot_hide_a_captured_tracer_from_primitive_inputs() -> None:
    captured: object | None = None

    @ad.primitive(name="tests.unified.captured_implementation_tracer")
    def primitive(x: np.ndarray) -> object:
        del x
        return captured

    def function(value: np.ndarray) -> object:
        nonlocal captured
        captured = value
        return primitive(value)

    with pytest.raises(TracingError, match=r"captured tracer.*explicit primitive argument"):
        ad.jvp(function)(np.ones(2), tangents=np.ones(2))


def test_static_pytree_metadata_cannot_hide_a_captured_tracer() -> None:
    captured: object | None = None

    @ad.primitive(name="tests.unified.static_captured_implementation_tracer")
    def primitive(x: np.ndarray) -> object:
        return {"value": x * x, "metadata": ad.pytree.static(captured)}

    def function(value: np.ndarray) -> object:
        nonlocal captured
        captured = value
        return primitive(value)

    with pytest.raises(TypeError, match=r"Static pytree metadata.*dynamic pytree leaf"):
        ad.jvp(function)(np.ones(2), tangents=np.ones(2))


def test_python_float_primitive_outputs_normalize_for_transforms() -> None:
    seen_rule_types: list[type[object]] = []

    @ad.primitive(name="tests.unified.python_float_output")
    def primitive(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    @primitive.def_abstract
    def abstract(x: object) -> ArraySpec:
        del x
        return ArraySpec((), "float64")

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> float:
        seen_rule_types.append(type(output))
        tangent = tangents[0]
        assert tangent is not None
        return float(2 * np.sum(primals[0] * tangent))

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        seen_rule_types.append(type(output))
        return (2 * primals[0] * cotangent,)

    x = np.array([1.0, -2.0, 3.0])
    assert type(primitive(x)) is float
    check_primitive(
        primitive,
        primals=(x,),
        check=("abstract", "jvp", "transpose", "stage"),
    )

    value, tangent = ad.jvp(primitive)(x, tangents=np.ones_like(x))
    gradient = ad.grad(primitive)(x)
    staged = ad.stage(primitive, specs=(ArraySpec(x.shape, x.dtype),))

    assert type(value) is np.ndarray
    assert type(tangent) is np.ndarray
    assert type(staged(x)) is np.ndarray
    assert value.shape == tangent.shape == staged(x).shape == ()
    assert_allclose(value, np.sum(x * x))
    assert_allclose(tangent, 2 * np.sum(x))
    assert_allclose(gradient, 2 * x)
    assert seen_rule_types
    assert all(rule_type is np.ndarray for rule_type in seen_rule_types)


def test_python_complex_primitive_outputs_normalize_for_transforms() -> None:
    seen_rule_types: list[type[object]] = []

    @ad.primitive(name="tests.unified.python_complex_output")
    def primitive(x: np.ndarray) -> complex:
        return complex(np.sum(x), np.sum(2 * x))

    @primitive.def_abstract
    def abstract(x: object) -> ArraySpec:
        del x
        return ArraySpec((), "complex128")

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> complex:
        del primals
        seen_rule_types.append(type(output))
        tangent = tangents[0]
        assert tangent is not None
        return complex(np.sum(tangent), np.sum(2 * tangent))

    x = np.array([1.0, -2.0, 3.0])
    tangent_seed = np.array([0.2, -0.3, 0.5])
    check_primitive(
        primitive,
        primals=(x,),
        tangents=(tangent_seed,),
        check=("abstract", "jvp", "stage"),
    )

    value, tangent = ad.jvp(primitive)(x, tangents=tangent_seed)
    staged = ad.stage(primitive, specs=(ArraySpec(x.shape, x.dtype),))

    assert type(primitive(x)) is complex
    assert type(value) is np.ndarray
    assert type(tangent) is np.ndarray
    assert type(staged(x)) is np.ndarray
    assert value.shape == tangent.shape == staged(x).shape == ()
    assert_allclose(value, complex(np.sum(x), np.sum(2 * x)))
    assert_allclose(tangent, complex(np.sum(tangent_seed), np.sum(2 * tangent_seed)))
    assert seen_rule_types
    assert all(rule_type is np.ndarray for rule_type in seen_rule_types)


@pytest.mark.parametrize(
    ("scalar_type", "value", "dtype"),
    [
        (bool, True, "bool"),
        (int, 3, "int64"),
    ],
)
def test_discrete_python_primitive_outputs_normalize_for_transforms(
    scalar_type: type[object],
    value: object,
    dtype: str,
) -> None:
    seen_rule_types: list[type[object]] = []

    @ad.primitive(name=f"tests.unified.python_{scalar_type.__name__}_output")
    def primitive(x: np.ndarray) -> object:
        del x
        return value

    @primitive.def_abstract
    def abstract(x: object) -> ArraySpec:
        del x
        return ArraySpec((), dtype)

    @primitive.def_jvp
    def jvp_rule(
        output: object,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> float:
        del primals, tangents
        seen_rule_types.append(type(output))
        return 0.0

    x = np.array([1.0, -2.0])
    direct = primitive(x)
    traced, tangent = ad.jvp(primitive)(x, tangents=np.ones_like(x))
    staged = ad.stage(primitive, specs=(ArraySpec(x.shape, x.dtype),))

    assert type(direct) is scalar_type
    assert type(traced) is np.ndarray
    assert type(staged(x)) is np.ndarray
    assert type(tangent) is np.ndarray
    assert traced.shape == tangent.shape == staged(x).shape == ()
    assert traced.dtype == staged(x).dtype == np.dtype(dtype)
    assert tangent == 0.0
    assert seen_rule_types == [np.ndarray]


def test_structured_primitive_rules_receive_public_output_pytrees() -> None:
    seen_jvp = False
    seen_transpose = False

    @ad.primitive(name="tests.unified.structured_output_rules")
    def primitive(x: np.ndarray) -> dict[str, object]:
        return {
            "square": x * x,
            "shift": x + 1,
            "metadata": ad.pytree.static("implementation-output"),
        }

    @primitive.def_abstract
    def abstract(x: object) -> dict[str, object]:
        return {
            "square": x.spec,  # type: ignore[attr-defined]
            "shift": x.spec,  # type: ignore[attr-defined]
            "metadata": ad.pytree.static("implementation-output"),
        }

    @primitive.def_jvp
    def jvp_rule(
        output: dict[str, object],
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> dict[str, object]:
        nonlocal seen_jvp
        seen_jvp = True
        assert cast("Any", output["metadata"]).value == "implementation-output"
        tangent = tangents[0]
        assert tangent is not None
        return {
            "square": 2 * primals[0] * tangent,
            "shift": tangent,
            "metadata": output["metadata"],
        }

    @primitive.def_transpose
    def transpose_rule(
        cotangent: dict[str, object],
        primals: tuple[np.ndarray, ...],
        output: dict[str, object],
    ) -> tuple[np.ndarray]:
        nonlocal seen_transpose
        seen_transpose = True
        assert cast("Any", output["metadata"]).value == "implementation-output"
        assert cast("Any", cotangent["metadata"]).value == "implementation-output"
        square_cotangent = cotangent["square"]
        shift_cotangent = cotangent["shift"]
        return (
            2 * primals[0] * (0 if square_cotangent is None else square_cotangent)
            + (0 if shift_cotangent is None else shift_cotangent),
        )

    x = np.array([0.25, -0.5, 1.0])
    direction = np.array([0.3, -0.2, 0.4])
    value, tangent = ad.jvp(primitive)(x, tangents=direction)

    assert_allclose(value["square"], x * x)
    assert_allclose(value["shift"], x + 1)
    assert_allclose(tangent["square"], 2 * x * direction)
    assert_allclose(tangent["shift"], direction)
    assert seen_jvp

    def loss(input_value: np.ndarray) -> np.ndarray:
        output = primitive(input_value)
        return np.sum(output["square"] + 3 * output["shift"])

    expected = 2 * x + 3
    assert_allclose(ad.grad(loss)(x), expected)
    assert seen_transpose
    assert_allclose(
        ad.grad(lambda input_value: np.sum(primitive(input_value)["square"]))(x),
        2 * x,
    )

    staged_gradient = ad.grad(ad.stage(loss, specs=(ad.ArraySpec(x.shape, x.dtype),)))
    assert_allclose(staged_gradient(x), expected)


def test_custom_primitive_traces_array_api_strict_outputs() -> None:
    @ad.primitive(name="tests.unified.array_api_strict")
    def primitive(x: object) -> object:
        return x * x  # type: ignore[operator]

    @primitive.def_jvp
    def jvp_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> object:
        del output
        tangent = tangents[0]
        assert tangent is not None
        return 2 * primals[0] * tangent  # type: ignore[operator]

    value = strict.asarray([1.0, -2.0, 3.0], dtype=strict.float32)
    gradient = ad.grad(
        lambda x: x.__array_namespace__().sum(primitive(x)),
    )(value)

    assert type(gradient) is type(value)
    assert_allclose(np.asarray(gradient), np.array([2.0, -4.0, 6.0]))


def test_staged_primitive_validates_static_output_metadata_exactly() -> None:
    @ad.primitive(name="tests.unified.static_output_mismatch")
    def primitive(x: np.ndarray) -> dict[str, object]:
        return {
            "value": x,
            "metadata": ad.pytree.static("concrete"),
        }

    @primitive.def_abstract
    def abstract(x: object) -> dict[str, object]:
        return {
            "value": x.spec,  # type: ignore[attr-defined]
            "metadata": ad.pytree.static("abstract"),
        }

    staged = ad.stage(
        primitive,
        specs=(ad.ArraySpec((2,), "float64"),),
    )

    with pytest.raises(ValueError, match="different structure"):
        staged(np.ones(2))


def test_structural_transpose_ignores_primal_only_jvp_work() -> None:
    @ad.primitive(name="tests.unified.primal_only_jvp_work")
    def primitive(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * x

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output
        tangent = tangents[0]
        assert tangent is not None
        coefficient = np.cumsum(np.diff(np.pad(primals[0], (1, 0))))
        return coefficient * tangent

    x = np.array([1.0, 2.0, -3.0])

    assert_allclose(ad.grad(lambda value: np.sum(primitive(value)))(x), x)


def test_check_primitive_uses_the_real_adjoint_for_complex_values() -> None:
    @ad.primitive(
        name="tests.unified.complex_check",
        static_argnames=("coefficient",),
    )
    def primitive(x: np.ndarray, coefficient: complex) -> np.ndarray:
        return coefficient * x

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
        coefficient: complex,
    ) -> np.ndarray:
        del output, primals
        tangent = tangents[0]
        assert tangent is not None
        return coefficient * tangent

    x = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    tangent = np.array([0.3 - 0.7j, 1.2 + 0.1j], dtype=np.complex64)
    cotangent = np.array([-0.4 + 1.1j, 0.8 - 0.2j], dtype=np.complex64)
    check_primitive(
        primitive,
        primals=(x,),
        static={"coefficient": 2 - 3j},
        tangents=(tangent,),
        cotangent=cotangent,
        check=("complex",),
        atol=2e-3,
        rtol=2e-3,
    )


def test_check_primitive_names_missing_and_inconsistent_rules() -> None:
    @ad.primitive(name="tests.unified.missing_jvp")
    def missing(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(
        MissingPrimitiveRuleError,
        match=r"tests.unified.missing_jvp.*'jvp'.*@primitive.def_jvp",
    ):
        check_primitive(missing, primals=(np.array([1.0]),), check=("jvp",))

    @ad.primitive(name="tests.unified.abstract_mismatch")
    def mismatch(x: np.ndarray) -> np.ndarray:
        return x

    @mismatch.def_abstract
    def abstract(x: object) -> ArraySpec:
        return ArraySpec(x.spec.shape, "float32")  # type: ignore[attr-defined]

    with pytest.raises(AssertionError, match="abstract output leaf 0 disagrees"):
        check_primitive(
            mismatch,
            primals=(np.array([1.0], dtype=np.float64),),
            check=("abstract",),
        )


def test_check_primitive_stage_rejects_input_mutation() -> None:
    @ad.primitive(name="tests.unified.mutating_stage_check")
    def mutating(x: np.ndarray) -> np.ndarray:
        x[0] += 1
        return x

    @mutating.def_abstract
    def abstract(x: object) -> ArraySpec:
        return x.spec  # type: ignore[attr-defined]

    with pytest.raises(AssertionError, match="compiled stage mutated an input"):
        check_primitive(
            mutating,
            primals=(np.array([1.0, 2.0]),),
            check=("stage",),
        )
