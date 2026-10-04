"""Focused contracts for the unified primitive-authoring surface."""

from __future__ import annotations

import inspect
import math
import operator
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st
from numpy.testing import assert_allclose

import advect as ad
from advect.core import ArraySpec, TracingError, _pytree
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


def test_static_argument_named_residual_keeps_the_ordinary_jvp_contract() -> None:
    @ad.primitive(name="tests.unified.static_residual", static_argnames=("residual",))
    def primitive(x: np.ndarray, *, residual: float = 1.0) -> np.ndarray:
        return x * residual

    @primitive.def_abstract
    def abstract(x: ad.AbstractValue, *, residual: float = 1.0) -> ad.ArraySpec:
        del residual
        return x.spec

    @primitive.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray, ...],
        *,
        residual: float = 1.0,
    ) -> np.ndarray:
        del output, primals
        return tangents[0] * residual

    x = np.array([0.5, 1.5])
    direction = np.array([0.25, -0.75])
    value, tangent = ad.jvp(lambda value: primitive(value, residual=3.0))(
        x,
        tangents=direction,
    )
    assert_allclose(value, 3 * x)
    assert_allclose(tangent, 3 * direction)
    check_primitive(
        primitive,
        primals=(x,),
        static={"residual": 3.0},
        tangents=(direction,),
    )


@pytest.mark.parametrize("has_residual", [False, True])
def test_positional_tangents_named_residual_keep_the_ordinary_jvp_contract(
    *, has_residual: bool
) -> None:
    token = object()
    released: list[object] = []

    @ad.primitive(
        name=f"tests.unified.tangents_named_residual_{has_residual}", residual=has_residual
    )
    def primitive(x: np.ndarray):
        output = x * x
        return (
            ad.PrimitiveResult(output, token, release=released.append) if has_residual else output
        )

    @primitive.def_jvp
    def jvp_rule(output: object, primals: tuple, residual: tuple):
        del output
        return 2 * primals[0] * residual[0]

    x = np.array([0.5, 1.5])
    direction = np.array([0.25, -0.75])
    value, tangent = ad.jvp(primitive)(x, tangents=direction)

    assert_allclose(value, x * x)
    assert_allclose(tangent, 2 * x * direction)
    assert released == ([token] if has_residual else [])


@pytest.mark.parametrize("has_residual", [False, True])
def test_static_primitive_residual_argument_does_not_collide_with_jvp_payload(
    *, has_residual: bool
) -> None:
    token = object()
    released: list[object] = []

    @ad.primitive(
        name=f"tests.unified.static_primitive_residual_{has_residual}",
        static_argnames=("_primitive_residual",),
        residual=has_residual,
    )
    def primitive(x: np.ndarray, *, _primitive_residual: float):
        output = x * _primitive_residual
        return (
            ad.PrimitiveResult(output, token, release=released.append) if has_residual else output
        )

    if has_residual:

        @primitive.def_jvp
        def residual_jvp(
            output: object,
            primals: tuple,
            tangents: tuple,
            *,
            _primitive_residual: float,
            residual: object,
        ):
            del output, primals
            assert residual is token
            assert released == []
            return _primitive_residual * tangents[0]

    else:

        @primitive.def_jvp
        def ordinary_jvp(
            output: object, primals: tuple, tangents: tuple, *, _primitive_residual: float
        ):
            del output, primals
            return _primitive_residual * tangents[0]

    x = np.array([0.5, 1.5])
    direction = np.array([0.25, -0.75])
    value, tangent = ad.jvp(lambda value: primitive(value, _primitive_residual=3.0))(
        x,
        tangents=direction,
    )

    assert_allclose(value, 3 * x)
    assert_allclose(tangent, 3 * direction)
    assert released == ([token] if has_residual else [])


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


def test_primitive_trace_rejects_unsupported_dynamic_contracts() -> None:
    @ad.primitive(name="tests.lifecycle.dynamic_config")
    def dynamic_config(value: object, config: object) -> object:
        del config
        return value

    with pytest.raises(TypeError, match="argument 'config' is not traceable"):
        ad.grad(lambda value: np.sum(dynamic_config(value, object())))(np.ones(2))

    @ad.primitive(name="tests.lifecycle.empty_output")
    def empty_output(value: object) -> dict[str, object]:
        del value
        return {}

    with pytest.raises(TypeError, match="at least one scalar/array leaf"):
        ad.grad(empty_output)(np.ones(2))


def test_primitive_names_argument_leaves_by_their_flattened_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @ad.primitive(name="tests.lifecycle.reordered_arguments", nondiff_argnames=("offset",))
    def shifted(value: object, offset: object) -> object:
        return cast("Any", value) + offset

    masks: list[tuple[bool, ...]] = []

    @shifted.def_jvp
    def shifted_jvp(_output: object, _primals: object, tangents: tuple[object, ...]) -> object:
        masks.append(tuple(tangent is None for tangent in tangents))
        return next(tangent for tangent in tangents if tangent is not None)

    def sorted_flatten(tree: dict[str, object]) -> tuple[tuple[object, ...], tuple[str, ...]]:
        keys = tuple(sorted(tree))
        return tuple(tree[key] for key in keys), keys

    def sorted_unflatten(keys: tuple[str, ...], children: tuple[object, ...]) -> dict[str, object]:
        return dict(zip(keys, children, strict=True))

    # A re-registered dict flattens ("offset", "value"), not in call order.
    monkeypatch.setitem(_pytree._REGISTRY, dict, (sorted_flatten, sorted_unflatten))
    offset = np.array([3.0, 4.0])
    _, tangent = ad.jvp(lambda value: shifted(value, offset))(
        np.array([1.0, 2.0]), tangents=np.array([1.0, -1.0])
    )
    assert masks == [(True, False)]
    np.testing.assert_array_equal(tangent, [1.0, -1.0])

    with pytest.raises(TypeError, match="argument 'offset' is not traceable"):
        ad.grad(lambda value: np.sum(shifted(value, object())))(np.ones(2))


def _identity(x: object) -> object:
    return x


def _variadic(*values: object) -> tuple[object, ...]:
    return values


@pytest.mark.parametrize(
    ("implementation", "options", "error", "match"),
    [
        (_identity, {"name": ""}, ValueError, "non-empty"),
        (_identity, {"name": "advect.reserved"}, ValueError, "reserved"),
        (
            _identity,
            {"name": "tests.lifecycle.empty_static", "static_argnames": ("",)},
            TypeError,
            "non-empty strings",
        ),
        (
            _identity,
            {"name": "tests.lifecycle.duplicate_static", "static_argnames": ("x", "x")},
            ValueError,
            "duplicates",
        ),
        (
            _identity,
            {
                "name": "tests.lifecycle.overlap",
                "static_argnames": ("x",),
                "nondiff_argnames": ("x",),
            },
            ValueError,
            "both static and nondifferentiable",
        ),
        (
            _identity,
            {"name": "tests.unified.unknown_declared_name", "static_argnames": ("config",)},
            ValueError,
            r"declares unknown argument.*config",
        ),
        (
            _identity,
            {"name": "tests.lifecycle.nonbool_residual", "residual": 1},
            TypeError,
            "boolean",
        ),
        (
            _identity,
            {"name": "tests.lifecycle.nonbool_variable_arity", "variable_output_arity": 1},
            TypeError,
            "boolean",
        ),
        (max, {"name": "tests.lifecycle.uninspectable"}, TypeError, "Cannot inspect"),
        (_variadic, {"name": "tests.lifecycle.variadic"}, TypeError, "fixed parameters"),
    ],
    ids=[
        "empty-name",
        "reserved-name",
        "empty-static-name",
        "duplicate-static-name",
        "static-and-nondiff",
        "unknown-declared-name",
        "nonbool-residual",
        "nonbool-variable-arity",
        "uninspectable",
        "variadic",
    ],
)
def test_primitive_rejects_invalid_declarations(
    implementation: Callable[..., object],
    options: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ad.primitive(implementation, **options)


def test_jvp_residual_requires_a_residual_primitive() -> None:
    @ad.primitive(name="tests.lifecycle.jvp_residual_without_declaration")
    def primitive(x: np.ndarray) -> np.ndarray:
        return x * x

    def invalid_jvp(_output, _primals, tangents, *, residual):
        return residual * tangents[0]

    with pytest.raises(TypeError, match="JVP residual requires residual=True"):
        primitive.def_jvp(invalid_jvp)

    primitive.def_jvp(lambda _output, primals, tangents: 2 * primals[0] * tangents[0])
    x = np.array([0.5, 1.5])
    value, tangent = ad.jvp(primitive)(x, tangents=np.ones_like(x))
    assert_allclose(value, x * x)
    assert_allclose(tangent, 2 * x)


def test_primitive_validates_calls_and_rule_registration() -> None:
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

    def positional_selection(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
        active_input_indices: tuple[int, ...] | None = None,
    ) -> tuple[object]:
        del primals, output, active_input_indices
        return (cotangent,)

    with pytest.raises(TypeError, match="active_input_indices must be keyword-only"):
        primitive.def_transpose(positional_selection)

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


def test_primitive_accepts_implementation_defaults_with_repr_only_identity() -> None:
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


def test_forward_mode_rejects_a_public_primitive_before_running_its_primal() -> None:
    calls = 0

    @ad.primitive(name="tests.additional_contracts.transpose_only")
    def transpose_only(value: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return value * value

    @transpose_only.def_transpose
    def transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        del output
        return (2.0 * primals[0] * cotangent,)

    with pytest.raises(ad.NoJVPError, match="no JVP rule is installed"):
        ad.jvp(transpose_only)(np.ones(2), tangents=np.ones(2))
    with pytest.raises(ad.NoJVPError, match="no JVP rule is installed"):
        ad.linearize(transpose_only, np.ones(2))

    assert calls == 0


def test_forward_mode_allows_a_transpose_only_primitive_on_an_enclosing_value() -> None:
    @ad.primitive(name="tests.additional_contracts.passive_transpose_only")
    def transpose_only(value: np.ndarray) -> np.ndarray:
        return value * value

    @transpose_only.def_transpose
    def transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        del output
        return (2 * primals[0] * cotangent,)

    def outer(value: np.ndarray) -> np.ndarray:
        primal, _tangent = ad.jvp(lambda active: active + transpose_only(value))(
            np.array(1.0),
            tangents=np.array(1.0),
        )
        return primal

    assert_allclose(ad.grad(outer)(np.array(2.0)), np.array(4.0))


@pytest.mark.parametrize("optional_residual", [False, True])
def test_check_primitive_accepts_residual_jvp_rules(*, optional_residual: bool) -> None:
    forwards: list[object] = []
    released: list[object] = []
    seen: list[object] = []

    @ad.primitive(
        name=f"tests.unified.residual_jvp_check_{optional_residual}",
        static_argnames=("scale",),
        nondiff_argnames=("offset",),
        residual=True,
    )
    def primitive(x: np.ndarray, scale: float, offset: np.ndarray):
        output = scale * x * x + offset
        residual = (output, 2 * scale * x.copy())
        forwards.append(residual)
        return ad.PrimitiveResult(output, residual, release=released.append)

    @primitive.def_abstract
    def abstract(x: ad.AbstractValue, scale: float, offset: ad.AbstractValue):
        del scale, offset
        return x.spec

    def apply_residual(output: object, tangents: tuple, residual: object):
        assert residual is not None
        assert any(residual is value for value in forwards)
        assert not any(residual is value for value in released)
        original_output, derivative = cast("tuple[np.ndarray, np.ndarray]", residual)
        assert output is original_output
        assert tangents[1] is None
        seen.append(residual)
        return derivative * tangents[0]

    if optional_residual:

        @primitive.def_jvp
        def optional_jvp(
            output: object,
            primals: tuple,
            tangents: tuple,
            *,
            scale: float,
            residual: object = None,
        ):
            del primals, scale
            return apply_residual(output, tangents, residual)

    else:

        @primitive.def_jvp
        def required_jvp(
            output: object,
            primals: tuple,
            tangents: tuple,
            *,
            scale: float,
            residual: object,
        ):
            del primals, scale
            return apply_residual(output, tangents, residual)

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple,
        output: object,
        residual: object,
        *,
        scale: float,
    ):
        del primals, output, scale
        _, derivative = cast("tuple[np.ndarray, np.ndarray]", residual)
        return cotangent * derivative, None

    x = np.array([0.5, 1.5])
    offset = np.array([3.0, -2.0])
    direction = np.array([0.25, -0.75])
    value, tangent = ad.jvp(lambda left, right: primitive(left, 2.0, right), argnums=(0, 1))(
        x,
        offset,
        tangents=(direction, np.full_like(offset, 123.0)),
    )
    assert_allclose(value, 2 * x * x + offset)
    assert_allclose(tangent, 4 * x * direction)
    assert len(forwards) == len(released) == len(seen) == 1

    check_primitive(
        primitive,
        primals=(x, offset),
        static={"scale": 2.0},
        tangents=(direction, np.full_like(offset, 123.0)),
        cotangent=np.array([1.5, -0.5]),
    )

    assert len(seen) > 1
    assert len(released) == len(forwards)
    assert {id(value) for value in released} == {id(value) for value in forwards}


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


def test_check_primitive_rejects_a_structurally_invalid_transpose() -> None:
    @ad.primitive(name="tests.unified.invalid_structured_transpose")
    def primitive(pair: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return pair[0] + pair[1]

    @primitive.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray]]:
        del primals, output
        return ((cotangent, cotangent),)

    pair = (np.array([1.0]), np.array([2.0]))
    with pytest.raises(
        RuntimeError,
        match="flat tuple with one contribution per dynamic input leaf",
    ):
        check_primitive(primitive, primals=(pair,), check=("transpose",))


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
        ad.vjp(function)(np.ones(2))


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
        ad.vjp(function)(np.ones(2))


@pytest.mark.parametrize(
    ("implementation", "jvp", "transpose", "dtype"),
    [
        pytest.param(
            lambda x: float(np.sum(x * x)),
            lambda x, t: float(2 * np.sum(x * t)),
            lambda x, ct: 2 * x * ct,
            "float64",
            id="float",
        ),
        pytest.param(
            lambda x: complex(np.sum(x), np.sum(2 * x)),
            lambda _x, t: complex(np.sum(t), np.sum(2 * t)),
            None,
            "complex128",
            id="complex",
        ),
        pytest.param(lambda _x: True, lambda _x, _t: 0.0, None, "bool", id="bool"),
        pytest.param(lambda _x: 3, lambda _x, _t: 0.0, None, "int64", id="int"),
    ],
)
def test_python_scalar_primitive_outputs_normalize_for_transforms(
    implementation: Callable[[np.ndarray], object],
    jvp: Callable[[np.ndarray, np.ndarray], object],
    transpose: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    dtype: str,
) -> None:
    seen_rule_types: list[type[object]] = []

    @ad.primitive(name=f"tests.unified.python_{dtype}_output")
    def primitive(x: np.ndarray) -> object:
        return implementation(x)

    @primitive.def_abstract
    def abstract(x: object) -> ArraySpec:
        del x
        return ArraySpec((), dtype)

    @primitive.def_jvp
    def jvp_rule(output: object, primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> object:
        seen_rule_types.append(type(output))
        return jvp(primals[0], tangents[0])

    if transpose is not None:

        @primitive.def_transpose
        def transpose_rule(cotangent: Any, primals: tuple[Any, ...], output: object) -> tuple[Any]:
            seen_rule_types.append(type(output))
            return (transpose(primals[0], cotangent),)

    x = np.array([1.0, -2.0, 3.0])
    seed = np.array([0.2, -0.3, 0.5])
    check = ("abstract", "jvp", "stage", *(("transpose",) if transpose else ()))
    check_primitive(primitive, primals=(x,), tangents=(seed,), check=check)
    checked_rule_types = set(seen_rule_types)
    seen_rule_types.clear()

    value, tangent = ad.jvp(primitive)(x, tangents=seed)
    staged = ad.stage(primitive, specs=(ArraySpec(x.shape, x.dtype),))(x)

    assert type(primitive(x)) is type(implementation(x))
    # Rules, traced results and staged results see rank-zero arrays.
    assert type(value) is type(tangent) is type(staged) is np.ndarray
    assert value.shape == tangent.shape == staged.shape == ()
    assert value.dtype == staged.dtype == np.dtype(dtype)
    assert_allclose(value, implementation(x))
    assert_allclose(tangent, jvp(x, seed))
    if transpose is not None:
        assert_allclose(ad.grad(primitive)(x), transpose(x, np.ones(())))
    assert checked_rule_types == {np.ndarray}
    # jvp and grad each call their rule exactly once; staging calls neither.
    assert seen_rule_types == [np.ndarray] * (1 if transpose is None else 2)


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


@pytest.mark.parametrize("first_arity", [1, 2])
def test_ordinary_primitive_keeps_one_fixed_output_arity(first_arity: int) -> None:
    first_split = first_arity == 2

    @ad.primitive(
        name=f"tests.unified.fixed_output_arity_{first_split}",
        static_argnames=("split",),
    )
    def primitive(value: np.ndarray, *, split: bool) -> object:
        return (value, value + 1) if split else value

    _output, pullback = ad.vjp(lambda value: primitive(value, split=first_split))(np.array(2.0))
    pullback.close()

    expected = "1 to 2" if not first_split else "2 to 1"
    with pytest.raises(ValueError, match=rf"changed its output count from {expected}"):
        ad.vjp(lambda value: primitive(value, split=not first_split))(np.array(2.0))


def test_variable_output_arity_is_owned_by_each_dynamic_call() -> None:
    @ad.primitive(
        name="tests.unified.variable_output_arity",
        static_argnames=("split",),
        variable_output_arity=True,
    )
    def primitive(value: np.ndarray, *, split: bool) -> object:
        return (value, value**2) if split else value**3

    @primitive.def_transpose
    def transpose_rule(
        cotangent: object,
        primals: tuple[np.ndarray, ...],
        output: object,
        *,
        split: bool,
    ) -> tuple[np.ndarray]:
        del output
        (value,) = primals
        if split:
            linear, quadratic = cast("tuple[np.ndarray, np.ndarray]", cotangent)
            return (linear + 2 * value * quadratic,)
        return (3 * value**2 * cast("np.ndarray", cotangent),)

    value = np.array(2.0)
    split_output, split_pullback = ad.vjp(lambda x: primitive(x, split=True))(value)
    single_output, single_pullback = ad.vjp(lambda x: primitive(x, split=False))(value)

    assert_allclose(split_output, (value, value**2))
    assert_allclose(split_pullback((np.array(3.0), np.array(4.0))), 19.0)
    assert_allclose(single_output, value**3)
    assert_allclose(single_pullback(np.array(5.0)), 60.0)
    with pytest.raises(ad.TracingError, match=r"variable output arity.*dynamic"):
        ad.stage(
            lambda x: primitive(x, split=False),
            specs=(ad.ArraySpec((), "float64"),),
        )


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


type _Matrix = tuple[tuple[complex, ...], ...]
_PROVIDERS = {"numpy": np, "array_api_strict": strict}


def _matmul(matrix: _Matrix, vector: Any) -> Any:
    namespace = vector.__array_namespace__()
    return namespace.asarray(matrix, dtype=vector.dtype) @ vector


@ad.primitive(name="tests.testing.linear_map", static_argnames=("matrix", "shift"))
def _linear_map(x: Any, matrix: _Matrix, shift: tuple[complex, ...] | None) -> Any:
    del shift
    return _matmul(matrix, x)


@_linear_map.def_abstract
def _linear_map_abstract(
    x: ad.AbstractValue,
    matrix: _Matrix,
    shift: tuple[complex, ...] | None,
) -> ad.ArraySpec:
    del shift
    return ad.ArraySpec((len(matrix),), x.spec.dtype)


@_linear_map.def_jvp
def _linear_map_jvp(
    output: Any,
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    matrix: _Matrix,
    shift: tuple[complex, ...] | None,
) -> Any:
    del output, primals, shift
    return _matmul(matrix, tangents[0])


@_linear_map.def_transpose
def _linear_map_transpose(
    cotangent: Any,
    primals: tuple[Any, ...],
    output: Any,
    matrix: _Matrix,
    shift: tuple[complex, ...] | None,
) -> tuple[Any]:
    """Return the real adjoint A^H c, displaced by ``shift`` when given."""
    del primals, output
    adjoint = tuple(zip(*((value.conjugate() for value in row) for row in matrix), strict=True))
    contribution = _matmul(adjoint, cotangent)
    if shift is not None:
        namespace = contribution.__array_namespace__()
        contribution = contribution + namespace.asarray(shift, dtype=contribution.dtype)
    return (contribution,)


class _LinearCase(NamedTuple):
    provider: str
    dtype: str
    matrix: _Matrix
    primal: tuple[complex, ...]
    tangent: tuple[complex, ...]
    cotangent: tuple[complex, ...]


@st.composite
def _linear_cases(draw: st.DrawFn) -> _LinearCase:
    """Draw ``A @ x`` with every real and imaginary part of magnitude in [0.5, 2]."""
    dtype = draw(st.sampled_from(("float64", "complex128")))
    part = st.builds(operator.mul, st.sampled_from((-1.0, 1.0)), st.floats(0.5, 2.0))
    entry = part if dtype == "float64" else st.builds(complex, part, part)
    rows, columns = draw(st.integers(1, 4)), draw(st.integers(1, 4))

    def vector(size: int) -> tuple[complex, ...]:
        return tuple(draw(entry) for _ in range(size))

    return _LinearCase(
        provider=draw(st.sampled_from(tuple(_PROVIDERS))),
        dtype=dtype,
        matrix=tuple(vector(columns) for _ in range(rows)),
        primal=vector(columns),
        tangent=vector(columns),
        cotangent=vector(rows),
    )


def _norm(values: tuple[complex, ...]) -> float:
    return math.sqrt(sum(abs(value) ** 2 for value in values))


@given(case=_linear_cases())
@example(
    case=_LinearCase(
        provider="array_api_strict",
        dtype="complex128",
        matrix=((1.0 + 2.0j, -0.5 + 1.0j),),
        primal=(1.0 - 1.0j, 2.0 + 0.5j),
        tangent=(0.5 + 0.5j, -1.0 + 2.0j),
        cotangent=(2.0 - 1.0j,),
    ),
)
@settings(deadline=None)
def test_check_primitive_accepts_exact_linear_rules_and_rejects_a_displaced_adjoint(
    case: _LinearCase,
) -> None:
    """A correct linear map passes; a transpose displaced by 0.1|c|/|t| t fails.

    Central differences of a linear map are exact up to rounding. The
    displacement changes Re<A^H c, t> by 0.1|c||t| >= 0.025, while the
    adjoint threshold is at most atol + rtol |A|_F |c||t| (about 1.2e-3 |c||t|
    for these bounded entries), so neither outcome depends on rounding.
    """
    namespace = _PROVIDERS[case.provider]
    dtype = getattr(namespace, case.dtype)
    primal, tangent, cotangent = (
        namespace.asarray(values, dtype=dtype)
        for values in (case.primal, case.tangent, case.cotangent)
    )
    complex_check = ("complex",) if case.dtype == "complex128" else ()
    check_primitive(
        _linear_map,
        primals=(primal,),
        static={"matrix": case.matrix, "shift": None},
        tangents=(tangent,),
        cotangent=cotangent,
        check=("abstract", "jvp", "transpose", *complex_check),
    )

    scale = 0.1 * _norm(case.cotangent) / _norm(case.tangent)
    shift = tuple(scale * value for value in case.tangent)
    with pytest.raises(AssertionError, match="violates the real-adjoint identity"):
        check_primitive(
            _linear_map,
            primals=(primal,),
            static={"matrix": case.matrix, "shift": shift},
            tangents=(tangent,),
            cotangent=cotangent,
            check=("transpose",),
        )


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

    for checks, message in (
        (("jvp",), r"tests.unified.missing_jvp.*'jvp'.*@primitive.def_jvp"),
        (("abstract",), "abstract.*@primitive.def_abstract"),
        (("transpose",), "transpose.*@primitive.def_transpose"),
    ):
        with pytest.raises(MissingPrimitiveRuleError, match=message):
            check_primitive(missing, primals=(np.array([1.0]),), check=checks)

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


def test_a_python_scalar_primitive_result_stays_weak_in_every_lifetime() -> None:
    # Staged execution normalized the declared weak result to a strong array:
    # "declared dtype=float32; produced dtype=float64".
    python_double = ad.primitive(name="tests.testing.weak_python_double")(lambda x: 2.0 * x)
    python_double.def_abstract(lambda x: x.spec)
    python_double.def_jvp(lambda _output, _primals, tangents: 2.0 * tangents[0])
    x = np.array([0.5, -1.0], np.float32)

    def scale(s: float, x: np.ndarray) -> np.ndarray:
        return python_double(s) * x

    program = ad.stage(scale, 3.0, x)
    results = (
        ad.jvp(scale, argnums=(0, 1))(3.0, x, tangents=(1.0, x))[0],
        program(3.0, x),
        ad.jvp(program, argnums=(0, 1))(3.0, x, tangents=(1.0, x))[0],
    )
    for result in results:
        assert result.dtype == scale(3.0, x).dtype == np.float32
        assert_allclose(result, scale(3.0, x))


_doubling_root = ad.implicit_root(
    lambda solution, params: solution - 2.0 * params,
    solve=lambda residual, initial: initial - residual(initial),
    linear_solve=lambda _operator, rhs: rhs,
)
_CONSTANT_DOUBLINGS: dict[str, Callable[[Any], Any]] = {
    "primitive": ad.primitive(name="tests.unified.constant_double")(lambda value: value + value),
    "checkpoint": ad.checkpoint(lambda value: value * 2.0),
    "implicit_root": lambda value: _doubling_root(value, initial=value),
}


@pytest.mark.parametrize("call", sorted(_CONSTANT_DOUBLINGS))
def test_a_primitive_call_on_constants_returns_a_traced_constant(call: str) -> None:
    # Returning the raw array lost the tracer: assigning a traced element into
    # it raised "setting an array element with a sequence", and
    # array_api_strict raised instead of deferring to the tracer it multiplies.
    double = _CONSTANT_DOUBLINGS[call]
    constant = np.array([1.0, 2.0, 3.0])
    x = np.array([0.5, -1.0, 2.0])

    def assign(x: np.ndarray) -> Any:
        doubled = double(constant)
        doubled[0] = x[1]
        return np.sum(doubled * x)

    assert_allclose(ad.grad(assign)(x), [x[1], x[0] + 4.0, 6.0])

    strict_x = strict.asarray(x)
    gradient = ad.grad(
        lambda x: x.__array_namespace__().sum(double(strict.asarray(constant)) * x),
    )(strict_x)
    assert type(gradient) is type(strict_x)
    assert_allclose(np.asarray(gradient), 2.0 * constant)


def test_a_python_scalar_primitive_result_of_a_held_array_stays_strong() -> None:
    # Holding the array operand made the result weak (float32), although
    # selecting it gives a strong result (float64).
    total = ad.primitive(name="tests.unified.python_total")(lambda value: float(np.sum(value)))
    total.def_jvp(lambda _output, _primals, tangents: np.sum(tangents[0]))
    constant = np.array([1.0, 2.0])
    x = np.array([0.5, -1.0], np.float32)

    def scale(constant: np.ndarray, x: np.ndarray) -> Any:
        return total(constant) * x

    held = ad.jvp(scale, argnums=1)(constant, x, tangents=x)[0]
    selected = ad.jvp(scale, argnums=(0, 1))(constant, x, tangents=(constant, x))[0]
    assert held.dtype == selected.dtype == np.float64
    assert_allclose(held, 3.0 * x)
    assert_allclose(selected, 3.0 * x)


def test_check_primitive_requires_the_weak_scalar_category_a_trace_records() -> None:
    # Like a Python operator's, a Python scalar returned from weak operands is
    # weak; a NumPy function of a Python scalar is strong (NEP 50).
    python_double = ad.primitive(name="tests.testing.python_double")(lambda x: 2.0 * x)
    python_double.def_abstract(lambda x: x.spec)
    python_double.def_jvp(lambda _output, _primals, tangents: 2.0 * tangents[0])
    check_primitive(python_double, primals=(3.0,), check=("abstract", "jvp", "stage"))

    numpy_double = ad.primitive(name="tests.testing.numpy_double")(lambda x: np.multiply(2.0, x))
    numpy_double.def_abstract(lambda x: x.spec)
    with pytest.raises(AssertionError, match="declares weak=True, but a dynamic trace records"):
        check_primitive(numpy_double, primals=(3.0,), check=("abstract",))

    python_strong = ad.primitive(name="tests.testing.python_strong")(lambda x: 2.0 * x)
    python_strong.def_abstract(lambda x: ArraySpec((), x.spec.dtype))
    with pytest.raises(AssertionError, match="declares weak=False, but a dynamic trace records"):
        check_primitive(python_strong, primals=(3.0,), check=("abstract",))


def test_check_primitive_handles_scalar_and_complex_defaults() -> None:
    scale = ad.primitive(name="tests.testing.default_scale")(lambda x: 2 * x)
    scale.def_abstract(lambda x: x.spec)
    scale.def_jvp(lambda _output, _primals, tangents: 2 * tangents[0])

    check_primitive(scale, primals=(2.0,), check=("transpose", "stage"))
    check_primitive(scale, primals=(np.array(1 + 2j),), check=("complex",))


def test_check_primitive_rejects_invalid_requests_and_trees() -> None:
    identity = ad.primitive(name="tests.testing.validated_identity")(lambda x: x)
    identity.def_abstract(lambda x: x.spec)
    identity.def_jvp(lambda _output, _primals, tangents: tangents[0])
    value = np.array([1.0, 2.0])

    cases = (
        (
            r"Unknown primitive check\(s\): later, mystery",
            {"primals": (value,), "check": ("mystery", "later")},
        ),
        ("epsilon must be positive", {"primals": (value,), "epsilon": 0}),
        (r"expected 1 dynamic primals.*got 0", {"primals": ()}),
        (
            "complex check requires at least one complex primal",
            {"primals": (value,), "check": ("complex",)},
        ),
        (
            "tangents must match the primals pytree",
            {"primals": (value,), "tangents": ({"value": value},), "check": ("jvp",)},
        ),
        (
            "cotangent must match the output pytree",
            {"primals": (value,), "cotangent": {"value": value}, "check": ("transpose",)},
        ),
    )
    for message, request in cases:
        with pytest.raises(ValueError, match=message):
            check_primitive(identity, **request)


def test_check_primitive_reports_malformed_rules() -> None:
    value = np.array([1.0, 2.0])

    invalid_abstract = ad.primitive(name="tests.testing.invalid_abstract")(lambda x: {"value": x})
    invalid_abstract.def_abstract(lambda x: x.spec)
    with pytest.raises(AssertionError, match="abstract output structure differs"):
        check_primitive(invalid_abstract, primals=(value,), check=("abstract",))

    invalid_leaf = ad.primitive(name="tests.testing.invalid_abstract_leaf")(lambda x: x)

    @invalid_leaf.def_abstract
    def invalid_leaf_abstract(x):
        del x
        return "not an array specification"

    with pytest.raises(AssertionError, match="abstract output leaf 0 is not an ArraySpec"):
        check_primitive(invalid_leaf, primals=(value,), check=("abstract",))

    wrong_jvp = ad.primitive(name="tests.testing.wrong_jvp")(lambda x: x * x)
    wrong_jvp.def_jvp(lambda output, _primals, _tangents: np.zeros_like(output))
    with pytest.raises(AssertionError, match="JVP disagrees with directional finite differences"):
        check_primitive(wrong_jvp, primals=(value,), check=("jvp",))

    wrong_structure = ad.primitive(name="tests.testing.wrong_jvp_structure")(lambda x: x * x)
    wrong_structure.def_jvp(lambda _output, _primals, tangents: {"value": tangents[0]})
    wrong_structure.def_transpose(lambda cotangent, primals, _output: (2 * primals[0] * cotangent,))
    with pytest.raises(AssertionError, match="JVP output structure differs"):
        check_primitive(wrong_structure, primals=(value,), check=("transpose",))


def test_check_primitive_accepts_an_omitted_nondiff_transpose_contribution() -> None:
    @ad.primitive(name="tests.testing.omitted_nondiff", nondiff_argnames=("offset",))
    def shift(value: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return value + offset

    @shift.def_transpose
    def transpose(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray, None]:
        del primals, output
        return cotangent, None

    check_primitive(
        shift,
        primals=(np.array([1.0]), np.array([2.0])),
        check=("transpose",),
    )


def test_check_primitive_enforces_residual_boundaries() -> None:
    square = ad.primitive(name="tests.testing.residual_square", residual=True)(
        lambda x: ad.PrimitiveResult(x * x, 2 * x)
    )
    square.def_jvp(lambda _output, primals, tangents: 2 * primals[0] * tangents[0])
    value = np.array([1.0, 2.0])

    for checks, message in (
        (("nested",), "first-order differentiation only"),
        (("transpose",), r"requires an explicit.*def_transpose"),
    ):
        with pytest.raises(MissingPrimitiveRuleError, match=message):
            check_primitive(square, primals=(value,), check=checks)


def test_check_primitive_attributes_nested_rule_failures() -> None:
    non_nested_jvp = ad.primitive(name="tests.testing.non_nested_jvp")(lambda x: x * x)

    @non_nested_jvp.def_jvp
    def jvp_rule(_output, primals, tangents):
        if not isinstance(primals[0], np.ndarray):
            raise TypeError("nested JVP unsupported")
        return 2 * primals[0] * tangents[0]

    non_nested_jvp.def_transpose(lambda cotangent, primals, _output: (2 * primals[0] * cotangent,))
    value = np.array([1.0, 2.0])
    with pytest.raises(AssertionError, match="JVP rule failed nested differentiation"):
        check_primitive(non_nested_jvp, primals=(value,), check=("nested",))

    non_nested_transpose = ad.primitive(name="tests.testing.non_nested_transpose")(lambda x: x * x)
    non_nested_transpose.def_jvp(lambda _output, primals, tangents: 2 * primals[0] * tangents[0])

    @non_nested_transpose.def_transpose
    def transpose_rule(cotangent, primals, _output):
        if not isinstance(primals[0], np.ndarray):
            raise TypeError("nested transpose unsupported")
        return (2 * primals[0] * cotangent,)

    with pytest.raises(AssertionError, match="transpose rule failed nested tracing"):
        check_primitive(non_nested_transpose, primals=(value,), check=("nested",))


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


def test_check_primitive_stage_rejects_state_dependent_execution() -> None:
    call_count = 0

    @ad.primitive(name="tests.testing.state_dependent")
    def state_dependent(x):
        nonlocal call_count
        call_count += 1
        return x + call_count

    state_dependent.def_abstract(lambda x: x.spec)
    with pytest.raises(AssertionError, match="compiled stage disagrees with concrete execution"):
        check_primitive(state_dependent, primals=(np.array([1.0]),), check=("stage",))


def test_check_primitive_stage_compares_boolean_leaves_exactly() -> None:
    # Regression: the stage check subtracted boolean inputs and outputs to
    # compare them, which NumPy rejects with a TypeError.
    @ad.primitive(name="tests.unified.boolean_stage_check")
    def negate(mask: np.ndarray) -> np.ndarray:
        return np.logical_not(mask)

    @negate.def_abstract
    def abstract(mask: object) -> ArraySpec:
        return mask.spec  # type: ignore[attr-defined]

    check_primitive(negate, primals=(np.array([True, False]),), check=("abstract", "stage"))
