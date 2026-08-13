"""Focused public regressions for core lifecycle boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import advect as ad


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
        (
            {
                "name": "tests.lifecycle.nonbool_variable_arity",
                "variable_output_arity": 1,
            },
            TypeError,
            "boolean",
        ),
    ],
)
def test_primitive_rejects_invalid_declarations(
    options: dict[str, Any], error: type[Exception], match: str
) -> None:
    def implementation(x: object) -> object:
        return x

    with pytest.raises(error, match=match):
        ad.primitive(implementation, **options)


def test_primitive_rejects_unsupported_signatures() -> None:
    with pytest.raises(TypeError, match="Cannot inspect"):
        ad.primitive(max, name="tests.lifecycle.uninspectable")

    def variadic(*values: object) -> tuple[object, ...]:
        return values

    with pytest.raises(TypeError, match="fixed parameters"):
        ad.primitive(variadic, name="tests.lifecycle.variadic")


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


def test_primitive_result_rejects_a_noncallable_release() -> None:
    with pytest.raises(TypeError, match="release must be callable"):
        ad.PrimitiveResult(output=1, residual=object(), release=1)  # type: ignore[arg-type]


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


def test_weak_array_specs_are_rank_zero_at_live_and_durable_boundaries() -> None:
    with pytest.raises(ValueError, match="rank-zero ArraySpec"):
        ad.ArraySpec((2,), "float32", weak=True)

    program = ad.stage(lambda value: value, specs=(ad.ArraySpec((2,), "float32"),))
    payload: Any = program.to_dict()
    payload["program"]["call_specs"][0]["weak"] = True

    with pytest.raises(ValueError, match="rank-zero ArraySpec"):
        ad.StagedProgram.from_dict(payload)
