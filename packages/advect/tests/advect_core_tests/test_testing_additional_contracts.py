"""Array API coverage for primitive-author validation."""

from __future__ import annotations

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad
from advect.testing import check_gradient, check_primitive


def test_check_primitive_accepts_an_array_api_provider() -> None:
    @ad.primitive(
        name="tests.testing.strict_scale",
        static_argnames=("scale",),
    )
    def scale(x: object, scale: float) -> object:
        return x * scale  # type: ignore[operator]

    @scale.def_abstract
    def abstract(x: ad.AbstractValue, scale: float) -> ad.ArraySpec:
        del scale
        return x.spec

    @scale.def_jvp
    def jvp_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
        scale: float,
    ) -> object:
        del output, primals
        tangent = tangents[0]
        assert tangent is not None
        return tangent * scale  # type: ignore[operator]

    value = strict.asarray([1.0, -2.0], dtype=strict.float64)
    direction = strict.asarray([0.25, 0.5], dtype=strict.float64)
    cotangent = strict.asarray([-1.0, 2.0], dtype=strict.float64)
    check_primitive(
        scale,
        primals=(value,),
        static={"scale": 2.5},
        tangents=(direction,),
        cotangent=cotangent,
        check=("abstract", "jvp", "transpose"),
    )


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


def test_check_primitive_names_missing_author_rules() -> None:
    missing = ad.primitive(name="tests.testing.missing_rules")(lambda x: x)
    value = np.array([1.0])

    for checks, message in (
        (("abstract",), "abstract.*@primitive.def_abstract"),
        (("transpose",), "transpose.*@primitive.def_transpose"),
    ):
        with pytest.raises(ad.MissingPrimitiveRuleError, match=message):
            check_primitive(missing, primals=(value,), check=checks)


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
        with pytest.raises(ad.MissingPrimitiveRuleError, match=message):
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


def test_check_gradient_validates_steps_and_the_reverse_adjoint() -> None:
    with pytest.raises(ValueError, match="epsilons must be a non-empty sequence"):
        check_gradient(lambda x: x * x, 2.0, epsilons=())

    wrong_adjoint = ad.primitive(name="tests.testing.wrong_adjoint")(lambda x: x * x)
    wrong_adjoint.def_jvp(lambda _output, primals, tangents: 2 * primals[0] * tangents[0])
    wrong_adjoint.def_transpose(lambda cotangent, _primals, _output: (np.zeros_like(cotangent),))
    with pytest.raises(AssertionError, match="reverse gradient disagreed with the JVP"):
        check_gradient(lambda x: np.sum(wrong_adjoint(x)), np.array([1.0, 2.0]))
