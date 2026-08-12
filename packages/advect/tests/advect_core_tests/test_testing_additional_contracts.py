"""Additional public contracts for primitive-author validation."""

from __future__ import annotations

from typing import Any, cast

import array_api_strict as strict
import numpy as np
import pytest

import advect as ad
from advect.testing import check_gradient, check_primitive


def test_check_primitive_accepts_full_scalar_check_with_static_and_nondiff_inputs() -> None:
    @ad.primitive(
        name="tests.testing.scalar_affine",
        static_argnames=("scale",),
        nondiff_argnames=("offset",),
    )
    def affine(x: float, offset: float, scale: float) -> float:
        return scale * x + offset

    @affine.def_abstract
    def abstract(
        x: ad.AbstractValue,
        offset: ad.AbstractValue,
        scale: float,
    ) -> ad.ArraySpec:
        del offset, scale
        return x.spec

    @affine.def_jvp
    def jvp_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
        scale: float,
    ) -> object:
        del output, primals
        tangent = tangents[0]
        assert tangent is not None
        assert tangents[1] is None
        return scale * cast("Any", tangent)

    @affine.def_transpose
    def transpose_rule(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
        scale: float,
    ) -> tuple[object, object]:
        del primals, output
        return scale * cast("Any", cotangent), 0.0 * cast("Any", cotangent)

    check_primitive(
        affine,
        primals=(2.0, 1.5),
        static={"scale": 3.0},
        check=("abstract", "jvp", "transpose", "nested", "stage"),
    )


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


def test_check_primitive_supplies_a_complex_direction() -> None:
    @ad.primitive(name="tests.testing.complex_scale")
    def scale(x: np.ndarray) -> np.ndarray:
        return 2 * x

    @scale.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output, primals
        tangent = tangents[0]
        assert tangent is not None
        return 2 * tangent

    check_primitive(
        scale,
        primals=(np.array([1 + 2j, -3 + 0.5j]),),
        check=("complex",),
    )


def test_check_primitive_rejects_invalid_requests() -> None:
    @ad.primitive(name="tests.testing.validated_identity")
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    @identity.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    @identity.def_jvp
    def jvp_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> object:
        del output, primals
        return tangents[0]

    value = np.array([1.0, 2.0])
    check_primitive(identity, primals=(value,), check=("abstract",))

    with pytest.raises(ValueError, match=r"Unknown primitive check\(s\): later, mystery"):
        check_primitive(identity, primals=(value,), check=("mystery", "later"))
    with pytest.raises(ValueError, match="epsilon must be positive"):
        check_primitive(identity, primals=(value,), epsilon=0)
    with pytest.raises(ValueError, match=r"expected 1 dynamic primals.*got 0"):
        check_primitive(identity, primals=())
    with pytest.raises(ValueError, match="complex check requires at least one complex primal"):
        check_primitive(identity, primals=(value,), check=("complex",))
    with pytest.raises(ValueError, match="tangents must match the primals pytree"):
        check_primitive(
            identity,
            primals=(value,),
            tangents=({"value": value},),
            check=("jvp",),
        )
    with pytest.raises(ValueError, match="cotangent must match the output pytree"):
        check_primitive(
            identity,
            primals=(value,),
            cotangent={"value": value},
            check=("transpose",),
        )


def test_check_primitive_names_each_missing_author_rule() -> None:
    @ad.primitive(name="tests.testing.missing_rules")
    def primitive(x: np.ndarray) -> np.ndarray:
        return x

    value = np.array([1.0])
    expected = (
        (("abstract",), "abstract.*@primitive.def_abstract"),
        (("jvp",), "jvp.*@primitive.def_jvp"),
        (("transpose",), "transpose.*@primitive.def_transpose"),
    )
    for checks, message in expected:
        with pytest.raises(ad.MissingPrimitiveRuleError, match=message):
            check_primitive(primitive, primals=(value,), check=checks)


def test_check_primitive_reports_malformed_abstract_and_jvp_rules() -> None:
    @ad.primitive(name="tests.testing.invalid_abstract")
    def invalid_abstract(x: np.ndarray) -> dict[str, np.ndarray]:
        return {"value": x}

    @invalid_abstract.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    with pytest.raises(AssertionError, match="abstract output structure differs"):
        check_primitive(
            invalid_abstract,
            primals=(np.array([1.0]),),
            check=("abstract",),
        )

    @ad.primitive(name="tests.testing.invalid_abstract_leaf")
    def invalid_leaf(x: np.ndarray) -> np.ndarray:
        return x

    @invalid_leaf.def_abstract
    def invalid_leaf_abstract(x: ad.AbstractValue) -> str:
        del x
        return "not an array specification"

    with pytest.raises(AssertionError, match="abstract output leaf 0 is not an ArraySpec"):
        check_primitive(
            invalid_leaf,
            primals=(np.array([1.0]),),
            check=("abstract",),
        )

    @ad.primitive(name="tests.testing.wrong_jvp")
    def wrong_jvp(x: np.ndarray) -> np.ndarray:
        return x * x

    @wrong_jvp.def_jvp
    def wrong_jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del primals, tangents
        return np.zeros_like(output)

    with pytest.raises(AssertionError, match="JVP disagrees with directional finite differences"):
        check_primitive(
            wrong_jvp,
            primals=(np.array([1.0, 2.0]),),
            check=("jvp",),
        )

    @ad.primitive(name="tests.testing.wrong_jvp_structure")
    def wrong_jvp_structure(x: np.ndarray) -> np.ndarray:
        return x * x

    @wrong_jvp_structure.def_jvp
    def wrong_structure_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> dict[str, object | None]:
        del output, primals
        return {"value": tangents[0]}

    @wrong_jvp_structure.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        del output
        return (2 * primals[0] * cotangent,)

    with pytest.raises(AssertionError, match="JVP output structure differs"):
        check_primitive(
            wrong_jvp_structure,
            primals=(np.array([1.0, 2.0]),),
            check=("transpose",),
        )


def test_check_primitive_accepts_an_omitted_nondiff_transpose_contribution() -> None:
    @ad.primitive(
        name="tests.testing.omitted_nondiff_contribution",
        nondiff_argnames=("offset",),
    )
    def shift(x: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return x + offset

    @shift.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray, None]:
        del primals, output
        return cotangent, None

    value = np.array([1.0, 2.0])
    check_primitive(
        shift,
        primals=(value, np.array([3.0, 4.0])),
        check=("transpose",),
    )


def test_check_primitive_enforces_residual_first_order_boundaries() -> None:
    @ad.primitive(name="tests.testing.residual_square", residual=True)
    def square(x: np.ndarray) -> ad.PrimitiveResult[np.ndarray]:
        return ad.PrimitiveResult(x * x, 2 * x)

    @square.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output
        tangent = tangents[0]
        assert tangent is not None
        return 2 * primals[0] * tangent

    value = np.array([1.0, 2.0])
    with pytest.raises(ad.MissingPrimitiveRuleError, match="first-order differentiation only"):
        check_primitive(square, primals=(value,), check=("nested",))
    with pytest.raises(ad.MissingPrimitiveRuleError, match=r"requires an explicit.*def_transpose"):
        check_primitive(square, primals=(value,), check=("transpose",))


def test_check_primitive_attributes_nested_rule_failures() -> None:
    @ad.primitive(name="tests.testing.non_nested_jvp")
    def non_nested_jvp(x: np.ndarray) -> np.ndarray:
        return x * x

    @non_nested_jvp.def_jvp
    def jvp_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> object:
        del output
        if type(primals[0]) is not np.ndarray:
            raise TypeError("nested JVP unsupported")
        tangent = tangents[0]
        assert tangent is not None
        return 2 * cast("Any", primals[0]) * cast("Any", tangent)

    @non_nested_jvp.def_transpose
    def transpose_rule(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
    ) -> tuple[object]:
        del output
        return (2 * cast("Any", primals[0]) * cast("Any", cotangent),)

    value = np.array([1.0, 2.0])
    with pytest.raises(AssertionError, match="JVP rule failed nested differentiation"):
        check_primitive(non_nested_jvp, primals=(value,), check=("nested",))

    @ad.primitive(name="tests.testing.non_nested_transpose")
    def non_nested_transpose(x: np.ndarray) -> np.ndarray:
        return x * x

    @non_nested_transpose.def_jvp
    def nested_jvp_rule(
        output: object,
        primals: tuple[object, ...],
        tangents: tuple[object | None, ...],
    ) -> object:
        del output
        tangent = tangents[0]
        assert tangent is not None
        return 2 * cast("Any", primals[0]) * cast("Any", tangent)

    @non_nested_transpose.def_transpose
    def non_nested_transpose_rule(
        cotangent: object,
        primals: tuple[object, ...],
        output: object,
    ) -> tuple[object]:
        del output
        if type(primals[0]) is not np.ndarray:
            raise TypeError("nested transpose unsupported")
        return (2 * cast("Any", primals[0]) * cast("Any", cotangent),)

    with pytest.raises(AssertionError, match="transpose rule failed nested tracing"):
        check_primitive(non_nested_transpose, primals=(value,), check=("nested",))


def test_check_primitive_stage_rejects_state_dependent_execution() -> None:
    call_count = 0

    @ad.primitive(name="tests.testing.state_dependent")
    def state_dependent(x: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return x + call_count

    @state_dependent.def_abstract
    def abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    with pytest.raises(AssertionError, match="compiled stage disagrees with concrete execution"):
        check_primitive(
            state_dependent,
            primals=(np.array([1.0]),),
            check=("stage",),
        )


def test_check_gradient_validates_steps_and_attributes_wrong_adjoint() -> None:
    with pytest.raises(ValueError, match="epsilons must be a non-empty sequence"):
        check_gradient(lambda x: x * x, 2.0, epsilons=())

    @ad.primitive(name="tests.testing.wrong_adjoint")
    def wrong_adjoint(x: np.ndarray) -> np.ndarray:
        return x * x

    @wrong_adjoint.def_jvp
    def jvp_rule(
        output: np.ndarray,
        primals: tuple[np.ndarray, ...],
        tangents: tuple[np.ndarray | None, ...],
    ) -> np.ndarray:
        del output
        tangent = tangents[0]
        assert tangent is not None
        return 2 * primals[0] * tangent

    @wrong_adjoint.def_transpose
    def transpose_rule(
        cotangent: np.ndarray,
        primals: tuple[np.ndarray, ...],
        output: np.ndarray,
    ) -> tuple[np.ndarray]:
        del primals, output
        return (np.zeros_like(cotangent),)

    with pytest.raises(AssertionError, match="reverse gradient disagreed with the JVP"):
        check_gradient(lambda x: np.sum(wrong_adjoint(x)), np.array([1.0, 2.0]))
