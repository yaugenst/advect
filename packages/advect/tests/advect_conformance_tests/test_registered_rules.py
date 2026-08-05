"""Exercise registered derivative functions directly, below public transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from advect.core._registry import get_registry
from advect_conformance_tests._builtin_cases import INVOCATIONS_BY_ID
from advect_conformance_tests._harness import Law, argument_tuples
from advect_conformance_tests._harness._rules import (
    check_raw_jvp,
    check_raw_vjp,
    check_registered_jvp,
    check_registered_vjp,
)
from advect_conformance_tests._raw_rule_cases import RAW_RULE_CASES

if TYPE_CHECKING:
    from hypothesis.strategies import DataObject

_SEARCH_EXAMPLES = max(2, min(50, settings.default.max_examples // 50))

_JVP_PARAMETERS = [
    pytest.param(identifier, variant, id=f"{identifier}-{case.variant_ids[variant]}")
    for identifier, case in INVOCATIONS_BY_ID.items()
    if Law.FINITE_DIFFERENCE in case.laws
    for variant in range(case.variant_count)
]


@pytest.mark.parametrize(("identifier", "variant"), _JVP_PARAMETERS)
@given(data=st.data())
@settings(
    max_examples=_SEARCH_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_registered_jvp_matches_raw_operation(
    identifier: str,
    variant: int,
    data: DataObject,
) -> None:
    case = INVOCATIONS_BY_ID[identifier]
    values = data.draw(argument_tuples(case, variant), label="arguments")
    check_registered_jvp(case, values, variant=variant)


def _explicit_vjp_parameters() -> list[object]:
    registry = get_registry()
    return [
        pytest.param(identifier, variant, id=f"{identifier}-{case.variant_ids[variant]}")
        for identifier, case in INVOCATIONS_BY_ID.items()
        if registry.has_vjp(case.op)
        for variant in range(case.variant_count)
    ]


@pytest.mark.parametrize(("identifier", "variant"), _explicit_vjp_parameters())
@given(data=st.data())
@settings(
    max_examples=_SEARCH_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_registered_vjp_is_adjoint_of_registered_jvp(
    identifier: str,
    variant: int,
    data: DataObject,
) -> None:
    case = INVOCATIONS_BY_ID[identifier]
    values = data.draw(argument_tuples(case, variant), label="arguments")
    check_registered_vjp(case, values, variant=variant)


def test_divide_broadcast_float32_vjp_regression() -> None:
    """Keep the saved float32 divide adjoint inside the unchanged tolerance."""
    case = INVOCATIONS_BY_ID["array.divide[numpy]"]
    variant = case.variant_ids.index("broadcast-float32")
    values = (
        np.array(
            [
                [[1.5, -0.78125, 0.75]],
                [[-0.78125, 1.6818099, -0.78125]],
            ],
            dtype=np.float32,
        ),
        np.array(
            [[[-1.0], [-1.0], [-1.0], [-0.2766159]]],
            dtype=np.float32,
        ),
    )

    # The fixed evaluation order gives an approximately 1.12e-6 difference
    # against this unchanged 2e-6 gate, retaining the observed ~1.8x margin.
    resolved = case.resolve_variant(variant)
    assert resolved.tolerance.adjoint_atol + resolved.tolerance.adjoint_rtol == pytest.approx(
        2e-6,
    )
    check_registered_vjp(case, values, variant=variant)


@pytest.mark.parametrize("case", RAW_RULE_CASES, ids=lambda case: case.op)
def test_raw_registered_jvp_matches_operation(case: object) -> None:
    check_raw_jvp(case)


_RAW_VJP_CASES = tuple(case for case in RAW_RULE_CASES if get_registry().has_vjp(case.op))


@pytest.mark.parametrize("case", _RAW_VJP_CASES, ids=lambda case: case.op)
def test_raw_registered_vjp_is_adjoint_of_jvp(case: object) -> None:
    check_raw_vjp(case)
