"""Exercise registered derivative functions directly, below public transforms.

Every drawn invocation cell of ``test_builtin_conformance`` also checks the
rules it captures; this module owns raw rules and saved rule regressions.
"""

from __future__ import annotations

import numpy as np
import pytest

from advect.core._registry import get_registry
from advect_conformance_tests._builtin_cases import INVOCATIONS_BY_ID
from advect_conformance_tests._harness._rules import (
    check_raw_jvp,
    check_raw_vjp,
    check_registered_vjp,
)
from advect_conformance_tests._raw_rule_cases import RAW_RULE_CASES


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
