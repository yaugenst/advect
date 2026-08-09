"""Search every declared public-transform contract with shrinkable inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

import advect as ad
from advect_conformance_tests._builtin_cases import (
    BUILTIN_INVOCATIONS,
    INVOCATIONS_BY_ID,
    STAGED_ONLY_INVOCATIONS,
)
from advect_conformance_tests._harness import Law, argument_tuples, check_law

if TYPE_CHECKING:
    from hypothesis.strategies import DataObject

    from advect_conformance_tests._harness import InvocationCase

# The repository profile owns search depth: fast=100 becomes two examples
# per matrix cell, while thorough=1000 becomes twenty. Keeping the conversion
# here avoids globally weakening unrelated Hypothesis tests.
_SEARCH_EXAMPLES = max(2, min(50, settings.default.max_examples // 50))

_LAW_PARAMETERS = [
    pytest.param(
        identifier,
        law,
        variant,
        id=f"{identifier}-{case.variant_ids[variant]}-{law.value}",
    )
    for identifier, case in INVOCATIONS_BY_ID.items()
    for variant in range(case.variant_count)
    for law in sorted(case.laws, key=lambda item: item.value)
]


@pytest.mark.parametrize(("identifier", "law", "variant"), _LAW_PARAMETERS)
@given(data=st.data())
@settings(
    max_examples=_SEARCH_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_invocation_satisfies_transform_law(
    identifier: str,
    law: Law,
    variant: int,
    data: DataObject,
) -> None:
    case = INVOCATIONS_BY_ID[identifier]
    values = data.draw(argument_tuples(case, variant), label="arguments")
    check_law(case, law, values, variant=variant)


def test_nanprod_matrix_float32_adjoint_regression() -> None:
    """Keep the saved low-precision product map inside the unchanged gate."""
    case = INVOCATIONS_BY_ID["array_ext.nanprod[numpy]"]
    variant = case.variant_ids.index("matrix-float32")
    values = (
        np.array(
            [
                [-2.423161, 0.99999, 0.97577816],
                [-2.5592031, -2.423161, 2.6907873],
            ],
            dtype=np.float32,
        ),
    )

    resolved = case.resolve_variant(variant)
    assert resolved.tolerance.adjoint_atol + resolved.tolerance.adjoint_rtol == pytest.approx(
        2e-6,
    )
    _value, tangent = ad.jvp(np.nanprod)(values[0], tangents=np.ones_like(values[0]))
    assert np.asarray(tangent).dtype == np.dtype("float32")
    check_law(case, Law.ADJOINT, values, variant=variant)


def test_prod_matrix_float32_adjoint_regression() -> None:
    """Keep the thorough-search product example inside its local ulp bound."""
    case = INVOCATIONS_BY_ID["array.prod[numpy]"]
    variant = case.variant_ids.index("matrix-float32")
    values = (
        np.array(
            [[-2.0, -3.0, -2.5], [2.125, 2.0, -2.0]],
            dtype=np.float32,
        ),
    )

    resolved = case.resolve_variant(variant)
    assert resolved.tolerance.adjoint_atol + resolved.tolerance.adjoint_rtol == pytest.approx(
        4e-6,
    )
    check_law(case, Law.ADJOINT, values, variant=variant)


@pytest.mark.parametrize(
    "identifier",
    ["array.prod[numpy]#1", "array_ext.nanprod[numpy]#1"],
)
def test_product_matrix_complex64_adjoint_regression(identifier: str) -> None:
    """Bound the saved complex product reduction association error locally."""
    case = INVOCATIONS_BY_ID[identifier]
    variant = case.variant_ids.index("matrix-complex64")
    values = (
        np.array(
            [
                [-0.1625314 + 2.995594j, -0.17556195 + 0.38360986j, 2.499981 + 0.0097656j],
                [-0.7022478 + 1.5344394j, -1.9997247 + 0.03318378j, 0.97783506 - 0.20937671j],
            ],
            dtype=np.complex64,
        ),
    )

    resolved = case.resolve_variant(variant)
    assert resolved.tolerance.adjoint_atol + resolved.tolerance.adjoint_rtol == pytest.approx(
        2e-5,
    )
    _value, tangent = ad.jvp(case.call)(values[0], tangents=np.ones_like(values[0]))
    assert np.asarray(tangent).dtype == np.dtype("complex64")
    check_law(case, Law.ADJOINT, values, variant=variant)


@pytest.mark.parametrize("case", STAGED_ONLY_INVOCATIONS, ids=lambda case: case.op)
@given(data=st.data())
@settings(max_examples=_SEARCH_EXAMPLES, deadline=None)
def test_non_differentiable_staged_extension_round_trips(
    case: InvocationCase,
    data: DataObject,
) -> None:
    values = data.draw(argument_tuples(case), label="arguments")
    check_law(case, Law.STAGED, values)


def test_declaration_tables_are_not_silently_empty() -> None:
    assert BUILTIN_INVOCATIONS
    assert STAGED_ONLY_INVOCATIONS
    assert len(INVOCATIONS_BY_ID) == len(BUILTIN_INVOCATIONS)


def test_search_depth_tracks_the_selected_hypothesis_profile() -> None:
    if settings.default.max_examples >= 1000:
        assert _SEARCH_EXAMPLES >= 20
    else:
        assert _SEARCH_EXAMPLES >= 2


@pytest.mark.parametrize("profile_name", ["advect", "thorough"])
def test_hypothesis_profile_scales_across_parametrized_items(profile_name: str) -> None:
    """Keep pytest's inherited settings chain cacheable for the full matrix."""
    inherited = settings.get_profile(profile_name)
    for _item in range(len(_LAW_PARAMETERS)):
        inherited = settings(parent=inherited)
        assert inherited.database is not None
