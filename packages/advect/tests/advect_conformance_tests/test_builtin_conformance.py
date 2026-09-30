"""Search every declared invocation's rules and transform laws with shrinkable inputs."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

import advect as ad
from advect.core._registry import get_registry
from advect_conformance_tests._builtin_cases import (
    DYNAMIC_ONLY_STAGING_INVOCATIONS,
    INVOCATIONS_BY_ID,
    STAGED_ONLY_INVOCATIONS,
)
from advect_conformance_tests._harness import (
    ClipRegions,
    ConformanceError,
    Distinct,
    Lattice,
    Law,
    Nonzero,
    Positive,
    Real,
    SeparatedFrom,
    Unit,
    argument_tuples,
    check_law,
)
from advect_conformance_tests._harness._rules import check_registered_jvp, check_registered_vjp

if TYPE_CHECKING:
    from hypothesis.strategies import DataObject

    from advect_conformance_tests._harness import InvocationCase

# The repository profile owns search depth: fast=100 becomes two examples
# per invocation-variant cell, while thorough=1000 becomes twenty. Keeping
# the conversion here avoids globally weakening unrelated Hypothesis tests.
_SEARCH_EXAMPLES = max(2, min(50, settings.default.max_examples // 50))

# A staged complex replacement of a real base replays NumPy's lossy-cast
# warning outside the invocation that silences it.
_CELL_WARNINGS = {"advect.index_update[numpy]#3": "ignore::numpy.exceptions.ComplexWarning"}
_CELLS = [
    (identifier, variant, variant_id)
    for identifier, case in INVOCATIONS_BY_ID.items()
    for variant, variant_id in enumerate(case.variant_ids)
]


@pytest.mark.parametrize(
    ("identifier", "variant"),
    [
        pytest.param(
            identifier,
            variant,
            id=f"{identifier}-{variant_id}",
            marks=[pytest.mark.filterwarnings(_CELL_WARNINGS[identifier])]
            if identifier in _CELL_WARNINGS
            else [],
        )
        for identifier, variant, variant_id in _CELLS
    ],
)
@given(data=st.data())
@settings(max_examples=_SEARCH_EXAMPLES, deadline=None)
def test_invocation_satisfies_its_rules_and_laws(
    identifier: str,
    variant: int,
    data: DataObject,
) -> None:
    """Check the captured rules, then every declared law, on one drawn input.

    The rule checks run first because they localise wrong mathematics; each
    failure names its boundary or law.
    """
    case = INVOCATIONS_BY_ID[identifier]
    values = data.draw(argument_tuples(case, variant), label="arguments")
    check_registered_jvp(case, values, variant=variant, data=data)
    if get_registry().has_vjp(case.op):
        check_registered_vjp(case, values, variant=variant, data=data)
    for law in Law:
        if law in case.laws:
            check_law(case, law, values, variant=variant, data=data)


# Elementwise domains only keep values off kinks, ties, zeros and domain
# edges; on the lattice the adjoint identity must hold there too. The masked
# conventions live in each frontend's rule, so the first invocation of each
# operation and frontend covers them.
_ELEMENTWISE_DOMAINS = (ClipRegions, Distinct, Nonzero, Positive, Real, SeparatedFrom, Unit)
_LATTICE_FORMS = {
    (case.op, case.frontend): (identifier, case)
    for identifier, case in reversed(INVOCATIONS_BY_ID.items())
    if Law.ADJOINT in case.laws
    and all(isinstance(argument.domain, _ELEMENTWISE_DOMAINS) for argument in case.arguments)
}
_LATTICE_CASES = {
    identifier: replace(
        case,
        arguments=tuple(replace(argument, domain=Lattice()) for argument in case.arguments),
    )
    for identifier, case in _LATTICE_FORMS.values()
}


@pytest.mark.parametrize("identifier", sorted(_LATTICE_CASES))
@given(data=st.data())
@settings(max_examples=_SEARCH_EXAMPLES, deadline=None)
def test_adjoint_holds_on_the_boundary_lattice(identifier: str, data: DataObject) -> None:
    """Forward and reverse modes implement one local linear map at boundaries."""
    case = _LATTICE_CASES[identifier]
    variant = data.draw(st.integers(0, case.variant_count - 1), label="variant")
    values = data.draw(argument_tuples(case, variant), label="arguments")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        check_law(case, Law.ADJOINT, values, variant=variant, data=data)


@pytest.mark.parametrize("identifier", sorted(DYNAMIC_ONLY_STAGING_INVOCATIONS))
@given(data=st.data())
@settings(max_examples=1, deadline=None)
def test_dynamic_only_invocation_refuses_to_stage(identifier: str, data: DataObject) -> None:
    """Observe each dynamic-only classification: a stageable form must leave the set."""
    case = INVOCATIONS_BY_ID[identifier]
    refusals = (NotImplementedError, TypeError, ValueError, ad.TracingError, ConformanceError)
    for variant, variant_id in enumerate(case.variant_ids):
        values = data.draw(argument_tuples(case, variant), label=f"{variant_id} arguments")
        with pytest.raises(refusals) as refusal:
            check_law(case, Law.STAGED, values, variant=variant, data=data)
        # Decompositions returning NumPy's named results do not stage that structure.
        if isinstance(refusal.value, ConformanceError):
            assert "structure differs" in str(refusal.value)


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


def test_divide_broadcast_float32_adjoint_regression() -> None:
    """Scale a cancellation-heavy adjoint comparison by its contributions."""
    case = INVOCATIONS_BY_ID["array.divide[numpy]"]
    variant = case.variant_ids.index("broadcast-float32")
    values = (
        np.array(
            [
                [[0.0, 1.75, -1.5]],
                [[1.125, -1.4375, 0.0]],
            ],
            dtype=np.float32,
        ),
        np.array(
            [[[-1.0], [-1.0], [0.25], [-1.0]]],
            dtype=np.float32,
        ),
    )

    resolved = case.resolve_variant(variant)
    assert resolved.tolerance.adjoint_atol + resolved.tolerance.adjoint_rtol == pytest.approx(
        2e-6,
    )
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
    ["array_ext.linalg.qr_r[numpy]", "array_ext.linalg.qr[array_api]"],
)
def test_qr_float32_pivot_sign_regression(identifier: str) -> None:
    """Align the R row whose Householder sign flips inside a float32 difference step."""
    case = INVOCATIONS_BY_ID[identifier]
    variant = case.variant_ids.index("3x3-float32")
    values = (
        np.array(
            [
                [5.4361438e-04, 1.1689512, -1.0150480],
                [-2.0470333, -1.5077639, 1.0155953],
                [-2.3139577e-02, -1.2404182, -1.3689437],
            ],
            dtype=np.float32,
        ),
    )

    check_registered_jvp(case, values, variant=variant)
    check_law(case, Law.FINITE_DIFFERENCE, values, variant=variant)


def test_interp_spanning_grid_keeps_knots_off_the_fixed_queries_regression() -> None:
    """Keep the grid drawn from cell weights 1, 1, 1/3, 1 off the fixed query 0.4.

    SpanningGrid put a knot exactly there, where interpolation is kinked, so
    the registered JVP differed from the central difference.
    """
    case = INVOCATIONS_BY_ID["array_ext.interp[numpy]#1"]
    grid = case.arguments[0].domain.knots([1.0, 1.0, 1 / 3, 1.0])
    values = (grid, np.array([0.5, -1.0, 1.5, 0.0, 2.0]))

    check_registered_jvp(case, values)
    check_law(case, Law.FINITE_DIFFERENCE, values)


@pytest.mark.parametrize("value", [-0.5, 0.0, 2.0])
def test_absolute_pullback_of_a_python_float_regression(value: float) -> None:
    """The real absolute pullback reads the dtype of a weak Python number."""
    assert ad.grad(np.abs)(value) == np.sign(value)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_clip_weak_bound_tie_regression(dtype: str) -> None:
    """A value equal to a weak bound rounded to its dtype is inside, as for the primal."""
    x = np.array([-0.6, 0.6], dtype=dtype)
    gradient = ad.grad(lambda value: np.sum(np.clip(value, -0.6, 0.6)))(x)
    np.testing.assert_array_equal(gradient, np.ones_like(x))


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
    check_law(case, Law.STAGED, values, data=data)
