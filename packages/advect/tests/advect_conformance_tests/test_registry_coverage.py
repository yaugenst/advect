"""Orthogonal completeness gates for operation and derivative contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from advect.core._array_api.evidence import operation_cases
from advect.core._array_api.frontend import _FUNCTION_SPECS
from advect.core._primitive_classification import STRUCTURAL_OPS as _STRUCTURAL_OPS
from advect.core._registry import OpRegistry, _register_builtin_ops, get_registry
from advect_conformance_tests._builtin_cases import (
    BUILTIN_INVOCATIONS,
    DYNAMIC_ONLY_STAGING_INVOCATIONS,
    INVOCATIONS_BY_ID,
    PORTABLE_ARRAY_API_INVOCATIONS,
    STAGED_ONLY_INVOCATIONS,
)
from advect_conformance_tests._harness import Frontend, Law
from advect_conformance_tests._raw_rule_cases import RAW_RULE_OPS

if TYPE_CHECKING:
    from advect.core._registry_types import OpDef

# Rules which are intentionally tested from raw operands because no supported
# frontend invocation reaches their registry ID.
_UNBOUND_OPS: dict[str, str] = {
    "array.full": "no frontend traces a differentiable fill value",
    "array_ext.true_divide": "NumPy aliases true_divide to the divide ufunc object",
}

# Staged composites lower entirely into other registered operations, so they
# have conformance cases but no native abstract result rule of their own.
_ABSTRACT_COMPOSITES = frozenset({"array_ext.gradient"})

# This namespace is reserved for direct SciPy primitives shipped by Advect.
# Other ``custom.*`` records are application or test extensions and do not
# belong to the product conformance inventory.
_BUNDLED_SCIPY_PREFIX = "custom.scipy."


def _expected_builtin_registry() -> OpRegistry:
    """Build the product registry without observing process-local test plugins."""
    registry = OpRegistry()
    _register_builtin_ops(registry)
    return registry


def _expected_definitions() -> dict[str, OpDef]:
    registry = _expected_builtin_registry()
    definitions = dict(registry._ops)
    # Bundled SciPy primitives are ordinary custom primitives and therefore
    # cannot be installed into an isolated registry. Importing the invocation
    # cases has already loaded ``advect.scipy``; the reserved live namespace is
    # therefore the authoritative product inventory. Do not derive it from the
    # cases whose completeness this module is meant to check.
    for definition in get_registry().definitions():
        if definition.name.startswith(_BUNDLED_SCIPY_PREFIX):
            definitions[definition.name] = definition
    return definitions


def test_declarations_name_known_operations() -> None:
    expected = set(_expected_definitions())
    unknown = sorted({case.op for case in BUILTIN_INVOCATIONS} - expected)
    assert not unknown, f"conformance declarations name unknown operations: {unknown}"


def test_every_bundled_scipy_primitive_has_an_invocation() -> None:
    expected = {name for name in _expected_definitions() if name.startswith(_BUNDLED_SCIPY_PREFIX)}
    covered = {case.op for case in (*BUILTIN_INVOCATIONS, *STAGED_ONLY_INVOCATIONS)}
    missing = sorted(expected - covered)
    assert not missing, f"bundled SciPy primitives lack invocation coverage: {missing}"


def test_operation_classification_is_a_complete_partition() -> None:
    definitions = _expected_definitions()
    covered = {case.op for case in BUILTIN_INVOCATIONS}
    nondifferentiable = {
        name
        for name, definition in definitions.items()
        if definition.non_differentiable_reason is not None
    }
    buckets = {
        "invocation": covered,
        "structural": set(_STRUCTURAL_OPS),
        "unbound": set(_UNBOUND_OPS),
        "non_differentiable": nondifferentiable,
    }

    memberships: dict[str, list[str]] = {}
    for bucket, names in buckets.items():
        for name in names:
            memberships.setdefault(name, []).append(bucket)
    overlap = {name: labels for name, labels in memberships.items() if len(labels) != 1}
    missing = sorted(set(definitions) - set(memberships))
    stale = sorted(set(memberships) - set(definitions))
    assert not overlap, f"operations belong to multiple coverage buckets: {overlap}"
    assert not missing, f"operations have no semantic coverage classification: {missing}"
    assert not stale, f"coverage classifications name removed operations: {stale}"


def test_every_differentiable_invocation_has_a_registered_jvp() -> None:
    definitions = _expected_definitions()
    missing = sorted({case.op for case in BUILTIN_INVOCATIONS if definitions[case.op].jvp is None})
    assert not missing, f"frontend-reachable differentiable operations lack JVPs: {missing}"


def test_every_invocation_with_a_jvp_has_direct_rule_coverage() -> None:
    # test_registered_rules parameterises exactly this set. The assertion keeps
    # a future law narrowing from silently dropping the direct-rule boundary.
    missing = sorted(
        invocation.op
        for invocation in BUILTIN_INVOCATIONS
        if Law.FINITE_DIFFERENCE not in invocation.laws
    )
    assert not missing, (
        "invocations with JVPs but no numerical rule reference need an explicit "
        f"raw rule contract: {missing}"
    )


def test_every_invocation_has_an_explicit_staging_classification() -> None:
    dynamic_only = {
        identifier
        for identifier, invocation in INVOCATIONS_BY_ID.items()
        if Law.STAGED not in invocation.laws
    }
    assert dynamic_only == DYNAMIC_ONLY_STAGING_INVOCATIONS


def test_staged_array_invocations_have_an_abstract_classification() -> None:
    definitions = _expected_definitions()
    staged = {
        invocation.op
        for invocation in BUILTIN_INVOCATIONS
        if Law.STAGED in invocation.laws and invocation.op.startswith(("array.", "array_ext."))
    }
    abstract_rules = {
        name for name, definition in definitions.items() if definition.abstract_schema is not None
    }
    memberships = {
        name: [
            label
            for label, names in (
                ("rule", abstract_rules),
                ("composite", _ABSTRACT_COMPOSITES),
            )
            if name in names
        ]
        for name in staged | _ABSTRACT_COMPOSITES
    }
    invalid = {name: labels for name, labels in memberships.items() if len(labels) != 1}
    assert not invalid, f"staged operations lack one abstract classification: {invalid}"

    unknown_rules = sorted(abstract_rules - set(definitions))
    assert not unknown_rules, f"abstract rules name unregistered operations: {unknown_rules}"


def test_every_abstract_rule_has_an_executable_staged_case() -> None:
    abstract_rules = {
        name
        for name, definition in _expected_definitions().items()
        if definition.abstract_schema is not None or definition.abstract_rule is not None
    }
    derivative_invocations = {
        invocation.op for invocation in BUILTIN_INVOCATIONS if Law.STAGED in invocation.laws
    }
    array_api_operations = {
        _FUNCTION_SPECS[case.path].op for case in operation_cases() if case.path in _FUNCTION_SPECS
    }
    staged_extensions = {invocation.op for invocation in STAGED_ONLY_INVOCATIONS}
    covered = derivative_invocations | array_api_operations | staged_extensions

    missing = sorted(abstract_rules - covered)
    stale_extensions = sorted(staged_extensions - abstract_rules)
    assert not missing, f"abstract rules lack executable staged coverage: {missing}"
    assert not stale_extensions, (
        f"staged extension cases no longer name abstract rules: {stale_extensions}"
    )


def test_portable_array_api_derivative_surface_has_invocation_coverage() -> None:
    definitions = _expected_definitions()
    qualified = {
        specification.op
        for case in operation_cases()
        if case.portable
        and case.path in _FUNCTION_SPECS
        and (specification := _FUNCTION_SPECS[case.path]).operands
        and definitions[specification.op].jvp is not None
    }
    declared = {invocation.op for invocation in PORTABLE_ARRAY_API_INVOCATIONS}
    assert all(
        invocation.frontend is Frontend.ARRAY_API for invocation in PORTABLE_ARRAY_API_INVOCATIONS
    )
    assert declared == qualified


def test_unbound_operations_still_have_rules() -> None:
    definitions = _expected_definitions()
    ruleless = sorted(
        name
        for name in _UNBOUND_OPS
        if definitions[name].jvp is None and definitions[name].vjp is None
    )
    assert not ruleless, f"unbound classifications no longer have rules: {ruleless}"


def test_every_rule_without_a_frontend_invocation_has_a_raw_case() -> None:
    definitions = _expected_definitions()
    required = {
        name
        for name in set(_STRUCTURAL_OPS) | set(_UNBOUND_OPS)
        if definitions[name].jvp is not None or definitions[name].vjp is not None
    }
    missing = sorted(required - RAW_RULE_OPS)
    stale = sorted(RAW_RULE_OPS - required)
    assert not missing, f"rules without frontend invocations lack raw cases: {missing}"
    assert not stale, f"raw rule cases now have frontend coverage or no rule: {stale}"


def test_non_differentiable_operations_have_no_derivative_rules() -> None:
    invalid = sorted(
        name
        for name, definition in _expected_definitions().items()
        if definition.non_differentiable_reason is not None
        and (definition.jvp is not None or definition.vjp is not None)
    )
    assert not invalid, f"non-differentiable operations also install derivative rules: {invalid}"


def test_process_registry_contains_the_deterministic_product_inventory() -> None:
    expected = set(_expected_definitions())
    missing = sorted(expected - set(get_registry()._ops))
    assert not missing, f"process registry is missing product operations: {missing}"
