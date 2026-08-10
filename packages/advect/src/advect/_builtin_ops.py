"""Complete built-in operation definitions assembled from co-located semantics."""

from __future__ import annotations

from typing import Any

from advect.autodiff.rules.array_family.jvp.registry import jvp_rule_items
from advect.autodiff.rules.array_family.vjp.registry import (
    non_differentiable_items,
    vjp_rule_items,
)
from advect.autodiff.rules.internal import _jvp_getoutput, _vjp_getoutput
from advect.core._abstract_domains import operation_semantics
from advect.core._registry_types import OpDef


def _output_arities() -> dict[str, int]:
    """Return the exceptional fixed arities; every other operation returns one value."""
    return {
        "array_ext.linalg.slogdet": 2,
        "array_ext.linalg.svd": 3,
        "array_ext.linalg.qr": 2,
        "array_ext.linalg.eig": 2,
        "array_ext.linalg.eigh": 2,
        "array_ext.modf": 2,
        "array_ext.frexp": 2,
        "array_ext.divmod": 2,
        "array_ext.unique_index": 2,
        "array.unique_inverse": 2,
        "array.unique_counts": 2,
        "array_ext.unique_index_inverse": 3,
        "array_ext.unique_index_counts": 3,
        "array_ext.unique_inverse_counts": 3,
        "array_ext.unique_index_inverse_counts": 4,
    }


def builtin_operation_definitions() -> tuple[OpDef, ...]:
    """Build one authoritative record for every operation shipped by Advect."""
    jvp_items = jvp_rule_items()
    non_differentiable = non_differentiable_items()
    # Rules also provide the OpDef fragment for dynamic-only operations. The
    # independent coverage tests reject names without semantics,
    # executable cases, frontend lowerings, or concrete evaluators.
    fields: dict[str, dict[str, Any]] = {
        name: {}
        for name in (
            "advect.input",
            "advect.const",
            "advect.getoutput",
            *(name for name, _rule in jvp_items),
            *(name for name, _reason in non_differentiable),
        )
    }

    for name, schema, evaluator in operation_semantics():
        fields.setdefault(name, {}).update(
            abstract_schema=schema,
            abstract_evaluator=evaluator,
        )

    for name, rule in jvp_items:
        fields[name]["jvp"] = rule
    for name, rule, needs_inputs, needs_output in vjp_rule_items():
        fields[name].update(
            vjp=rule,
            vjp_needs_inputs=needs_inputs,
            vjp_needs_output=needs_output,
        )
    for name, reason in non_differentiable:
        fields[name]["non_differentiable_reason"] = reason

    fields["advect.getoutput"].update(
        jvp=_jvp_getoutput,
        vjp=_vjp_getoutput,
        vjp_needs_inputs=False,
        vjp_needs_output=False,
    )

    arities = _output_arities()
    missing = arities.keys() - fields.keys()
    if missing:
        message = f"Output arities name undefined operations: {sorted(missing)!r}"
        raise RuntimeError(message)
    return tuple(
        OpDef(name=name, num_outputs=arities.get(name, 1), **fields[name])
        for name in sorted(fields)
    )


__all__ = ["builtin_operation_definitions"]
