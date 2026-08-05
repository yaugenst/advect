"""Canonical operation-name helpers for array-family ops."""

from __future__ import annotations


def _canonical_array_family_op_name(suffix: str) -> str:
    """Resolve one conventional frontend suffix from canonical semantics."""
    # Import lazily so the naming helper remains below the abstract-definition
    # authority without pulling the autodiff rule modules into a root import.
    from advect.core._abstract_domains import operation_semantics  # noqa: PLC0415

    candidate = f"array.{suffix}"
    return (
        candidate
        if any(name == candidate for name, _schema, _evaluator in operation_semantics())
        else f"array_ext.{suffix}"
    )
