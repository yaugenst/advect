"""Co-located schemas and evaluators for abstract array semantics."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from advect.core._abstract_domains import (
    contractions,
    creation,
    elementwise,
    fft,
    indexing,
    linalg,
    reductions,
    shape,
    signal,
)

if TYPE_CHECKING:
    from advect.core._abstract_model import AbstractRule, ResultEvaluator

_DOMAINS = (elementwise, creation, reductions, shape, indexing, contractions, linalg, fft, signal)


def _merge_unique[T](
    tables: tuple[dict[str, T], ...],
    *,
    label: str,
) -> dict[str, T]:
    merged: dict[str, T] = {}
    for table in tables:
        duplicates = merged.keys() & table.keys()
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise RuntimeError(f"Duplicate abstract {label} registrations: {names}")
        merged.update(table)
    return merged


@lru_cache(maxsize=1)
def operation_semantics() -> tuple[tuple[str, AbstractRule, ResultEvaluator], ...]:
    """Return each built-in operation's schema with its abstract evaluator."""
    rules = _merge_unique(tuple(domain.RULES for domain in _DOMAINS), label="operation")
    evaluators = _merge_unique(tuple(domain.EVALUATORS for domain in _DOMAINS), label="result-kind")
    declared_kinds = {rule.kind for rule in rules.values()}
    missing_evaluators = declared_kinds - evaluators.keys()
    orphan_evaluators = evaluators.keys() - declared_kinds
    if missing_evaluators or orphan_evaluators:
        raise RuntimeError(
            "Abstract result evaluators do not match declared rule kinds: "
            f"missing={sorted(missing_evaluators)!r}, "
            f"orphan={sorted(orphan_evaluators)!r}"
        )
    return tuple((name, schema, evaluators[schema.kind]) for name, schema in sorted(rules.items()))


__all__ = ["operation_semantics"]
