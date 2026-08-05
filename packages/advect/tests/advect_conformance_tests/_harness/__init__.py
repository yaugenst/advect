"""Test-internal primitive conformance vocabulary."""

from __future__ import annotations

from advect_conformance_tests._harness._cases import (
    DEFAULT_LAWS,
    Argument,
    InputVariant,
    InvocationCase,
    Law,
    NumericalReference,
    Tolerance,
    argument_tuples,
)
from advect_conformance_tests._harness._domains import (
    ClipRegions,
    Distinct,
    Domain,
    Increasing,
    Interior,
    Nonzero,
    Positive,
    Real,
    SeparatedFrom,
    SpanningGrid,
    StableEigensystem,
    SymmetricPositiveDefinite,
    Unit,
    WellConditioned,
)
from advect_conformance_tests._harness._frontends import Frontend
from advect_conformance_tests._harness._laws import ConformanceError, check_law

__all__ = [
    "DEFAULT_LAWS",
    "Argument",
    "ClipRegions",
    "ConformanceError",
    "Distinct",
    "Domain",
    "Frontend",
    "Increasing",
    "InputVariant",
    "Interior",
    "InvocationCase",
    "Law",
    "Nonzero",
    "NumericalReference",
    "Positive",
    "Real",
    "SeparatedFrom",
    "SpanningGrid",
    "StableEigensystem",
    "SymmetricPositiveDefinite",
    "Tolerance",
    "Unit",
    "WellConditioned",
    "argument_tuples",
    "check_law",
]
