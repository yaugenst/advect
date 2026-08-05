"""Public contracts for Advect's backend-neutral core."""

from __future__ import annotations

from advect.core._abstract import AbstractValue, ArraySpec
from advect.core._context import debug
from advect.core._errors import (
    AdvectError,
    EscapedTracerError,
    MutationError,
    NoJVPError,
    NoVJPError,
    NumericsError,
    StaleViewError,
    TracingError,
)
from advect.core._primitive import MissingPrimitiveRuleError, primitive
from advect.core._residual import PrimitiveResult
from advect.core._stage import (
    ConstantRecord,
    OptimizationPass,
    OptimizationReport,
    StagedProgram,
    StagedTrace,
    StaticSpec,
    TracedNode,
    stage,
)

__all__ = [
    "AbstractValue",
    "AdvectError",
    "ArraySpec",
    "ConstantRecord",
    "EscapedTracerError",
    "MissingPrimitiveRuleError",
    "MutationError",
    "NoJVPError",
    "NoVJPError",
    "NumericsError",
    "OptimizationPass",
    "OptimizationReport",
    "PrimitiveResult",
    "StagedProgram",
    "StagedTrace",
    "StaleViewError",
    "StaticSpec",
    "TracedNode",
    "TracingError",
    "debug",
    "primitive",
    "stage",
]
