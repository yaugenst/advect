"""Declarative primitive invocations and their Hypothesis strategies."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import hypothesis.strategies as st
import numpy as np

from advect_conformance_tests._harness._frontends import Frontend as _Frontend

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from hypothesis.strategies import DrawFn, SearchStrategy

    from advect_conformance_tests._harness._domains import Domain
    from advect_conformance_tests._harness._frontends import Frontend

__all__ = [
    "DEFAULT_LAWS",
    "Argument",
    "InputVariant",
    "InvocationCase",
    "Law",
    "NumericalReference",
    "Tolerance",
    "argument_tuples",
]


class Law(Enum):
    """One independently falsifiable public-transform contract."""

    PRIMAL = "primal"
    FINITE_DIFFERENCE = "finite_difference"
    ADJOINT = "adjoint"
    DEPENDENCE = "dependence"
    STRUCTURE = "structure"
    NO_INPUT_MUTATION = "no_input_mutation"
    DTYPE = "dtype"
    SECOND_ORDER = "second_order"
    STAGED = "staged"


class NumericalReference(Enum):
    """How a directional derivative is anchored outside Advect."""

    CENTRAL = "central"
    COMPLEX_STEP = "complex_step"


# Dependence is intentionally absent.  A nonzero derivative at every sampled
# point is a stronger, domain-specific promise than differentiability and must
# be opted into by the invocation that can justify it.
DEFAULT_LAWS: frozenset[Law] = frozenset(
    {
        Law.PRIMAL,
        Law.FINITE_DIFFERENCE,
        Law.ADJOINT,
        Law.STRUCTURE,
        Law.NO_INPUT_MUTATION,
        Law.DTYPE,
    },
)


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Comparison tolerances for one invocation."""

    primal_rtol: float = 1e-12
    primal_atol: float = 1e-12
    adjoint_rtol: float = 1e-9
    adjoint_atol: float = 1e-9
    finite_difference_rtol: float = 1e-5
    finite_difference_atol: float = 1e-6
    finite_difference_step: float = 1e-6
    complex_step_rtol: float = 1e-11
    complex_step_atol: float = 1e-12
    complex_step: float = 1e-20

    def scaled(self, factor: float) -> Tolerance:
        """Loosen derivative tolerances by an explicit visible factor."""
        return Tolerance(
            primal_rtol=self.primal_rtol,
            primal_atol=self.primal_atol,
            adjoint_rtol=self.adjoint_rtol * factor,
            adjoint_atol=self.adjoint_atol * factor,
            finite_difference_rtol=self.finite_difference_rtol * factor,
            finite_difference_atol=self.finite_difference_atol * factor,
            finite_difference_step=self.finite_difference_step,
            complex_step_rtol=self.complex_step_rtol * factor,
            complex_step_atol=self.complex_step_atol * factor,
            complex_step=self.complex_step,
        )


@dataclass(frozen=True, slots=True)
class InputVariant:
    """One named shape, dtype, and tolerance specialization."""

    name: str
    shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    dtypes: Mapping[str, str] = field(default_factory=dict)
    tolerance: Tolerance | None = None
    numerical_reference: NumericalReference | None = None

    def __post_init__(self) -> None:
        if not self.name:
            msg = "InputVariant name must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Argument:
    """One positional argument and the domain from which it is drawn."""

    name: str
    domain: Domain
    shape: tuple[int, ...] = ()
    dtype: str = "float64"
    differentiable: bool = True

    def strategy(
        self,
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[Any]:
        """Build the concrete value strategy after dependencies are known."""
        dtype = np.dtype(self.dtype)
        return self.domain.strategy(self.shape, dtype, drawn)


@dataclass(frozen=True, slots=True)
class InvocationCase:
    """One supported way of reaching a canonical registry operation.

    Several invocations may name the same ``op``: NumPy and Array API
    frontends, positional and keyword signatures, or materially different
    static attributes are distinct contracts even when they share one rule.
    """

    op: str
    call: Callable[..., Any]
    arguments: tuple[Argument, ...]
    frontend: Frontend = _Frontend.NUMPY
    static: Mapping[str, Any] = field(default_factory=dict)
    laws: frozenset[Law] = DEFAULT_LAWS
    tolerance: Tolerance = Tolerance()
    variants: tuple[InputVariant, ...] = ()
    numerical_reference: NumericalReference = NumericalReference.CENTRAL
    # Indices promised to have a locally nonzero derivative throughout the
    # declared domain.  Merely being differentiable does not imply this.
    dependence_indices: frozenset[int] = frozenset()
    reason: str = ""

    def __post_init__(self) -> None:
        differentiable = set(self.differentiable_indices)
        if not set(self.dependence_indices) <= differentiable:
            msg = f"{self.op}: dependence_indices must name differentiable arguments"
            raise ValueError(msg)
        has_dependence = Law.DEPENDENCE in self.laws
        if has_dependence != bool(self.dependence_indices):
            msg = f"{self.op}: Law.DEPENDENCE and dependence_indices must be declared together"
            raise ValueError(msg)
        if self.laws != DEFAULT_LAWS and not self.reason:
            msg = f"{self.op}: narrowing or extending laws requires a reason"
            raise ValueError(msg)
        if self.numerical_reference is NumericalReference.COMPLEX_STEP and any(
            np.dtype(argument.dtype).kind == "c" for argument in self.arguments
        ):
            msg = f"{self.op}: complex-step is only valid for real primal domains"
            raise ValueError(msg)
        argument_names = {argument.name for argument in self.arguments}
        variant_names: set[str] = set()
        for variant in self.variants:
            if variant.name in variant_names:
                msg = f"{self.op}: duplicate input variant name {variant.name!r}"
                raise ValueError(msg)
            variant_names.add(variant.name)
            unknown = (set(variant.shapes) | set(variant.dtypes)) - argument_names
            if unknown:
                msg = f"{self.op}: input variant names unknown arguments {sorted(unknown)!r}"
                raise ValueError(msg)
            for dtype in variant.dtypes.values():
                np.dtype(dtype)

    @property
    def differentiable_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, argument in enumerate(self.arguments) if argument.differentiable
        )

    @property
    def variant_ids(self) -> tuple[str, ...]:
        if not self.variants:
            return ("default",)
        return tuple(variant.name for variant in self.variants)

    @property
    def variant_count(self) -> int:
        return max(len(self.variants), 1)

    def resolve_variant(self, index: int) -> InvocationCase:
        """Return a declaration with one input specialization applied."""
        if not self.variants:
            if index != 0:
                raise IndexError(index)
            return self
        variant = self.variants[index]
        arguments = tuple(
            replace(
                argument,
                shape=variant.shapes.get(argument.name, argument.shape),
                dtype=variant.dtypes.get(argument.name, argument.dtype),
            )
            for argument in self.arguments
        )
        return replace(
            self,
            arguments=arguments,
            tolerance=self.tolerance if variant.tolerance is None else variant.tolerance,
            numerical_reference=(
                self.numerical_reference
                if variant.numerical_reference is None
                else variant.numerical_reference
            ),
            variants=(),
        )


@st.composite
def argument_tuples(
    draw: DrawFn,
    case: InvocationCase,
    variant: int = 0,
) -> tuple[Any, ...]:
    """Draw a full argument tuple in cross-domain dependency order."""
    resolved = case.resolve_variant(variant)
    drawn: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for argument in _dependency_order(resolved):
        value = draw(argument.strategy(drawn))
        drawn[argument.name] = value
        values[argument.name] = value
    return tuple(values[argument.name] for argument in resolved.arguments)


def _dependency_order(case: InvocationCase) -> tuple[Argument, ...]:
    remaining = list(case.arguments)
    resolved: set[str] = set()
    ordered: list[Argument] = []
    while remaining:
        ready = [
            argument
            for argument in remaining
            if all(name in resolved for name in argument.domain.depends_on)
        ]
        if not ready:
            unresolved = ", ".join(argument.name for argument in remaining)
            msg = f"{case.op}: cyclic or unsatisfied argument dependency: {unresolved}"
            raise ValueError(msg)
        for argument in ready:
            ordered.append(argument)
            resolved.add(argument.name)
            remaining.remove(argument)
    return tuple(ordered)
