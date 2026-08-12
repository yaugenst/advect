"""Shared value and rule types for abstract array evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Shape/dtype contract for one staged array input or result.

    Examples
    --------
    >>> import advect as ad
    >>> spec = ad.ArraySpec((2, 3), "float64")
    >>> spec.shape, spec.dtype
    ((2, 3), 'float64')
    """

    shape: tuple[int, ...]
    dtype: Any
    device: str | None = None
    weak: bool = False

    def __post_init__(self) -> None:
        invalid_dimension = any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in self.shape
        )
        if invalid_dimension:
            msg = f"ArraySpec dimensions must be non-negative integers, got {self.shape!r}"
            raise ValueError(msg)
        if self.weak and self.shape:
            raise ValueError("Only rank-zero ArraySpec values can be weak scalars")


@dataclass(frozen=True, slots=True)
class AbstractValue:
    """A payload-free value passed to custom primitive abstract rules.

    Examples
    --------
    >>> import advect as ad
    >>> abstract = ad.AbstractValue(ad.ArraySpec((4,), "float32"))
    >>> abstract.spec.shape
    (4,)
    """

    spec: ArraySpec


@dataclass(frozen=True, slots=True)
class AbstractRule:
    """Closed frontend call schema linked to one domain evaluator."""

    kind: str
    operands: int
    positional_attrs: tuple[str, ...] = ()
    allowed_attrs: frozenset[str] = frozenset()
    required_attrs: frozenset[str] = frozenset()
    sequence_operand: bool = False
    generic_only: bool = False


type ResultEvaluator = Callable[
    [Sequence[ArraySpec], Mapping[str, Any]],
    tuple[ArraySpec, ...],
]


def rule(
    kind: str,
    operands: int,
    *,
    positional: tuple[str, ...] = (),
    allowed: tuple[str, ...] = (),
    required: tuple[str, ...] = (),
    sequence: bool = False,
    generic_only: bool = False,
) -> AbstractRule:
    """Declare one frontend call schema without introducing another registry."""
    return AbstractRule(
        kind=kind,
        operands=operands,
        positional_attrs=positional,
        allowed_attrs=frozenset(allowed),
        required_attrs=frozenset(required),
        sequence_operand=sequence,
        generic_only=generic_only,
    )
