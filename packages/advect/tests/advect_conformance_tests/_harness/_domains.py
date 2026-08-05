"""Hypothesis domains for smooth, well-conditioned primitive inputs.

A domain constructs valid values.  It does not draw a seed for a second random
number generator and it does not reject invalid values with ``assume``.  This
keeps Hypothesis in control of generation and shrinking all the way to the
array that reproduces a failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, override

import hypothesis.strategies as st
import numpy as np
from hypothesis.extra import numpy as hnp

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hypothesis.strategies import SearchStrategy

__all__ = [
    "ClipRegions",
    "Distinct",
    "Domain",
    "Increasing",
    "Interior",
    "Nonzero",
    "Positive",
    "Real",
    "SeparatedFrom",
    "SpanningGrid",
    "StableEigensystem",
    "SymmetricPositiveDefinite",
    "Unit",
    "WellConditioned",
]

_DEFAULT_CONDITION_LIMIT = 30.0
_MATRIX_RANK = 2


class Domain(Protocol):
    """Construct one concrete argument strategy."""

    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[Any]:
        """Return values of ``shape`` and ``dtype`` satisfying this domain."""
        ...

    @property
    def condition_note(self) -> str:
        """Explain the guarantees relevant to derivative checks."""
        ...

    @property
    def depends_on(self) -> tuple[str, ...]:
        """Arguments which must be drawn before this domain is constructed."""
        ...


def _float_width(dtype: np.dtype[Any]) -> int:
    return 32 if dtype.itemsize <= 4 else 64


def _bounded_float(
    dtype: np.dtype[Any],
    *,
    low: float,
    high: float,
) -> SearchStrategy[float]:
    if _float_width(dtype) == 32:
        low = float(np.float32(low))
        high = float(np.float32(high))
    return st.floats(
        min_value=low,
        max_value=high,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        width=_float_width(dtype),
    )


def _array(
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    elements: SearchStrategy[Any],
) -> SearchStrategy[np.ndarray[Any, Any]]:
    return hnp.arrays(dtype=dtype, shape=shape, elements=elements)


def _real_array(
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
) -> SearchStrategy[np.ndarray[Any, Any]]:
    if np.issubdtype(dtype, np.complexfloating):
        component_dtype = np.dtype("float32" if dtype.itemsize <= 8 else "float64")
        component = _bounded_float(component_dtype, low=low, high=high)
        elements = st.tuples(component, component).map(lambda pair: complex(*pair))
        return _array(dtype, shape, elements)
    return _array(dtype, shape, _bounded_float(dtype, low=low, high=high))


def _count(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


class _BaseDomain:
    __slots__ = ()

    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[Any]:
        del shape, dtype, drawn
        raise NotImplementedError

    @property
    def condition_note(self) -> str:
        return type(self).__name__

    @property
    def depends_on(self) -> tuple[str, ...]:
        return ()


class Real(_BaseDomain):
    """Finite real or complex values with bounded magnitude."""

    __slots__ = ("_scale",)

    def __init__(self, scale: float = 1.0) -> None:
        if scale <= 0:
            msg = "Real scale must be positive"
            raise ValueError(msg)
        self._scale = scale

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        return _real_array(dtype, shape, low=-2.0 * self._scale, high=2.0 * self._scale)

    @override
    @property
    def condition_note(self) -> str:
        return f"finite values in [{-2.0 * self._scale}, {2.0 * self._scale}]"


class Positive(_BaseDomain):
    """Strictly positive values bounded away from zero."""

    __slots__ = ("_high", "_low")

    def __init__(self, low: float = 0.25, high: float = 4.0) -> None:
        if not 0 < low <= high:
            msg = "Positive bounds must satisfy 0 < low <= high"
            raise ValueError(msg)
        self._low = low
        self._high = high

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if np.issubdtype(dtype, np.complexfloating):
            msg = "Positive is only defined for real dtypes"
            raise TypeError(msg)
        return _array(
            dtype,
            shape,
            _bounded_float(dtype, low=self._low, high=self._high),
        )

    @override
    @property
    def condition_note(self) -> str:
        return f"values in [{self._low}, {self._high}]"


class Nonzero(_BaseDomain):
    """Values whose magnitude is bounded away from zero."""

    __slots__ = ("_high", "_margin")

    def __init__(self, margin: float = 0.25, high: float = 3.0) -> None:
        if not 0 < margin <= high:
            msg = "Nonzero bounds must satisfy 0 < margin <= high"
            raise ValueError(msg)
        self._margin = margin
        self._high = high

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if np.issubdtype(dtype, np.complexfloating):
            component_dtype = np.dtype("float32" if dtype.itemsize <= 8 else "float64")
            magnitude = _bounded_float(
                component_dtype,
                low=self._margin,
                high=self._high,
            )
            phase = _bounded_float(component_dtype, low=-np.pi, high=np.pi)

            def polar(pair: tuple[float, float]) -> Any:
                return dtype.type(pair[0] * np.exp(1j * pair[1]))

            elements = st.tuples(magnitude, phase).map(polar)
        else:
            negative = _bounded_float(dtype, low=-self._high, high=-self._margin)
            positive = _bounded_float(dtype, low=self._margin, high=self._high)
            elements = st.one_of(negative, positive)
        return _array(dtype, shape, elements)

    @override
    @property
    def condition_note(self) -> str:
        return f"magnitude in [{self._margin}, {self._high}]"


class Unit(_BaseDomain):
    """Real values strictly inside ``(-1, 1)``."""

    __slots__ = ("_margin",)

    def __init__(self, margin: float = 0.15) -> None:
        if not 0 < margin < 1:
            msg = "Unit margin must lie in (0, 1)"
            raise ValueError(msg)
        self._margin = margin

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if np.issubdtype(dtype, np.complexfloating):
            msg = "Unit is only defined for real dtypes"
            raise TypeError(msg)
        limit = 1.0 - self._margin
        return _array(dtype, shape, _bounded_float(dtype, low=-limit, high=limit))

    @override
    @property
    def condition_note(self) -> str:
        return f"values in (-{1.0 - self._margin}, {1.0 - self._margin})"


class ClipRegions(_BaseDomain):
    """Values safely below, inside, and above fixed clipping bounds."""

    __slots__ = ("_excursion", "_lower", "_margin", "_upper")

    def __init__(
        self,
        lower: float,
        upper: float,
        *,
        margin: float = 0.05,
        excursion: float = 0.5,
    ) -> None:
        if not lower + margin < upper - margin:
            msg = "ClipRegions requires non-overlapping interior margins"
            raise ValueError(msg)
        if margin <= 0 or excursion <= margin:
            msg = "ClipRegions requires 0 < margin < excursion"
            raise ValueError(msg)
        self._lower = lower
        self._upper = upper
        self._margin = margin
        self._excursion = excursion

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if np.issubdtype(dtype, np.complexfloating):
            msg = "ClipRegions is only defined for ordered real dtypes"
            raise TypeError(msg)

        below = _bounded_float(
            dtype,
            low=self._lower - self._excursion,
            high=self._lower - self._margin,
        )
        interior = _bounded_float(
            dtype,
            low=self._lower + self._margin,
            high=self._upper - self._margin,
        )
        above = _bounded_float(
            dtype,
            low=self._upper + self._margin,
            high=self._upper + self._excursion,
        )
        count = _count(shape)
        required = (
            (interior,)
            if count == 1
            else (interior, st.one_of(below, above))
            if count == 2
            else (below, interior, above)
        )
        remaining = count - len(required)
        tail = st.lists(
            st.one_of(below, interior, above),
            min_size=remaining,
            max_size=remaining,
        )
        permutation = st.permutations(tuple(range(count)))

        def build(data: tuple[tuple[float, ...], list[float], list[int]]) -> np.ndarray[Any, Any]:
            anchors, rest, order = data
            values = np.asarray((*anchors, *rest), dtype=dtype)
            return values[np.asarray(order)].reshape(shape)

        return st.tuples(st.tuples(*required), tail, permutation).map(build)

    @override
    @property
    def condition_note(self) -> str:
        return (
            f"covers clip regions at least {self._margin} away from {self._lower} and {self._upper}"
        )


class Distinct(_BaseDomain):
    """Finite real values separated by a minimum gap."""

    __slots__ = ("_gap", "_scale")

    def __init__(self, gap: float = 0.1, scale: float = 1.0) -> None:
        if gap <= 0 or scale <= 0:
            msg = "Distinct gap and scale must be positive"
            raise ValueError(msg)
        self._gap = gap
        self._scale = scale

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if np.issubdtype(dtype, np.complexfloating):
            msg = "Distinct is only defined for ordered real dtypes"
            raise TypeError(msg)
        count = _count(shape)
        steps = st.lists(
            _bounded_float(
                dtype,
                low=self._gap,
                high=self._gap + self._scale,
            ),
            min_size=count,
            max_size=count,
        )
        permutation = st.permutations(tuple(range(count)))

        def build(data: tuple[list[float], list[int]]) -> np.ndarray[Any, Any]:
            increments, order = data
            values = np.cumsum(np.asarray(increments, dtype=np.float64))
            values -= float(values[0] + values[-1]) / 2.0
            return values[np.asarray(order)].reshape(shape).astype(dtype)

        return st.tuples(steps, permutation).map(build)

    @override
    @property
    def condition_note(self) -> str:
        return f"pairwise separation of at least {self._gap}"


class SeparatedFrom(_BaseDomain):
    """Values separated elementwise from another argument.

    Signs alternate, so binary selection primitives exercise both branches
    while remaining a fixed positive distance from their nondifferentiable
    equality boundary.
    """

    __slots__ = ("_high", "_margin", "_other")

    def __init__(self, other: str, margin: float = 0.25, high: float = 1.0) -> None:
        if not 0 < margin <= high:
            msg = "SeparatedFrom bounds must satisfy 0 < margin <= high"
            raise ValueError(msg)
        self._other = other
        self._margin = margin
        self._high = high

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        base = np.asarray(drawn[self._other])
        if base.shape != shape:
            msg = f"SeparatedFrom expected shape {base.shape} from '{self._other}', got {shape}"
            raise ValueError(msg)
        magnitudes = _array(
            dtype,
            shape,
            _bounded_float(dtype, low=self._margin, high=self._high),
        )
        signs = np.ones(base.size, dtype=np.float64)
        signs[1::2] = -1.0
        signs = signs.reshape(shape)
        return magnitudes.map(lambda amount: (base + signs * amount).astype(dtype))

    @override
    @property
    def condition_note(self) -> str:
        return f"alternating above/below '{self._other}' by at least {self._margin}"

    @override
    @property
    def depends_on(self) -> tuple[str, ...]:
        return (self._other,)


class Increasing(_BaseDomain):
    """A strictly increasing one-dimensional grid."""

    __slots__ = ("_gap", "_start")

    def __init__(self, gap: float = 0.2, start: float = 0.0) -> None:
        if gap <= 0:
            msg = "Increasing gap must be positive"
            raise ValueError(msg)
        self._gap = gap
        self._start = start

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if len(shape) != 1:
            msg = f"Increasing requires a 1-D shape, got {shape}"
            raise ValueError(msg)
        steps = st.lists(
            _bounded_float(dtype, low=self._gap, high=2.0 * self._gap),
            min_size=shape[0],
            max_size=shape[0],
        )
        return steps.map(
            lambda values: (self._start + np.cumsum(np.asarray(values, dtype=np.float64))).astype(
                dtype
            ),
        )

    @override
    @property
    def condition_note(self) -> str:
        return f"strictly increasing with gaps of at least {self._gap}"


class Interior(_BaseDomain):
    """Points away from every knot of another argument's grid.

    At least one point is always in an interior cell.  Remaining points may be
    below or above the grid, so interpolation's clamped branches are still
    exercised without making the derivative promise vacuous.
    """

    __slots__ = ("_grid", "_margin", "_outside_fraction")

    def __init__(
        self,
        grid: str,
        margin: float = 0.15,
        outside_fraction: float = 0.25,
    ) -> None:
        if not 0 < margin < 0.5:
            msg = "Interior margin must lie in (0, 0.5)"
            raise ValueError(msg)
        if not 0 <= outside_fraction <= 1:
            msg = "outside_fraction must lie in [0, 1]"
            raise ValueError(msg)
        self._grid = grid
        self._margin = margin
        self._outside_fraction = outside_fraction

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        grid = np.asarray(drawn[self._grid])
        count = _count(shape)
        if grid.ndim != 1 or grid.size < 2:
            msg = "Interior requires a one-dimensional grid with at least two points"
            raise ValueError(msg)
        cell = st.integers(min_value=0, max_value=grid.size - 2)
        offset = _bounded_float(dtype, low=self._margin, high=1.0 - self._margin)
        interior = st.tuples(cell, offset).map(
            lambda pair: float(grid[pair[0]] + pair[1] * (grid[pair[0] + 1] - grid[pair[0]])),
        )
        span = float(grid[-1] - grid[0])
        excursion = _bounded_float(dtype, low=0.1 * span, high=0.5 * span)
        below = excursion.map(lambda amount: float(grid[0] - amount))
        above = excursion.map(lambda amount: float(grid[-1] + amount))
        outside_weight = round(8 * self._outside_fraction)
        element = st.one_of(
            *([interior] * max(1, 8 - outside_weight)),
            *([below, above] * max(1, outside_weight // 2)),
        )
        tail = st.lists(element, min_size=max(0, count - 1), max_size=max(0, count - 1))
        return st.tuples(interior, tail).map(
            lambda pair: np.asarray((pair[0], *pair[1]), dtype=dtype).reshape(shape),
        )

    @override
    @property
    def condition_note(self) -> str:
        return f"cell interiors of '{self._grid}', with at least one in-range point"

    @override
    @property
    def depends_on(self) -> tuple[str, ...]:
        return (self._grid,)


class SpanningGrid(_BaseDomain):
    """A strictly increasing grid that brackets a fixed span."""

    __slots__ = ("_high", "_low")

    def __init__(self, low: float = -1.0, high: float = 1.0) -> None:
        if low >= high:
            msg = "SpanningGrid requires low < high"
            raise ValueError(msg)
        self._low = low
        self._high = high

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if len(shape) != 1 or shape[0] < 2:
            msg = f"SpanningGrid requires a 1-D shape of length >= 2, got {shape}"
            raise ValueError(msg)
        count = shape[0] - 2
        # Positive weights normalised to the span guarantee strict ordering
        # without rejection, even after Hypothesis shrinks every raw value.
        weights = st.lists(
            _bounded_float(dtype, low=0.2, high=1.0),
            min_size=shape[0] - 1,
            max_size=shape[0] - 1,
        )

        def build(raw: list[float]) -> np.ndarray[Any, Any]:
            cumulative = np.cumsum(np.asarray(raw, dtype=np.float64))
            interior = self._low + (self._high - self._low) * cumulative[:-1] / cumulative[-1]
            values = np.concatenate(([self._low], interior[:count], [self._high]))
            return values.astype(dtype)

        return weights.map(build)

    @override
    @property
    def condition_note(self) -> str:
        return f"strictly increasing grid spanning [{self._low}, {self._high}]"


def _matrix_noise_strategy(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    magnitude: float,
) -> SearchStrategy[np.ndarray[Any, Any]]:
    return _real_array(dtype, shape, low=-magnitude, high=magnitude)


class SymmetricPositiveDefinite(_BaseDomain):
    """Hermitian positive-definite matrices with separated eigenvalues."""

    __slots__ = ("_condition",)

    def __init__(self, condition: float = _DEFAULT_CONDITION_LIMIT) -> None:
        if condition <= 1:
            msg = "condition must exceed one"
            raise ValueError(msg)
        self._condition = condition

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if len(shape) < _MATRIX_RANK or shape[-1] != shape[-2]:
            msg = f"SymmetricPositiveDefinite requires a square shape, got {shape}"
            raise ValueError(msg)
        size = shape[-1]
        eigenvalues = np.linspace(1.0, min(self._condition, 4.0), size)

        def build(noise: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            seed = np.sin(
                np.arange(size * size, dtype=np.float64).reshape(size, size) + 0.75,
            )
            seed += np.eye(size)
            dense_seed = np.broadcast_to(seed.astype(dtype), shape)
            basis, _ = np.linalg.qr(dense_seed + noise)
            transpose = np.swapaxes(np.conjugate(basis), -1, -2)
            matrix = (basis * eigenvalues[..., None, :]) @ transpose
            return ((matrix + np.swapaxes(np.conjugate(matrix), -1, -2)) / 2).astype(dtype)

        return _matrix_noise_strategy(shape, dtype, magnitude=0.15).map(build)

    @override
    @property
    def condition_note(self) -> str:
        return (
            "Hermitian positive definite with separated eigenvalues and "
            f"condition number at most {min(self._condition, 4.0)}"
        )


class StableEigensystem(_BaseDomain):
    """Square matrices away from eigenvalue ordering and phase boundaries."""

    __slots__ = ()

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if len(shape) < _MATRIX_RANK or shape[-1] != shape[-2]:
            msg = f"StableEigensystem requires a square shape, got {shape}"
            raise ValueError(msg)
        size = shape[-1]

        def build(noise: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            diagonal = np.linspace(1.0, 4.0, size).astype(dtype)
            identity = np.eye(size, dtype=dtype)
            base = identity * diagonal[..., None, :]
            upper = np.triu(noise, k=1)
            return np.broadcast_to(base, shape) + upper

        return _matrix_noise_strategy(shape, dtype, magnitude=0.05).map(build)

    @override
    @property
    def condition_note(self) -> str:
        return "upper triangular with ordered, separated eigenvalues and stable phase pivots"


class WellConditioned(_BaseDomain):
    """Full-rank matrices kept uniformly away from singularity."""

    __slots__ = ("_condition",)

    def __init__(self, condition: float = _DEFAULT_CONDITION_LIMIT) -> None:
        if condition <= 1:
            msg = "condition must exceed one"
            raise ValueError(msg)
        self._condition = condition

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        if len(shape) < _MATRIX_RANK:
            msg = f"WellConditioned requires a matrix shape, got {shape}"
            raise ValueError(msg)
        rows, columns = shape[-2:]
        rank = min(rows, columns)

        def build(noise: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            diagonal = np.linspace(1.0, min(self._condition, 3.0), rank)
            left_seed = np.sin(
                np.arange(rows * rank, dtype=np.float64).reshape(rows, rank) + 1.0,
            )
            right_seed = np.cos(
                np.arange(columns * rank, dtype=np.float64).reshape(columns, rank) + 0.5,
            )
            left_seed += np.eye(rows, rank)
            right_seed += np.eye(columns, rank)
            left, _ = np.linalg.qr(left_seed)
            right, _ = np.linalg.qr(right_seed)
            dense_base = (left * diagonal[None, :]) @ right.T
            if np.issubdtype(dtype, np.complexfloating):
                phase_seed = np.sin(
                    np.arange(rows * columns, dtype=np.float64).reshape(rows, columns) + 0.37,
                )
                dense_base = dense_base + 0.1j * phase_seed
            base = np.broadcast_to(dense_base.astype(dtype), shape)
            return (base + noise).astype(dtype)

        # ||noise||_2 <= sqrt(rows*columns)*0.02, so the smallest singular
        # value stays comfortably above zero for the small conformance shapes.
        return _matrix_noise_strategy(shape, dtype, magnitude=0.02).map(build)

    @override
    @property
    def condition_note(self) -> str:
        return "full rank with singular values bounded away from zero"
