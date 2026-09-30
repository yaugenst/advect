"""Hypothesis domains for smooth, well-conditioned primitive inputs.

A domain constructs valid values.  It does not draw a seed for a second random
number generator and it does not reject invalid values with ``assume``.  This
keeps Hypothesis in control of generation and shrinking all the way to the
array that reproduces a failure.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any, override

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
    "HermitianTriangle",
    "Increasing",
    "Interior",
    "Lattice",
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

_MATRIX_RANK = 2


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


class Domain:
    """Construct one concrete argument strategy."""

    __slots__ = ()

    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[Any]:
        """Return values of ``shape`` and ``dtype`` satisfying this domain."""
        del shape, dtype, drawn
        raise NotImplementedError

    @property
    def condition_note(self) -> str:
        """Explain the guarantees relevant to derivative checks."""
        return type(self).__name__

    @property
    def depends_on(self) -> tuple[str, ...]:
        """Arguments which must be drawn before this domain is constructed."""
        return ()


class Real(Domain):
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


class Positive(Domain):
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


class Nonzero(Domain):
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


class Unit(Domain):
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


class Lattice(Domain):
    """Exact kinks, ties, zeros and domain edges, shrinking to zero.

    The points are exact in every float dtype, so forward and reverse modes
    see identical primals even where finite differences are meaningless.
    """

    __slots__ = ()
    _POINTS = (0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0)

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        del drawn
        point = st.sampled_from(self._POINTS)
        if np.issubdtype(dtype, np.complexfloating):
            return _array(dtype, shape, st.tuples(point, point).map(lambda pair: complex(*pair)))
        return _array(dtype, shape, point)

    @override
    @property
    def condition_note(self) -> str:
        return f"lattice points {self._POINTS}"


class ClipRegions(Domain):
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


class Distinct(Domain):
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


class SeparatedFrom(Domain):
    """Values separated elementwise from another argument.

    Each element lies above or below the other argument by a drawn side, so
    binary selection primitives exercise both branches while remaining a
    fixed positive distance from their nondifferentiable equality boundary.
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
        offsets = Nonzero(margin=self._margin, high=self._high).strategy(shape, dtype, drawn)
        return offsets.map(lambda offset: (base + offset).astype(dtype))

    @override
    @property
    def condition_note(self) -> str:
        return f"above or below '{self._other}' by at least {self._margin}"

    @override
    @property
    def depends_on(self) -> tuple[str, ...]:
        return (self._other,)


class Increasing(Domain):
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


class Interior(Domain):
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


class SpanningGrid(Domain):
    """A strictly increasing grid that brackets a fixed span.

    Interior knots keep at least ``_MARGIN`` from every ``avoid`` point, such
    as a fixed interpolation query, where a piecewise linear map of the grid
    is kinked.
    """

    __slots__ = ("_avoid", "_high", "_low")
    _MARGIN = 0.1

    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        *,
        avoid: tuple[float, ...] = (),
    ) -> None:
        if low >= high:
            msg = "SpanningGrid requires low < high"
            raise ValueError(msg)
        margin = self._MARGIN
        # A point more than the margin beyond the span is never near a knot;
        # each other window must lie inside the span, apart from the others.
        self._avoid = tuple(
            sorted(point for point in avoid if low - margin <= point <= high + margin)
        )
        edges = (edge for point in self._avoid for edge in (point - margin, point + margin))
        if any(left >= right for left, right in pairwise((low, *edges, high))):
            msg = "SpanningGrid avoided windows must be disjoint and inside the span"
            raise ValueError(msg)
        self._low = low
        self._high = high

    def knots(self, weights: list[float]) -> np.ndarray[Any, Any]:
        """Place the endpoints and one interior knot per weight but the last."""
        width = 2 * self._MARGIN
        free = self._high - self._low - width * len(self._avoid)
        cumulative = np.cumsum(np.asarray(weights, dtype=np.float64))
        interior = self._low + free * cumulative[:-1] / cumulative[-1]
        # Skipping each avoided window in turn keeps the knots strictly increasing.
        for point in self._avoid:
            interior = np.where(interior >= point - self._MARGIN, interior + width, interior)
        return np.concatenate(([self._low], interior, [self._high]))

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
        # Positive weights normalised to the span guarantee strict ordering
        # without rejection, even after Hypothesis shrinks every raw value.
        weights = st.lists(
            _bounded_float(dtype, low=0.2, high=1.0),
            min_size=shape[0] - 1,
            max_size=shape[0] - 1,
        )
        return weights.map(lambda raw: self.knots(raw).astype(dtype))

    @override
    @property
    def condition_note(self) -> str:
        avoided = f", knots at least {self._MARGIN} from {self._avoid}" if self._avoid else ""
        return f"strictly increasing grid spanning [{self._low}, {self._high}]{avoided}"


def _diagonal(matrix: np.ndarray[Any, Any], size: int) -> np.ndarray[Any, Any]:
    return np.diagonal(matrix, axis1=-2, axis2=-1)[..., :size]


class SymmetricPositiveDefinite(Domain):
    """Hermitian positive-definite matrices with separated eigenvalues."""

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
            msg = f"SymmetricPositiveDefinite requires a square shape, got {shape}"
            raise ValueError(msg)
        size = shape[-1]

        def build(noise: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            # The noise tilts the basis and jitters each eigenvalue by at most
            # 0.2 around 1..4, so the spectrum stays separated.
            eigenvalues = np.linspace(1.0, 4.0, size) + 0.4 * np.real(_diagonal(noise, size))
            seed = np.sin(np.arange(size * size, dtype=np.float64).reshape(size, size) + 0.75)
            basis, _ = np.linalg.qr((seed + np.eye(size)).astype(dtype) + noise)
            transpose = np.swapaxes(np.conjugate(basis), -1, -2)
            matrix = (basis * eigenvalues[..., None, :]) @ transpose
            return ((matrix + np.swapaxes(np.conjugate(matrix), -1, -2)) / 2).astype(dtype)

        return _real_array(dtype, shape, low=-0.5, high=0.5).map(build)

    @override
    @property
    def condition_note(self) -> str:
        return "Hermitian positive definite with eigenvalues in [0.8, 4.2] separated by 0.6"


class HermitianTriangle(Domain):
    """A ``SymmetricPositiveDefinite`` triangle beside unrelated data in the other.

    NumPy's Hermitian routines read only their ``UPLO`` triangle, so the laws
    see the unread triangle's data as inert rather than as a mirrored copy.
    """

    __slots__ = ("_uplo",)

    def __init__(self, uplo: str) -> None:
        self._uplo = uplo.upper()

    @override
    def strategy(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        drawn: Mapping[str, Any],
    ) -> SearchStrategy[np.ndarray[Any, Any]]:
        read = np.triu if self._uplo == "U" else np.tril
        return st.tuples(
            SymmetricPositiveDefinite().strategy(shape, dtype, drawn),
            _real_array(dtype, shape, low=-4.0, high=4.0),
        ).map(lambda pair: read(pair[0]) + pair[1] - read(pair[1]))

    @override
    @property
    def condition_note(self) -> str:
        return f"Hermitian positive definite in UPLO={self._uplo!r}, unrelated data elsewhere"


class StableEigensystem(Domain):
    """Dense non-normal matrices away from eigenvalue ordering and phase boundaries.

    ``S diag(lambda) S^-1`` with ``S = I + N`` and ``|N| <= 0.2`` keeps each
    eigenvector's largest component on the diagonal. Complex dtypes draw a
    genuinely complex spectrum.
    """

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
            # Real parts stay 1.5 apart up to a 0.25 jitter; complex dtypes
            # also draw imaginary parts in [-0.6, 0.6].
            jitter = _diagonal(noise, size)
            eigenvalues = np.linspace(1.0, 4.0, size) + 1.25 * np.real(jitter)
            if np.iscomplexobj(noise):
                eigenvalues = eigenvalues + 3j * np.imag(jitter)
            vectors = np.eye(size) + noise
            return (vectors @ (eigenvalues[..., :, None] * np.linalg.inv(vectors))).astype(dtype)

        return _real_array(dtype, shape, low=-0.2, high=0.2).map(build)

    @override
    @property
    def condition_note(self) -> str:
        return "non-normal with separated eigenvalues and diagonal eigenvector pivots"


class WellConditioned(Domain):
    """Full-rank matrices kept uniformly away from singularity."""

    __slots__ = ()

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
            # The real part tilts both singular bases, jitters the singular
            # values by at most 0.2 around 1..3, and flips the first row's
            # sign (and so the determinant's) when its last entry is negative.
            # Complex dtypes add a fixed imaginary part and at most 0.02 of
            # drawn noise, staying off LAPACK's complex QR gauge boundary at
            # real-valued matrices.
            real = np.real(noise)
            singular = np.linspace(1.0, 3.0, rank) + 0.4 * _diagonal(real, rank)
            left_seed = np.sin(np.arange(rows * rank, dtype=np.float64).reshape(rows, rank) + 1.0)
            right_seed = np.cos(
                np.arange(columns * rank, dtype=np.float64).reshape(columns, rank) + 0.5
            )
            left, _ = np.linalg.qr(left_seed + np.eye(rows, rank) + real[..., :, :rank])
            right, _ = np.linalg.qr(
                right_seed + np.eye(columns, rank) + np.swapaxes(real[..., :rank, :], -1, -2)
            )
            matrix = (left * singular[..., None, :]) @ np.swapaxes(right, -1, -2)
            matrix[..., :1, :] *= np.where(real[..., -1:, -1:] < 0, -1.0, 1.0)
            if np.iscomplexobj(noise):
                phase = np.sin(
                    np.arange(rows * columns, dtype=np.float64).reshape(rows, columns) + 0.37
                )
                matrix = matrix + 1j * (0.1 * phase + 0.04 * np.imag(noise))
            return matrix.astype(dtype)

        return _real_array(dtype, shape, low=-0.5, high=0.5).map(build)

    @override
    @property
    def condition_note(self) -> str:
        return "singular values in [0.8, 3.2] with drawn bases and determinant sign"
