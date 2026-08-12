"""Explicit real adjoints for the linear basis used by structural JVPs."""

from __future__ import annotations

from functools import partial
from math import comb, prod
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _array_constructor_like,
    _moveaxis,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    _conjugate_transpose as _h,
)

_PAD_PAIR_LENGTH = 2
_MIN_GRADIENT_POINTS = 2
_MIN_SECOND_ORDER_GRADIENT_POINTS = 3
_SECOND_EDGE_ORDER = 2
_CROSS_VECTOR_LENGTH = 3


def _shape(value: object) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in cast("Any", value).shape)


def _normalize_axis(axis: int, *, ndim: int) -> int:
    normalized = axis
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        msg = f"axis {axis} is out of bounds for rank {ndim}"
        raise ValueError(msg)
    return normalized


def _axis_slice(
    *,
    ndim: int,
    axis: int,
    start: int | None = None,
    stop: int | None = None,
    index: int | None = None,
) -> tuple[int | slice, ...]:
    result: list[int | slice] = [slice(None)] * ndim
    result[axis] = index if index is not None else slice(start, stop)
    return tuple(result)


def _vjp_concatenate(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    axis: int | None = 0,
    **attrs: Any,
) -> tuple[xp.ndarray, ...]:
    """Split a concatenated cotangent back into its source arrays."""
    _ = ans, attrs
    if axis is None:
        flat = xp.reshape(g, (-1,))
        offset = 0
        outputs: list[xp.ndarray] = []
        for value in inputs:
            size = prod(_shape(value))
            outputs.append(xp.reshape(flat[offset : offset + size], _shape(value)))
            offset += size
        return tuple(outputs)

    normalized_axis = _normalize_axis(axis, ndim=g.ndim)
    offset = 0
    outputs = []
    for value in inputs:
        width = _shape(value)[normalized_axis]
        index = _axis_slice(
            ndim=g.ndim,
            axis=normalized_axis,
            start=offset,
            stop=offset + width,
        )
        outputs.append(g[index])
        offset += width
    return tuple(outputs)


def _vjp_stack(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    axis: int = 0,
    **attrs: Any,
) -> tuple[xp.ndarray, ...]:
    """Remove the inserted stack axis for each source cotangent."""
    _ = ans, attrs
    normalized_axis = _normalize_axis(axis, ndim=g.ndim)
    return tuple(
        g[_axis_slice(ndim=g.ndim, axis=normalized_axis, index=index)]
        for index in range(len(inputs))
    )


def _vjp_ravel(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Restore the source shape after flattening."""
    _ = ans, rest, attrs
    return (xp.reshape(g, _shape(x)),)


def _vjp_swapaxes(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    axis1: int = 0,
    axis2: int = 1,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Swap the same two axes in the cotangent."""
    _ = ans, inputs, attrs
    return (xp.swapaxes(g, axis1, axis2),)


def _vjp_flip(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    axis: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Reverse the same axes in the cotangent."""
    _ = ans, inputs, attrs
    return (xp.flip(g, axis=axis),)


_vjp_fliplr = partial(_vjp_flip, axis=1)
_vjp_flipud = partial(_vjp_flip, axis=0)


def _vjp_roll(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    shift: int | tuple[int, ...] = 0,
    axis: int | tuple[int, ...] | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Roll by the inverse displacement."""
    _ = ans, inputs, attrs
    inverse_shift = tuple(-component for component in shift) if isinstance(shift, tuple) else -shift
    return (xp.roll(g, shift=inverse_shift, axis=axis),)


def _vjp_rot90(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    k: int = 1,
    axes: tuple[int, int] = (0, 1),
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, inputs, attrs
    return (xp.rot90(g, k=-k, axes=axes),)


def _vjp_rollaxis(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    axis: int,
    start: int = 0,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Move the rolled output axis back to its source position."""
    _ = ans, inputs, attrs
    source_axis = _normalize_axis(axis, ndim=g.ndim)
    destination = start
    if destination < 0:
        destination += g.ndim
    if destination < 0 or destination > g.ndim:
        msg = f"start {start} is out of bounds for rank {g.ndim}"
        raise ValueError(msg)
    if source_axis < destination:
        destination -= 1
    return (_moveaxis(g, destination, source_axis),)


def _vjp_triangular(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    k: int = 0,
    upper: bool,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    _ = ans, inputs, attrs
    return (xp.triu(g, k=k) if upper else xp.tril(g, k=k),)


_vjp_triu = partial(_vjp_triangular, upper=True)
_vjp_tril = partial(_vjp_triangular, upper=False)
_vjp_atleast = _vjp_ravel


def _vjp_diag(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    k: int = 0,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Apply the adjoint diagonal map."""
    _ = ans, inputs, attrs
    return (xp.diag(g, k=k),)


def _diagonal_positions(
    *,
    first_length: int,
    second_length: int,
    offset: int,
) -> tuple[tuple[int, int], ...]:
    first_start = max(0, -offset)
    second_start = max(0, offset)
    length = min(first_length - first_start, second_length - second_start)
    return tuple((first_start + index, second_start + index) for index in range(max(0, length)))


def _vjp_diagonal(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Scatter a diagonal cotangent back into the source axes."""
    _ = ans, rest, attrs
    first_axis = _normalize_axis(axis1, ndim=x.ndim)
    second_axis = _normalize_axis(axis2, ndim=x.ndim)
    if first_axis == second_axis:
        msg = "diagonal axes must be distinct"
        raise ValueError(msg)
    result = _array_constructor_like(g, "zeros_like", x)
    positions = _diagonal_positions(
        first_length=int(x.shape[first_axis]),
        second_length=int(x.shape[second_axis]),
        offset=offset,
    )
    for diagonal_index, (first_index, second_index) in enumerate(positions):
        destination: list[int | slice] = [slice(None)] * x.ndim
        destination[first_axis] = first_index
        destination[second_axis] = second_index
        source: list[int | slice] = [slice(None)] * g.ndim
        source[-1] = diagonal_index
        result[tuple(destination)] += g[tuple(source)]
    return (result,)


def _vjp_trace(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Scatter one trace cotangent across the selected diagonal."""
    _ = ans, rest, attrs
    first_axis = _normalize_axis(axis1, ndim=x.ndim)
    second_axis = _normalize_axis(axis2, ndim=x.ndim)
    if first_axis == second_axis:
        msg = "trace axes must be distinct"
        raise ValueError(msg)
    result = _array_constructor_like(g, "zeros_like", x)
    positions = _diagonal_positions(
        first_length=int(x.shape[first_axis]),
        second_length=int(x.shape[second_axis]),
        offset=offset,
    )
    for first_index, second_index in positions:
        destination: list[int | slice] = [slice(None)] * x.ndim
        destination[first_axis] = first_index
        destination[second_axis] = second_index
        result[tuple(destination)] += g
    return (result,)


def _vjp_cumsum(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    axis: int | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Apply a reverse cumulative sum."""
    _ = ans, rest, attrs
    if axis is None:
        flat = xp.reshape(g, (-1,))
        pulled = xp.flip(xp.cumsum(xp.flip(flat, axis=0), axis=0), axis=0)
        return (xp.reshape(pulled, _shape(x)),)
    normalized_axis = _normalize_axis(axis, ndim=g.ndim)
    return (
        xp.flip(
            xp.cumsum(xp.flip(g, axis=normalized_axis), axis=normalized_axis),
            axis=normalized_axis,
        ),
    )


def _normalized_pad_width(
    pad_width: int | tuple[int, int] | tuple[tuple[int, int], ...],
    *,
    ndim: int,
) -> tuple[tuple[int, int], ...]:
    if isinstance(pad_width, int):
        return ((pad_width, pad_width),) * ndim
    raw = tuple(pad_width)
    if len(raw) == _PAD_PAIR_LENGTH and all(isinstance(value, int) for value in raw):
        before, after = cast("tuple[int, int]", raw)
        return ((before, after),) * ndim
    result = tuple((int(pair[0]), int(pair[1])) for pair in cast("tuple[Any, ...]", raw))
    if len(result) == 1:
        return result * ndim
    if len(result) != ndim:
        msg = f"pad_width has {len(result)} axes for rank {ndim}"
        raise ValueError(msg)
    return result


def _vjp_pad(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    pad_width: int | tuple[int, int] | tuple[tuple[int, int], ...] = 0,
    mode: str = "constant",
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Crop the cotangent of constant padding."""
    _ = ans, rest, attrs
    if mode != "constant":
        msg = f"numpy.pad transpose only supports mode='constant' (got {mode!r})"
        raise NotImplementedError(msg)
    widths = _normalized_pad_width(pad_width, ndim=x.ndim)
    index = tuple(
        slice(before, before + size)
        for (before, _after), size in zip(widths, _shape(x), strict=True)
    )
    return (g[index],)


def _static_axis_extent(value: object | None, *, axis: int, ndim: int) -> int:
    if value is None:
        return 0
    shape = getattr(value, "shape", ())
    if not shape:
        return 1
    value_shape = tuple(int(dimension) for dimension in shape)
    normalized = _normalize_axis(axis, ndim=ndim)
    return value_shape[normalized]


def _vjp_diff(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    n: int = 1,
    axis: int = -1,
    prepend: object | None = None,
    append: object | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose finite differences, including static prepend/append values."""
    _ = ans, rest, attrs
    order = n
    if order < 0:
        msg = f"numpy.diff transpose requires n >= 0 (got {n})"
        raise ValueError(msg)
    if order == 0:
        return (g,)

    normalized_axis = _normalize_axis(axis, ndim=x.ndim)
    input_length = _shape(x)[normalized_axis]
    output_length = _shape(g)[normalized_axis]
    prepend_length = _static_axis_extent(prepend, axis=normalized_axis, ndim=x.ndim)
    _ = append

    # The zero-times-sum term lifts a concrete zero base into an enclosing
    # trace before functional indexed updates add tracer cotangents.
    result = _array_constructor_like(g, "zeros_like", x)
    for k in range(order + 1):
        coefficient = (-1) ** (order - k) * comb(order, k)
        offset = prepend_length - k
        destination_start = max(0, -offset)
        destination_stop = min(input_length, output_length - offset)
        if destination_start >= destination_stop:
            continue
        source_start = destination_start + offset
        source_stop = destination_stop + offset
        destination = _axis_slice(
            ndim=x.ndim,
            axis=normalized_axis,
            start=destination_start,
            stop=destination_stop,
        )
        source = _axis_slice(
            ndim=g.ndim,
            axis=normalized_axis,
            start=source_start,
            stop=source_stop,
        )
        result[destination] += coefficient * g[source]
    return (result,)


def _vjp_repeat(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    repeats: int = 1,
    axis: int | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Sum cotangents from each repeated copy."""
    _ = ans, rest, attrs
    repeat_count = repeats
    if repeat_count < 0:
        msg = f"repeat transpose requires repeats >= 0 (got {repeats})"
        raise ValueError(msg)

    source_shape = _shape(x)
    if axis is None:
        source_size = 1
        for extent in source_shape:
            source_size *= extent
        grouped = xp.reshape(g, (source_size, repeat_count))
        return (xp.reshape(xp.sum(grouped, axis=1), source_shape),)

    normalized_axis = _normalize_axis(axis, ndim=x.ndim)
    grouped_shape = (
        *source_shape[: normalized_axis + 1],
        repeat_count,
        *source_shape[normalized_axis + 1 :],
    )
    grouped = xp.reshape(g, grouped_shape)
    return (xp.sum(grouped, axis=normalized_axis + 1),)


def _vjp_tile(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    reps: int | tuple[int, ...] = 1,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Sum cotangents over every tiled copy."""
    _ = ans, rest, attrs
    repetitions = (reps,) if isinstance(reps, int) else tuple(reps)
    if any(value < 0 for value in repetitions):
        msg = f"tile transpose requires non-negative reps (got {reps!r})"
        raise ValueError(msg)

    source_shape = _shape(x)
    rank = max(len(source_shape), len(repetitions))
    padded_source = (1,) * (rank - len(source_shape)) + source_shape
    padded_repetitions = (1,) * (rank - len(repetitions)) + repetitions
    grouped_shape = tuple(
        extent
        for repetition, source_extent in zip(
            padded_repetitions,
            padded_source,
            strict=True,
        )
        for extent in (repetition, source_extent)
    )
    grouped = xp.reshape(g, grouped_shape)
    repetition_axes = tuple(range(0, 2 * rank, 2))
    reduced = xp.sum(grouped, axis=repetition_axes)
    return (xp.reshape(reduced, source_shape),)


def _vjp_gradient(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    axis: int = 0,
    edge_order: int = 1,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose NumPy's unit-spacing first-order finite-difference stencil."""
    _ = ans, rest, attrs
    order = edge_order
    if order not in {1, 2}:
        msg = f"gradient transpose only supports edge_order=1 or 2 (got {edge_order})"
        raise NotImplementedError(msg)

    normalized_axis = _normalize_axis(axis, ndim=x.ndim)
    length = int(x.shape[normalized_axis])
    minimum_length = (
        _MIN_SECOND_ORDER_GRADIENT_POINTS if order == _SECOND_EDGE_ORDER else _MIN_GRADIENT_POINTS
    )
    if length < minimum_length:
        msg = (
            f"gradient transpose with edge_order={order} requires at least "
            f"{minimum_length} points along its axis"
        )
        raise ValueError(msg)

    result = _array_constructor_like(g, "zeros_like", x)
    first = _axis_slice(ndim=x.ndim, axis=normalized_axis, index=0)
    second = _axis_slice(ndim=x.ndim, axis=normalized_axis, index=1)
    third = _axis_slice(ndim=x.ndim, axis=normalized_axis, index=2)
    antepenultimate = _axis_slice(ndim=x.ndim, axis=normalized_axis, index=length - 3)
    penultimate = _axis_slice(ndim=x.ndim, axis=normalized_axis, index=length - 2)
    last = _axis_slice(ndim=x.ndim, axis=normalized_axis, index=length - 1)
    if order == 1:
        result[first] += -g[first]
        result[second] += g[first]
        result[penultimate] += -g[last]
        result[last] += g[last]
    else:
        result[first] += -1.5 * g[first]
        result[second] += 2.0 * g[first]
        result[third] += -0.5 * g[first]
        result[antepenultimate] += 0.5 * g[last]
        result[penultimate] += -2.0 * g[last]
        result[last] += 1.5 * g[last]
    if length > _MIN_GRADIENT_POINTS:
        interior = _axis_slice(
            ndim=x.ndim,
            axis=normalized_axis,
            start=1,
            stop=length - 1,
        )
        before = _axis_slice(
            ndim=x.ndim,
            axis=normalized_axis,
            start=0,
            stop=length - 2,
        )
        after = _axis_slice(
            ndim=x.ndim,
            axis=normalized_axis,
            start=2,
            stop=length,
        )
        result[before] += -0.5 * g[interior]
        result[after] += 0.5 * g[interior]
    return (result,)


def _vjp_inner(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose NumPy inner products of arbitrary-rank operands."""
    _ = ans, rest, attrs
    if a.ndim == 0 or b.ndim == 0:
        return g * xp.conj(b), g * xp.conj(a)

    a_prefix_rank = a.ndim - 1
    b_prefix_rank = b.ndim - 1
    g_b_axes = tuple(range(a_prefix_rank, a_prefix_rank + b_prefix_rank))
    b_prefix_axes = tuple(range(b_prefix_rank))
    a_grad = xp.tensordot(g, xp.conj(b), axes=(g_b_axes, b_prefix_axes))

    a_prefix_axes = tuple(range(a_prefix_rank))
    g_a_axes = tuple(range(a_prefix_rank))
    b_grad = xp.tensordot(xp.conj(a), g, axes=(a_prefix_axes, g_a_axes))
    if b_prefix_rank:
        b_grad = _moveaxis(b_grad, 0, -1)
    return a_grad, b_grad


def _vjp_outer(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose a flattened outer product."""
    _ = ans, rest, attrs
    a_flat = xp.reshape(a, (-1,))
    b_flat = xp.reshape(b, (-1,))
    a_grad = xp.reshape(xp.matmul(g, xp.conj(b_flat)), _shape(a))
    b_grad = xp.reshape(xp.matmul(xp.conj(a_flat), g), _shape(b))
    return a_grad, b_grad


def _vjp_cross(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose a three-dimensional vector cross product."""
    _ = ans, rest, attrs
    a_axis = _normalize_axis(axis if axis is not None else axisa, ndim=a.ndim)
    b_axis = _normalize_axis(axis if axis is not None else axisb, ndim=b.ndim)
    if int(a.shape[a_axis]) != _CROSS_VECTOR_LENGTH or int(b.shape[b_axis]) != _CROSS_VECTOR_LENGTH:
        msg = "cross transpose supports only three-component vectors"
        raise NotImplementedError(msg)
    g_axis = _normalize_axis(axis if axis is not None else axisc, ndim=g.ndim)
    a_grad = xp.cross(
        xp.conj(b),
        g,
        axisa=b_axis,
        axisb=g_axis,
        axisc=a_axis,
        axis=axis,
    )
    b_grad = xp.cross(
        g,
        xp.conj(a),
        axisa=g_axis,
        axisb=a_axis,
        axisc=b_axis,
        axis=axis,
    )
    return a_grad, b_grad


def _vjp_kron(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose a Kronecker product without materializing a basis."""
    _ = ans, rest, attrs
    rank = max(a.ndim, b.ndim)
    if rank == 0:
        return g * xp.conj(b), g * xp.conj(a)

    a_shape = (1,) * (rank - a.ndim) + _shape(a)
    b_shape = (1,) * (rank - b.ndim) + _shape(b)
    grouped_shape = tuple(
        extent
        for a_extent, b_extent in zip(a_shape, b_shape, strict=True)
        for extent in (a_extent, b_extent)
    )
    grouped = xp.reshape(g, grouped_shape)
    a_broadcast_shape = tuple(extent for value in a_shape for extent in (value, 1))
    b_broadcast_shape = tuple(extent for value in b_shape for extent in (1, value))
    a_grad = xp.sum(
        grouped * xp.conj(xp.reshape(b, b_broadcast_shape)),
        axis=tuple(range(1, 2 * rank, 2)),
    )
    b_grad = xp.sum(
        grouped * xp.conj(xp.reshape(a, a_broadcast_shape)),
        axis=tuple(range(0, 2 * rank, 2)),
    )
    return (
        xp.reshape(a_grad, _shape(a)),
        xp.reshape(b_grad, _shape(b)),
    )


def _vjp_linspace(
    ans: xp.ndarray,
    *inputs: xp.ndarray,
    g: xp.ndarray,
    num: int = 50,
    endpoint: bool = True,
    axis: int = 0,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Transpose the affine interpolation from start and stop."""
    _ = ans, inputs, attrs
    sample_count = num
    normalized_axis = _normalize_axis(axis, ndim=g.ndim)
    denominator = sample_count - 1 if endpoint and sample_count > 1 else max(sample_count, 1)
    positions = xp.arange(sample_count, dtype=xp.real(g).dtype) / denominator
    if endpoint and sample_count == 1:
        positions = positions * 0
    coefficient_shape = [1] * g.ndim
    coefficient_shape[normalized_axis] = sample_count
    stop_weight = xp.reshape(positions, tuple(coefficient_shape))
    start_grad = xp.sum(g * (1 - stop_weight), axis=normalized_axis)
    stop_grad = xp.sum(g * stop_weight, axis=normalized_axis)
    return cast("xp.ndarray", start_grad), cast("xp.ndarray", stop_grad)


def _vjp_solve(
    ans: xp.ndarray,
    a: xp.ndarray,
    b: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    **attrs: Any,
) -> tuple[xp.ndarray, xp.ndarray]:
    """Real adjoint of ``solve(a, b)`` for vector or matrix right-hand sides."""
    _ = rest, attrs
    rhs_grad = xp.linalg.solve(_h(a), g)
    if ans.ndim == a.ndim - 1:
        matrix_grad = -rhs_grad[..., :, None] * xp.conj(ans[..., None, :])
    else:
        matrix_grad = -xp.matmul(rhs_grad, _h(ans))
    return cast("xp.ndarray", matrix_grad), cast("xp.ndarray", rhs_grad)
