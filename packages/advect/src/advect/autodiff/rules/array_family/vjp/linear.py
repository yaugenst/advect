"""Explicit real adjoints for the linear basis used by structural JVPs."""

from __future__ import annotations

from functools import partial
from math import prod
from typing import Any, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _array_constructor_like,
    _moveaxis,
    _scalar_like,
    _zero_pad_axis,
    xp,
)
from advect.autodiff.rules.array_family._transpose_utils import (
    _axis_slice,
    _conjugate_transpose as _h,
    _difference,
    _normalize_axis,
    _ravel_axes,
    _shape_of,
    _tile_layout,
)
from advect.autodiff.rules.array_family.vjp.linalg.contractions import _contraction_vjp

_PAD_PAIR_LENGTH = 2
_MIN_GRADIENT_POINTS = 2
_MIN_SECOND_ORDER_GRADIENT_POINTS = 3
_SECOND_EDGE_ORDER = 2
_CROSS_VECTOR_LENGTH = 3
_MATRIX_RANK = 2


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
            size = prod(_shape_of(value))
            outputs.append(xp.reshape(flat[offset : offset + size], _shape_of(value)))
            offset += size
        return tuple(outputs)

    normalized_axis = _normalize_axis(axis, ndim=g.ndim, op_name="concatenate")
    offset = 0
    outputs = []
    for value in inputs:
        width = _shape_of(value)[normalized_axis]
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
    normalized_axis = _normalize_axis(axis, ndim=g.ndim, op_name="stack")
    return tuple(
        g[_axis_slice(ndim=g.ndim, axis=normalized_axis, index=index)]
        for index in range(len(inputs))
    )


def _vjp_ravel(
    ans: xp.ndarray,
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    order: str | None = None,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Restore the source shape in the order NumPy's ravel read the primal."""
    _ = ans, rest, attrs
    axes = _ravel_axes(x, order)
    source_shape = _shape_of(x)
    restored = xp.reshape(g, tuple(source_shape[axis] for axis in axes))
    if axes == tuple(sorted(axes)):
        return (restored,)
    return (xp.permute_dims(restored, tuple(axes.index(axis) for axis in range(len(axes)))),)


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
    source_axis = _normalize_axis(axis, ndim=g.ndim, op_name="rollaxis")
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
    x: xp.ndarray,
    *rest: xp.ndarray,
    g: xp.ndarray,
    k: int = 0,
    **attrs: Any,
) -> tuple[xp.ndarray]:
    """Transpose building a diagonal matrix or extracting a matrix diagonal."""
    _ = rest, attrs
    if x.ndim == _MATRIX_RANK:
        # The source matrix may be rectangular; its shape fixes the adjoint.
        return _vjp_diagonal(ans, x, g=g, offset=k)
    return (xp.diag(g, k=k),)


def _matrix_axes(x: xp.ndarray, axis1: int, axis2: int, *, operation: str) -> tuple[int, int]:
    first_axis = _normalize_axis(axis1, ndim=x.ndim, op_name=operation)
    second_axis = _normalize_axis(axis2, ndim=x.ndim, op_name=operation)
    if first_axis == second_axis:
        msg = f"{operation} axes must be distinct"
        raise ValueError(msg)
    return first_axis, second_axis


def _place_on_diagonal(
    g: xp.ndarray,
    spread: xp.ndarray,
    x: xp.ndarray,
    *,
    offset: int,
    axes: tuple[int, int],
) -> xp.ndarray:
    """Select ``spread`` on the ``offset`` diagonal of ``axes`` and zero elsewhere.

    Selecting rather than multiplying by a 0/1 basis keeps non-finite
    cotangent entries on the diagonal; the zeros are built in the trace of ``g``.
    """
    rows, columns = (int(x.shape[axis]) for axis in axes)
    mask = _array_constructor_like(g, "eye", rows, columns, k=offset, dtype=xp.bool)
    placed = xp.where(mask, spread, _scalar_like(0, g))
    return _moveaxis(placed, (-2, -1), axes)


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
    """Spread a diagonal cotangent back into the source axes."""
    _ = ans, rest, attrs
    axes = _matrix_axes(x, axis1, axis2, operation="diagonal")
    rows, columns = (int(x.shape[axis]) for axis in axes)
    # Diagonal entry d sits in row d above the main diagonal and in column d
    # below it, so zero-pad the cotangent to that axis and select the diagonal.
    batch = _shape_of(g)[:-1]
    length = _shape_of(g)[-1]
    if offset >= 0:
        rows_first = _zero_pad_axis(g, axis=-1, before=0, after=rows - length)
        spread = xp.reshape(rows_first, (*batch, rows, 1))
    else:
        columns_first = _zero_pad_axis(g, axis=-1, before=0, after=columns - length)
        spread = xp.reshape(columns_first, (*batch, 1, columns))
    return (_place_on_diagonal(g, spread, x, offset=offset, axes=axes),)


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
    """Spread one trace cotangent across the selected diagonal."""
    _ = ans, rest, attrs
    axes = _matrix_axes(x, axis1, axis2, operation="trace")
    spread = xp.reshape(g, (*_shape_of(g), 1, 1))
    return (_place_on_diagonal(g, spread, x, offset=offset, axes=axes),)


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
        return (xp.reshape(pulled, _shape_of(x)),)
    normalized_axis = _normalize_axis(axis, ndim=g.ndim, op_name="cumsum")
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
        for (before, _after), size in zip(widths, _shape_of(x), strict=True)
    )
    return (g[index],)


def _static_axis_extent(value: object | None, *, axis: int, ndim: int) -> int:
    if value is None:
        return 0
    shape = getattr(value, "shape", ())
    if not shape:
        return 1
    value_shape = tuple(int(dimension) for dimension in shape)
    normalized = _normalize_axis(axis, ndim=ndim, op_name="diff")
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
) -> tuple[xp.ndarray, ...]:
    """Transpose finite differences, including prepend and append operands.

    The adjoint of ``diff`` over the extended axis is
    ``(-1)**n * diff(pad(g, n, n), n)``; each operand receives its segment.
    """
    _ = ans
    order = n
    if order < 0:
        msg = f"numpy.diff transpose requires n >= 0 (got {n})"
        raise ValueError(msg)
    if order == 0:
        return (g,)

    normalized_axis = _normalize_axis(axis, ndim=x.ndim, op_name="diff")
    operands = iter(rest)
    prepend_is_input = bool(attrs.get("_advect_diff_prepend_input", False))
    append_is_input = bool(attrs.get("_advect_diff_append_input", False))
    prepend_value = next(operands) if prepend_is_input else prepend
    append_value = next(operands) if append_is_input else append
    prepend_length = _static_axis_extent(prepend_value, axis=normalized_axis, ndim=x.ndim)
    input_stop = prepend_length + _shape_of(x)[normalized_axis]
    append_stop = input_stop + _static_axis_extent(
        append_value,
        axis=normalized_axis,
        ndim=x.ndim,
    )

    extended = _difference(
        _zero_pad_axis(g, axis=normalized_axis, before=order, after=order),
        order=order,
        axis=normalized_axis,
    )
    if order % 2:
        extended = -extended

    def segment(start: int, stop: int) -> xp.ndarray:
        return extended[_axis_slice(ndim=x.ndim, axis=normalized_axis, start=start, stop=stop)]

    contributions = [segment(prepend_length, input_stop)]
    if prepend_is_input:
        contributions.append(segment(0, prepend_length))
    if append_is_input:
        contributions.append(segment(input_stop, append_stop))
    return tuple(contributions)


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

    source_shape = _shape_of(x)
    if axis is None or not source_shape:
        # NumPy repeats a rank-0 source along axis 0 or -1 as its one-element
        # flattening.
        grouped = xp.reshape(g, (prod(source_shape), repeat_count))
        return (xp.reshape(xp.sum(grouped, axis=1), source_shape),)

    normalized_axis = _normalize_axis(axis, ndim=x.ndim, op_name="repeat")
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
    source_shape = _shape_of(x)
    _, pairs = _tile_layout(source_shape, reps)
    if any(count < 0 for count in pairs[::2]):
        msg = f"tile transpose requires non-negative reps (got {reps!r})"
        raise ValueError(msg)
    reduced = xp.sum(xp.reshape(g, pairs), axis=tuple(range(0, len(pairs), 2)))
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

    normalized_axis = _normalize_axis(axis, ndim=x.ndim, op_name="gradient")
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

    def along(start: int, stop: int) -> tuple[int | slice, ...]:
        return _axis_slice(ndim=x.ndim, axis=normalized_axis, start=start, stop=stop)

    def place(value: xp.ndarray, start: int) -> xp.ndarray:
        width = _shape_of(value)[normalized_axis]
        return _zero_pad_axis(
            value,
            axis=normalized_axis,
            before=start,
            after=length - start - width,
        )

    # One-sided edge stencils, then the central interior stencil.
    head_weights, tail_weights = (
        ((-1.0, 1.0), (-1.0, 1.0)) if order == 1 else ((-1.5, 2.0, -0.5), (0.5, -2.0, 1.5))
    )
    first = g[along(0, 1)]
    last = g[along(length - 1, length)]
    head = xp.concatenate(tuple(weight * first for weight in head_weights), axis=normalized_axis)
    tail = xp.concatenate(tuple(weight * last for weight in tail_weights), axis=normalized_axis)
    result = place(head, 0) + place(tail, length - len(tail_weights))
    if length > _MIN_GRADIENT_POINTS:
        interior = 0.5 * g[along(1, length - 1)]
        result = result + place(interior, 2) - place(interior, 0)
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
    return cast(
        "tuple[xp.ndarray, xp.ndarray]",
        _contraction_vjp(
            a=a,
            b=b,
            g=g,
            a_axes=(a.ndim - 1,),
            b_axes=(b.ndim - 1,),
            op_name="numpy.inner",
        ),
    )


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
    a_grad = xp.reshape(xp.matmul(g, xp.conj(b_flat)), _shape_of(a))
    b_grad = xp.reshape(xp.matmul(xp.conj(a_flat), g), _shape_of(b))
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
    a_axis = _normalize_axis(axis if axis is not None else axisa, ndim=a.ndim, op_name="cross")
    b_axis = _normalize_axis(axis if axis is not None else axisb, ndim=b.ndim, op_name="cross")
    if int(a.shape[a_axis]) != _CROSS_VECTOR_LENGTH or int(b.shape[b_axis]) != _CROSS_VECTOR_LENGTH:
        msg = "cross transpose supports only three-component vectors"
        raise NotImplementedError(msg)
    g_axis = _normalize_axis(axis if axis is not None else axisc, ndim=g.ndim, op_name="cross")
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

    a_shape = (1,) * (rank - a.ndim) + _shape_of(a)
    b_shape = (1,) * (rank - b.ndim) + _shape_of(b)
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
        xp.reshape(a_grad, _shape_of(a)),
        xp.reshape(b_grad, _shape_of(b)),
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
    normalized_axis = _normalize_axis(axis, ndim=g.ndim, op_name="linspace")
    denominator = sample_count - 1 if endpoint and sample_count > 1 else max(sample_count, 1)
    positions = xp.arange(sample_count, dtype=xp.real(g).dtype) / denominator
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
        matrix_grad = -xp.multiply(rhs_grad[..., :, None], xp.conj(ans[..., None, :]))
    else:
        matrix_grad = -xp.matmul(rhs_grad, _h(ans))
    return cast("xp.ndarray", matrix_grad), cast("xp.ndarray", rhs_grad)
