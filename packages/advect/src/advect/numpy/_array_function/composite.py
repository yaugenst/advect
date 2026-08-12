# ruff: noqa: ANN401
# Composite lowerings intentionally accept both concrete arrays and tracers.
"""NumPy conveniences lowered to Advect's existing differentiable operations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced

np: Any = _numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionHandler

type CompositeResult = tuple[Any, Any]

_BINARY_ARITY = 2
_MATRIX_RANK = 2
_NEGATIVE_SPECTRAL_ORDER = -2
_TERNARY_ARITY = 3


def _ndim(value: object) -> int:
    ndim = getattr(value, "ndim", None)
    return int(ndim) if ndim is not None else int(np.ndim(value))


def _normalize_axes(axis: object, ndim: int) -> tuple[int, ...]:
    if isinstance(axis, (int, np.integer)):
        raw = (axis,)
    elif isinstance(axis, Iterable):
        raw = tuple(axis)
    else:
        msg = f"axis {axis!r} is not iterable"
        raise TypeError(msg)
    normalized = tuple(int(item) if int(item) >= 0 else int(item) + ndim for item in raw)
    if any(item < 0 or item >= ndim for item in normalized):
        msg = f"axis {axis!r} is out of bounds for ndim={ndim}"
        raise TracingError(msg)
    if len(set(normalized)) != len(normalized):
        msg = f"axis {axis!r} contains duplicates"
        raise TracingError(msg)
    return normalized


def _finish(
    result: object,
    *,
    traced_type: type[TracedArrayLike],
) -> CompositeResult:
    def rebuild(template: object, items: list[object]) -> object:
        if isinstance(template, list):
            return items
        if type(template) is tuple:
            return tuple(items)
        return type(template)(*items)

    def snapshot_tree(value: object) -> tuple[object, object]:
        if isinstance(value, traced_type):
            node_id, concrete = _snapshot_traced(value)
            return concrete, node_id
        if isinstance(value, (tuple, list)):
            children = [snapshot_tree(item) for item in value]
            return (
                rebuild(value, [concrete for concrete, _node_id in children]),
                rebuild(value, [node_id for _concrete, node_id in children]),
            )
        msg = (
            "A composite NumPy lowering did not produce traced array output; "
            f"got {type(value).__name__}"
        )
        raise TracingError(msg)

    return snapshot_tree(result)


def _sequence_arg(
    name: str,
    args: tuple[Any, ...],
) -> tuple[Any, ...]:
    arrays = args[0]
    if not isinstance(arrays, (tuple, list)) or not arrays:
        msg = f"numpy.{name} requires a non-empty tuple or list during tracing"
        raise TracingError(msg)
    return tuple(arrays)


def _hstack_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    arrays = args[0]
    if not isinstance(arrays, (tuple, list)) or not arrays:
        msg = "numpy.hstack requires a non-empty tuple or list during tracing"
        raise TracingError(msg)
    promoted = tuple(np.atleast_1d(item) for item in arrays)
    axis = 0 if promoted[0].ndim == 1 else 1
    return _finish(
        np.concatenate(promoted, axis=axis, **kwargs),
        traced_type=traced_type,
    )


def _vstack_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    arrays = args[0]
    if not isinstance(arrays, (tuple, list)) or not arrays:
        msg = "numpy.vstack requires a non-empty tuple or list during tracing"
        raise TracingError(msg)
    promoted = tuple(np.atleast_2d(item) for item in arrays)
    return _finish(
        np.concatenate(promoted, axis=0, **kwargs),
        traced_type=traced_type,
    )


def _dstack_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    arrays = _sequence_arg("dstack", args)
    promoted = tuple(np.atleast_3d(item) for item in arrays)
    return _finish(np.concatenate(promoted, axis=2), traced_type=traced_type)


def _column_stack_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    arrays = _sequence_arg("column_stack", args)
    columns = tuple(
        np.reshape(item, (-1, 1)) if _ndim(item) < _MATRIX_RANK else item for item in arrays
    )
    return _finish(np.concatenate(columns, axis=1), traced_type=traced_type)


def _append_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    array, values = args[:2]
    axis = args[2] if len(args) == _TERNARY_ARITY else kwargs.get("axis")
    if axis is None:
        array = np.ravel(array)
        values = np.ravel(values)
        axis = 0
    return _finish(
        np.concatenate((array, values), axis=int(axis)),
        traced_type=traced_type,
    )


def _delete_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    array, obj = args[:2]
    if isinstance(obj, traced_type):
        msg = "numpy.delete obj= must be static because it controls the output shape"
        raise TracingError(msg)
    axis_raw = args[2] if len(args) == _TERNARY_ARITY else kwargs.get("axis")
    if axis_raw is None:
        array = np.ravel(array)
        axis = 0
    else:
        axis = int(axis_raw)
    size = int(array.shape[axis])
    keep = np.delete(np.arange(size), obj)
    return _finish(np.take(array, keep, axis=axis), traced_type=traced_type)


def _diagflat_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    k = int(args[1] if len(args) == _BINARY_ARITY else kwargs.get("k", 0))
    return _finish(np.diag(np.ravel(args[0]), k=k), traced_type=traced_type)


def _ediff1d_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    positional_names = ("to_end", "to_begin")
    values = dict(kwargs)
    values.update(zip(positional_names, args[1:], strict=False))
    pieces: list[object] = []
    if values.get("to_begin") is not None:
        pieces.append(np.atleast_1d(values["to_begin"]))
    pieces.append(np.diff(np.ravel(args[0])))
    if values.get("to_end") is not None:
        pieces.append(np.atleast_1d(values["to_end"]))
    result = pieces[0] if len(pieces) == 1 else np.concatenate(tuple(pieces))
    return _finish(result, traced_type=traced_type)


def _resize_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    array, new_shape_raw = args
    if isinstance(new_shape_raw, traced_type):
        msg = "numpy.resize new_shape must be static during tracing"
        raise TracingError(msg)
    shape_array = np.asarray(new_shape_raw)
    new_shape = (
        (int(shape_array.item()),)
        if shape_array.ndim == 0
        else tuple(int(item) for item in shape_array.tolist())
    )
    output_size = math.prod(new_shape)
    flat = np.ravel(array)
    if output_size == 0:
        result = np.reshape(flat[:0], new_shape)
    elif flat.size == 0:
        result = np.zeros(new_shape, dtype=array.dtype) + np.sum(array) * 0
    else:
        repetitions = math.ceil(output_size / int(flat.size))
        result = np.reshape(np.tile(flat, repetitions)[:output_size], new_shape)
    return _finish(result, traced_type=traced_type)


def _meshgrid_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if not args:
        msg = "numpy.meshgrid requires at least one input during tracing"
        raise TracingError(msg)
    if not bool(kwargs.get("copy", True)):
        msg = "numpy.meshgrid(copy=False) returns aliasing views; use copy=True during tracing"
        raise TracingError(msg)
    indexing = str(kwargs.get("indexing", "xy"))
    if indexing not in {"ij", "xy"}:
        msg = "numpy.meshgrid indexing= must be 'ij' or 'xy'"
        raise TracingError(msg)
    sparse = bool(kwargs.get("sparse", False))
    anchor = next(item for item in args if isinstance(item, traced_type))
    vectors = tuple(
        np.ravel(item) if isinstance(item, traced_type) else np.ravel(item) + np.sum(anchor) * 0
        for item in args
    )
    sizes = tuple(int(item.shape[0]) for item in vectors)
    rank = len(vectors)
    outputs: list[object] = []
    for position, vector in enumerate(vectors):
        axis = position
        if indexing == "xy" and rank >= _BINARY_ARITY:
            if position == 0:
                axis = 1
            elif position == 1:
                axis = 0
        shape = [1] * rank
        shape[axis] = sizes[position]
        reshaped = np.reshape(vector, tuple(shape))
        if sparse:
            outputs.append(np.copy(reshaped))
            continue
        target = list(sizes)
        if indexing == "xy" and rank >= _BINARY_ARITY:
            target[0], target[1] = target[1], target[0]
        outputs.append(np.copy(np.broadcast_to(reshaped, tuple(target))))
    return _finish(tuple(outputs), traced_type=traced_type)


def _average_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = dict(kwargs)
    values.update(zip(("axis", "weights", "returned"), args[1:], strict=False))
    array = args[0]
    axis = values.get("axis")
    weights = values.get("weights")
    returned = bool(values.get("returned", False))
    keepdims = bool(values.get("keepdims", False))
    if weights is None:
        result = np.mean(array, axis=axis, keepdims=keepdims)
        if not returned:
            return _finish(result, traced_type=traced_type)
        count = (
            array.size
            if axis is None
            else math.prod(array.shape[item] for item in _normalize_axes(axis, array.ndim))
        )
        weight_sum = np.ones_like(result) * count
        return _finish((result, weight_sum), traced_type=traced_type)

    normalized_weights = weights
    weight_shape = tuple(int(size) for size in _concrete_array(weights).shape)
    array_shape = tuple(int(size) for size in array.shape)
    if weight_shape != array_shape:
        if axis is None:
            msg = "Axis must be specified when shapes of a and weights differ."
            raise TypeError(msg)
        axes = _normalize_axes(axis, array.ndim)
        expected_shape = tuple(array_shape[item] for item in axes)
        if weight_shape != expected_shape:
            msg = "Shape of weights must be consistent with shape of a along specified axis."
            raise ValueError(msg)
        shape = [1] * array.ndim
        for weight_axis, array_axis in enumerate(axes):
            shape[array_axis] = weight_shape[weight_axis]
        normalized_weights = np.reshape(weights, tuple(shape))
    concrete_weights = _concrete_array(normalized_weights)
    if np.any(np.sum(concrete_weights, axis=axis, keepdims=keepdims) == 0):
        msg = "Weights sum to zero, can't be normalized"
        raise ZeroDivisionError(msg)
    numerator = np.sum(array * normalized_weights, axis=axis, keepdims=keepdims)
    denominator = np.sum(normalized_weights, axis=axis, keepdims=keepdims)
    result = numerator / denominator
    if not returned:
        return _finish(result, traced_type=traced_type)
    return _finish((result, np.ones_like(result) * denominator), traced_type=traced_type)


def _ptp_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    positional_names = ("axis",)
    values = dict(kwargs)
    values.update(zip(positional_names, args[1:], strict=False))
    axis = values.get("axis")
    keepdims = bool(values.get("keepdims", False))
    result = np.max(args[0], axis=axis, keepdims=keepdims) - np.min(
        args[0],
        axis=axis,
        keepdims=keepdims,
    )
    return _finish(result, traced_type=traced_type)


def _trapezoid_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    positional_names = ("x", "dx", "axis")
    values = dict(kwargs)
    values.update(zip(positional_names, args[1:], strict=False))
    y = args[0]
    x = values.get("x")
    axis = int(values.get("axis", -1))
    normalized_axis = axis if axis >= 0 else axis + y.ndim
    lower = [slice(None)] * y.ndim
    upper = [slice(None)] * y.ndim
    lower[normalized_axis] = slice(None, -1)
    upper[normalized_axis] = slice(1, None)
    pair_sum = y[tuple(lower)] + y[tuple(upper)]
    if x is None:
        spacing: object = values.get("dx", 1.0)
    else:
        spacing = np.diff(x, axis=axis if _ndim(x) > 1 else -1)
        if _ndim(spacing) == 1 and y.ndim > 1:
            shape = [1] * y.ndim
            shape[normalized_axis] = int(spacing.shape[0])
            spacing = np.reshape(spacing, tuple(shape))
    result = np.sum(pair_sum * spacing * 0.5, axis=axis)
    return _finish(result, traced_type=traced_type)


def _nan_scan_handler(
    *,
    product: bool,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    positional_names = ("axis", "dtype")
    values = dict(kwargs)
    values.update(zip(positional_names, args[1:], strict=False))
    replacement = 1.0 if product else 0.0
    cleaned = np.where(np.isnan(args[0]), replacement, args[0])
    function = np.cumprod if product else np.cumsum
    return _finish(function(cleaned, **values), traced_type=traced_type)


def _nancumsum_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    return _nan_scan_handler(
        product=False,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _nancumprod_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    return _nan_scan_handler(
        product=True,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _round_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    decimals = int(args[1] if len(args) == _BINARY_ARITY else kwargs.get("decimals", 0))
    scale = 10.0**decimals
    return _finish(np.rint(args[0] * scale) / scale, traced_type=traced_type)


def _fix_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    x = args[0]
    return _finish(np.where(x >= 0, np.floor(x), np.ceil(x)), traced_type=traced_type)


def _vdot_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if len(args) != _BINARY_ARITY or kwargs:
        msg = "numpy.vdot expects two arrays during tracing"
        raise TracingError(msg)
    return _finish(
        np.vecdot(np.ravel(args[0]), np.ravel(args[1])),
        traced_type=traced_type,
    )


def _matrix_power_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    matrix, exponent_raw = args
    if isinstance(exponent_raw, traced_type) or not isinstance(
        exponent_raw,
        (int, np.integer),
    ):
        msg = "numpy.linalg.matrix_power exponent must be a static integer"
        raise TracingError(msg)
    exponent = int(exponent_raw)
    if matrix.ndim < _MATRIX_RANK or matrix.shape[-2] != matrix.shape[-1]:
        msg = "numpy.linalg.matrix_power requires square matrices"
        raise TracingError(msg)
    if exponent == 0:
        identity = np.eye(int(matrix.shape[-1]), dtype=matrix.dtype)
        result = np.zeros_like(matrix) + identity
        return _finish(result, traced_type=traced_type)
    base = np.linalg.inv(matrix) if exponent < 0 else matrix
    remaining = abs(exponent)
    result: object | None = None
    while remaining:
        if remaining & 1:
            result = base if result is None else np.matmul(result, base)
        remaining >>= 1
        if remaining:
            base = np.matmul(base, base)
    return _finish(cast("object", result), traced_type=traced_type)


def _multi_dot_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    arrays = args[0]
    if not isinstance(arrays, (tuple, list)) or len(arrays) < _BINARY_ARITY:
        msg = "numpy.linalg.multi_dot requires at least two arrays"
        raise TracingError(msg)
    result = arrays[0]
    for operand in arrays[1:]:
        result = np.dot(result, operand)
    return _finish(result, traced_type=traced_type)


def _tensorinv_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    tensor = args[0]
    ind_raw = args[1] if len(args) == _BINARY_ARITY else kwargs.get("ind", 2)
    if isinstance(ind_raw, traced_type) or not isinstance(ind_raw, (int, np.integer)):
        msg = "numpy.linalg.tensorinv ind must be a static integer"
        raise TracingError(msg)
    ind = int(ind_raw)
    if ind <= 0 or ind >= tensor.ndim:
        msg = "numpy.linalg.tensorinv ind must split the tensor dimensions"
        raise TracingError(msg)
    left_shape = tuple(int(size) for size in tensor.shape[:ind])
    right_shape = tuple(int(size) for size in tensor.shape[ind:])
    left_size = math.prod(left_shape)
    right_size = math.prod(right_shape)
    if left_size != right_size:
        msg = "numpy.linalg.tensorinv requires equal products on both sides of ind"
        raise TracingError(msg)
    matrix = np.reshape(tensor, (right_size, left_size))
    return _finish(
        np.reshape(np.linalg.inv(matrix), (*right_shape, *left_shape)),
        traced_type=traced_type,
    )


def _tensorsolve_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    tensor, right = args[:2]
    axes_raw = args[2] if len(args) == _TERNARY_ARITY else kwargs.get("axes")
    if axes_raw is not None:
        axes = _normalize_axes(axes_raw, tensor.ndim)
        permutation = tuple(axis for axis in range(tensor.ndim) if axis not in set(axes)) + axes
        tensor = np.transpose(tensor, permutation)
    solution_shape = tuple(int(size) for size in tensor.shape[right.ndim :])
    size = math.prod(solution_shape)
    if tensor.size != size * size:
        msg = "numpy.linalg.tensorsolve requires a square flattened operator"
        raise TracingError(msg)
    matrix = np.reshape(tensor, (size, size))
    vector = np.ravel(right)
    return _finish(
        np.reshape(np.linalg.solve(matrix, vector), solution_shape),
        traced_type=traced_type,
    )


def _cond_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    matrix = args[0]
    order = args[1] if len(args) == _BINARY_ARITY else kwargs.get("p")
    if order is None or order in {2, _NEGATIVE_SPECTRAL_ORDER}:
        singular_values = np.linalg.svdvals(matrix)
        result = (
            singular_values[..., -1] / singular_values[..., 0]
            if order == _NEGATIVE_SPECTRAL_ORDER
            else singular_values[..., 0] / singular_values[..., -1]
        )
    else:
        result = np.linalg.norm(matrix, ord=order, axis=(-2, -1)) * np.linalg.norm(
            np.linalg.inv(matrix),
            ord=order,
            axis=(-2, -1),
        )
    return _finish(result, traced_type=traced_type)


def _first_traced(
    values: object,
    *,
    traced_type: type[TracedArrayLike],
) -> TracedArrayLike | None:
    if isinstance(values, traced_type):
        return values
    if isinstance(values, (tuple, list)):
        for value in values:
            found = _first_traced(value, traced_type=traced_type)
            if found is not None:
                return found
    return None


def _lift_composite_constant(value: object, anchor: TracedArrayLike) -> Any:
    return np.asarray(value) + np.sum(anchor) * 0


def _concrete_array(value: object) -> Any:
    current = value
    while callable(getattr(current, "_advect_snapshot", None)):
        _node_id, nested = _snapshot_traced(current)
        if nested is current:
            break
        current = nested
    return np.asarray(current)


def _broadcast_arrays_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    if bool(kwargs.get("subok", False)):
        msg = "numpy.broadcast_arrays(subok=True) is not supported during tracing"
        raise TracingError(msg)
    anchor = _first_traced(args, traced_type=traced_type)
    if anchor is None:
        msg = "numpy.broadcast_arrays requires a traced operand"
        raise TracingError(msg)
    shape = np.broadcast_shapes(*(value.shape for value in args))
    arrays = tuple(
        value if isinstance(value, traced_type) else _lift_composite_constant(value, anchor)
        for value in args
    )
    return _finish(
        tuple(np.broadcast_to(value, shape) for value in arrays),
        traced_type=traced_type,
    )


def _select_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    conditions, choices = args[:2]
    if not isinstance(conditions, (tuple, list)) or not isinstance(choices, (tuple, list)):
        msg = "numpy.select condlist and choicelist must be sequences"
        raise TracingError(msg)
    if not conditions or len(conditions) != len(choices):
        msg = "numpy.select requires equally sized non-empty condition and choice lists"
        raise TracingError(msg)
    default = args[2] if len(args) == _TERNARY_ARITY else kwargs.get("default", 0)
    anchor = _first_traced((conditions, choices, default), traced_type=traced_type)
    if anchor is None:
        msg = "numpy.select requires a traced operand"
        raise TracingError(msg)
    result: object = (
        default if isinstance(default, traced_type) else _lift_composite_constant(default, anchor)
    )
    for condition, choice in reversed(tuple(zip(conditions, choices, strict=True))):
        selected = (
            choice if isinstance(choice, traced_type) else _lift_composite_constant(choice, anchor)
        )
        result = np.where(condition, selected, result)
    return _finish(result, traced_type=traced_type)


def _piecewise_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    x, conditions_raw, functions_raw, *function_args = args
    conditions = (
        list(conditions_raw) if isinstance(conditions_raw, (tuple, list)) else [conditions_raw]
    )
    functions = list(functions_raw) if isinstance(functions_raw, (tuple, list)) else [functions_raw]
    if len(functions) not in {len(conditions), len(conditions) + 1}:
        msg = "numpy.piecewise funclist must match condlist or contain one default"
        raise TracingError(msg)
    default_fn = functions.pop() if len(functions) > len(conditions) else 0
    anchor = _first_traced((x, conditions), traced_type=traced_type)
    if anchor is None:
        msg = "numpy.piecewise requires a traced input or condition"
        raise TracingError(msg)
    masks = tuple(np.broadcast_to(condition, x.shape) for condition in conditions)

    def expanded_value(candidate: object, mask: object) -> object:
        selected = x[mask]
        value = candidate(selected, *function_args, **kwargs) if callable(candidate) else candidate
        if not isinstance(value, traced_type):
            value = _lift_composite_constant(value, anchor)
        value_size = int(value.size)
        selected_count = int(np.count_nonzero(_concrete_array(mask)))
        if value_size == 1:
            return np.broadcast_to(value, x.shape)
        if value_size != selected_count:
            msg = (
                "numpy.piecewise callable output must be scalar or match the number "
                "of selected elements"
            )
            raise TracingError(msg)
        flat_mask = np.ravel(_concrete_array(mask))
        positions = np.maximum(np.cumsum(flat_mask) - 1, 0)
        return np.reshape(np.take(np.ravel(value), positions), x.shape)

    occupied: object = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        occupied = np.logical_or(occupied, mask)
    default_mask = np.logical_not(occupied)
    result = expanded_value(default_fn, default_mask)
    # NumPy assigns in condlist order, so later overlapping conditions win.
    for mask, function in zip(masks, functions, strict=True):
        result = np.where(mask, expanded_value(function, mask), result)
    return _finish(result, traced_type=traced_type)


def _choose_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    indices, choices = args
    if not isinstance(choices, (tuple, list)) or not choices:
        msg = "numpy.choose requires a non-empty choice sequence"
        raise TracingError(msg)
    mode = str(kwargs.get("mode", "raise"))
    if mode == "wrap":
        indices = np.remainder(indices, len(choices))
    elif mode == "clip":
        indices = np.clip(indices, 0, len(choices) - 1)
    elif mode == "raise":
        concrete_indices = _concrete_array(indices)
        if np.any((concrete_indices < 0) | (concrete_indices >= len(choices))):
            msg = "invalid entry in choice array"
            raise ValueError(msg)
    else:
        msg = "numpy.choose mode must be raise, wrap, or clip"
        raise TracingError(msg)
    anchor = _first_traced((indices, choices), traced_type=traced_type)
    if anchor is None:
        msg = "numpy.choose requires a traced operand"
        raise TracingError(msg)
    normalized = tuple(
        choice if isinstance(choice, traced_type) else _lift_composite_constant(choice, anchor)
        for choice in choices
    )
    result = normalized[0]
    for index, choice in enumerate(normalized[1:], start=1):
        result = np.where(indices == index, choice, result)
    return _finish(result, traced_type=traced_type)


def _compress_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    condition, array = args[:2]
    condition_value = (
        _snapshot_traced(condition)[1]
        if isinstance(condition, traced_type)
        else np.asarray(condition)
    )
    axis_raw = args[2] if len(args) == _TERNARY_ARITY else kwargs.get("axis")
    axis = None if axis_raw is None else int(axis_raw)
    source = np.ravel(array) if axis is None else array
    source_axis = 0 if axis is None else axis
    limit = int(source.shape[source_axis])
    indices = np.flatnonzero(np.ravel(condition_value)[:limit])
    return _finish(
        np.take(source, indices, axis=source_axis),
        traced_type=traced_type,
    )


def _extract_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
) -> CompositeResult:
    return _compress_handler(graph, traced_type, args, {"axis": None})


def _vander_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    values = dict(kwargs)
    values.update(zip(("N", "increasing"), args[1:], strict=False))
    x = args[0]
    columns_raw = values.get("N")
    columns = int(x.size if columns_raw is None else columns_raw)
    if columns < 0:
        msg = "numpy.vander N must be non-negative"
        raise TracingError(msg)
    increasing = bool(values.get("increasing", False))
    exponents = range(columns) if increasing else range(columns - 1, -1, -1)
    if x.ndim != 1:
        msg = "numpy.vander input must be one-dimensional"
        raise TracingError(msg)
    if columns == 0:
        result = np.zeros((int(x.shape[0]), 0), dtype=x.dtype) + np.sum(x) * 0
    else:
        result = np.stack(tuple(x**exponent for exponent in exponents), axis=-1)
    return _finish(result, traced_type=traced_type)


def _cov_handler(  # noqa: C901, PLR0915 - one closed NumPy signature
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    positional_names = ("y", "rowvar", "bias", "ddof", "fweights", "aweights")
    values = dict(kwargs)
    values.update(zip(positional_names, args[1:], strict=False))

    matrix = args[0]
    additional = values.get("y")
    rowvar = bool(values.get("rowvar", True))
    bias = bool(values.get("bias", False))
    ddof_raw = values.get("ddof")
    if ddof_raw is not None and not isinstance(ddof_raw, (int, np.integer)):
        msg = "ddof must be integer"
        raise ValueError(msg)
    ddof = (0 if bias else 1) if ddof_raw is None else int(ddof_raw)
    dtype = values.get("dtype")
    if dtype is None:
        dtype = np.result_type(
            matrix,
            *(() if additional is None else (additional,)),
            np.float64,
        )

    data = np.astype(np.atleast_2d(matrix), dtype)
    if not rowvar and _ndim(matrix) != 1:
        data = np.transpose(data)
    if additional is not None:
        other = np.astype(np.atleast_2d(additional), dtype)
        if not rowvar and _ndim(additional) != 1:
            other = np.transpose(other)
        data = np.concatenate((data, other), axis=0)

    frequency = values.get("fweights")
    analytic = values.get("aweights")
    observation_count = int(data.shape[1])

    def validate_weights(weight: object, *, name: str, integral: bool) -> None:
        concrete = _concrete_array(weight)
        if concrete.ndim > 1:
            msg = f"{name} must be 1-D"
            raise RuntimeError(msg)
        if concrete.ndim == 0 or concrete.shape[0] != observation_count:
            msg = f"incompatible numbers of samples and {name}"
            raise RuntimeError(msg)
        if integral and np.any(concrete != np.round(concrete)):
            msg = "fweights must be integer"
            raise TypeError(msg)
        if np.any(concrete < 0):
            msg = f"{name} cannot be negative"
            raise ValueError(msg)

    if frequency is not None:
        validate_weights(frequency, name="fweights", integral=True)
    if analytic is not None:
        validate_weights(analytic, name="aweights", integral=False)
    weights: object | None = None
    if frequency is not None:
        weights = frequency
    if analytic is not None:
        weights = analytic if weights is None else weights * analytic
    if weights is not None and np.sum(_concrete_array(weights)) == 0:
        msg = "Weights sum to zero, can't be normalized"
        raise ZeroDivisionError(msg)

    if weights is None:
        average = np.mean(data, axis=1)
        factor: object = int(data.shape[1]) - ddof
    else:
        average, weight_sum_array = np.average(
            data,
            axis=1,
            weights=weights,
            returned=True,
        )
        weight_sum = weight_sum_array[0]
        if ddof == 0:
            factor = weight_sum
        elif analytic is None:
            factor = weight_sum - ddof
        else:
            factor = weight_sum - ddof * np.sum(weights * analytic) / weight_sum

    centered = data - average[:, None]
    weighted = centered if weights is None else centered * weights
    covariance = np.dot(centered, np.conjugate(np.transpose(weighted))) / factor
    return _finish(np.squeeze(covariance), traced_type=traced_type)


def _corrcoef_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> CompositeResult:
    positional_names = ("y", "rowvar", "bias", "ddof")
    values = dict(kwargs)
    values.update(zip(positional_names, args[1:], strict=False))
    covariance_args = (args[0],)
    covariance_kwargs = {
        key: value for key, value in values.items() if key in {"dtype", "rowvar", "y"}
    }
    covariance_value, covariance_node = _cov_handler(
        graph,
        traced_type,
        covariance_args,
        covariance_kwargs,
    )
    traced_ctor = cast("Callable[..., TracedArrayLike]", traced_type)
    covariance: Any = traced_ctor(
        value=covariance_value,
        node_id=cast("int", covariance_node),
        recorder=graph,
    )
    if covariance.ndim == 0:
        return _finish(covariance / covariance, traced_type=traced_type)
    variance = np.real(np.diag(covariance))
    standard_deviation = np.sqrt(variance)
    result = covariance / standard_deviation[:, None] / standard_deviation[None, :]
    if np.issubdtype(result.dtype, np.complexfloating):
        result = np.clip(np.real(result), -1, 1) + 1j * np.clip(
            np.imag(result),
            -1,
            1,
        )
    else:
        result = np.clip(result, -1, 1)
    return _finish(result, traced_type=traced_type)


def register_composite_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register conveniences that need no new primitive semantics."""
    handlers[np.hstack] = _hstack_handler
    handlers[np.vstack] = _vstack_handler
    handlers[np.dstack] = _dstack_handler
    handlers[np.column_stack] = _column_stack_handler
    handlers[np.append] = _append_handler
    handlers[np.delete] = _delete_handler
    handlers[np.diagflat] = _diagflat_handler
    handlers[np.ediff1d] = _ediff1d_handler
    handlers[np.resize] = _resize_handler
    handlers[np.meshgrid] = _meshgrid_handler
    handlers[np.average] = _average_handler
    handlers[np.ptp] = _ptp_handler
    handlers[np.trapezoid] = _trapezoid_handler
    handlers[np.nancumsum] = _nancumsum_handler
    handlers[np.nancumprod] = _nancumprod_handler
    handlers[np.round] = _round_handler
    handlers[np.around] = _round_handler
    handlers[np.fix] = _fix_handler
    handlers[np.vdot] = _vdot_handler
    handlers[np.linalg.matrix_power] = _matrix_power_handler
    handlers[np.linalg.multi_dot] = _multi_dot_handler
    handlers[np.linalg.tensorinv] = _tensorinv_handler
    handlers[np.linalg.tensorsolve] = _tensorsolve_handler
    handlers[np.linalg.cond] = _cond_handler
    handlers[np.broadcast_arrays] = _broadcast_arrays_handler
    handlers[np.select] = _select_handler
    handlers[np.piecewise] = _piecewise_handler
    handlers[np.choose] = _choose_handler
    handlers[np.compress] = _compress_handler
    handlers[np.extract] = _extract_handler
    handlers[np.vander] = _vander_handler
    handlers[np.cov] = _cov_handler
    handlers[np.corrcoef] = _corrcoef_handler
    concat = np.__dict__.get("concat")
    row_stack = np.__dict__.get("row_stack")
    trapz = np.__dict__.get("trapz")
    if concat is not None:
        handlers[concat] = handlers[np.concatenate]
    if row_stack is not None:
        handlers[row_stack] = _vstack_handler
    if trapz is not None:
        handlers[trapz] = _trapezoid_handler
