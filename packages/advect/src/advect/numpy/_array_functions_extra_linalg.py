from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - typed module and dynamic lowering namespace

from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_functions_extra_common import _binary_handler
from advect.numpy._array_functions_extra_composite import (
    _finish,
    _first_traced,
)
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._protocol_array_function_common import _add_backend_node, _get_node, _get_value

if TYPE_CHECKING:
    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike

_BINARY_ARG_COUNT = 2
_MATRIX_RANK = 2
_MIN_EINSUM_ARGS = 2
_TENSORDOT_DEFAULT_AXES = 2
_TENSORDOT_MAX_ARGS = 3


def _bind_optional_positionals(
    *,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    required: int,
    optional: tuple[str, ...],
    keyword_only: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if len(args) < required or len(args) > required + len(optional):
        msg = f"numpy.linalg.{name} received an invalid positional signature during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - (set(optional) | set(keyword_only))
    if unsupported:
        msg = f"numpy.linalg.{name} kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for parameter, value in zip(optional, args[required:], strict=False):
        if parameter in values:
            msg = f"numpy.linalg.{name} received {parameter} twice"
            raise TracingError(msg)
        values[parameter] = value
    return values


def _real_epsilon(dtype: object) -> float:
    real_dtype = np.empty((), dtype=np.dtype(dtype)).real.dtype
    return float(np.finfo(real_dtype).eps)


def _matrix_rank_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    values = _bind_optional_positionals(
        name="matrix_rank",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("tol", "hermitian"),
        keyword_only=frozenset({"rtol"}),
    )
    matrix = args[0]
    if int(matrix.ndim) < 1:
        msg = "numpy.linalg.matrix_rank requires an array with ndim >= 1"
        raise TracingError(msg)
    tolerance = values.get("tol")
    relative_tolerance = values.get("rtol")
    if tolerance is not None and relative_tolerance is not None:
        msg = "numpy.linalg.matrix_rank cannot receive both tol and rtol"
        raise TracingError(msg)

    if int(matrix.ndim) == 1:
        singular_values = np.reshape(np.linalg.norm(matrix), (1,))
    elif 0 in tuple(int(size) for size in matrix.shape[-2:]):
        batch_shape = tuple(int(size) for size in matrix.shape[:-2])
        zero = np.astype(np.sum(matrix) * 0, np.int64)
        return _finish(np.broadcast_to(zero, batch_shape), traced_type=traced_type)
    elif bool(values.get("hermitian", False)):
        if int(matrix.shape[-2]) != int(matrix.shape[-1]):
            msg = "numpy.linalg.matrix_rank(hermitian=True) requires square matrices"
            raise TracingError(msg)
        singular_values = np.sort(np.absolute(np.linalg.eigvalsh(matrix)), axis=-1)[..., ::-1]
    else:
        singular_values = np.linalg.svdvals(matrix)

    maximum = np.max(singular_values, axis=-1, keepdims=True)
    if tolerance is None:
        if relative_tolerance is None:
            relative_tolerance = max(int(size) for size in matrix.shape) * _real_epsilon(
                matrix.dtype
            )
        tolerance = maximum * relative_tolerance
    rank = np.sum(np.greater(singular_values, tolerance), axis=-1)
    return _finish(np.astype(rank, np.int64), traced_type=traced_type)


def _lstsq_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    values = _bind_optional_positionals(
        name="lstsq",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARG_COUNT,
        optional=("rcond",),
    )
    matrix, right = args[:2]
    anchor = _first_traced((matrix, right), traced_type=traced_type)
    if anchor is None:  # pragma: no cover - NumPy dispatch requires a tracer
        msg = "numpy.linalg.lstsq requires a traced matrix or right-hand side"
        raise TracingError(msg)
    if int(matrix.ndim) != _MATRIX_RANK:
        msg = "numpy.linalg.lstsq matrix must be two-dimensional"
        raise TracingError(msg)
    if int(right.ndim) not in {1, 2} or int(right.shape[0]) != int(matrix.shape[0]):
        msg = "numpy.linalg.lstsq right-hand side must match the matrix row count"
        raise TracingError(msg)
    rcond = values.get("rcond")
    if isinstance(rcond, traced_type):
        msg = "numpy.linalg.lstsq rcond must be static because it controls numerical rank"
        raise TracingError(msg)
    if rcond is None:
        rcond = max(int(size) for size in matrix.shape) * _real_epsilon(matrix.dtype)
    rcond_value = float(rcond)

    solution = np.matmul(np.linalg.pinv(matrix, rcond=rcond_value), right)
    singular_values = np.linalg.svdvals(matrix)
    if isinstance(singular_values, traced_type):
        _node_id, concrete_singular_values = _snapshot_traced(singular_values)
    else:
        concrete_singular_values = singular_values
        singular_array = np.asarray(concrete_singular_values)
        # Keep the mixed result traceable while preserving this static output's real dtype.
        zero = np.astype(np.real(np.sum(anchor)) * 0, singular_array.dtype)
        singular_values = zero + singular_array
    singular_array = np.asarray(concrete_singular_values)
    rank = (
        int(np.sum(singular_array > singular_array[0] * rcond_value)) if singular_array.size else 0
    )
    rows, columns = (int(size) for size in matrix.shape)
    if rows > columns and rank == columns:
        residual = right - np.matmul(matrix, solution)
        residuals = np.atleast_1d(np.sum(np.real(np.conjugate(residual) * residual), axis=0))
    else:
        residuals = np.zeros((0,), dtype=solution.dtype) + np.sum(solution) * 0
    rank_value = np.astype(np.sum(solution) * 0 + rank, np.int32)
    return _finish(
        (solution, residuals, rank_value, singular_values),
        traced_type=traced_type,
    )


np: Any = _numpy


def _inner_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    return _binary_handler(
        op_name="numpy.inner",
        np_func=np.inner,
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _outer_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    return _binary_handler(
        op_name="numpy.outer",
        np_func=np.outer,
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _kron_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    return _binary_handler(
        op_name="numpy.kron",
        np_func=np.kron,
        graph=graph,
        traced_type=traced_type,
        args=args,
        kwargs=kwargs,
    )


def _cross_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("axisa", "axisb", "axisc", "axis")
    if len(args) < _BINARY_ARG_COUNT or len(args) > _BINARY_ARG_COUNT + len(positional_names):
        msg = "numpy.cross expects (a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)"
        raise TracingError(msg)
    unsupported = set(kwargs) - set(positional_names)
    if unsupported:
        msg = f"numpy.cross kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    values = dict(kwargs)
    for name, value in zip(positional_names, args[_BINARY_ARG_COUNT:], strict=False):
        if name in values:
            msg = f"numpy.cross received {name} twice"
            raise TracingError(msg)
        values[name] = value

    a, b = args[:2]
    result = np.cross(_get_value(a, traced_type), _get_value(b, traced_type), **values)

    attrs: dict[str, Any] = {}
    for key in ("axisa", "axisb", "axisc", "axis"):
        if key in values and values[key] is not None:
            attrs[key] = int(values[key])

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.cross"),
        inputs=(
            _get_node(a, graph, traced_type),
            _get_node(b, graph, traced_type),
        ),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _tensordot_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    if len(args) < _BINARY_ARG_COUNT or len(args) > _TENSORDOT_MAX_ARGS:
        msg = "numpy.tensordot expects (a, b, axes=...) during tracing"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"axes"}
    if unsupported:
        msg = f"numpy.tensordot kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    a, b = args[0], args[1]
    axes = (
        args[_BINARY_ARG_COUNT]
        if len(args) == _TENSORDOT_MAX_ARGS
        else kwargs.get("axes", _TENSORDOT_DEFAULT_AXES)
    )
    if len(args) == _TENSORDOT_MAX_ARGS and "axes" in kwargs:
        msg = "numpy.tensordot received axes twice"
        raise TracingError(msg)
    result = np.tensordot(_get_value(a, traced_type), _get_value(b, traced_type), axes=axes)

    attrs: dict[str, Any] = {}
    if isinstance(axes, (tuple, list, np.ndarray)):
        axes_arr = np.asarray(axes)
        if axes_arr.ndim == 1:
            attrs["axes"] = tuple(int(item) for item in axes_arr.tolist())
        else:
            attrs["axes"] = tuple(tuple(int(item) for item in row) for row in axes_arr.tolist())
    else:
        attrs["axes"] = int(axes)

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.tensordot"),
        inputs=(
            _get_node(a, graph, traced_type),
            _get_node(b, graph, traced_type),
        ),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id


def _resolve_einsum_string_form(
    args: tuple[Any, ...],
    *,
    traced_type: type[TracedArrayLike],
) -> tuple[str, tuple[Any, ...], tuple[Any, ...]]:
    if len(args) < _MIN_EINSUM_ARGS:
        msg = "numpy.einsum expects a subscript string and at least one operand"
        raise TracingError(msg)
    subscripts = args[0]
    if not isinstance(subscripts, str):
        msg = "numpy.einsum string form requires a subscript string"
        raise TracingError(msg)
    operands = args[1:]
    values = tuple(_get_value(op, traced_type) for op in operands)
    normalized = _normalize_einsum_syntax(
        (subscripts, *(_einsum_shape_dummy(value) for value in values))
    )
    return normalized, operands, values


def _einsum_shape_dummy(value: object) -> _numpy.ndarray[Any, Any]:
    array_value = cast("Any", value)
    shape = tuple(int(size) for size in array_value.shape)
    scalar = np.empty((), dtype=np.dtype(array_value.dtype))
    return cast(
        "_numpy.ndarray[Any, Any]",
        np.lib.stride_tricks.as_strided(
            scalar,
            shape=shape,
            strides=(0,) * len(shape),
            writeable=False,
        ),
    )


def _normalize_einsum_syntax(
    einsum_operands: tuple[Any, ...],
) -> str:
    try:
        try:
            einsum_module = importlib.import_module("numpy._core.einsumfunc")
        except ModuleNotFoundError:
            einsum_module = importlib.import_module("numpy.core.einsumfunc")
        parser_name = "_parse_einsum_input"
        parse_einsum_input = getattr(einsum_module, parser_name)
        in_subscripts, out_subscripts, _ = parse_einsum_input(einsum_operands)
    except Exception as exc:  # pragma: no cover - defensive normalization path
        msg = f"numpy.einsum syntax parsing failed: {exc}"
        raise TracingError(msg) from exc

    return f"{in_subscripts}->{out_subscripts}"


def _resolve_einsum_sublist_form(
    args: tuple[Any, ...],
    *,
    traced_type: type[TracedArrayLike],
) -> tuple[str, tuple[Any, ...], tuple[Any, ...]]:
    end = -1 if len(args) % 2 else None
    operands = args[:end:2]
    if not operands:
        msg = "numpy.einsum sublist form requires at least one operand"
        raise TracingError(msg)

    if end is None:
        labels = args[1::2]
        output_labels: Any | None = None
    else:
        labels = args[1:-1:2]
        output_labels = args[-1]
    if len(labels) != len(operands):
        msg = "numpy.einsum sublist form requires one label-list per operand"
        raise TracingError(msg)

    values = tuple(_get_value(operand, traced_type) for operand in operands)
    einsum_operands: list[Any] = []
    for value, label in zip(values, labels, strict=True):
        einsum_operands.extend((_einsum_shape_dummy(value), label))
    if output_labels is not None:
        einsum_operands.append(output_labels)

    subscripts = _normalize_einsum_syntax(tuple(einsum_operands))
    return subscripts, operands, values


def _einsum_calculation_dtype(
    values: tuple[Any, ...],
    *,
    dtype: object,
) -> object:
    if dtype is not None:
        return np.dtype(dtype)
    operand_dtypes = tuple(np.dtype(value.dtype) for value in values)
    return np.result_type(*operand_dtypes)


def _diagonalize_einsum_operand(
    operand: object,
    term: str,
) -> tuple[object, str]:
    labels = list(term)
    while len(labels) != len(set(labels)):
        repeated = next(label for label in labels if labels.count(label) > 1)
        axis1 = labels.index(repeated)
        axis2 = labels.index(repeated, axis1 + 1)
        operand = np.diagonal(operand, axis1=axis1, axis2=axis2)
        labels = [label for axis, label in enumerate(labels) if axis not in {axis1, axis2}]
        labels.append(repeated)
    return operand, "".join(labels)


def _canonicalize_einsum_operands(
    subscripts: str,
    operands: tuple[Any, ...],
    *,
    calculation_dtype: object,
) -> tuple[str, tuple[Any, ...]]:
    lhs, output_term = subscripts.split("->", maxsplit=1)
    terms = lhs.split(",")
    if len(terms) != len(operands):
        msg = f"numpy.einsum operand count does not match its normalized subscripts: {subscripts!r}"
        raise TracingError(msg)

    lowered = list(operands)
    for index, (operand, term) in enumerate(zip(lowered, terms, strict=True)):
        lowered[index], terms[index] = _diagonalize_einsum_operand(operand, term)

    output_labels = set(output_term)
    for index, (operand, term) in enumerate(zip(lowered, terms, strict=True)):
        other_labels = set().union(
            *(set(candidate) for position, candidate in enumerate(terms) if position != index)
        )
        reduced_axes = tuple(
            axis
            for axis, label in enumerate(term)
            if label not in output_labels and label not in other_labels
        )
        if not reduced_axes:
            continue
        lowered[index] = np.sum(
            operand,
            axis=reduced_axes,
            dtype=calculation_dtype,
        )
        terms[index] = "".join(label for axis, label in enumerate(term) if axis not in reduced_axes)

    return f"{','.join(terms)}->{output_term}", tuple(lowered)


def _einsum_call(
    subscripts: str,
    values: tuple[Any, ...],
    *,
    optimize: object,
    dtype: object,
    order: object,
    casting: object,
) -> object:
    call_kwargs: dict[str, object] = {
        "casting": casting,
        "optimize": optimize,
        "order": order,
    }
    if dtype is not None:
        call_kwargs["dtype"] = dtype
    einsum_fn = cast("Any", np.einsum)
    return cast(
        "object",
        einsum_fn(
            subscripts,
            *values,
            **call_kwargs,
        ),
    )


def _einsum_node_inputs(
    *,
    operands: tuple[Any, ...],
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
) -> tuple[Any, ...]:
    return tuple(_get_node(op, graph, traced_type) for op in operands)


def _einsum_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    if not args:
        msg = "numpy.einsum expects at least one operand"
        raise TracingError(msg)
    unsupported = set(kwargs) - {"optimize", "dtype", "order", "casting"}
    if unsupported:
        msg = f"numpy.einsum kwargs not supported during tracing: {sorted(unsupported)}"
        raise TracingError(msg)

    optimize = kwargs.get("optimize")
    dtype = kwargs.get("dtype")
    order = kwargs.get("order", "K")
    casting = kwargs.get("casting", "safe")

    if isinstance(args[0], str):
        subscripts, operands, values = _resolve_einsum_string_form(args, traced_type=traced_type)
    else:
        subscripts, operands, values = _resolve_einsum_sublist_form(args, traced_type=traced_type)
    subscripts, operands = _canonicalize_einsum_operands(
        subscripts,
        operands,
        calculation_dtype=_einsum_calculation_dtype(values, dtype=dtype),
    )
    values = tuple(_get_value(operand, traced_type) for operand in operands)

    result = cast(
        "Any",
        _einsum_call(
            subscripts,
            values,
            optimize=optimize,
            dtype=dtype,
            order=order,
            casting=casting,
        ),
    )

    attrs: dict[str, Any] = {
        "subscripts": subscripts,
        "order": order,
        "casting": casting,
    }
    if optimize is not None:
        attrs["optimize"] = optimize
    if dtype is not None:
        attrs["dtype"] = str(np.dtype(dtype))
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op("numpy.einsum"),
        inputs=_einsum_node_inputs(
            operands=operands,
            graph=graph,
            traced_type=traced_type,
        ),
        value=result,
        attrs=attrs,
        shape=result.shape,
        dtype=result.dtype,
    )
    return result, node_id
