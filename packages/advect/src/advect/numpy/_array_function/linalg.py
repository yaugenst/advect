"""Linalg-related ``__array_function__`` handlers."""

from __future__ import annotations

import importlib
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._array_api.results import restore_array_api_result
from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import (
    _concrete_array,
    _finish,
    _first_traced,
    _lift_composite_constant,
    _lift_lapack_rank,
)
from advect.numpy._array_function.emission import (
    _add_backend_node,
    _emit,
    _get_array_value,
    _get_node,
    _get_value,
    _make_binary_handler,
    _result_shape_and_dtype,
)
from advect.numpy._array_function.normalization import _bind_optional_positionals
from advect.numpy._composite_lowering import operand_dtype
from advect.numpy._op_bindings import canonicalize_numpy_op

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import (
        ArrayFunctionHandler,
        ArrayFunctionResult,
    )

np: Any = _numpy


def _record(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    name: str,
    operands: tuple[Any, ...],
    value: object,
    attrs: dict[str, Any] | None = None,
) -> ArrayFunctionResult:
    """Record ``numpy.linalg.<name>``; a tuple result gets one getoutput node per field."""
    op = f"numpy.linalg.{name}"
    if not isinstance(value, tuple):
        return _emit(graph, traced_type, op, operands, value, attrs)
    outputs = tuple(value)
    shape, dtype = _result_shape_and_dtype(outputs[0])
    parent_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(op),
        inputs=tuple(_get_node(operand, graph, traced_type) for operand in operands),
        value=outputs,
        attrs={} if attrs is None else attrs,
        shape=shape,
        dtype=dtype,
    )
    node_ids = tuple(
        _add_backend_node(
            graph=graph,
            op="advect.getoutput",
            inputs=(parent_id,),
            value=output,
            attrs={"index": index, "num_outputs": len(outputs)},
        )
        for index, output in enumerate(outputs)
    )
    return restore_array_api_result(f"linalg.{name}", outputs), node_ids


def _plain_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    _kwargs: dict[str, Any],
    *,
    name: str,
) -> ArrayFunctionResult:
    """Record a control-free ``np.linalg`` function of its positional arrays."""
    # One trace level down, so an enclosing trace records the call as well.
    value = getattr(np.linalg, name)(*(_get_value(arg, traced_type) for arg in args))
    return _record(graph, traced_type, name, args, value)


def _svd_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.svd as a multi-output op, or svdvals when compute_uv=False."""
    values = dict(zip(("full_matrices", "compute_uv", "hermitian"), args[1:], strict=False))
    values |= kwargs
    a = args[0]
    hermitian = bool(values.get("hermitian", False))
    if not values.get("compute_uv", True):
        value = np.linalg.svd(_get_value(a, traced_type), compute_uv=False, hermitian=hermitian)
        return _record(graph, traced_type, "svdvals", (a,), value, {"hermitian": hermitian})
    attrs = {
        "full_matrices": bool(values.get("full_matrices", True)),
        "compute_uv": True,
        "hermitian": hermitian,
    }
    value = np.linalg.svd(_get_value(a, traced_type), **attrs)
    return _record(graph, traced_type, "svd", (a,), value, attrs)


def _normalize_norm_axis(axis: object) -> int | tuple[int, ...] | None:
    if axis is None:
        return None
    if isinstance(axis, np.integer):
        return int(axis)
    if isinstance(axis, int):
        return axis
    if isinstance(axis, (tuple, list)):
        return tuple(int(item) for item in axis)
    msg = f"np.linalg.norm(axis={axis!r}) is not supported during tracing"
    raise TracingError(msg)


def _normalize_norm_ord(ord_value: object) -> str | int | float | None:
    if ord_value is None:
        return None
    if isinstance(ord_value, np.integer):
        return int(ord_value)
    if isinstance(ord_value, np.floating):
        return float(ord_value)
    if isinstance(ord_value, (int, float, str)):
        return ord_value
    msg = f"np.linalg.norm(ord={ord_value!r}) is not supported during tracing"
    raise TracingError(msg)


def _norm_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.norm as a single-output op."""
    values = dict(zip(("ord", "axis", "keepdims"), args[1:], strict=False)) | kwargs
    a = args[0]
    ord_norm = _normalize_norm_ord(values.get("ord"))
    axis_norm = _normalize_norm_axis(values.get("axis"))
    keepdims = bool(values.get("keepdims", False))
    result = np.linalg.norm(
        _get_value(a, traced_type), ord=ord_norm, axis=axis_norm, keepdims=keepdims
    )
    attrs: dict[str, Any] = {"keepdims": keepdims}
    if ord_norm is not None:
        attrs["ord"] = ord_norm
    if axis_norm is not None:
        attrs["axis"] = axis_norm
    return _record(graph, traced_type, "norm", (a,), result, attrs)


def _cholesky_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.cholesky as a single-output op."""
    a = args[0]
    upper = kwargs.get("upper", False)
    if type(upper) is not bool:
        msg = "np.linalg.cholesky upper= must be a bool during tracing"
        raise TracingError(msg)
    value = np.linalg.cholesky(_get_value(a, traced_type), upper=upper)
    return _record(graph, traced_type, "cholesky", (a,), value, {"upper": upper})


def _pinv_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.pinv as a single-output op."""
    values = dict(kwargs) | dict(zip(("rcond", "hermitian"), args[1:], strict=False))
    if "rcond" in values and "rtol" in values:
        msg = "np.linalg.pinv accepts only one of rcond= and rtol="
        raise TracingError(msg)
    for name in ("rcond", "rtol"):
        if isinstance(values.get(name), traced_type):
            msg = f"np.linalg.pinv {name}= must be static because it controls numerical rank"
            raise TracingError(msg)
    attrs = {name: values[name] for name in ("rcond", "rtol") if name in values}
    if "hermitian" in values:
        attrs["hermitian"] = bool(values["hermitian"])
    a = args[0]
    value = np.linalg.pinv(_get_value(a, traced_type), **attrs)
    return _record(graph, traced_type, "pinv", (a,), value, attrs)


def _qr_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.qr as a multi-output op, or a single-output qr_r op when mode='r'."""
    a = args[0]
    mode = args[1] if len(args) > 1 else kwargs.get("mode", "reduced")
    if mode is None:
        mode = "reduced"
    if mode not in {"reduced", "complete", "r"}:
        msg = (
            f"np.linalg.qr(mode={mode!r}) is not supported during tracing because it can change "
            "the output arity. Use mode='reduced', mode='complete', or mode='r'."
        )
        raise TracingError(msg)
    value = np.linalg.qr(_get_value(a, traced_type), mode=mode)
    return _record(graph, traced_type, "qr_r" if mode == "r" else "qr", (a,), value, {"mode": mode})


def _hermitian_eigen_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
) -> ArrayFunctionResult:
    """Handle np.linalg.eigh and eigvalsh with a normalized UPLO= control."""
    raw = args[1] if len(args) > 1 else kwargs.get("UPLO", "L")
    uplo = str(raw).upper()
    if uplo not in {"L", "U"}:
        msg = f"np.linalg.{name}(UPLO={raw!r}) is not supported during tracing. Use 'L' or 'U'."
        raise TracingError(msg)
    a = args[0]
    value = getattr(np.linalg, name)(_get_value(a, traced_type), UPLO=uplo)
    return _record(graph, traced_type, name, (a,), value, {"UPLO": uplo})


_BINARY_ARG_COUNT = 2
_MATRIX_RANK = 2
_TENSORDOT_DEFAULT_AXES = 2
_TENSORDOT_MAX_ARGS = 3


def _matrix_rank_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    values = _bind_optional_positionals(
        name="linalg.matrix_rank",
        args=args,
        kwargs=kwargs,
        required=1,
        optional=("tol", "hermitian"),
        keyword_only=frozenset({"rtol"}),
    )
    matrix = args[0]
    tolerances = {name: values[name] for name in ("tol", "rtol") if values.get(name) is not None}
    if len(tolerances) > 1:
        msg = "numpy.linalg.matrix_rank cannot receive both tol and rtol"
        raise TracingError(msg)
    hermitian = bool(values.get("hermitian", False))
    if hermitian and int(matrix.ndim) >= _MATRIX_RANK and matrix.shape[-2] != matrix.shape[-1]:
        msg = "numpy.linalg.matrix_rank(hermitian=True) requires square matrices"
        raise TracingError(msg)
    # The rank is integer-valued and has no derivative, so NumPy computes it concretely.
    rank = np.linalg.matrix_rank(
        _concrete_array(matrix),
        hermitian=hermitian,
        **{name: _concrete_array(value) for name, value in tolerances.items()},
    )
    anchor = _first_traced((matrix, *tolerances.values()), traced_type=traced_type)
    return _finish(_lift_composite_constant(rank, anchor), traced_type=traced_type)


def _lstsq_handler(
    _graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    values = _bind_optional_positionals(
        name="linalg.lstsq",
        args=args,
        kwargs=kwargs,
        required=_BINARY_ARG_COUNT,
        optional=("rcond",),
    )
    matrix, right = args[:2]
    anchor = _first_traced((matrix, right), traced_type=traced_type)
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
        # NumPy solves in double precision, so its default uses float64 epsilon.
        rcond = max(int(size) for size in matrix.shape) * np.finfo(np.float64).eps
    rcond_value = float(rcond)

    solution = np.matmul(np.linalg.pinv(matrix, rcond=rcond_value), right)
    singular_values = np.linalg.svdvals(matrix)
    if isinstance(singular_values, traced_type):
        _node_id, concrete_singular_values = _snapshot_traced(singular_values)
    else:
        concrete_singular_values = singular_values
        singular_values = _lift_composite_constant(concrete_singular_values, anchor)
    singular_array = np.asarray(concrete_singular_values)
    rank = (
        int(np.sum(singular_array > singular_array[0] * rcond_value)) if singular_array.size else 0
    )
    rows, columns = (int(size) for size in matrix.shape)
    if rows > columns and rank == columns:
        residual = right - np.matmul(matrix, solution)
        residuals = np.atleast_1d(np.sum(np.real(np.conjugate(residual) * residual), axis=0))
    else:
        residuals = _lift_composite_constant(np.zeros((0,), dtype=solution.dtype), solution)
    rank_value = _lift_lapack_rank(rank, solution)
    return _finish(
        (solution, residuals, rank_value, singular_values),
        traced_type=traced_type,
    )


def _cross_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    positional_names = ("axisa", "axisb", "axisc", "axis")
    values = dict(kwargs)
    values.update(dict(zip(positional_names, args[_BINARY_ARG_COUNT:], strict=False)))

    a, b = args[:2]
    result = np.cross(_get_value(a, traced_type), _get_value(b, traced_type), **values)

    attrs: dict[str, Any] = {}
    for key in ("axisa", "axisb", "axisc", "axis"):
        if key in values and values[key] is not None:
            attrs[key] = int(values[key])

    return _emit(graph, traced_type, "numpy.cross", (a, b), result, attrs)


def _tensordot_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
    a, b = args[0], args[1]
    axes = (
        args[_BINARY_ARG_COUNT]
        if len(args) == _TENSORDOT_MAX_ARGS
        else kwargs.get("axes", _TENSORDOT_DEFAULT_AXES)
    )
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

    return _emit(graph, traced_type, "numpy.tensordot", (a, b), result, attrs)


def _resolve_einsum_string_form(
    args: tuple[Any, ...],
    *,
    traced_type: type[TracedArrayLike],
) -> tuple[str, tuple[Any, ...], tuple[Any, ...]]:
    subscripts = args[0]
    operands = args[1:]
    values = tuple(_get_array_value(op, traced_type) for op in operands)
    normalized = _normalize_einsum_syntax(
        (subscripts, *(_einsum_shape_dummy(value) for value in values))
    )
    return normalized, operands, values


def _einsum_shape_dummy(value: object) -> _numpy.ndarray[Any, Any]:
    array_value = cast("Any", value)
    shape = tuple(int(size) for size in array_value.shape)
    scalar = np.empty((), dtype=operand_dtype(array_value))
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
        einsum_module = importlib.import_module("numpy._core.einsumfunc")
        parser_name = "_parse_einsum_input"
        parse_einsum_input = getattr(einsum_module, parser_name)
        in_subscripts, out_subscripts, _ = parse_einsum_input(einsum_operands)
    except Exception as exc:
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

    values = tuple(_get_array_value(operand, traced_type) for operand in operands)
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
    operand_dtypes = tuple(operand_dtype(value) for value in values)
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


def _einsum_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, int]:
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
    return _emit(graph, traced_type, "numpy.einsum", operands, result, attrs)


def register_linalg_handlers(
    handlers: dict[Callable[..., Any], ArrayFunctionHandler],
) -> None:
    """Register linalg-related array functions."""
    for name in ("det", "inv", "eigvals", "svdvals", "eig", "slogdet", "solve"):
        handlers[getattr(np.linalg, name)] = partial(_plain_handler, name=name)
    for name in ("eigh", "eigvalsh"):
        handlers[getattr(np.linalg, name)] = partial(_hermitian_eigen_handler, name=name)
    handlers[np.linalg.svd] = _svd_handler
    handlers[np.linalg.qr] = _qr_handler
    handlers[np.linalg.norm] = _norm_handler
    handlers[np.linalg.cholesky] = _cholesky_handler
    handlers[np.linalg.pinv] = _pinv_handler
    handlers[np.linalg.lstsq] = _lstsq_handler
    handlers[np.linalg.matrix_rank] = _matrix_rank_handler
    for function in (np.inner, np.outer, np.kron):
        handlers[function] = _make_binary_handler(function, f"numpy.{function.__name__}")
    handlers[np.cross] = _cross_handler
    handlers[np.tensordot] = _tensordot_handler
    handlers[np.einsum] = _einsum_handler
