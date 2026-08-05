# ruff: noqa: PLR2004  # Numeric arity checks in handler validation
"""Linalg-related ``__array_function__`` handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._array_api_results import restore_array_api_result
from advect.core._errors import TracingError
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._protocol_array_function_common import (
    _add_backend_node,
    _get_node,
    _get_value,
    _result_shape_and_dtype,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._protocol_array_function_common import ArrayFunctionResult

np: Any = _numpy


def _op_name(suffix: str) -> str:
    return f"numpy.linalg.{suffix}"


def _reject_duplicate_positionals(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    names: tuple[str, ...],
    *,
    function: str,
) -> None:
    for index, name in enumerate(names, start=1):
        if len(args) > index and name in kwargs:
            msg = f"np.linalg.{function} received {name} twice"
            raise TypeError(msg)


def _slogdet_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.slogdet as a multi-output op."""
    expected_outputs = 2
    if not args:
        msg = "np.linalg.slogdet requires an input array"
        raise ValueError(msg)
    if kwargs:
        msg = f"np.linalg.slogdet does not support kwargs during tracing: {sorted(kwargs)}"
        raise TracingError(msg)

    a = args[0]
    result = np.linalg.slogdet(_get_value(a, traced_type))
    outputs = tuple(result)
    if len(outputs) != expected_outputs:
        msg = f"np.linalg.slogdet returned {len(outputs)} outputs, expected {expected_outputs}"
        raise ValueError(msg)

    output_meta = tuple(_result_shape_and_dtype(out) for out in outputs)
    output_shapes = tuple(shape for shape, _ in output_meta)
    output_dtypes = tuple(dtype for _, dtype in output_meta)

    parent_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("slogdet")),
        inputs=(_get_node(a, graph, traced_type),),
        value=outputs,
        attrs={},
        shape=output_shapes[0],
        dtype=output_dtypes[0],
        num_outputs=len(outputs),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    node_ids: list[int] = []
    for index, (output, shape, dtype) in enumerate(
        zip(outputs, output_shapes, output_dtypes, strict=True)
    ):
        node_id = _add_backend_node(
            graph=graph,
            op="advect.getoutput",
            inputs=(parent_id,),
            value=output,
            attrs={"index": index, "num_outputs": len(outputs)},
            shape=shape,
            dtype=dtype,
        )
        node_ids.append(node_id)

    return restore_array_api_result("linalg.slogdet", outputs), tuple(node_ids)


def _svd_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.svd as a multi-output op, or svdvals when compute_uv=False."""
    pos_full_matrices = 1
    pos_compute_uv = 2
    pos_hermitian = 3
    max_svd_args = 4

    expected_outputs = 3
    if not args:
        msg = "np.linalg.svd requires an input array"
        raise ValueError(msg)

    allowed_kwargs = frozenset({"full_matrices", "compute_uv", "hermitian"})
    unsupported = set(kwargs) - allowed_kwargs
    if unsupported:
        msg = f"np.linalg.svd does not support kwargs during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    _reject_duplicate_positionals(
        args,
        kwargs,
        ("full_matrices", "compute_uv", "hermitian"),
        function="svd",
    )

    a = args[0]
    full_matrices = (
        args[pos_full_matrices]
        if len(args) > pos_full_matrices
        else kwargs.get("full_matrices", True)
    )
    compute_uv = (
        args[pos_compute_uv] if len(args) > pos_compute_uv else kwargs.get("compute_uv", True)
    )
    hermitian = args[pos_hermitian] if len(args) > pos_hermitian else kwargs.get("hermitian", False)
    if len(args) > max_svd_args:
        msg = "np.linalg.svd supports at most 4 positional arguments"
        raise TypeError(msg)

    if compute_uv is False:
        svals = np.linalg.svd(
            _get_value(a, traced_type),
            compute_uv=False,
            hermitian=bool(hermitian),
        )
        svals_shape, svals_dtype = _result_shape_and_dtype(svals)
        node_id = _add_backend_node(
            graph=graph,
            op=canonicalize_numpy_op(_op_name("svdvals")),
            inputs=(_get_node(a, graph, traced_type),),
            value=svals,
            attrs={"hermitian": bool(hermitian)},
            shape=svals_shape,
            dtype=svals_dtype,
        )
        return svals, node_id

    result = np.linalg.svd(
        _get_value(a, traced_type),
        full_matrices=bool(full_matrices),
        compute_uv=True,
        hermitian=bool(hermitian),
    )
    outputs = tuple(result)
    if len(outputs) != expected_outputs:
        msg = f"np.linalg.svd returned {len(outputs)} outputs, expected {expected_outputs}"
        raise ValueError(msg)

    output_meta = tuple(_result_shape_and_dtype(out) for out in outputs)
    output_shapes = tuple(shape for shape, _ in output_meta)
    output_dtypes = tuple(dtype for _, dtype in output_meta)

    parent_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("svd")),
        inputs=(_get_node(a, graph, traced_type),),
        value=outputs,
        attrs={
            "full_matrices": bool(full_matrices),
            "compute_uv": True,
            "hermitian": bool(hermitian),
        },
        shape=output_shapes[0],
        dtype=output_dtypes[0],
        num_outputs=len(outputs),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    node_ids: list[int] = []
    for index, (output, shape, dtype) in enumerate(
        zip(outputs, output_shapes, output_dtypes, strict=True)
    ):
        node_id = _add_backend_node(
            graph=graph,
            op="advect.getoutput",
            inputs=(parent_id,),
            value=output,
            attrs={"index": index, "num_outputs": len(outputs)},
            shape=shape,
            dtype=dtype,
        )
        node_ids.append(node_id)

    return outputs, tuple(node_ids)


def _svdvals_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.svdvals as a single-output op."""
    max_svdvals_args = 1
    if not args:
        msg = "np.linalg.svdvals requires an input array"
        raise ValueError(msg)
    if kwargs:
        msg = f"np.linalg.svdvals does not support kwargs during tracing: {sorted(kwargs)}"
        raise TracingError(msg)
    if len(args) > max_svdvals_args:
        msg = "np.linalg.svdvals supports at most 1 positional argument"
        raise TypeError(msg)

    a = args[0]
    svals = np.linalg.svdvals(_get_value(a, traced_type))
    svals_shape, svals_dtype = _result_shape_and_dtype(svals)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("svdvals")),
        inputs=(_get_node(a, graph, traced_type),),
        value=svals,
        attrs={},
        shape=svals_shape,
        dtype=svals_dtype,
    )
    return svals, node_id


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
    pos_ord = 1
    pos_axis = 2
    pos_keepdims = 3
    max_norm_args = 4

    if not args:
        msg = "np.linalg.norm requires an input array"
        raise ValueError(msg)

    allowed_kwargs = frozenset({"ord", "axis", "keepdims"})
    unsupported = set(kwargs) - allowed_kwargs
    if unsupported:
        msg = f"np.linalg.norm does not support kwargs during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    _reject_duplicate_positionals(
        args,
        kwargs,
        ("ord", "axis", "keepdims"),
        function="norm",
    )

    if len(args) > max_norm_args:
        msg = "np.linalg.norm supports at most 4 positional arguments"
        raise TypeError(msg)

    a = args[0]
    ord_raw = args[pos_ord] if len(args) > pos_ord else kwargs.get("ord")
    axis_raw = args[pos_axis] if len(args) > pos_axis else kwargs.get("axis")
    keepdims = (
        bool(args[pos_keepdims])
        if len(args) > pos_keepdims
        else bool(kwargs.get("keepdims", False))
    )

    ord_norm = _normalize_norm_ord(ord_raw)
    axis_norm = _normalize_norm_axis(axis_raw)

    norm_fn = cast("Callable[..., Any]", np.linalg.norm)
    # Unwrap one trace layer only so nested autodiff records the norm in the
    # enclosing graph instead of materializing a constant result.
    a_value = _get_value(a, traced_type)
    result = norm_fn(
        a_value,
        ord=ord_norm,
        axis=axis_norm,
        keepdims=keepdims,
    )
    result_shape, result_dtype = _result_shape_and_dtype(result)

    attrs: dict[str, Any] = {"keepdims": keepdims}
    if ord_norm is not None:
        attrs["ord"] = ord_norm
    if axis_norm is not None:
        attrs["axis"] = axis_norm

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("norm")),
        inputs=(_get_node(a, graph, traced_type),),
        value=result,
        attrs=attrs,
        shape=result_shape,
        dtype=result_dtype,
    )
    return result, node_id


def _det_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.det as a single-output op."""
    max_det_args = 1
    if not args:
        msg = "np.linalg.det requires an input array"
        raise ValueError(msg)
    if kwargs:
        msg = f"np.linalg.det does not support kwargs during tracing: {sorted(kwargs)}"
        raise TracingError(msg)
    if len(args) > max_det_args:
        msg = "np.linalg.det supports at most 1 positional argument"
        raise TypeError(msg)

    a = args[0]
    # Preserve an enclosing trace for higher-order derivatives.
    a_value = _get_value(a, traced_type)
    det_value = np.linalg.det(a_value)
    det_shape, det_dtype = _result_shape_and_dtype(det_value)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("det")),
        inputs=(_get_node(a, graph, traced_type),),
        value=det_value,
        attrs={},
        shape=det_shape,
        dtype=det_dtype,
    )
    return det_value, node_id


def _inv_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.inv as a single-output op."""
    max_inv_args = 1
    if not args:
        msg = "np.linalg.inv requires an input array"
        raise ValueError(msg)
    if kwargs:
        msg = f"np.linalg.inv does not support kwargs during tracing: {sorted(kwargs)}"
        raise TracingError(msg)
    if len(args) > max_inv_args:
        msg = "np.linalg.inv supports at most 1 positional argument"
        raise TypeError(msg)

    a = args[0]
    # Preserve an enclosing trace for higher-order derivatives.
    a_value = _get_value(a, traced_type)
    inv_value = np.linalg.inv(a_value)
    inv_shape, inv_dtype = _result_shape_and_dtype(inv_value)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("inv")),
        inputs=(_get_node(a, graph, traced_type),),
        value=inv_value,
        attrs={},
        shape=inv_shape,
        dtype=inv_dtype,
    )
    return inv_value, node_id


def _cholesky_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.cholesky as a single-output op."""
    max_cholesky_args = 1
    if not args:
        msg = "np.linalg.cholesky requires an input array"
        raise ValueError(msg)
    unsupported = set(kwargs) - {"upper"}
    if unsupported:
        msg = f"np.linalg.cholesky does not support kwargs during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    if len(args) > max_cholesky_args:
        msg = "np.linalg.cholesky supports at most 1 positional argument"
        raise TypeError(msg)

    a = args[0]
    upper = kwargs.get("upper", False)
    if type(upper) is not bool:
        msg = "np.linalg.cholesky upper= must be a bool during tracing"
        raise TracingError(msg)
    chol_value = np.linalg.cholesky(_get_value(a, traced_type), upper=upper)
    chol_shape, chol_dtype = _result_shape_and_dtype(chol_value)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("cholesky")),
        inputs=(_get_node(a, graph, traced_type),),
        value=chol_value,
        attrs={"upper": upper},
        shape=chol_shape,
        dtype=chol_dtype,
    )
    return chol_value, node_id


def _solve_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.solve as a single-output op."""
    if len(args) != 2:
        msg = "np.linalg.solve expects (a, b) during tracing"
        raise TypeError(msg)
    if kwargs:
        msg = f"np.linalg.solve does not support kwargs during tracing: {sorted(kwargs)}"
        raise TracingError(msg)

    a, b = args
    solve_value = np.linalg.solve(_get_value(a, traced_type), _get_value(b, traced_type))
    solve_shape, solve_dtype = _result_shape_and_dtype(solve_value)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("solve")),
        inputs=(
            _get_node(a, graph, traced_type),
            _get_node(b, graph, traced_type),
        ),
        value=solve_value,
        attrs={},
        shape=solve_shape,
        dtype=solve_dtype,
    )
    return solve_value, node_id


def _pinv_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.pinv as a single-output op."""
    if not args or len(args) > 3:
        msg = "np.linalg.pinv expects (a, rcond=None, hermitian=False) during tracing"
        raise TypeError(msg)
    allowed_kwargs = frozenset({"rcond", "hermitian", "rtol"})
    unsupported = set(kwargs) - allowed_kwargs
    if unsupported:
        msg = f"np.linalg.pinv does not support kwargs during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    _reject_duplicate_positionals(
        args,
        kwargs,
        ("rcond", "hermitian"),
        function="pinv",
    )

    a = args[0]
    values = dict(kwargs)
    values.update(dict(zip(("rcond", "hermitian"), args[1:], strict=False)))
    if "rcond" in values and "rtol" in values:
        msg = "np.linalg.pinv accepts only one of rcond= and rtol="
        raise TracingError(msg)
    for name in ("rcond", "rtol"):
        if isinstance(values.get(name), traced_type):
            msg = f"np.linalg.pinv {name}= must be static because it controls numerical rank"
            raise TracingError(msg)

    call_kwargs: dict[str, Any] = {}
    if "rcond" in values:
        call_kwargs["rcond"] = values["rcond"]
    if "rtol" in values:
        call_kwargs["rtol"] = values["rtol"]
    if "hermitian" in values:
        call_kwargs["hermitian"] = bool(values["hermitian"])

    pinv_value = np.linalg.pinv(_get_value(a, traced_type), **call_kwargs)
    pinv_shape, pinv_dtype = _result_shape_and_dtype(pinv_value)
    attrs: dict[str, Any] = {}
    if "rcond" in values:
        attrs["rcond"] = values["rcond"]
    if "rtol" in values:
        attrs["rtol"] = values["rtol"]
    if "hermitian" in values:
        attrs["hermitian"] = bool(values["hermitian"])

    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("pinv")),
        inputs=(_get_node(a, graph, traced_type),),
        value=pinv_value,
        attrs=attrs,
        shape=pinv_shape,
        dtype=pinv_dtype,
    )
    return pinv_value, node_id


def _qr_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.qr as a multi-output op, or a single-output qr_r op when mode='r'."""
    pos_mode = 1
    max_qr_args = 2

    expected_outputs = 2
    if not args:
        msg = "np.linalg.qr requires an input array"
        raise ValueError(msg)

    allowed_kwargs = frozenset({"mode"})
    unsupported = set(kwargs) - allowed_kwargs
    if unsupported:
        msg = f"np.linalg.qr does not support kwargs during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    _reject_duplicate_positionals(args, kwargs, ("mode",), function="qr")

    a = args[0]
    mode = args[pos_mode] if len(args) > pos_mode else kwargs.get("mode", "reduced")
    if len(args) > max_qr_args:
        msg = "np.linalg.qr supports at most 2 positional arguments"
        raise TypeError(msg)

    if mode is None:
        mode = "reduced"

    if mode == "r":
        r_value = np.linalg.qr(_get_value(a, traced_type), mode="r")
        r_shape, r_dtype = _result_shape_and_dtype(r_value)
        node_id = _add_backend_node(
            graph=graph,
            op=canonicalize_numpy_op(_op_name("qr_r")),
            inputs=(_get_node(a, graph, traced_type),),
            value=r_value,
            attrs={"mode": "r"},
            shape=r_shape,
            dtype=r_dtype,
        )
        return r_value, node_id

    if mode not in {"reduced", "complete"}:
        msg = (
            f"np.linalg.qr(mode={mode!r}) is not supported during tracing because it can change "
            "the output arity. Use mode='reduced', mode='complete', or mode='r'."
        )
        raise TracingError(msg)

    result = np.linalg.qr(_get_value(a, traced_type), mode=mode)
    outputs = tuple(result)
    if len(outputs) != expected_outputs:
        msg = f"np.linalg.qr returned {len(outputs)} outputs, expected {expected_outputs}"
        raise ValueError(msg)

    output_meta = tuple(_result_shape_and_dtype(out) for out in outputs)
    output_shapes = tuple(shape for shape, _ in output_meta)
    output_dtypes = tuple(dtype for _, dtype in output_meta)

    parent_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("qr")),
        inputs=(_get_node(a, graph, traced_type),),
        value=outputs,
        attrs={"mode": mode},
        shape=output_shapes[0],
        dtype=output_dtypes[0],
        num_outputs=len(outputs),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    node_ids: list[int] = []
    for index, (output, shape, dtype) in enumerate(
        zip(outputs, output_shapes, output_dtypes, strict=True)
    ):
        node_id = _add_backend_node(
            graph=graph,
            op="advect.getoutput",
            inputs=(parent_id,),
            value=output,
            attrs={"index": index, "num_outputs": len(outputs)},
            shape=shape,
            dtype=dtype,
        )
        node_ids.append(node_id)

    return outputs, tuple(node_ids)


def _eig_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.eig as a multi-output op."""
    expected_outputs = 2
    max_eig_args = 1

    if not args:
        msg = "np.linalg.eig requires an input array"
        raise ValueError(msg)
    if kwargs:
        msg = f"np.linalg.eig does not support kwargs during tracing: {sorted(kwargs)}"
        raise TracingError(msg)
    if len(args) > max_eig_args:
        msg = "np.linalg.eig supports at most 1 positional argument"
        raise TypeError(msg)

    a = args[0]
    result = np.linalg.eig(_get_value(a, traced_type))
    outputs = tuple(result)
    if len(outputs) != expected_outputs:
        msg = f"np.linalg.eig returned {len(outputs)} outputs, expected {expected_outputs}"
        raise ValueError(msg)

    output_meta = tuple(_result_shape_and_dtype(out) for out in outputs)
    output_shapes = tuple(shape for shape, _ in output_meta)
    output_dtypes = tuple(dtype for _, dtype in output_meta)

    parent_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("eig")),
        inputs=(_get_node(a, graph, traced_type),),
        value=outputs,
        attrs={},
        shape=output_shapes[0],
        dtype=output_dtypes[0],
        num_outputs=len(outputs),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    node_ids: list[int] = []
    for index, (output, shape, dtype) in enumerate(
        zip(outputs, output_shapes, output_dtypes, strict=True)
    ):
        node_id = _add_backend_node(
            graph=graph,
            op="advect.getoutput",
            inputs=(parent_id,),
            value=output,
            attrs={"index": index, "num_outputs": len(outputs)},
            shape=shape,
            dtype=dtype,
        )
        node_ids.append(node_id)

    return outputs, tuple(node_ids)


def _eigh_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.eigh as a multi-output op."""
    pos_uplo = 1
    max_eigh_args = 2
    expected_outputs = 2

    if not args:
        msg = "np.linalg.eigh requires an input array"
        raise ValueError(msg)

    allowed_kwargs = frozenset({"UPLO"})
    unsupported = set(kwargs) - allowed_kwargs
    if unsupported:
        msg = f"np.linalg.eigh does not support kwargs during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    _reject_duplicate_positionals(args, kwargs, ("UPLO",), function="eigh")

    if len(args) > max_eigh_args:
        msg = "np.linalg.eigh supports at most 2 positional arguments"
        raise TypeError(msg)

    a = args[0]
    uplo = args[pos_uplo] if len(args) > pos_uplo else kwargs.get("UPLO", "L")
    uplo_raw = str(uplo)
    if uplo_raw in {"L", "l"}:
        uplo_norm: Literal["L", "U", "l", "u"] = "L"
    elif uplo_raw in {"U", "u"}:
        uplo_norm = "U"
    else:
        msg = f"np.linalg.eigh(UPLO={uplo!r}) is not supported during tracing. Use 'L' or 'U'."
        raise TracingError(msg)

    result = np.linalg.eigh(_get_value(a, traced_type), UPLO=uplo_norm)
    outputs = tuple(result)
    if len(outputs) != expected_outputs:
        msg = f"np.linalg.eigh returned {len(outputs)} outputs, expected {expected_outputs}"
        raise ValueError(msg)

    output_meta = tuple(_result_shape_and_dtype(out) for out in outputs)
    output_shapes = tuple(shape for shape, _ in output_meta)
    output_dtypes = tuple(dtype for _, dtype in output_meta)

    parent_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("eigh")),
        inputs=(_get_node(a, graph, traced_type),),
        value=outputs,
        attrs={"UPLO": uplo_norm},
        shape=output_shapes[0],
        dtype=output_dtypes[0],
        num_outputs=len(outputs),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    node_ids: list[int] = []
    for index, (output, shape, dtype) in enumerate(
        zip(outputs, output_shapes, output_dtypes, strict=True)
    ):
        node_id = _add_backend_node(
            graph=graph,
            op="advect.getoutput",
            inputs=(parent_id,),
            value=output,
            attrs={"index": index, "num_outputs": len(outputs)},
            shape=shape,
            dtype=dtype,
        )
        node_ids.append(node_id)

    return outputs, tuple(node_ids)


def _eigvals_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.eigvals as a single-output op."""
    max_eigvals_args = 1
    if not args:
        msg = "np.linalg.eigvals requires an input array"
        raise ValueError(msg)
    if kwargs:
        msg = f"np.linalg.eigvals does not support kwargs during tracing: {sorted(kwargs)}"
        raise TracingError(msg)
    if len(args) > max_eigvals_args:
        msg = "np.linalg.eigvals supports at most 1 positional argument"
        raise TypeError(msg)

    a = args[0]
    eigvals = np.linalg.eigvals(_get_value(a, traced_type))
    eigvals_shape, eigvals_dtype = _result_shape_and_dtype(eigvals)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("eigvals")),
        inputs=(_get_node(a, graph, traced_type),),
        value=eigvals,
        attrs={},
        shape=eigvals_shape,
        dtype=eigvals_dtype,
    )
    return eigvals, node_id


def _eigvalsh_handler(
    graph: DynamicTape,
    traced_type: type[TracedArrayLike],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ArrayFunctionResult:
    """Handle np.linalg.eigvalsh as a single-output op."""
    pos_uplo = 1
    max_eigvalsh_args = 2
    if not args:
        msg = "np.linalg.eigvalsh requires an input array"
        raise ValueError(msg)

    allowed_kwargs = frozenset({"UPLO"})
    unsupported = set(kwargs) - allowed_kwargs
    if unsupported:
        msg = f"np.linalg.eigvalsh does not support kwargs during tracing: {sorted(unsupported)}"
        raise TracingError(msg)
    _reject_duplicate_positionals(args, kwargs, ("UPLO",), function="eigvalsh")

    if len(args) > max_eigvalsh_args:
        msg = "np.linalg.eigvalsh supports at most 2 positional arguments"
        raise TypeError(msg)

    a = args[0]
    uplo = args[pos_uplo] if len(args) > pos_uplo else kwargs.get("UPLO", "L")
    uplo_raw = str(uplo)
    if uplo_raw in {"L", "l"}:
        uplo_norm: Literal["L", "U", "l", "u"] = "L"
    elif uplo_raw in {"U", "u"}:
        uplo_norm = "U"
    else:
        msg = f"np.linalg.eigvalsh(UPLO={uplo!r}) is not supported during tracing. Use 'L' or 'U'."
        raise TracingError(msg)

    eigvals = np.linalg.eigvalsh(_get_value(a, traced_type), UPLO=uplo_norm)
    eigvals_shape, eigvals_dtype = _result_shape_and_dtype(eigvals)
    node_id = _add_backend_node(
        graph=graph,
        op=canonicalize_numpy_op(_op_name("eigvalsh")),
        inputs=(_get_node(a, graph, traced_type),),
        value=eigvals,
        attrs={"UPLO": uplo_norm},
        shape=eigvals_shape,
        dtype=eigvals_dtype,
    )
    return eigvals, node_id


def register_linalg_handlers(handlers: dict[Callable[..., Any], Callable[..., Any]]) -> None:
    """Register linalg-related array functions."""
    handlers[np.linalg.slogdet] = _slogdet_handler
    handlers[np.linalg.svd] = _svd_handler
    handlers[np.linalg.svdvals] = _svdvals_handler
    handlers[np.linalg.qr] = _qr_handler
    handlers[np.linalg.eig] = _eig_handler
    handlers[np.linalg.eigh] = _eigh_handler
    handlers[np.linalg.eigvals] = _eigvals_handler
    handlers[np.linalg.eigvalsh] = _eigvalsh_handler
    handlers[np.linalg.norm] = _norm_handler
    handlers[np.linalg.det] = _det_handler
    handlers[np.linalg.inv] = _inv_handler
    handlers[np.linalg.cholesky] = _cholesky_handler
    handlers[np.linalg.solve] = _solve_handler
    handlers[np.linalg.pinv] = _pinv_handler
