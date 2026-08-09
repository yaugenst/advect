# ruff: noqa: PLR2004  # Numeric arity checks in handler validation
"""Linalg-related ``__array_function__`` handlers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as _numpy  # noqa: ICN001 - concrete namespace with dynamic protocol operands

from advect.core._array_api.results import restore_array_api_result
from advect.core._errors import TracingError
from advect.core._protocols import _snapshot_traced
from advect.numpy._array_function.composite import _finish, _first_traced
from advect.numpy._array_function.emission import (
    _add_backend_node,
    _get_node,
    _get_value,
    _result_shape_and_dtype,
)
from advect.numpy._array_function.normalization import _binary_handler
from advect.numpy._op_bindings import canonicalize_numpy_op

if TYPE_CHECKING:
    from collections.abc import Callable

    from advect.core._native import DynamicTape
    from advect.core._protocols import TracedArrayLike
    from advect.numpy._array_function.emission import ArrayFunctionResult

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
