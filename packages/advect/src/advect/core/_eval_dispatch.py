# ruff: noqa: ANN401, C901, PLR0911, PLR2004
"""Shared concrete evaluator binding for staged replay and autodiff."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable
from functools import cache
from typing import TYPE_CHECKING, Any, cast

from advect.core._abstract_helpers import accumulation_dtype, dtype_name, normalize_axis
from advect.core._array_api.providers import (
    ResolvedArrayNamespace,
    _array_namespace_can_donate,
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._array_protocol_helpers import materialize_weak_scalar_operands
from advect.core._backends import get_hook
from advect.core._basic_index import decode_basic_index
from advect.core._graph_attrs import decode_graph_attrs_from_native
from advect.core._primitive import evaluate_primitive
from advect.core._primitive_call import _infer_namespace

if TYPE_CHECKING:
    from collections.abc import Mapping

type BoundEvaluator = Callable[[tuple[Any, ...], Any | None, int | None], Any]

_ALIASING_ARRAY_LEAVES = frozenset(
    {
        "broadcast_to",
        "expand_dims",
        "reshape",
        "squeeze",
        "transpose",
    }
)

_OWNED_ARRAY_LEAVES = frozenset(
    {
        "absolute",
        "add",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctan2",
        "arctanh",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "ceil",
        "conjugate",
        "cos",
        "cosh",
        "divide",
        "equal",
        "exp",
        "expm1",
        "floor",
        "floor_divide",
        "fft",
        "fftfreq",
        "fftn",
        "fftshift",
        "greater",
        "greater_equal",
        "hypot",
        "hfft",
        "isfinite",
        "isinf",
        "isnan",
        "ifft",
        "ifftn",
        "ifftshift",
        "ihfft",
        "irfft",
        "irfftn",
        "less",
        "less_equal",
        "log",
        "log1p",
        "log2",
        "log10",
        "logaddexp",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "matmul",
        "maximum",
        "minimum",
        "multiply",
        "negative",
        "not_equal",
        "ones_like",
        "positive",
        "power",
        "reciprocal",
        "remainder",
        "rint",
        "sign",
        "sin",
        "sinh",
        "solve",
        "sqrt",
        "square",
        "subtract",
        "tan",
        "tanh",
        "trunc",
        "where",
        "zeros_like",
        "rfft",
        "rfftfreq",
        "rfftn",
        "searchsorted",
        "sort",
        "argsort",
        "take",
        "take_along_axis",
    }
)

_BINARY_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "add": operator.add,
    "bitwise_and": operator.and_,
    "bitwise_or": operator.or_,
    "bitwise_xor": operator.xor,
    "divide": operator.truediv,
    "equal": operator.eq,
    "floor_divide": operator.floordiv,
    "greater": operator.gt,
    "greater_equal": operator.ge,
    "less": operator.lt,
    "less_equal": operator.le,
    "matmul": operator.matmul,
    "multiply": operator.mul,
    "not_equal": operator.ne,
    "power": operator.pow,
    "remainder": operator.mod,
    "subtract": operator.sub,
}

_SCALAR_UNARY_OPERATORS: dict[str, Callable[[Any], Any]] = {
    "absolute": abs,
    "conjugate": lambda value: value.conjugate(),
    "imag": lambda value: value.imag,
    "negative": operator.neg,
    "positive": operator.pos,
    "real": lambda value: value.real,
}

_PYTHON_SCALARS = frozenset({bool, complex, float, int})
_NUMPY_ALIASES = {"absolute": "abs"}
_PORTABLE_ALIASES = {
    "absolute": "abs",
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctan2": "atan2",
    "arctanh": "atanh",
    "concatenate": "concat",
    "conjugate": "conj",
    "cumprod": "cumulative_prod",
    "cumsum": "cumulative_sum",
    "invert": "bitwise_invert",
    "left_shift": "bitwise_left_shift",
    "power": "pow",
    "rint": "round",
    "right_shift": "bitwise_right_shift",
    "transpose": "permute_dims",
}
_LINALG_MEMBERS = frozenset({"cross", "diagonal", "outer", "trace", "vecdot"})
_ASARRAY = (("asarray",), ("asarray",))
_PERMUTE_DIMS = (("permute_dims",), ("permute_dims",))

# How a leaf passes operands and static attrs to its namespace member; plain
# ``function(*operands, **attrs)`` needs no entry.
_ARRAY_CALL_KINDS = {
    **dict.fromkeys(
        ("broadcast_to", "moveaxis", "repeat", "reshape", "tile", "transpose"), "operand"
    ),
    **dict.fromkeys(("arange", "empty", "eye", "linspace", "ones", "zeros"), "creation"),
    **dict.fromkeys(("fftfreq", "rfftfreq"), "frequency"),
    **dict.fromkeys(("argsort", "sort"), "sort"),
    **dict.fromkeys(("concatenate", "stack"), "sequence"),
    **dict.fromkeys(("diagonal", "trace"), "diagonal"),
    "astype": "astype",
    "clip": "clip",
    "full": "full",
    "pinv": "pinv",
}
_REQUIRED = object()
# Static attrs passed positionally, as (name, default) pairs.
_POSITIONAL_ATTRS: dict[str, tuple[tuple[str, object], ...]] = {
    "arange": (("start", _REQUIRED), ("stop", None), ("step", 1)),
    "broadcast_to": (("shape", _REQUIRED),),
    "empty": (("shape", _REQUIRED),),
    "eye": (("n_rows", _REQUIRED), ("n_cols", None)),
    "fftfreq": (("n", _REQUIRED),),
    "full": (("shape", _REQUIRED),),
    "linspace": (("start", _REQUIRED), ("stop", _REQUIRED), ("num", _REQUIRED)),
    "moveaxis": (("source", _REQUIRED), ("destination", _REQUIRED)),
    "ones": (("shape", _REQUIRED),),
    "repeat": (("repeats", _REQUIRED),),
    "reshape": (("shape", _REQUIRED),),
    "rfftfreq": (("n", _REQUIRED),),
    "tile": (("reps", _REQUIRED),),
    "zeros": (("shape", _REQUIRED),),
}


def _validate_backend_namespace(op: str, backend_name: str, namespace: Any | None) -> None:
    if backend_name != "numpy" or namespace is None:
        return
    raw_namespace = getattr(namespace, "raw_namespace", namespace)
    backend = _get_backend_key_from_namespace(raw_namespace)
    # Preserve Advect's internal abstract namespace used to construct nested staged transforms.
    if backend == "advect.array_api" or (
        backend is not None and backend.split(".", 1)[0] == "numpy"
    ):
        return
    provider = backend if backend is not None else type(raw_namespace).__name__
    raise TypeError(f"NumPy-authored node {op!r} requires NumPy replay, got {provider!r}")


def bind_native_node_evaluator(op: str, attrs: Mapping[str, Any]) -> BoundEvaluator:
    """Decode one native attribute snapshot and bind its stable evaluator."""
    evaluator = bind_node_evaluator(op, decode_graph_attrs_from_native(attrs))
    metadata = cast("Any", evaluator)
    if op in {"advect.copy", "advect.index_update"}:
        metadata.__advect_owned_output__ = True
    if op == "advect.index_update":
        metadata.__advect_donation_positions__ = (0,)
    leaf_name = op.rsplit(".", 1)[-1]
    if op == "advect.getitem" or leaf_name in _ALIASING_ARRAY_LEAVES:
        metadata.__advect_alias_positions__ = (0,)
    elif op.startswith(("array.", "array_ext.")) and leaf_name in _OWNED_ARRAY_LEAVES:
        metadata.__advect_owned_output__ = True
    return evaluator


def has_core_evaluator(op: str) -> bool:
    """Return whether a structural operation has a built-in evaluator."""
    return op in _CORE_BINDERS


def bind_node_evaluator(op: str, attrs: Mapping[str, Any]) -> BoundEvaluator:
    """Resolve stable evaluator dispatch and attribute decoding once per graph."""
    bind_core = _CORE_BINDERS.get(op)
    if bind_core is not None and not (op == "advect.copy" and "_advect_backend" in attrs):
        return bind_core(attrs)

    if op.startswith("custom."):

        def evaluate_custom(
            input_vals: tuple[Any, ...],
            context: Any | None = None,
            _donation_position: int | None = None,
        ) -> Any:
            return evaluate_primitive(op, input_vals, attrs, namespace=context)

        return evaluate_custom

    backend_name = attrs.get("_advect_backend")
    if isinstance(backend_name, str) and backend_name:
        backend_evaluate_op = get_hook(f"{backend_name}.evaluate_op")
        if backend_evaluate_op is not None:
            return _bind_backend_op(op, attrs, backend_name, backend_evaluate_op)
    if op.startswith(("array.", "array_ext.")):
        return _bind_array_op(op, attrs)
    raise ValueError(f"No evaluator for staged operation {op!r}")


def _bind_backend_op(
    op: str,
    attrs: Mapping[str, Any],
    backend_name: str,
    backend_evaluate_op: Callable[..., Any],
) -> BoundEvaluator:
    decoder = get_hook(f"{backend_name}.decode_attrs")
    decoded_attrs = decoder(op, attrs) if decoder is not None else attrs
    bind_evaluator = get_hook(f"{backend_name}.bind_evaluator")
    bound = None if bind_evaluator is None else bind_evaluator(op, decoded_attrs)
    if bound is None:

        def evaluate_backend(
            input_vals: tuple[Any, ...],
            context: Any | None = None,
            _donation_position: int | None = None,
        ) -> Any:
            _validate_backend_namespace(op, backend_name, context)
            return backend_evaluate_op(op, input_vals, decoded_attrs)

        return evaluate_backend

    members = _array_members(op) if op.startswith(("array.", "array_ext.")) else None
    # An instance-specific namespace (a nested trace) replays through the
    # portable array evaluator, which only such calls bind.
    portable = cache(lambda: _bind_array_op(op, attrs))

    def evaluate_bound(
        input_vals: tuple[Any, ...],
        context: Any | None = None,
        _donation_position: int | None = None,
    ) -> Any:
        _validate_backend_namespace(op, backend_name, context)
        if members is not None:
            runtime_namespace = _instance_specific_namespace(input_vals)
            if runtime_namespace is not None:
                try:
                    _namespace_member(runtime_namespace, members)
                except AttributeError:
                    pass
                else:
                    return portable()(input_vals, runtime_namespace, None)
        return bound(input_vals)

    return evaluate_bound


def _bind_getoutput_evaluator(attrs: Mapping[str, Any]) -> BoundEvaluator:
    index = attrs.get("index")
    num_outputs = attrs.get("num_outputs")
    if not isinstance(index, int):
        raise TypeError("advect.getoutput requires integer 'index' attr")
    if not isinstance(num_outputs, int):
        raise TypeError("advect.getoutput requires integer 'num_outputs' attr")
    if index < 0 or index >= num_outputs:
        raise IndexError(f"advect.getoutput index {index} out of range for {num_outputs} outputs")

    def evaluate(
        input_vals: tuple[Any, ...],
        _context: Any | None = None,
        _donation_position: int | None = None,
    ) -> Any:
        if len(input_vals) != 1:
            raise ValueError("advect.getoutput expects a single input value")
        parent_value = input_vals[0]
        if not isinstance(parent_value, tuple):
            raise TypeError("advect.getoutput input must be a tuple of outputs")
        if len(parent_value) != num_outputs:
            raise ValueError(
                f"advect.getoutput expected {num_outputs} outputs but got {len(parent_value)}"
            )
        return parent_value[index]

    return evaluate


def _bind_getitem_evaluator(attrs: Mapping[str, Any]) -> BoundEvaluator:
    index = decode_basic_index(attrs.get("index"))

    def evaluate(
        input_vals: tuple[Any, ...],
        _context: Any | None = None,
        _donation_position: int | None = None,
    ) -> Any:
        if len(input_vals) != 1:
            raise ValueError("advect.getitem expects one input")
        return input_vals[0][index]

    return evaluate


def _bind_copy_evaluator(attrs: Mapping[str, Any]) -> BoundEvaluator:
    order = attrs.get("order")

    def evaluate(
        input_vals: tuple[Any, ...],
        context: Any | None = None,
        _donation_position: int | None = None,
    ) -> Any:
        if len(input_vals) != 1:
            raise ValueError("advect.copy expects one input")
        value = input_vals[0]
        copy_value = getattr(value, "copy", None)
        if callable(copy_value):
            return copy_value() if order is None else copy_value(order=order)
        namespace = context if context is not None else _infer_namespace(input_vals)
        asarray = None if namespace is None else getattr(namespace, "asarray", None)
        if callable(asarray):
            return asarray(value, copy=True)
        raise TypeError(f"Cannot copy staged value of type {type(value).__name__}")

    return evaluate


def _bind_index_update_evaluator(attrs: Mapping[str, Any]) -> BoundEvaluator:
    index = decode_basic_index(attrs.get("index"))
    mode = attrs.get("mode", "set")
    if mode not in {"add", "set"}:
        raise ValueError(f"Unsupported index_update mode {mode!r}")

    def evaluate(
        input_vals: tuple[Any, ...],
        context: Any | None = None,
        donation_position: int | None = None,
    ) -> Any:
        if len(input_vals) != 2:
            raise ValueError("advect.index_update expects array and replacement inputs")
        result = (
            input_vals[0]
            if donation_position == 0 and _can_donate_array(input_vals[0])
            else _bind_copy_evaluator({})((input_vals[0],), context, None)
        )
        if mode == "add":
            result[index] += input_vals[1]
        else:
            result[index] = input_vals[1]
        return result

    return evaluate


def _can_donate_array(value: Any) -> bool:
    """Return whether an internal array buffer is safe for staged reuse."""
    if callable(getattr(value, "_advect_snapshot", None)):
        return False
    flags = getattr(value, "flags", None)
    if flags is None or getattr(value, "base", None) is not None:
        return False
    if not bool(getattr(flags, "owndata", False)):
        return False
    writable = getattr(flags, "writeable", getattr(flags, "writable", None))
    if writable is not None:
        return bool(writable)
    return _array_namespace_can_donate(value)


def _instance_specific_namespace(values: object) -> Any | None:
    """Resolve an invocation-local namespace nested inside replay inputs."""
    if bool(
        getattr(type(values), "__advect_namespace_is_instance_specific__", False),
    ):
        return _get_array_namespace(values)
    if isinstance(values, (tuple, list)):
        for value in values:
            namespace = _instance_specific_namespace(value)
            if namespace is not None:
                return namespace
    if isinstance(values, dict):
        for value in values.values():
            namespace = _instance_specific_namespace(value)
            if namespace is not None:
                return namespace
    return None


@cache
def _array_members(op: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return an array operation's NumPy and portable Array API member paths."""
    path = op.removeprefix("array_ext.").removeprefix("array.")
    *prefix, leaf = path.split(".")
    numpy = (*prefix, _NUMPY_ALIASES.get(leaf, leaf))
    if path in _LINALG_MEMBERS:
        prefix = ["linalg"]
    return numpy, (*prefix, _PORTABLE_ALIASES.get(leaf, leaf))


def _namespace_member(
    namespace: Any,
    members: tuple[tuple[str, ...], tuple[str, ...]],
) -> Callable[..., Any]:
    parts = members[getattr(namespace, "__name__", "") != "numpy"]
    target = namespace
    for part in parts:
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"Array namespace member {'.'.join(parts)!r} is not callable")
    return cast("Callable[..., Any]", target)


def _execution_device(device_key: str, inputs: tuple[Any, ...], namespace: Any) -> Any:
    candidates = [
        getattr(value, "device", None)
        for value in inputs
        if getattr(value, "device", None) is not None
    ]
    namespace_info = getattr(namespace, "__array_namespace_info__", None)
    if callable(namespace_info):
        devices = getattr(namespace_info(), "devices", None)
        if callable(devices):
            available_devices = devices()
            if not isinstance(available_devices, Iterable):
                raise TypeError("Array namespace devices() must return an iterable")
            candidates.extend(available_devices)
    device = next((candidate for candidate in candidates if str(candidate) == device_key), None)
    if device is None:
        raise ValueError(f"Array API device {device_key!r} is unavailable at execution")
    return device


def _bind_array_op(op: str, attrs: Mapping[str, Any]) -> BoundEvaluator:  # noqa: PLR0915
    """Resolve one portable array operation's calling convention once per node."""
    members = _array_members(op)
    leaf = op.rsplit(".", 1)[-1]
    kind = _ARRAY_CALL_KINDS.get(leaf)
    binary = _BINARY_OPERATORS.get(leaf)
    unary = _SCALAR_UNARY_OPERATORS.get(leaf)
    accumulates = leaf in {"cumprod", "cumsum", "prod", "sum"}
    attr_version = attrs.get("_advect_array_api_version")
    device_key = attrs.get("_advect_device")
    clip_bounds = (
        (
            bool(attrs.get("_advect_clip_min_is_input", False)),
            bool(attrs.get("_advect_clip_max_is_input", False)),
        )
        if kind == "clip"
        else (False, False)
    )
    tolerance = attrs.get("_advect_pinv_tolerance") if kind == "pinv" else None
    as_array = kind == "astype" and bool(attrs.get("_advect_array_api_asarray", False))
    static_kwargs = {key: value for key, value in attrs.items() if not key.startswith("_advect_")}
    positional: tuple[Any, ...] = ()
    if leaf in _POSITIONAL_ATTRS:
        positional = tuple(
            static_kwargs.pop(name) if default is _REQUIRED else static_kwargs.pop(name, default)
            for name, default in _POSITIONAL_ATTRS[leaf]
        )
    elif leaf == "transpose":
        axes = static_kwargs.pop("axes", None)
        positional = () if axes is None else (axes,)

    def evaluate(  # noqa: PLR0912 - one explicit portable execution schema
        inputs: tuple[Any, ...],
        context: Any | None = None,
        _donation_position: int | None = None,
    ) -> Any:
        if binary is not None and len(inputs) == 2:
            left, right = inputs
            if (
                (type(left) in _PYTHON_SCALARS and type(right) in _PYTHON_SCALARS)
                or callable(getattr(left, "_advect_snapshot", None))
                or callable(getattr(right, "_advect_snapshot", None))
            ):
                return binary(left, right)
        if unary is not None and len(inputs) == 1 and type(inputs[0]) in _PYTHON_SCALARS:
            return unary(inputs[0])
        resolved = context if context is not None else _infer_namespace(inputs)
        if resolved is None:
            raise RuntimeError(f"Cannot execute {op!r} without an array namespace")
        namespace = (
            resolved.raw_namespace if isinstance(resolved, ResolvedArrayNamespace) else resolved
        )
        is_numpy = getattr(namespace, "__name__", "") == "numpy"
        if not is_numpy:
            inputs = cast(
                "tuple[Any, ...]",
                materialize_weak_scalar_operands(op, inputs, namespace=namespace),
            )
        kwargs = dict(static_kwargs)
        if accumulates and inputs and kwargs.get("dtype") is None:
            requested = (
                attr_version
                if isinstance(attr_version, str)
                else getattr(resolved, "_advect_requested_array_api_version", None)
            )
            if isinstance(requested, str):
                target_dtype = accumulation_dtype(inputs[0].dtype, array_api_version=requested)
                if target_dtype != dtype_name(inputs[0].dtype):
                    kwargs["dtype"] = target_dtype
        if isinstance(device_key, str):
            kwargs["device"] = _execution_device(device_key, inputs, namespace)
        if kwargs.get("dtype") is not None:
            kwargs["dtype"] = getattr(namespace, str(kwargs["dtype"]), kwargs["dtype"])

        if kind == "astype":
            dtype = kwargs.pop("dtype", None)
            if as_array:
                if dtype is not None:
                    kwargs["dtype"] = dtype
                return _namespace_member(namespace, _ASARRAY)(inputs[0], **kwargs)
            if dtype is None:
                raise TypeError("astype requires a dtype")
            scalar_type = getattr(namespace, "generic", None)
            if type(inputs[0]) in _PYTHON_SCALARS or (
                isinstance(scalar_type, type) and isinstance(inputs[0], scalar_type)
            ):
                return _namespace_member(namespace, _ASARRAY)(inputs[0], dtype=dtype)
            return _namespace_member(namespace, members)(inputs[0], dtype, **kwargs)
        if kind == "diagonal" and not is_numpy:
            rank = len(inputs[0].shape)
            first_axis = normalize_axis(kwargs.pop("axis1", 0), rank)
            second_axis = normalize_axis(kwargs.pop("axis2", 1), rank)
            axes = (
                *(axis for axis in range(rank) if axis not in {first_axis, second_axis}),
                first_axis,
                second_axis,
            )
            if axes != tuple(range(rank)):
                inputs = (_namespace_member(namespace, _PERMUTE_DIMS)(inputs[0], axes),)
        elif kind == "sort" and is_numpy and bool(kwargs.pop("descending", False)):
            raise NotImplementedError(
                "Portable staged descending sort is not supported on NumPy; "
                "use an Array API provider or sort ascending."
            )
        function = _namespace_member(namespace, members)
        if kind == "operand":
            return function(inputs[0], *positional, **kwargs)
        if kind == "creation":
            return function(*positional, **kwargs)
        if kind == "frequency" and is_numpy:
            dtype = kwargs.pop("dtype")
            return function(*positional, **kwargs).astype(dtype, copy=False)
        if kind == "frequency":
            return function(*positional, **kwargs)
        if kind == "full":
            return function(*positional, inputs[0], **kwargs)
        if kind == "sequence":
            return function(inputs, **kwargs)
        if kind == "clip":
            values = iter(inputs[1:])
            lower = next(values) if clip_bounds[0] else None
            upper = next(values) if clip_bounds[1] else None
            return function(inputs[0], min=lower, max=upper, **kwargs)
        if kind == "pinv":
            if tolerance is not None:
                if tolerance not in {"rcond", "rtol"} or len(inputs) != 2:
                    raise ValueError("pinv tolerance metadata does not match its operands")
                kwargs[str(tolerance)] = inputs[1]
            return function(inputs[0], **kwargs)
        return function(*inputs, **kwargs)

    return evaluate


_CORE_BINDERS: dict[str, Callable[[Mapping[str, Any]], BoundEvaluator]] = {
    "advect.copy": _bind_copy_evaluator,
    "advect.getitem": _bind_getitem_evaluator,
    "advect.getoutput": _bind_getoutput_evaluator,
    "advect.index_update": _bind_index_update_evaluator,
}


def _decode_attrs_for_vjp(op: str, attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Decode backend-owned attrs from an already materialized node view."""
    if not attrs:
        return {}
    if len(attrs) == 1 and "_advect_backend" in attrs:
        return {}
    materialized_attrs = dict(attrs)
    backend_name = materialized_attrs.get("_advect_backend")
    if isinstance(backend_name, str) and backend_name:
        backend_decoder = get_hook(f"{backend_name}.decode_attrs")
        if backend_decoder is not None:
            decoded = backend_decoder(op, materialized_attrs)
            return dict(cast("dict[str, Any]", decoded))
    if "." not in op:
        return materialized_attrs
    op_ns = op.split(".", 1)[0]
    decoder = get_hook(f"{op_ns}.decode_attrs")
    if decoder is None:
        return materialized_attrs
    decoded = decoder(op, materialized_attrs)
    return dict(cast("dict[str, Any]", decoded))


__all__ = [
    "BoundEvaluator",
    "_decode_attrs_for_vjp",
    "bind_native_node_evaluator",
    "bind_node_evaluator",
    "has_core_evaluator",
]
