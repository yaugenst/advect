# ruff: noqa: ANN401, C901, PLR0911, PLR2004
"""Shared concrete evaluator binding for staged replay and autodiff."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache
from typing import TYPE_CHECKING, Any, cast

from advect.core._abstract_helpers import (
    ABSTRACT_NAMESPACE_NAME,
    PYTHON_SCALAR_TYPES,
    accumulation_dtype,
    dtype_name,
    normalize_axis,
)
from advect.core._array_api.providers import (
    ResolvedArrayNamespace,
    _array_namespace_can_donate,
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._array_family_ops import ARRAY_API_TO_CANONICAL
from advect.core._array_protocol_helpers import (
    PYTHON_BINARY_OPERATORS,
    PYTHON_OPERATOR_ATTR,
    PYTHON_UNARY_OPERATORS,
    literal_is_weak,
    materialize_weak_scalar_operands,
)
from advect.core._backends import get_hook
from advect.core._basic_index import decode_basic_index
from advect.core._graph_attrs import decode_graph_attrs_from_native
from advect.core._primitive import evaluate_primitive
from advect.core._primitive_call import _infer_namespace
from advect.core._pytree import tree_map
from advect.core._scatter_add import scatter_add

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

_NUMPY_ALIASES = {"absolute": "abs"}
# Portable Array API members of canonical leaves; transpose replays as
# permute_dims rather than the matrix-only matrix_transpose.
_PORTABLE_ALIASES = {
    leaf: path for path, leaf in ARRAY_API_TO_CANONICAL.items() if "." not in path
} | {"transpose": "permute_dims"}
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
    if backend == ABSTRACT_NAMESPACE_NAME or (
        backend is not None and backend.split(".", 1)[0] == "numpy"
    ):
        return
    provider = backend if backend is not None else type(raw_namespace).__name__
    raise TypeError(f"NumPy-authored node {op!r} requires NumPy replay, got {provider!r}")


def shared_operand_positions(op: str) -> tuple[int, ...] | None:
    """Return the operand positions whose storage a result of *op* may share.

    An empty tuple marks a freshly owned result, and ``None`` any operand.
    """
    leaf_name = op.rsplit(".", 1)[-1]
    if op in {"advect.copy", "advect.index_update", "advect.scatter_add"}:
        return ()
    if op == "advect.getitem" or leaf_name in _ALIASING_ARRAY_LEAVES:
        return (0,)
    if op.startswith(("array.", "array_ext.")) and leaf_name in _OWNED_ARRAY_LEAVES:
        return ()
    return None


def bind_native_node_evaluator(op: str, attrs: Mapping[str, Any]) -> BoundEvaluator:
    """Decode one native attribute snapshot and bind its stable evaluator."""
    if not attrs:
        return _bind_attributeless_evaluator(op)
    return _bind_native_node_evaluator(op, attrs)


@cache
def _bind_attributeless_evaluator(op: str) -> BoundEvaluator:
    # A bound evaluator is a pure function of its op and attributes, so the
    # attributeless nodes that dominate derivative graphs share one closure
    # rather than retaining one per node.
    return _bind_native_node_evaluator(op, {})


def _bind_native_node_evaluator(op: str, attrs: Mapping[str, Any]) -> BoundEvaluator:
    evaluator = bind_node_evaluator(op, decode_graph_attrs_from_native(attrs))
    metadata = cast("Any", evaluator)
    positions = shared_operand_positions(op)
    if positions == ():
        metadata.__advect_owned_output__ = True
    elif positions is not None:
        metadata.__advect_alias_positions__ = positions
    if op == "advect.index_update":
        metadata.__advect_donation_positions__ = (0,)
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
    portable: BoundEvaluator | None = None

    def evaluate_bound(
        input_vals: tuple[Any, ...],
        context: Any | None = None,
        _donation_position: int | None = None,
    ) -> Any:
        nonlocal portable
        _validate_backend_namespace(op, backend_name, context)
        if members is not None:
            nested = _instance_specific_value(input_vals)
            if nested is not None:
                runtime_namespace = _get_array_namespace(nested)
                try:
                    _namespace_member(runtime_namespace, members)
                except AttributeError:
                    # The backend frontend evaluates an extension that the nested
                    # namespace lacks; keep its results on the nested trace's tracers.
                    adopt = getattr(nested, "_advect_adopt", None)
                    result = bound(input_vals)
                    return result if adopt is None else tree_map(adopt, result)
                if portable is None:
                    portable = _bind_array_op(op, attrs)
                return portable(input_vals, runtime_namespace, None)
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
        raise ValueError(f"advect.getoutput index {index} out of range for {num_outputs} outputs")

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


def _bind_scatter_add_evaluator(attrs: Mapping[str, Any]) -> BoundEvaluator:
    axis = attrs["axis"]
    size = attrs["size"]

    def evaluate(
        input_vals: tuple[Any, ...],
        context: Any | None = None,
        _donation_position: int | None = None,
    ) -> Any:
        if len(input_vals) != 2:
            raise ValueError("advect.scatter_add expects values and indices inputs")
        values, indices = input_vals
        return scatter_add(values, indices, axis=axis, size=size, namespace=context)

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


def _instance_specific_value(values: object) -> Any | None:
    """Return the first replay input whose namespace is invocation-local."""
    if bool(
        getattr(type(values), "__advect_namespace_is_instance_specific__", False),
    ):
        return values
    if isinstance(values, dict):
        values = tuple(values.values())
    if isinstance(values, (tuple, list)):
        for value in values:
            nested = _instance_specific_value(value)
            if nested is not None:
                return nested
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


# Advect's abstract namespace serves the cumulative scans under NumPy's
# spelling at every revision, as derivative rules record them; the standard
# names enter only in 2023.12 and 2024.12.
_ABSTRACT_SCANS = {"cumulative_prod": "cumprod", "cumulative_sum": "cumsum"}


def _namespace_member(
    namespace: Any,
    members: tuple[tuple[str, ...], tuple[str, ...]],
) -> Callable[..., Any]:
    name = getattr(namespace, "__name__", "")
    parts = members[name != "numpy"]
    if name == ABSTRACT_NAMESPACE_NAME and parts[-1] in _ABSTRACT_SCANS:
        parts = (*parts[:-1], _ABSTRACT_SCANS[parts[-1]])
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


# A comparison reflects through the other operand's mirrored method.
_MIRRORED_COMPARISONS = {"eq": "eq", "ge": "le", "gt": "lt", "le": "ge", "lt": "gt", "ne": "ne"}


def _reflected_method(binary: Callable[[Any, Any], Any]) -> str:
    """Return the right operand's method that Python calls for ``binary`` reflected."""
    stem = binary.__name__.rstrip("_")
    mirrored = _MIRRORED_COMPARISONS.get(stem)
    return f"__r{stem}__" if mirrored is None else f"__{mirrored}__"


def _reflected_operator(
    binary: Callable[[Any, Any], Any],
    reflected: str,
    left: Any,
    right: Any,
) -> Any:
    """Apply Python's operator to a right-hand tracer through its reflection first.

    A provider array such as array_api_strict's raises instead of returning
    ``NotImplemented``, so Python must not ask the left operand first.
    """
    result = getattr(right, reflected)(left)
    return binary(left, right) if result is NotImplemented else result


def _bind_array_op(op: str, attrs: Mapping[str, Any]) -> BoundEvaluator:  # noqa: PLR0915
    """Resolve one portable array operation's calling convention once per node."""
    members = _array_members(op)
    leaf = op.rsplit(".", 1)[-1]
    kind = _ARRAY_CALL_KINDS.get(leaf)
    binary = PYTHON_BINARY_OPERATORS.get(leaf)
    reflected = None if binary is None else _reflected_method(binary)
    # Python's operator computed this weak result, and computes it again on the
    # weak replay values: Python scalars, or tracers whose operators keep them weak.
    python_operator = (
        (binary or PYTHON_UNARY_OPERATORS.get(leaf)) if attrs.get(PYTHON_OPERATOR_ATTR) else None
    )
    accumulates = leaf in {"cumprod", "cumsum", "prod", "sum", "trace"}
    attr_version = attrs.get("_advect_array_api_version")
    # A NumPy-authored node declared NumPy's accumulation dtype, which no
    # replay namespace's Array API revision may change.
    backend_authored = "_advect_backend" in attrs
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
        if python_operator is not None:
            return python_operator(*inputs)
        if binary is not None and len(inputs) == 2:
            left, right = inputs
            left_traced = callable(getattr(left, "_advect_snapshot", None))
            # A tracer applies the operator through its own frontend, except that
            # weak scalars alone call the function, whose result is strong.
            if (left_traced or callable(getattr(right, "_advect_snapshot", None))) and not (
                literal_is_weak(left) and literal_is_weak(right)
            ):
                return (
                    binary(left, right)
                    if left_traced
                    else _reflected_operator(binary, cast("str", reflected), left, right)
                )
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
            if backend_authored:
                kwargs["dtype"] = accumulation_dtype(inputs[0].dtype)
            elif isinstance(requested, str):
                target_dtype = accumulation_dtype(inputs[0].dtype, array_api_version=requested)
                if target_dtype != dtype_name(inputs[0].dtype):
                    kwargs["dtype"] = target_dtype
        if isinstance(device_key, str):
            kwargs["device"] = _execution_device(device_key, inputs, namespace)
        if kwargs.get("dtype") is not None:
            kwargs["dtype"] = getattr(namespace, str(kwargs["dtype"]), kwargs["dtype"])

        if kind == "astype":
            if as_array:
                if is_numpy and callable(getattr(inputs[0], "_advect_snapshot", None)):
                    # NumPy's asarray dispatches a traced operand only through like=.
                    kwargs["like"] = inputs[0]
                return _namespace_member(namespace, _ASARRAY)(inputs[0], **kwargs)
            dtype = kwargs.pop("dtype", None)
            if dtype is None:
                raise TypeError("astype requires a dtype")
            scalar_type = getattr(namespace, "generic", None)
            if type(inputs[0]) in PYTHON_SCALAR_TYPES or (
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
    "advect.scatter_add": _bind_scatter_add_evaluator,
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
    "shared_operand_positions",
]
