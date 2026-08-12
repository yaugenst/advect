# ruff: noqa: ANN401, C901, PLR0911, PLR2004
"""Shared concrete evaluator binding for staged replay and autodiff."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

from advect.core._abstract_helpers import accumulation_dtype, dtype_name, normalize_axis
from advect.core._array_api.providers import (
    _array_namespace_can_donate,
    _get_array_namespace,
    _get_backend_key_from_namespace,
)
from advect.core._array_protocol_helpers import materialize_weak_scalar_operands
from advect.core._backend_hooks import resolve_backend_hooks
from advect.core._backends import get_hook
from advect.core._basic_index import decode_basic_index
from advect.core._graph_attrs import decode_graph_attrs_from_native
from advect.core._primitive import evaluate_primitive

if TYPE_CHECKING:
    from collections.abc import Mapping

type BoundEvaluator = Callable[[tuple[Any, ...], Any | None, int | None], Any]

_CORE_EVALUATOR_OPS = frozenset(
    {"advect.copy", "advect.getitem", "advect.getoutput", "advect.index_update"}
)

_ALIASING_ARRAY_LEAVES = frozenset(
    {
        "broadcast_to",
        "expand_dims",
        "permute_dims",
        "reshape",
        "squeeze",
        "transpose",
    }
)

_OWNED_ARRAY_LEAVES = frozenset(
    {
        "abs",
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
        "bitwise_invert",
        "bitwise_left_shift",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "ceil",
        "conj",
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


def _selected_array_api_version(
    namespace: Any,
    attrs: Mapping[str, Any],
) -> str | None:
    version = attrs.get("_advect_array_api_version")
    if isinstance(version, str):
        return version
    requested = getattr(namespace, "_advect_requested_array_api_version", None)
    return requested if isinstance(requested, str) else None


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
    return op in _CORE_EVALUATOR_OPS


def bind_node_evaluator(op: str, attrs: Mapping[str, Any]) -> BoundEvaluator:
    """Resolve stable evaluator dispatch and attribute decoding once per graph."""
    if op == "advect.getoutput":
        return _bind_getoutput_evaluator(attrs)
    if op == "advect.getitem":
        return _bind_getitem_evaluator(attrs)
    if op == "advect.copy" and "_advect_backend" not in attrs:
        return _bind_copy_evaluator(attrs)
    if op == "advect.index_update":
        return _bind_index_update_evaluator(attrs)

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
            decoder = get_hook(f"{backend_name}.decode_attrs")
            decoded_attrs = decoder(op, attrs) if decoder is not None else attrs
            bind_evaluator = get_hook(f"{backend_name}.bind_evaluator")
            if bind_evaluator is not None:
                bound = bind_evaluator(op, decoded_attrs)
                if bound is not None:

                    def evaluate_bound(
                        input_vals: tuple[Any, ...],
                        context: Any | None = None,
                        _donation_position: int | None = None,
                    ) -> Any:
                        _validate_backend_namespace(op, backend_name, context)
                        if op.startswith(("array.", "array_ext.")):
                            runtime_namespace = _instance_specific_namespace(input_vals)
                            if runtime_namespace is not None:
                                path = op.removeprefix("array_ext.").removeprefix("array.")
                                try:
                                    _namespace_function(runtime_namespace, path)
                                except AttributeError:
                                    pass
                                else:
                                    return _evaluate_array_op(
                                        op,
                                        input_vals,
                                        attrs,
                                        runtime_namespace,
                                    )
                        return bound(input_vals)

                    return evaluate_bound

            def evaluate_backend(
                input_vals: tuple[Any, ...],
                context: Any | None = None,
                _donation_position: int | None = None,
            ) -> Any:
                _validate_backend_namespace(op, backend_name, context)
                return backend_evaluate_op(op, input_vals, decoded_attrs)

            return evaluate_backend

    if op.startswith(("array.", "array_ext.")):

        def evaluate_array(
            input_vals: tuple[Any, ...],
            context: Any | None = None,
            _donation_position: int | None = None,
        ) -> Any:
            namespace = context if context is not None else _namespace_from_inputs(input_vals)
            return _evaluate_array_op(op, input_vals, attrs, namespace)

        return evaluate_array

    # Some custom backend namespaces intentionally resolve from runtime input
    # types. Preserve that dynamic path rather than guessing at compile time.
    def evaluate_dynamic(
        input_vals: tuple[Any, ...],
        context: Any | None = None,
        _donation_position: int | None = None,
    ) -> Any:
        return evaluate_node_value(op, input_vals, attrs, namespace=context)

    return evaluate_dynamic


def evaluate_node_value(
    op: str,
    input_vals: tuple[Any, ...],
    attrs: Mapping[str, Any],
    *,
    namespace: Any | None = None,
) -> Any:
    """Evaluate a single op given concrete inputs."""
    if op in _CORE_EVALUATOR_OPS and not (op == "advect.copy" and "_advect_backend" in attrs):
        return bind_node_evaluator(op, attrs)(input_vals, namespace, None)
    if op.startswith("custom."):
        return evaluate_primitive(op, input_vals, attrs, namespace=namespace)
    backend_name = attrs.get("_advect_backend")
    if isinstance(backend_name, str) and backend_name:
        backend_evaluate_op = get_hook(f"{backend_name}.evaluate_op")
        if backend_evaluate_op is not None:
            _validate_backend_namespace(op, backend_name, namespace)
            decoder = get_hook(f"{backend_name}.decode_attrs")
            decoded_attrs = decoder(op, attrs) if decoder is not None else attrs
            return backend_evaluate_op(op, input_vals, decoded_attrs)
    if op.startswith(("array.", "array_ext.")):
        runtime_namespace = (
            namespace if namespace is not None else _namespace_from_inputs(input_vals)
        )
        return _evaluate_array_op(op, input_vals, attrs, runtime_namespace)
    backend_evaluate_op, decode_attrs = resolve_backend_hooks(op, input_vals)
    decoded_attrs = decode_attrs(op, attrs) if decode_attrs is not None else attrs
    return backend_evaluate_op(op, input_vals, decoded_attrs)


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
        namespace = context if context is not None else _namespace_from_inputs(input_vals)
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


def _namespace_from_inputs(inputs: tuple[Any, ...]) -> Any | None:
    for value in inputs:
        namespace = _get_array_namespace(value)
        if namespace is not None:
            return namespace
    return None


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


def _namespace_function(namespace: Any, path: str) -> Callable[..., Any]:
    target = namespace
    if getattr(namespace, "__name__", "") == "numpy":
        aliases = {"absolute": "abs"}
    else:
        aliases = {
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
        if path in {"cross", "diagonal", "outer", "trace", "vecdot"}:
            path = f"linalg.{path}"
    parts = path.split(".")
    parts[-1] = aliases.get(parts[-1], parts[-1])
    for part in parts:
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"Array namespace member {path!r} is not callable")
    return cast("Callable[..., Any]", target)


def _evaluate_array_op(  # noqa: PLR0912, PLR0915 - one explicit portable execution schema
    op: str,
    inputs: tuple[Any, ...],
    attrs: Mapping[str, Any],
    namespace: Any | None,
) -> Any:
    path = op.removeprefix("array_ext.").removeprefix("array.")
    leaf_name = path.rsplit(".", 1)[-1]
    if (
        leaf_name in _BINARY_OPERATORS
        and len(inputs) == 2
        and (
            all(type(value) in {bool, complex, float, int} for value in inputs)
            or any(callable(getattr(value, "_advect_snapshot", None)) for value in inputs)
        )
    ):
        return _BINARY_OPERATORS[leaf_name](*inputs)
    scalar_unary = _SCALAR_UNARY_OPERATORS.get(leaf_name)
    if (
        scalar_unary is not None
        and len(inputs) == 1
        and type(inputs[0])
        in {
            bool,
            complex,
            float,
            int,
        }
    ):
        return scalar_unary(inputs[0])
    if namespace is None:
        raise RuntimeError(f"Cannot execute {op!r} without an array namespace")
    array_api_version = _selected_array_api_version(namespace, attrs)
    if getattr(namespace, "__name__", "") != "numpy":
        inputs = cast(
            "tuple[Any, ...]",
            materialize_weak_scalar_operands(
                op,
                inputs,
                namespace=namespace,
            ),
        )
    kwargs = {key: value for key, value in attrs.items() if not key.startswith("_advect_")}
    if (
        leaf_name in {"cumprod", "cumsum", "prod", "sum"}
        and inputs
        and kwargs.get("dtype") is None
        and array_api_version is not None
    ):
        target_dtype = accumulation_dtype(
            inputs[0].dtype,
            array_api_version=array_api_version,
        )
        if target_dtype != dtype_name(inputs[0].dtype):
            kwargs["dtype"] = target_dtype
    device_key = attrs.get("_advect_device")
    if isinstance(device_key, str):
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
        device = next(
            (candidate for candidate in candidates if str(candidate) == device_key),
            None,
        )
        if device is None:
            raise ValueError(f"Array API device {device_key!r} is unavailable at execution")
        kwargs["device"] = device
    if kwargs.get("dtype") is not None:
        dtype = kwargs["dtype"]
        kwargs["dtype"] = getattr(namespace, str(dtype), dtype)
    if leaf_name in {"reshape", "broadcast_to"}:
        return _namespace_function(namespace, path)(inputs[0], kwargs.pop("shape"), **kwargs)
    if leaf_name == "clip":
        values = iter(inputs[1:])
        lower = next(values) if bool(attrs.get("_advect_clip_min_is_input", False)) else None
        upper = next(values) if bool(attrs.get("_advect_clip_max_is_input", False)) else None
        return _namespace_function(namespace, path)(inputs[0], min=lower, max=upper, **kwargs)
    if leaf_name == "pinv":
        tolerance = attrs.get("_advect_pinv_tolerance")
        if tolerance is not None:
            if tolerance not in {"rcond", "rtol"} or len(inputs) != 2:
                raise ValueError("pinv tolerance metadata does not match its operands")
            kwargs[str(tolerance)] = inputs[1]
        return _namespace_function(namespace, path)(inputs[0], **kwargs)
    if leaf_name in {"diagonal", "trace"} and getattr(namespace, "__name__", "") != "numpy":
        rank = len(inputs[0].shape)
        first_axis = normalize_axis(kwargs.pop("axis1", 0), rank)
        second_axis = normalize_axis(kwargs.pop("axis2", 1), rank)
        axes = (
            *(axis for axis in range(rank) if axis not in {first_axis, second_axis}),
            first_axis,
            second_axis,
        )
        if axes != tuple(range(rank)):
            inputs = (_namespace_function(namespace, "permute_dims")(inputs[0], axes),)
    if leaf_name in {"empty", "ones", "zeros"}:
        return _namespace_function(namespace, path)(kwargs.pop("shape"), **kwargs)
    if leaf_name == "eye":
        n_rows = kwargs.pop("n_rows")
        n_cols = kwargs.pop("n_cols", None)
        return _namespace_function(namespace, path)(n_rows, n_cols, **kwargs)
    if leaf_name == "arange":
        start = kwargs.pop("start")
        stop = kwargs.pop("stop", None)
        step = kwargs.pop("step", 1)
        return _namespace_function(namespace, path)(start, stop, step, **kwargs)
    if leaf_name == "linspace":
        start = kwargs.pop("start")
        stop = kwargs.pop("stop")
        num = kwargs.pop("num")
        return _namespace_function(namespace, path)(start, stop, num, **kwargs)
    if leaf_name in {"fftfreq", "rfftfreq"}:
        n = kwargs.pop("n")
        if getattr(namespace, "__name__", "") == "numpy":
            dtype = kwargs.pop("dtype")
            result = _namespace_function(namespace, path)(n, **kwargs)
            return result.astype(dtype, copy=False)
        return _namespace_function(namespace, path)(n, **kwargs)
    if leaf_name in {"argsort", "sort"} and getattr(namespace, "__name__", "") == "numpy":
        descending = bool(kwargs.pop("descending", False))
        if descending:
            raise NotImplementedError(
                "Portable staged descending sort is not supported on NumPy; "
                "use an Array API provider or sort ascending."
            )
        return _namespace_function(namespace, path)(*inputs, **kwargs)
    if leaf_name == "moveaxis":
        return _namespace_function(namespace, path)(
            inputs[0],
            kwargs.pop("source"),
            kwargs.pop("destination"),
            **kwargs,
        )
    if leaf_name == "repeat":
        return _namespace_function(namespace, path)(
            inputs[0],
            kwargs.pop("repeats"),
            **kwargs,
        )
    if leaf_name == "tile":
        return _namespace_function(namespace, path)(
            inputs[0],
            kwargs.pop("reps"),
            **kwargs,
        )
    if leaf_name == "full":
        return _namespace_function(namespace, path)(
            kwargs.pop("shape"),
            inputs[0],
            **kwargs,
        )
    if leaf_name in {"permute_dims", "transpose"}:
        axes = kwargs.pop("axes", None)
        function = _namespace_function(namespace, path)
        return (
            function(inputs[0], **kwargs) if axes is None else function(inputs[0], axes, **kwargs)
        )
    if leaf_name == "astype":
        dtype = kwargs.pop("dtype", None)
        if bool(attrs.get("_advect_array_api_asarray", False)):
            if dtype is not None:
                kwargs["dtype"] = dtype
            return _namespace_function(namespace, "asarray")(inputs[0], **kwargs)
        if dtype is None:
            raise TypeError("astype requires a dtype")
        scalar_type = getattr(namespace, "generic", None)
        if type(inputs[0]) in {bool, complex, float, int} or (
            isinstance(scalar_type, type) and isinstance(inputs[0], scalar_type)
        ):
            return _namespace_function(namespace, "asarray")(inputs[0], dtype=dtype)
        return _namespace_function(namespace, path)(inputs[0], dtype, **kwargs)
    if leaf_name in {"concat", "concatenate", "stack"}:
        return _namespace_function(namespace, path)(inputs, **kwargs)
    return _namespace_function(namespace, path)(*inputs, **kwargs)


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
    "evaluate_node_value",
    "has_core_evaluator",
]
