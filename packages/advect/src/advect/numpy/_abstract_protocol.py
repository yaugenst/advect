"""NumPy protocol lowering for payload-free staged arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, override

import numpy as np

from advect.core._abstract import (
    AbstractArray,
    _alias_result,
    _lift,
    _record_abstract_op,
    _traces,
)
from advect.core._errors import TracingError
from advect.numpy._abstract_calls import _empty_out, _numpy_array, apply_numpy, can_cast_dtype
from advect.numpy._array_function.composite import _map_tree
from advect.numpy._array_function.registry import (
    _STATIC_ARRAY_FUNCTIONS,
    ARRAY_FUNCTION_HANDLERS,
)
from advect.numpy._array_function.runtime import _LIKE_DISPATCH_CONSTRUCTORS
from advect.numpy._composite_lowering import lower_ufunc_method
from advect.numpy._constructors import _normalize_order, construct_abstract
from advect.numpy._signature import (
    keyword_optional_positionals,
    normalize_required_positionals,
    positional_parameters,
)
from advect.numpy._stage_lifecycle import stage_context
from advect.numpy._traced_array import _SEMANTIC_ALIAS_FUNCTIONS, squares_bool_array
from advect.numpy._traced_array_indexing import normalize_integer_scalars

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from contextlib import AbstractContextManager

    from advect.core._native import DynamicTape
    from advect.core._protocols import ArrayLike


class _NamedProtocol(Protocol):
    __name__: str


# NumPy 2.0 rounds booleans and integers in a floating loop; later releases
# keep their dtype, as the canonical rounding operations do.
_ROUNDING_UFUNCS = frozenset({"ceil", "floor", "trunc"})


def _in_rounding_loop(ufunc: np.ufunc, value: object) -> object:
    """Cast an integral operand into the loop the installed NumPy selects."""
    if not isinstance(value, AbstractArray):
        return value
    dtype = np.dtype(value.spec.dtype)
    if dtype.kind not in "biu":
        return value
    loop = ufunc.resolve_dtypes((dtype, None))[0]
    return value if loop == dtype else value.astype(loop)


def abstract_array_ufunc(
    self: AbstractArray,
    ufunc: _NamedProtocol,
    method: str,
    *inputs: object,
    **kwargs: object,
) -> AbstractArray:
    """Lower one NumPy ufunc call into the staged canonical graph."""
    if method != "__call__":
        return lower_ufunc_method(ufunc, method, inputs, kwargs)
    if ufunc.__name__ in _ROUNDING_UFUNCS and len(inputs) == 1 and not kwargs:
        inputs = (_in_rounding_loop(cast("np.ufunc", ufunc), inputs[0]),)
    if not _empty_out(kwargs.get("out")):
        kwargs["_advect_ufunc_call"] = True
    return _numpy_array(self._trace, ufunc.__name__, inputs, kwargs)


def abstract_array_function(
    self: AbstractArray,
    func: _NamedProtocol,
    types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 - NumPy's protocol returns heterogeneous pytrees
    """Lower one NumPy array-function call into the staged canonical graph."""
    del types
    module = str(getattr(func, "__module__", "numpy"))
    name = func.__name__
    # NumPy 2.0 and 2.1 define scimath in numpy.lib._scimath_impl.
    if getattr(np.lib.scimath, name, None) is func:
        raise TracingError(
            f"numpy.lib.scimath.{name} is dynamic-only because its output dtype "
            "can depend on runtime values"
        )
    if module.startswith("numpy.linalg"):
        name = f"linalg.{name}"
    elif module.startswith("numpy.fft"):
        name = f"fft.{name}"
    positional = positional_parameters(func)
    args, kwargs = normalize_required_positionals(positional, args, kwargs, func=func)
    args, kwargs = keyword_optional_positionals(positional, args, kwargs)
    if module.startswith("numpy") and name in {"array", "asarray", "asanyarray"}:
        return construct_abstract(name, self, args, kwargs)
    if func not in ARRAY_FUNCTION_HANDLERS and func not in _STATIC_ARRAY_FUNCTIONS:
        raise TracingError(f"Array function 'numpy.{name}' is not supported during staging")
    if name == "copy":
        if set(kwargs) - {"order", "subok"}:
            raise TracingError("numpy.copy expects (a, order='K', subok=False) during staging")
        if bool(kwargs.get("subok", False)):
            raise TracingError(
                "numpy.copy(subok=True) is not supported during staging because "
                "durable programs do not preserve ndarray subclass identity"
            )
        source = cast("_NumpyAbstractArray", _lift(self._trace, args[0]))
        return source.copy(order=str(kwargs.get("order", "K")))
    result = apply_numpy(self._trace, name, args, kwargs)
    if name in _SEMANTIC_ALIAS_FUNCTIONS and isinstance(args[0], AbstractArray):
        # NumPy returns a view, so writes and stale reads fail as they do dynamically.
        return _alias_result(args[0], cast("AbstractArray", result))
    return result


class _NumpyAbstractArray(AbstractArray):
    """Payload-free staged value that owns NumPy's foreign protocols."""

    __slots__ = ()

    @staticmethod
    @override
    def _advect_stage_context(
        _captures: Sequence[tuple[str, object]],
    ) -> AbstractContextManager[None]:
        return stage_context(_captures)

    @override
    def __getitem__(self, index: object) -> AbstractArray:
        return super().__getitem__(normalize_integer_scalars(index))

    @override
    def __setitem__(self, index: object, value: object) -> None:
        super().__setitem__(normalize_integer_scalars(index), value)

    @property
    @override
    def real(self) -> AbstractArray:
        # NumPy's components are views, so writes and stale reads fail as for other views.
        return _alias_result(self, super().real)

    @property
    @override
    def imag(self) -> AbstractArray:
        return _alias_result(self, super().imag)

    @override
    def _assignment_value(self, value: AbstractArray, shape: tuple[int, ...]) -> AbstractArray:
        # NumPy drops a value's leading unit dimensions beyond the target rank.
        extra = value.ndim - len(shape)
        if extra > 0 and all(size == 1 for size in value.shape[:extra]):
            return value.reshape(value.shape[extra:])
        return value

    @override
    def _augmented_result(self, result: AbstractArray) -> AbstractArray:
        # NumPy evaluates `a op= b` as `op(a, b, out=a)`, which casts the
        # promoted result back to a's dtype under same_kind, whatever the
        # operand is. The cast is a neutral astype, so any provider replays it.
        if result.spec.dtype == self.spec.dtype or not can_cast_dtype(
            result.spec.dtype, self.spec.dtype, casting="same_kind"
        ):
            return result
        return result.astype(self.spec.dtype)

    @override
    def astype(self, dtype: object, **kwargs: object) -> AbstractArray:
        if any(name in kwargs for name in ("casting", "order", "subok")):
            return cast(
                "AbstractArray",
                apply_numpy(self._trace, "astype", (self, dtype), kwargs),
            )
        return super().astype(dtype, **kwargs)

    @override
    def copy(self, order: str | None = None) -> AbstractArray:
        if order is None:
            return super().copy()
        return cast(
            "AbstractArray",
            _record_abstract_op(
                self._trace,
                "advect.copy",
                (self,),
                {"order": _normalize_order(order, default="C")},
                graph_attrs={"_advect_backend": "numpy"},
            ),
        )

    @override
    def _squares(self, exponent: object) -> bool:
        return squares_bool_array(self, exponent)

    def __pow__(self, other: object) -> AbstractArray:
        if squares_bool_array(self, other):
            return cast("AbstractArray", np.square(self))
        return cast("Any", super()).__pow__(other)

    # The methods bind their arguments exactly as NumPy's functions do.
    @override
    def sum(self, *args: object, **kwargs: object) -> AbstractArray:
        return cast("Any", np.sum)(self, *args, **kwargs)

    @override
    def mean(self, *args: object, **kwargs: object) -> AbstractArray:
        return cast("Any", np.mean)(self, *args, **kwargs)

    def __array_ufunc__(
        self,
        ufunc: _NamedProtocol,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> AbstractArray:
        if _wraps_nested(self, inputs) or (kwargs and _wraps_nested(self, kwargs.values())):
            return NotImplemented
        return abstract_array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def __array_function__(
        self,
        func: _NamedProtocol,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401 - NumPy's protocol returns heterogeneous pytrees
        if _wraps_nested(self, args) or _wraps_nested(self, kwargs.values()):
            return NotImplemented
        return abstract_array_function(self, func, types, args, kwargs)


def _wraps_nested(self: AbstractArray, values: Iterable[object]) -> bool:
    """Return whether a nested dynamic trace's tracer of this trace's values is in ``values``.

    That tracer's frontend records the call in its own trace, as its reflected
    operator does, instead of this trace reading it as a constant.
    """
    for value in values:
        if isinstance(value, (tuple, list)):
            if _wraps_nested(self, value):
                return True
        elif not isinstance(value, AbstractArray) and _traces(value, self._trace):
            return True
    return False


def as_numpy_nested(value: object) -> Any:  # noqa: ANN401 - optional protocol conversion
    """Wrap an Array API tracer whose nested payload is an abstract staged value."""
    snapshot = getattr(value, "_advect_snapshot", None)
    if not callable(snapshot):
        return NotImplemented
    node_id, wrapped = cast("tuple[int, object]", cast("Any", snapshot)())
    # A staged value is its own payload: the enclosing stage's, not a nested tracer.
    if wrapped is value or not bool(getattr(type(wrapped), "__advect_abstract_array__", False)):
        return NotImplemented
    recorder = getattr(value, "recorder", None)
    if recorder is None:
        return NotImplemented
    from advect.numpy._traced_array import TracedArray  # noqa: PLC0415 - avoid init cycle

    return TracedArray(
        cast("ArrayLike", wrapped),
        node_id,
        cast("DynamicTape", recorder),
    )


def _as_numpy_nested_tree(tree: object) -> tuple[Any, bool]:
    """Wrap each nested Array API tracer leaf; report whether any was wrapped."""
    changed = False

    def convert(value: object) -> object:
        nonlocal changed
        nested = as_numpy_nested(value)
        if nested is NotImplemented:
            return value
        changed = True
        return nested

    return _map_tree(convert, tree), changed


def nested_array_ufunc(
    _tracer: object,
    ufunc: _NamedProtocol,
    method: str,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 - NumPy's protocol returns heterogeneous pytrees
    """Let NumPy bind a call encountered inside an Array API nested trace."""
    (converted_inputs, converted_kwargs), changed = _as_numpy_nested_tree((inputs, kwargs))
    if not changed:
        return NotImplemented
    call = cast("Callable[..., object]", getattr(ufunc, method))
    return call(*converted_inputs, **converted_kwargs)


def nested_array_function(
    _tracer: object,
    function: _NamedProtocol,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 - NumPy's protocol returns heterogeneous pytrees
    """Let NumPy bind an array function inside an Array API nested trace."""
    if getattr(function, "__name__", "") in _LIKE_DISPATCH_CONSTRUCTORS and "like" not in kwargs:
        # NumPy consumed like= to dispatch a constructor here; rebinding it
        # must dispatch again rather than convert its traced operands.
        kwargs = {**kwargs, "like": _tracer}
    (converted_args, converted_kwargs), changed = _as_numpy_nested_tree((args, kwargs))
    if not changed:
        return NotImplemented
    return cast("Callable[..., object]", function)(*converted_args, **converted_kwargs)


__all__ = [
    "abstract_array_function",
    "abstract_array_ufunc",
    "as_numpy_nested",
    "nested_array_function",
    "nested_array_ufunc",
]
