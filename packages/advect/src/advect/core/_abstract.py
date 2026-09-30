# ruff: noqa: SLF001
"""Payload-free arrays for explicit, conservative abstract staging.

Only operations declared in :mod:`advect.core._abstract_domains` are stageable.
Each has a stable primitive ID, an explicit operand schema, and a domain-local
abstract result rule. Unknown operations fail instead of guessing a result.
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NoReturn, cast

from advect.core._abstract_helpers import (
    ABSTRACT_NAMESPACE_NAME,
    DTYPE_NAMES as _DTYPE_NAMES,
    PYTHON_SCALAR_TYPES as _PYTHON_SCALAR_TYPES,
    _staged_dtype,
    broadcast_shape as _broadcast_shape,
    can_cast_dtype as _can_cast_dtype,
    coerced_dtype as _coerced_dtype,
    discovered_dtype as _discovered_dtype,
    dtype_kind_bits as _dtype_kind_bits,
    dtype_name as _dtype_name,
    promote_dtype as _promote_dtype,
    safely_casts as _safely_casts,
    shape_tuple as _shape_tuple,
    value_spec as _value_spec,
)
from advect.core._abstract_model import AbstractValue, ArraySpec
from advect.core._array_api.frontend import (
    _ARITHMETIC_OPERATORS,
    _COMPARISON_OPERATORS,
    _FUNCTION_SPECS,
    _INTERNAL_FUNCTION_SPECS,
    _STAGED_ARRAY_API_COMPOSITES,
    ArrayAPINamespace,
    _staged_array_api_composite,
    bind_array_api_call,
    lower_array_api_call,
)
from advect.core._array_api.profiles import materialize_array_api_profile
from advect.core._array_api.results import restore_array_api_result
from advect.core._array_protocol_helpers import (
    PYTHON_OPERATOR_ATTR,
    python_scalar_operands,
    select_item,
)
from advect.core._basic_index import encode_basic_index, normalize_basic_index
from advect.core._context import (
    _peek_pending_update,
    _set_pending_update,
    _take_pending_update,
    get_source_location,
)
from advect.core._errors import (
    EscapedTracerError,
    MutationError,
    StaleViewError,
    TracingError,
    _array_conversion_error,
)
from advect.core._graph_attrs import encode_graph_attrs_for_native
from advect.core._protocols import _innermost
from advect.core._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from contextlib import AbstractContextManager

    from advect.core._array_api.profiles import ArrayAPIProfile
    from advect.core._native import GraphBuilder

_ASARRAY_LITERAL_TYPES = (tuple, list, *_PYTHON_SCALAR_TYPES)


@dataclass(frozen=True, slots=True)
class _StrongScalarConstant:
    """Array-shaped rank-zero constant with an explicit staged dtype."""

    value: object
    dtype: object
    shape: tuple[()] = ()

    def __getitem__(self, index: object) -> object:
        if index != ():
            raise IndexError(index)
        return self.value


@dataclass(frozen=True, slots=True)
class _Finfo:
    bits: int
    dtype: object
    eps: float
    max: float
    min: float
    smallest_normal: float


@dataclass(frozen=True, slots=True)
class _Iinfo:
    bits: int
    dtype: object
    max: int
    min: int


@dataclass(slots=True)
class _AbstractCell:
    """Mutable source wrapper state pointing at the current immutable SSA value."""

    node_id: int
    spec: ArraySpec
    owned: bool
    layout: str | None = None
    epoch: int = 0


@dataclass(frozen=True, slots=True)
class _AbstractView:
    """Conservative tracer-only alias relationship."""

    root: _AbstractCell
    epoch: int
    index: object | None


@dataclass(frozen=True, slots=True)
class _PendingIndexUpdate:
    """Acknowledgement for Python's getitem/iadd/setitem protocol."""

    destination: _AbstractCell
    epoch: int
    index: object
    replacement: AbstractArray

    @property
    def complete_without_setitem(self) -> bool:
        """Mark an already-applied update whose generated setitem is optional."""
        return True


def _provider_name(namespace: object) -> str:
    return str(getattr(namespace, "__name__", type(namespace).__name__))


class AbstractTrace:
    """Construction state shared by all wrappers in one abstract trace."""

    __slots__ = (
        "add_constant",
        "array_api_profile",
        "array_api_version",
        "array_factory",
        "builder",
        "dtype_namespace",
        "dtype_objects",
        "open",
    )

    def __init__(
        self,
        builder: GraphBuilder,
        *,
        array_api_version: str,
        add_constant: Callable[[Any, ArraySpec], int],
        array_factory: type[AbstractArray],
        dtype_namespace: object,
    ) -> None:
        self.builder = builder
        self.array_api_version = array_api_version
        self.array_api_profile: ArrayAPIProfile = materialize_array_api_profile(array_api_version)
        self.add_constant = add_constant
        self.array_factory = array_factory
        if isinstance(dtype_namespace, ArrayAPINamespace):
            # An enclosing dynamic transform proxies the examples' provider.
            dtype_namespace = dtype_namespace.raw_namespace
        if isinstance(dtype_namespace, AbstractNamespace):
            # stage() inside another stage() trace presents the enclosing
            # provider; that trace's namespace would record each empty() probe.
            dtype_namespace = dtype_namespace._trace.dtype_namespace
        self.dtype_namespace = dtype_namespace
        self.dtype_objects: dict[str, object] = {}
        self.open = True

    def dtype_object(self, name: str) -> object:
        """Return the provider dtype object that user code sees for a staged dtype.

        Staged specs hold canonical names. The provider that staging resolved
        for this trace presents them, so ``x.dtype`` equals what its arrays report.
        """
        dtype = self.dtype_objects.get(name)
        if dtype is None:
            namespace = cast("Any", self.dtype_namespace)
            provider_dtype = getattr(namespace, name, None)
            if provider_dtype is None:
                raise TypeError(
                    f"Array provider {_provider_name(namespace)!r} has no {name} dtype "
                    "to present for this staged value"
                )
            dtype = namespace.empty((0,), dtype=provider_dtype).dtype
            self.dtype_objects[name] = dtype
        return dtype

    def require_open(self) -> None:
        if not self.open:
            raise EscapedTracerError("An abstract tracer escaped the stage() trace that created it")


def _leaves(value: object) -> Iterator[object]:
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _sequence_dtype(value: object) -> str:
    """Return the dtype NumPy's array coercion discovers for *value*."""
    return _coerced_dtype(map(_discovered_dtype, _leaves(value)))


def _sequence_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    child_shapes = tuple(_sequence_shape(child) for child in value)
    if child_shapes and any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise ValueError("asarray() requires a rectangular nested sequence")
    return (len(value), *(child_shapes[0] if child_shapes else ()))


def _array_api_op(path: str) -> str:
    function_spec = _FUNCTION_SPECS.get(path) or _INTERNAL_FUNCTION_SPECS.get(path)
    if function_spec is not None:
        definition = get_registry().get_optional(function_spec.op)
        if definition is not None and definition.abstract_schema is not None:
            return function_spec.op
    raise NotImplementedError(
        f"Array API function {path!r} has no abstract staging rule. "
        "Define it as an Advect primitive with def_abstract()."
    )


class AbstractNamespace:
    """Array API namespace bound to one abstract trace."""

    __slots__ = ("_prefix", "_trace")

    def __init__(self, trace: AbstractTrace, *, prefix: str = "") -> None:
        self._trace = trace
        self._prefix = prefix

    @property
    def __name__(self) -> str:
        return ABSTRACT_NAMESPACE_NAME

    @property
    def __array_api_version__(self) -> str:
        return self._trace.array_api_version

    @property
    def linalg(self) -> AbstractNamespace:
        return AbstractNamespace(self._trace, prefix="linalg.")

    @property
    def fft(self) -> AbstractNamespace | Callable[..., AbstractArray]:
        if self._prefix:
            return cast("Callable[..., AbstractArray]", self.__getattr__("fft"))
        return AbstractNamespace(self._trace, prefix="fft.")

    def __array_namespace_info__(self) -> AbstractNamespace:
        """Identify this invocation-local namespace as Array API compatible."""
        return self

    def result_type(self, *values: object) -> object:
        """Evaluate dtype promotion as abstract compile-time metadata."""
        if not values:
            raise TypeError("result_type() requires at least one argument")
        specs: list[ArraySpec] = []
        for value in values:
            if isinstance(value, AbstractArray):
                specs.append(value.spec)
                continue
            dtype = getattr(value, "dtype", None)
            shape = getattr(value, "shape", None)
            if dtype is not None and shape is not None:
                specs.append(ArraySpec(tuple(int(size) for size in shape), dtype))
                continue
            normalized_dtype = _dtype_name(value)
            if normalized_dtype in _DTYPE_NAMES:
                specs.append(ArraySpec((), normalized_dtype))
                continue
            specs.append(_value_spec(value))
        return self._trace.dtype_object(_promote_dtype(specs))

    def isdtype(self, dtype: object, kind: object) -> bool:
        """Evaluate standard dtype-category queries at staging time."""
        if isinstance(kind, tuple):
            return any(self.isdtype(dtype, item) for item in kind)
        dtype_name = _dtype_name(dtype)
        if not isinstance(kind, str):
            return dtype_name == _dtype_name(kind)
        dtype_kind, _bits = _dtype_kind_bits(dtype_name)
        categories = {
            "bool": {"bool"},
            "complex floating": {"complex"},
            "integral": {"int", "uint"},
            "numeric": {"complex", "float", "int", "uint"},
            "real floating": {"float"},
            "signed integer": {"int"},
            "unsigned integer": {"uint"},
        }
        accepted = categories.get(kind)
        return dtype_name == kind if accepted is None else dtype_kind in accepted

    def can_cast(self, from_: object, to: object) -> bool:
        """Evaluate the profile's lossless dtype-cast relation at staging time."""
        source = (
            from_.spec.dtype if isinstance(from_, AbstractArray) else getattr(from_, "dtype", from_)
        )
        source_name = _dtype_name(source)
        target_name = _dtype_name(to)
        if source_name not in _DTYPE_NAMES or target_name not in _DTYPE_NAMES:
            raise TypeError(f"Unsupported dtype pair for can_cast(): {source!r}, {to!r}")
        return _can_cast_dtype(source_name, target_name)

    def finfo(self, type_: object) -> _Finfo:
        """Return deterministic floating-point metadata during staging."""
        dtype = (
            type_.spec.dtype if isinstance(type_, AbstractArray) else getattr(type_, "dtype", type_)
        )
        dtype_name = _dtype_name(dtype)
        real_dtype = {
            "complex64": "float32",
            "complex128": "float64",
        }.get(dtype_name, dtype_name)
        if real_dtype == "float32":
            return _Finfo(
                bits=32,
                dtype=self._trace.dtype_object("float32"),
                eps=2.0**-23,
                max=float.fromhex("0x1.fffffep+127"),
                min=-float.fromhex("0x1.fffffep+127"),
                smallest_normal=2.0**-126,
            )
        if real_dtype == "float64":
            return _Finfo(
                bits=64,
                dtype=self._trace.dtype_object("float64"),
                eps=2.0**-52,
                max=float.fromhex("0x1.fffffffffffffp+1023"),
                min=-float.fromhex("0x1.fffffffffffffp+1023"),
                smallest_normal=2.0**-1022,
            )
        raise TypeError(f"finfo() requires a floating-point dtype, got {dtype!r}")

    def iinfo(self, type_: object) -> _Iinfo:
        """Return deterministic integer metadata during staging."""
        dtype = (
            type_.spec.dtype if isinstance(type_, AbstractArray) else getattr(type_, "dtype", type_)
        )
        dtype_name = _dtype_name(dtype)
        kind, bits = _dtype_kind_bits(dtype_name)
        if kind == "int":
            return _Iinfo(
                bits=bits,
                dtype=self._trace.dtype_object(dtype_name),
                max=(1 << (bits - 1)) - 1,
                min=-(1 << (bits - 1)),
            )
        if kind == "uint":
            return _Iinfo(
                bits=bits,
                dtype=self._trace.dtype_object(dtype_name),
                max=(1 << bits) - 1,
                min=0,
            )
        raise TypeError(f"iinfo() requires an integer dtype, got {dtype!r}")

    def _advect_materialize_constant(self, value: object, spec: ArraySpec) -> AbstractArray:
        """Lift a closed staged constant without converting it through Python."""
        return _constant(self._trace, value, spec)

    def __getattr__(self, name: str) -> object:
        if not self._prefix and name in _DTYPE_NAMES:
            namespace = self._trace.dtype_namespace
            if not hasattr(namespace, name):
                message = f"Array provider {_provider_name(namespace)!r} has no {name} dtype"
                raise AttributeError(message)
            return self._trace.dtype_object(name)
        path = f"{self._prefix}{name}"
        if not self._trace.array_api_profile.admits(path) and path not in _INTERNAL_FUNCTION_SPECS:
            message = (
                f"Array API function {path!r} is not available in the selected "
                f"{self._trace.array_api_version} revision"
            )
            raise AttributeError(message)
        if path in _STAGED_ARRAY_API_COMPOSITES:

            def composite_operation(*args: object, **kwargs: object) -> object:
                root = AbstractNamespace(self._trace)
                return _staged_array_api_composite(path, root, args, kwargs)

            composite_operation.__name__ = name
            return composite_operation

        _array_api_op(path)

        def operation(
            *args: object,
            **kwargs: object,
        ) -> AbstractArray | tuple[AbstractArray, ...]:
            return cast(
                "AbstractArray | tuple[AbstractArray, ...]",
                _apply_array_api(self._trace, path, args, kwargs),
            )

        operation.__name__ = name
        return operation


class AbstractArray:
    """An array-shaped SSA value with no readable payload."""

    __slots__ = ("_cell", "_trace", "_view")
    __array_priority__ = 100_000
    __advect_abstract_array__ = True
    __advect_namespace_is_instance_specific__ = True

    def __init__(
        self,
        trace: AbstractTrace,
        node_id: int,
        spec: ArraySpec,
        *,
        owned: bool = True,
        view: _AbstractView | None = None,
        layout: str | None = None,
    ) -> None:
        self._trace = trace
        self._cell = _AbstractCell(node_id, spec, owned, layout)
        self._view = view

    @staticmethod
    def _advect_stage_context(
        _captures: Sequence[tuple[str, object]],
    ) -> AbstractContextManager[None]:
        """Return the selected frontend's abstract-staging lifecycle scope."""
        return nullcontext()

    def _require(self, *, allow_pending: bool = False) -> None:
        self._trace.require_open()
        pending = _peek_pending_update(self._trace.builder)
        if pending is not None and not allow_pending:
            if bool(getattr(pending, "complete_without_setitem", False)):
                _take_pending_update(self._trace.builder)
            else:
                message = getattr(pending, "unconsumed_message", None)
                if not isinstance(message, str):
                    message = "A staged indexed augmented assignment was not completed"
                raise TracingError(message)
        if self._view is not None and self._view.root.epoch != self._view.epoch:
            raise StaleViewError(
                "A staged view was used after its base changed. Copy the view or reorder "
                "the base update."
            )

    def _require_mutable(self, operation: str) -> None:
        self._require()
        if self._view is not None:
            raise MutationError(
                f"Cannot perform {operation} through a staged view. Update the base with "
                "one basic index expression or call `.copy()` first."
            )
        if not self._cell.owned:
            raise MutationError(
                f"Cannot perform {operation} on a staged input or captured value. "
                "Call `.copy()` before mutating it."
            )

    def _augmented_result(self, result: AbstractArray) -> AbstractArray:
        """Return the value that an augmented assignment stores in this array.

        Portable augmented assignment never changes dtype, so the promoted
        result is kept and a dtype change is rejected. A frontend whose
        in-place operators cast the result back to the destination overrides
        this.
        """
        return result

    def _assignment_value(self, value: AbstractArray, shape: tuple[int, ...]) -> AbstractArray:
        """Return the value that a view assignment broadcasts into ``shape``.

        Portable assignment broadcasts the value as given. A frontend whose
        view assignment first drops extra leading unit dimensions overrides
        this. An index that selects one element stores the value as given.
        """
        del shape
        return value

    def _squares(self, exponent: object) -> bool:
        """Return whether Python's ``self ** exponent`` squares this array.

        Portable ``**`` raises to a power. A frontend whose power operator
        squares instead, with another result dtype, overrides this.
        """
        del exponent
        return False

    def _commit(self, replacement: AbstractArray) -> None:
        replacement._require()
        self._cell.node_id = replacement.node_id
        self._cell.spec = replacement.spec
        self._cell.layout = replacement._cell.layout
        self._cell.epoch += 1

    def advect_require_mutable(self, operation: str) -> None:
        """Backend-neutral protocol hook used before functional ``out=``."""
        self._require_mutable(operation)

    def advect_replace(
        self,
        *,
        value: object,
        node_id: int,
        operation: str,
    ) -> None:
        """Backend-neutral protocol hook for committing a functional write."""
        self._require_mutable(operation)
        if not isinstance(value, AbstractArray):
            raise TypeError("A staged functional replacement must be an AbstractArray")
        if value._trace is not self._trace:
            raise TracingError("A staged functional replacement belongs to another trace")
        if value.node_id != node_id:
            raise TracingError("A staged functional replacement node does not match its value")
        self._commit(value)

    def _root_cell(self) -> _AbstractCell:
        return self._view.root if self._view is not None else self._cell

    @property
    def recorder(self) -> GraphBuilder:
        """Return the owning recorder for nested transform dispatch."""
        return self._trace.builder

    def _advect_snapshot(self) -> tuple[int, AbstractArray]:
        """Return this outer trace value as the payload of a nested trace."""
        self._require()
        return self._cell.node_id, self

    def _advect_snapshot_in_active_trace(self) -> tuple[int, AbstractArray]:
        self._require()
        return self._cell.node_id, self

    def _advect_scalar_cotangent(self) -> AbstractArray:
        """Create a typed scalar seed without retaining the primal output."""
        self._require()
        if self.shape != ():
            raise TypeError("A scalar cotangent seed requires a rank-zero value")
        spec = ArraySpec((), self.spec.dtype, device=self.device)
        return _constant(self._trace, _StrongScalarConstant(1.0, self.spec.dtype), spec)

    @property
    def _advect_weak(self) -> bool:
        """Return the serialized weak-scalar category of this SSA value."""
        return self.spec.weak

    @property
    def _advect_layout(self) -> str | None:
        """Return a layout guarantee known from staged allocation semantics."""
        self._require()
        return self._cell.layout

    def _advect_mark_weak(self) -> None:
        """Mark this rank-zero SSA value as weak inside an enclosing stage."""
        self._require()
        if self.shape != ():
            raise ValueError("Only rank-zero abstract values can be weak scalars")
        self._cell.spec = replace(self._cell.spec, weak=True)

    @property
    def node_id(self) -> int:
        self._require()
        return self._cell.node_id

    @property
    def spec(self) -> ArraySpec:
        self._require()
        return self._cell.spec

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> object:
        return self._trace.dtype_object(self.spec.dtype)

    @property
    def device(self) -> str | None:
        return self.spec.device

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def real(self) -> AbstractArray:
        return _apply_array(self._trace, "real", (self,), {})

    @property
    def imag(self) -> AbstractArray:
        return _apply_array(self._trace, "imag", (self,), {})

    @property
    def T(self) -> AbstractArray:  # noqa: N802 - NumPy spelling
        return _apply_array(self._trace, "permute_dims", (self,), {"axes": None})

    def __array_namespace__(self, *, api_version: str | None = None) -> AbstractNamespace:
        selected = self._trace.array_api_version
        if api_version not in (None, selected):
            message = (
                f"Array API version {api_version!r} requested, but this staged trace "
                f"targets {selected!r}"
            )
            raise ValueError(message)
        self._require()
        return AbstractNamespace(self._trace)

    def __array__(
        self,
        dtype: object | None = None,
        copy: bool | None = None,  # noqa: FBT001 - NumPy protocol signature
    ) -> NoReturn:
        del dtype, copy
        raise TracingError(_array_conversion_error())

    def __bool__(self) -> NoReturn:
        raise TracingError(
            "Python control flow cannot depend on an abstract staged value; use where() "
            "or an explicit array control-flow primitive"
        )

    def __iter__(self) -> NoReturn:
        raise TracingError("Iteration over an abstract staged array is data-dependent")

    def __len__(self) -> NoReturn:
        raise TracingError("len() on an abstract staged array is not allowed; use x.shape")

    def __getitem__(self, index: object) -> AbstractArray:
        return _apply_getitem(self, index)

    def __setitem__(self, index: object, value: object) -> None:
        self._require(allow_pending=True)
        encoded, target_spec = _basic_index_spec(self, index, allow_pending=True)
        pending = _peek_pending_update(self._trace.builder)
        if pending is not None:
            if not isinstance(pending, _PendingIndexUpdate):
                _take_pending_update(self._trace.builder)
                raise MutationError(
                    "A pending staged indexed update was redirected to the wrong assignment"
                )
            if value is not pending.replacement:
                _take_pending_update(self._trace.builder)
                pending = None
            else:
                _take_pending_update(self._trace.builder)
        if pending is not None:
            if self._view is not None:
                raise MutationError(
                    "Nested staged subscript mutation is unsupported. Rewrite "
                    "`field[i][j] += value` as `field[i, j] += value`."
                )
            if (
                pending.destination is not self._cell
                or pending.epoch != self._cell.epoch
                or pending.index != encoded
            ):
                raise MutationError(
                    "The pending staged update does not match this base, index, or epoch"
                )
            return
        if isinstance(value, _PendingIndexUpdate):
            raise MutationError("This staged indexed-update token has expired")
        replacement = _lift(self._trace, value)
        if not _selects_element(encoded, target_spec.shape):
            replacement = self._assignment_value(replacement, target_spec.shape)

        self._require_mutable("item assignment")
        if _broadcast_shape(replacement.shape, target_spec.shape) != target_spec.shape:
            raise ValueError(
                f"Cannot assign shape {replacement.shape!r} into indexed shape "
                f"{target_spec.shape!r}"
            )
        updated = _emit(
            self._trace,
            "advect.index_update",
            (self, replacement),
            {"index": encoded},
            self.spec,
        )
        self._commit(updated)

    def astype(self, dtype: object, **kwargs: object) -> AbstractArray:
        return cast(
            "AbstractArray",
            _apply_array_api(self._trace, "astype", (self, dtype), kwargs),
        )

    def copy(self) -> AbstractArray:
        return cast(
            "AbstractArray",
            _record_abstract_op(self._trace, "advect.copy", (self,), {}),
        )

    def reshape(self, *shape: object, **kwargs: object) -> AbstractArray:
        if not shape:
            raise TypeError("reshape() missing required argument 'shape'")
        target = shape[0] if len(shape) == 1 else shape
        return cast(
            "AbstractArray",
            _apply_array_api(self._trace, "reshape", (self, target), kwargs),
        )

    def item(self, *args: object) -> AbstractArray:
        """Represent scalar extraction without requiring a concrete payload."""
        return cast("AbstractArray", select_item(self, args))

    def sum(self, *args: object, **kwargs: object) -> AbstractArray:
        return cast(
            "AbstractArray",
            _apply_array_api(self._trace, "sum", (self, *args), kwargs),
        )

    def mean(self, *args: object, **kwargs: object) -> AbstractArray:
        return cast(
            "AbstractArray",
            _apply_array_api(self._trace, "mean", (self, *args), kwargs),
        )


def _new_abstract_array(
    trace: AbstractTrace,
    node_id: int,
    spec: ArraySpec,
    *,
    owned: bool = True,
    view: _AbstractView | None = None,
    layout: str | None = None,
) -> AbstractArray:
    """Construct the concrete abstract tracer selected for this trace."""
    if type(spec.dtype) is not str or spec.dtype not in _DTYPE_NAMES:
        spec = replace(spec, dtype=_staged_dtype(spec.dtype))
    value = trace.array_factory(
        trace,
        node_id,
        spec,
        owned=owned,
        view=view,
        layout=layout,
    )
    if not isinstance(value, AbstractArray):
        raise TypeError("The abstract-array factory must return an AbstractArray")
    return value


def _constant(trace: AbstractTrace, value: object, spec: ArraySpec) -> AbstractArray:
    """Lift one closed constant as a borrowed value of this trace."""
    return _new_abstract_array(trace, trace.add_constant(value, spec), spec, owned=False)


def _traces(value: object, trace: AbstractTrace) -> bool:
    """Return whether a tracer wraps a value of ``trace``."""
    payload = _innermost(value)
    return isinstance(payload, AbstractArray) and payload._trace is trace


def _binary_method(
    name: str,
    *,
    reverse: bool = False,
) -> Callable[[AbstractArray, object], AbstractArray]:
    def method(self: AbstractArray, other: object) -> AbstractArray:
        # A dynamic tracer that wraps this trace's values records the operation
        # through its own reflected operator rather than becoming a constant.
        if not isinstance(other, AbstractArray) and _traces(other, self._trace):
            return NotImplemented
        args = (other, self) if reverse else (self, other)
        return _apply_array(self._trace, name, args, {}, by_operator=True)

    return method


for _dunder, _operation in _ARITHMETIC_OPERATORS.items():
    setattr(AbstractArray, f"__{_dunder}__", _binary_method(_operation))
    setattr(AbstractArray, f"__r{_dunder}__", _binary_method(_operation, reverse=True))
for _dunder, _operation in _COMPARISON_OPERATORS.items():
    setattr(AbstractArray, f"__{_dunder}__", _binary_method(_operation))


def _unary_method(name: str) -> Callable[[AbstractArray], AbstractArray]:
    def method(self: AbstractArray) -> AbstractArray:
        return _apply_array(self._trace, name, (self,), {}, by_operator=True)

    return method


def _augmented_replacement(self: AbstractArray, name: str, other: object) -> AbstractArray:
    replacement = self._augmented_result(_apply_array(self._trace, name, (self, other), {}))
    if replacement.shape != self.shape or replacement.spec.dtype != self.spec.dtype:
        raise MutationError(
            f"Augmented {name} would change shape or dtype from {self.spec!r} to "
            f"{replacement.spec!r}"
        )
    return replacement


def _inplace_method(name: str) -> Callable[[AbstractArray, object], object]:
    def method(self: AbstractArray, other: object) -> object:
        if self.spec.weak:
            # A Python scalar has no in-place operator; Python rebinds the name.
            return NotImplemented
        view = self._view
        if view is not None:
            if not view.root.owned:
                raise MutationError(
                    "Cannot mutate a staged input through an indexed view. Call `.copy()` "
                    "on the base before the indexed assignment."
                )
            if view.index is None:
                raise MutationError(
                    "Mutation through a nested or reshaped staged view is unsupported. "
                    "Use one basic index on the base or call `.copy()` first."
                )
            replacement = _augmented_replacement(self, name, other)
            root = _new_abstract_array(
                self._trace,
                view.root.node_id,
                view.root.spec,
                owned=view.root.owned,
            )
            updated = _emit(
                self._trace,
                "advect.index_update",
                (root, replacement),
                {"index": view.index},
                view.root.spec,
            )
            view.root.node_id = updated.node_id
            view.root.spec = updated.spec
            view.root.epoch += 1
            self._cell.node_id = replacement.node_id
            self._cell.spec = replacement.spec
            self._view = _AbstractView(
                root=view.root,
                epoch=view.root.epoch,
                index=view.index,
            )
            pending = _PendingIndexUpdate(
                destination=view.root,
                epoch=view.root.epoch,
                index=view.index,
                replacement=self,
            )
            _set_pending_update(self._trace.builder, pending)
            return self

        self._require_mutable(f"augmented {name}")
        replacement = _augmented_replacement(self, name, other)
        self._commit(replacement)
        return self

    return method


AbstractArray.__neg__ = _unary_method("negative")
AbstractArray.__pos__ = _unary_method("positive")
AbstractArray.__abs__ = _unary_method("abs")
AbstractArray.__invert__ = _unary_method("bitwise_invert")
for _dunder, _operation in _ARITHMETIC_OPERATORS.items():
    setattr(AbstractArray, f"__i{_dunder}__", _inplace_method(_operation))


def _lift(trace: AbstractTrace, value: object) -> AbstractArray:
    if isinstance(value, AbstractArray):
        value._require()
        if value._trace is not trace:
            raise TracingError("Cannot mix values from different abstract traces")
        return value
    if isinstance(value, (tuple, list)):
        return _assemble(trace, value, _sequence_dtype(value))
    return _constant(trace, value, _value_spec(value))


def _encodes(leaf: object, dtype: str) -> bool:
    """Whether a sequence constant converts *leaf* to *dtype* as NumPy's coercion does.

    Coercion converts a Python scalar to the target dtype, which the constant
    encoder reproduces. It casts a NumPy scalar or array like ``astype``, which
    the encoder reproduces for a rank-zero value that a safe cast admits or that
    lies in the target's range.
    """
    if type(leaf) in _PYTHON_SCALAR_TYPES:
        return True
    if isinstance(leaf, AbstractArray):
        return False
    shape, source = getattr(leaf, "shape", None), getattr(leaf, "dtype", None)
    if shape is None or source is None:
        # A Python scalar subclass; the encoder rejects any other object.
        return True
    return not shape and (_safely_casts(source, dtype) or _encodes_in_range(leaf, source, dtype))


# Largest finite magnitude of each floating width the constant encoder packs.
_FLOAT_MAX = {16: 65504.0, 32: (2 - 2**-23) * 2.0**127, 64: sys.float_info.max}
# Integers of at most this magnitude convert to float64 exactly.
_EXACT_FLOAT64_INTEGER = 2**53
_FLOAT64_BITS = 64


def _fits_float(value: float, bits: int) -> bool:
    return not math.isfinite(value) or abs(value) <= _FLOAT_MAX[bits]


def _encodes_in_range(leaf: object, source: object, dtype: str) -> bool:
    """Whether the encoder converts a rank-zero NumPy leaf to *dtype* as ``astype`` does.

    The encoder raises on overflow and wrap-around where ``astype`` wraps or
    saturates, but converts an in-range value identically.
    """
    kind, bits = _dtype_kind_bits(dtype)
    source_kind = _dtype_kind_bits(source)[0]
    convert = complex if source_kind == "complex" else float if source_kind == "float" else int
    value: Any = convert(cast("Any", leaf))
    if kind == "bool":
        return True
    if source_kind == "complex":
        width = bits // 2
        return (
            kind == "complex" and _fits_float(value.real, width) and _fits_float(value.imag, width)
        )
    if kind in {"int", "uint"}:
        if isinstance(value, float) and not math.isfinite(value):
            return False
        low = -(2 ** (bits - 1)) if kind == "int" else 0
        high = 2 ** (bits - 1) - 1 if kind == "int" else 2**bits - 1
        return low <= int(value) <= high
    width = bits if kind == "float" else bits // 2
    if (
        source_kind in {"int", "uint"}
        and width < _FLOAT64_BITS
        and abs(value) > _EXACT_FLOAT64_INTEGER
    ):
        # The encoder would round through float64 first, rounding twice.
        return False
    return _fits_float(float(value), width)


def _assemble(trace: AbstractTrace, value: object, dtype: str) -> AbstractArray:
    """Build *value* with every leaf cast to *dtype*, as NumPy's coercion does."""
    if isinstance(value, (tuple, list)):
        if all(_encodes(leaf, dtype) for leaf in _leaves(value)):
            return _constant(trace, value, ArraySpec(_sequence_shape(value), dtype))
        children = tuple(_assemble(trace, item, dtype) for item in value)
        return cast("AbstractArray", _apply_array_api(trace, "stack", (children,), {"axis": 0}))
    if type(value) in _PYTHON_SCALAR_TYPES:
        return _constant(trace, _StrongScalarConstant(value, dtype), ArraySpec((), dtype))
    leaf = _lift(trace, value)
    if leaf.spec.dtype == dtype:
        return leaf
    return cast("AbstractArray", _apply_array_api(trace, "astype", (leaf, dtype), {}))


def _emitted_layout(
    op: str,
    inputs: Sequence[AbstractArray],
    attrs: Mapping[str, Any],
) -> str | None:
    leaf = op.rsplit(".", 1)[-1]
    layout: str | None = None
    if leaf in {"empty", "eye", "full", "linspace", "ones", "zeros"}:
        order = str(attrs.get("order", "C"))
        layout = order if order in {"C", "F"} else "C"
    elif leaf in {"empty_like", "full_like", "ones_like", "zeros_like"}:
        order = str(attrs.get("order", "K"))
        layout = order if order in {"C", "F"} else inputs[0]._cell.layout if inputs else None
    elif leaf == "astype" and inputs:
        order = str(attrs.get("order", "K"))
        layout = order if order in {"C", "F"} else inputs[0]._cell.layout
    elif op == "advect.copy" and inputs and "order" in attrs:
        order = str(attrs.get("order", "K"))
        if order in {"C", "F"}:
            layout = order
        elif order == "A":
            layout = "F" if inputs[0]._cell.layout == "F" else "C"
        else:
            layout = inputs[0]._cell.layout
    return layout


def _emit(
    trace: AbstractTrace,
    op: str,
    inputs: Sequence[AbstractArray],
    attrs: Mapping[str, Any],
    spec: ArraySpec,
) -> AbstractArray:
    trace.require_open()
    closed_attrs = dict(attrs)
    node_id = _append_node(
        trace,
        op=op,
        inputs=tuple(value.node_id for value in inputs),
        attrs=closed_attrs,
        shape=spec.shape,
        dtype=spec.dtype,
    )
    return _new_abstract_array(
        trace,
        node_id,
        spec,
        owned=True,
        layout=_emitted_layout(op, inputs, attrs),
    )


def _emit_outputs(
    trace: AbstractTrace,
    op: str,
    inputs: Sequence[AbstractArray],
    attrs: Mapping[str, Any],
    specs: tuple[ArraySpec, ...],
) -> tuple[AbstractArray, ...]:
    """Emit one fixed-arity parent and explicit projections for every result."""
    trace.require_open()
    closed_attrs = dict(attrs)
    parent_id = _append_node(
        trace,
        op=op,
        inputs=tuple(value.node_id for value in inputs),
        attrs=closed_attrs,
        shape=specs[0].shape,
        dtype=specs[0].dtype,
        num_outputs=len(specs),
        output_shapes=tuple(spec.shape for spec in specs),
        output_dtypes=tuple(spec.dtype for spec in specs),
    )
    outputs: list[AbstractArray] = []
    for index, spec in enumerate(specs):
        output_id = _append_node(
            trace,
            op="advect.getoutput",
            inputs=(parent_id,),
            attrs={"index": index, "num_outputs": len(specs)},
            shape=spec.shape,
            dtype=spec.dtype,
        )
        outputs.append(_new_abstract_array(trace, output_id, spec, owned=True))
    return tuple(outputs)


def _append_node(  # noqa: PLR0913 - mirrors the native node schema
    trace: AbstractTrace,
    *,
    op: str,
    inputs: Sequence[int],
    attrs: Mapping[str, Any],
    shape: tuple[int, ...],
    dtype: object,
    num_outputs: int = 1,
    output_shapes: Sequence[tuple[int, ...]] | None = None,
    output_dtypes: Sequence[Any] | None = None,
) -> int:
    """Append one validated instruction directly to the native stage builder."""
    registry = get_registry()
    op_def = registry.get_optional(op)
    if op_def is None:
        raise ValueError(
            f"Op '{op}' is not registered. Import its frontend or "
            "define it as an Advect primitive before staging."
        )
    if op_def.num_outputs != num_outputs:
        raise ValueError(
            f"Op '{op}' expects num_outputs={op_def.num_outputs}, got num_outputs={num_outputs}"
        )
    return trace.builder.append_node(
        op,
        inputs,
        encode_graph_attrs_for_native(attrs),
        shape,
        dtype,
        schema_version=op_def.schema_version,
        num_outputs=num_outputs,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        source_location=get_source_location(),
    )


def _result_specs(
    op: str,
    specs: Sequence[ArraySpec],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    """Return the fixed output contract for one abstract operation."""
    definition = get_registry().get(op)
    evaluator = definition.abstract_evaluator
    if evaluator is None:
        raise AssertionError(f"Operation {op!r} has no abstract evaluator")
    return evaluator(specs, attrs)


def _record_abstract_op(
    trace: AbstractTrace,
    op: str,
    raw_operands: Sequence[object],
    raw_attrs: Mapping[str, object],
    *,
    abstract_attrs: Mapping[str, object] | None = None,
    graph_attrs: Mapping[str, object] | None = None,
    by_operator: bool = False,
) -> AbstractArray | tuple[AbstractArray, ...]:
    """Record one canonical operation after a frontend has bound its call.

    *by_operator* says that a Python operator spelled the call, which on weak
    operands computes as Python does and keeps a weak result.
    """
    trace.require_open()
    operands = tuple(_lift(trace, value) for value in raw_operands)
    attrs = dict(raw_attrs)
    definition = get_registry().get(op)
    rule = definition.abstract_schema
    if rule is None:
        raise AssertionError(f"Operation {op!r} has no abstract schema")

    public_attrs = {name for name in attrs if not name.startswith("_advect_")}
    unexpected = public_attrs - rule.allowed_attrs
    if unexpected:
        raise TypeError(
            f"Abstract staging of {op} does not support attributes {tuple(sorted(unexpected))!r}"
        )
    # A None value selects a provider default, which staging cannot know.
    missing = rule.required_attrs - {name for name in public_attrs if attrs[name] is not None}
    if missing:
        raise TypeError(f"Abstract staging of {op} requires {tuple(sorted(missing))!r}")

    for name in ("shape", "axes"):
        if name in attrs and attrs[name] is not None:
            if name == "axes" and rule.kind == "tensordot":
                continue
            attrs[name] = _shape_tuple(attrs[name])
    if attrs.get("dtype") is not None:
        attrs["dtype"] = _dtype_name(attrs["dtype"])

    evaluation_attrs = attrs if abstract_attrs is None else {**attrs, **abstract_attrs}
    emitted_attrs = attrs if graph_attrs is None else {**attrs, **graph_attrs}
    operand_specs = [operand.spec for operand in operands]
    python_operands = python_scalar_operands(op, operand_specs, by_operator=by_operator)
    specs = _result_specs(
        op,
        operand_specs if python_operands is None else python_operands,
        evaluation_attrs,
    )
    if len(specs) > 1:
        return _emit_outputs(trace, op, operands, emitted_attrs, specs)
    spec = specs[0]
    if python_operands is not None:
        # A Python-scalar result promotes weakly against the arrays it meets,
        # and replay computes it with Python's operator again.
        spec = replace(spec, weak=True)
        emitted_attrs = {**emitted_attrs, PYTHON_OPERATOR_ATTR: True}
    result = _emit(trace, op, operands, emitted_attrs, spec)
    if rule.kind in {
        "broadcast_to",
        "diagonal",
        "expand_dims",
        "moveaxis",
        "reshape",
        "squeeze",
        "transpose",
    }:
        return _alias_result(operands[0], result)
    return result


def _apply_array(
    trace: AbstractTrace,
    raw_name: str,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
    *,
    by_operator: bool = False,
) -> AbstractArray:
    """Apply one single-output provider-neutral Array API operation."""
    result = _apply_array_api(trace, raw_name, raw_args, raw_kwargs, by_operator=by_operator)
    if not isinstance(result, AbstractArray):
        msg = f"Single-output abstract operation {raw_name!r} returned metadata or a tuple"
        raise TypeError(msg)
    return result


def _alias_result(
    source: AbstractArray,
    result: AbstractArray,
    *,
    index: object | None = None,
) -> AbstractArray:
    root = source._root_cell()
    return _new_abstract_array(
        result._trace,
        result.node_id,
        result.spec,
        owned=False,
        view=_AbstractView(root=root, epoch=root.epoch, index=index),
        layout=source._cell.layout,
    )


def _array_api_asarray(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    """Bind the Array API constructor without retaining nested tracer payloads."""
    args = list(raw_args)
    kwargs = dict(raw_kwargs)
    if not args:
        raise TypeError("asarray() requires an input")
    raw_value = args.pop(0)
    if args:
        kwargs.setdefault("dtype", args.pop(0))
    if args or set(kwargs) - {"copy", "device", "dtype"}:
        raise TypeError("Abstract staging supports only asarray(obj, dtype=, device=, copy=)")
    copy = kwargs.get("copy")
    if copy is not None and type(copy) is not bool:
        raise TypeError("asarray copy must be a bool or None")
    dtype = None if kwargs.get("dtype") is None else _dtype_name(kwargs["dtype"])

    if isinstance(raw_value, _ASARRAY_LITERAL_TYPES) and copy is False:
        raise ValueError(
            "asarray(copy=False) cannot construct an array from a sequence or Python scalar"
        )
    value = (
        _assemble(trace, raw_value, dtype or _sequence_dtype(raw_value))
        if isinstance(raw_value, _ASARRAY_LITERAL_TYPES)
        else _lift(trace, raw_value)
    )
    target_dtype = value.spec.dtype if dtype is None else dtype
    device = kwargs.get("device")
    target_device = value.device if device is None else str(device)
    if copy is False and (
        target_dtype != value.spec.dtype
        or (value.device is not None and target_device != value.device)
    ):
        raise ValueError("asarray(copy=False) cannot satisfy the requested dtype or device")
    attrs: dict[str, object] = {
        "_advect_array_api_asarray": True,
        "copy": copy,
        "dtype": target_dtype,
    }
    if device is not None:
        attrs["_advect_device"] = target_device
    result = cast(
        "AbstractArray",
        _record_abstract_op(
            trace,
            "array.astype",
            (value,),
            attrs,
            abstract_attrs={"_advect_array_api_version": trace.array_api_version},
        ),
    )
    unchanged = target_dtype == value.spec.dtype and target_device == value.device
    return _alias_result(value, result) if copy is not True and unchanged else result


def _apply_array_api(
    trace: AbstractTrace,
    path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    by_operator: bool = False,
) -> Any:  # noqa: ANN401 - Array API calls may return structured results
    """Bind one provider-neutral call and record its canonical operation."""
    trace.require_open()
    if path == "asarray":
        return _array_api_asarray(trace, args, kwargs)
    lowered = lower_array_api_call(path, AbstractNamespace(trace), args, kwargs)
    if lowered is not NotImplemented:
        return lowered

    binding = bind_array_api_call(path, args, kwargs)
    result = _record_abstract_op(
        trace,
        binding.op,
        binding.operands,
        binding.attrs,
        abstract_attrs={"_advect_array_api_version": trace.array_api_version},
        by_operator=by_operator,
    )
    if not isinstance(result, tuple):
        return result
    return restore_array_api_result(path, result)


def _basic_index_spec(
    value: AbstractArray,
    index: object,
    *,
    allow_pending: bool = False,
) -> tuple[list[dict[str, object]], ArraySpec]:
    value._require(allow_pending=allow_pending)
    items = normalize_basic_index(index)
    encoded = encode_basic_index(items)
    if sum(item is Ellipsis for item in items) > 1:
        raise IndexError("Only one ellipsis is allowed")
    consumed = sum(item is not None and item is not Ellipsis for item in items)
    source_spec = value._cell.spec
    source_rank = len(source_spec.shape)
    if consumed > source_rank:
        raise IndexError(f"Too many indices for an array with rank {source_rank}")
    expanded: list[object] = []
    for item in items:
        if item is Ellipsis:
            expanded.extend(slice(None) for _ in range(source_rank - consumed))
        else:
            expanded.append(item)
    if Ellipsis not in items:
        expanded.extend(slice(None) for _ in range(source_rank - consumed))

    shape: list[int] = []
    source_axis = 0
    for item in expanded:
        if item is None:
            shape.append(1)
            continue
        size = source_spec.shape[source_axis]
        source_axis += 1
        if isinstance(item, int):
            if not -size <= item < size:
                raise IndexError(f"Index {item} is out of bounds for axis of size {size}")
            continue
        start, stop, step = cast("slice", item).indices(size)
        shape.append(len(range(start, stop, step)))
    return encoded, ArraySpec(tuple(shape), source_spec.dtype)


def _selects_element(encoded: list[dict[str, object]], shape: tuple[int, ...]) -> bool:
    """Return whether an encoded basic index selects one element instead of a view."""
    return not shape and all(item["type"] == "int" for item in encoded)


def _apply_getitem(value: AbstractArray, index: object) -> AbstractArray:
    encoded, result_spec = _basic_index_spec(value, index)
    result = _emit(value._trace, "advect.getitem", (value,), {"index": encoded}, result_spec)
    if _selects_element(encoded, result_spec.shape):
        return result
    return _alias_result(value, result, index=encoded if value._view is None else None)


__all__ = [
    "AbstractArray",
    "AbstractNamespace",
    "AbstractTrace",
    "AbstractValue",
    "ArraySpec",
]
