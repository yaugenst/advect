# ruff: noqa: PLR2004, SLF001
"""Payload-free arrays for explicit, conservative abstract staging.

Only operations declared in :mod:`advect.core._abstract_domains` are stageable.
Each has a stable primitive ID, an explicit operand schema, and a domain-local
abstract result rule. Unknown operations fail instead of guessing a result.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NoReturn, cast

from advect.core._abstract_helpers import (
    broadcast_shape as _broadcast_shape,
    dtype_kind_bits as _dtype_kind_bits,
    dtype_name as _dtype_name,
    normalize_axis as _normalize_axis,
    promote_dtype as _promote_dtype,
    shape_tuple as _shape_tuple,
)
from advect.core._abstract_model import AbstractValue, ArraySpec
from advect.core._array_api.frontend import (
    _FUNCTION_SPECS,
    _INTERNAL_FUNCTION_SPECS,
    _STAGED_ARRAY_API_COMPOSITES,
    _staged_array_api_composite,
    bind_array_api_call,
)
from advect.core._array_api.profiles import materialize_array_api_profile
from advect.core._array_api.results import restore_array_api_result
from advect.core._array_protocol_helpers import normalize_item_index
from advect.core._basic_index import encode_basic_index
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
from advect.core._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager

    from advect.core._array_api.profiles import ArrayAPIProfile
    from advect.core._native import GraphBuilder


_DTYPE_NAMES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }
)
_SAFE_CAST_TARGETS: dict[str, frozenset[str]] = {
    "bool": frozenset({"bool"}),
    "int8": frozenset({"int8", "int16", "int32", "int64"}),
    "int16": frozenset({"int16", "int32", "int64"}),
    "int32": frozenset({"int32", "int64"}),
    "int64": frozenset({"int64"}),
    "uint8": frozenset({"uint8", "uint16", "uint32", "uint64", "int16", "int32", "int64"}),
    "uint16": frozenset({"uint16", "uint32", "uint64", "int32", "int64"}),
    "uint32": frozenset({"uint32", "uint64", "int64"}),
    "uint64": frozenset({"uint64"}),
    "float16": frozenset({"float16", "float32", "float64", "complex64", "complex128"}),
    "float32": frozenset({"float32", "float64", "complex64", "complex128"}),
    "float64": frozenset({"float64", "complex128"}),
    "complex64": frozenset({"complex64", "complex128"}),
    "complex128": frozenset({"complex128"}),
}


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
    dtype: str
    eps: float
    max: float
    min: float
    smallest_normal: float


@dataclass(frozen=True, slots=True)
class _Iinfo:
    bits: int
    dtype: str
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


class AbstractTrace:
    """Construction state shared by all wrappers in one abstract trace."""

    __slots__ = (
        "add_constant",
        "array_api_profile",
        "array_api_version",
        "array_factory",
        "builder",
        "open",
        "profile",
    )

    def __init__(
        self,
        builder: GraphBuilder,
        *,
        profile: str,
        array_api_version: str,
        add_constant: Callable[[Any, ArraySpec], int],
        array_factory: type[AbstractArray],
    ) -> None:
        self.builder = builder
        self.profile = profile
        self.array_api_version = array_api_version
        self.array_api_profile: ArrayAPIProfile = materialize_array_api_profile(array_api_version)
        self.add_constant = add_constant
        self.array_factory = array_factory
        self.open = True

    def require_open(self) -> None:
        if not self.open:
            raise EscapedTracerError("An abstract tracer escaped the stage() trace that created it")


def _scalar_spec(value: object) -> ArraySpec:
    if isinstance(value, bool):
        return ArraySpec((), "bool", weak=True)
    if isinstance(value, complex):
        return ArraySpec((), "complex128", weak=True)
    if isinstance(value, float):
        return ArraySpec((), "float64", weak=True)
    if isinstance(value, int):
        return ArraySpec((), "int64", weak=True)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        raise TypeError(f"Cannot stage concrete operand of type {type(value).__name__}")
    return ArraySpec(tuple(int(size) for size in shape), dtype)


def _nested_sequence_spec(value: object, dtype: object | None) -> ArraySpec:
    leaf_specs: list[ArraySpec] = []

    def visit(item: object) -> tuple[int, ...]:
        if not isinstance(item, (tuple, list)):
            leaf_specs.append(_scalar_spec(item))
            return ()
        child_shapes = tuple(visit(child) for child in item)
        if child_shapes and any(shape != child_shapes[0] for shape in child_shapes[1:]):
            raise ValueError("asarray() requires a rectangular nested sequence")
        return (len(item), *(child_shapes[0] if child_shapes else ()))

    shape = visit(value)
    if dtype is None:
        dtype = "float64" if not leaf_specs else _promote_dtype(leaf_specs)
    return ArraySpec(shape, _dtype_name(dtype))


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
        return "advect.array_api"

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

    def result_type(self, *values: object) -> str:
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
            specs.append(_scalar_spec(value))
        return _promote_dtype(specs)

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
        source = from_.dtype if isinstance(from_, AbstractArray) else getattr(from_, "dtype", from_)
        source_name = _dtype_name(source)
        target_name = _dtype_name(to)
        targets = _SAFE_CAST_TARGETS.get(source_name)
        if targets is None or target_name not in _DTYPE_NAMES:
            raise TypeError(f"Unsupported dtype pair for can_cast(): {source!r}, {to!r}")
        return target_name in targets

    def finfo(self, type_: object) -> _Finfo:
        """Return deterministic floating-point metadata during staging."""
        dtype = type_.dtype if isinstance(type_, AbstractArray) else getattr(type_, "dtype", type_)
        dtype_name = _dtype_name(dtype)
        real_dtype = {
            "complex64": "float32",
            "complex128": "float64",
        }.get(dtype_name, dtype_name)
        if real_dtype == "float32":
            return _Finfo(
                bits=32,
                dtype="float32",
                eps=2.0**-23,
                max=float.fromhex("0x1.fffffep+127"),
                min=-float.fromhex("0x1.fffffep+127"),
                smallest_normal=2.0**-126,
            )
        if real_dtype == "float64":
            return _Finfo(
                bits=64,
                dtype="float64",
                eps=2.0**-52,
                max=float.fromhex("0x1.fffffffffffffp+1023"),
                min=-float.fromhex("0x1.fffffffffffffp+1023"),
                smallest_normal=2.0**-1022,
            )
        raise TypeError(f"finfo() requires a floating-point dtype, got {dtype!r}")

    def iinfo(self, type_: object) -> _Iinfo:
        """Return deterministic integer metadata during staging."""
        dtype = type_.dtype if isinstance(type_, AbstractArray) else getattr(type_, "dtype", type_)
        dtype_name = _dtype_name(dtype)
        kind, bits = _dtype_kind_bits(dtype_name)
        if kind == "int":
            return _Iinfo(
                bits=bits,
                dtype=dtype_name,
                max=(1 << (bits - 1)) - 1,
                min=-(1 << (bits - 1)),
            )
        if kind == "uint":
            return _Iinfo(
                bits=bits,
                dtype=dtype_name,
                max=(1 << bits) - 1,
                min=0,
            )
        raise TypeError(f"iinfo() requires an integer dtype, got {dtype!r}")

    def _advect_materialize_constant(self, value: object, spec: ArraySpec) -> AbstractArray:
        """Lift a closed staged constant without converting it through Python."""
        node_id = self._trace.add_constant(value, spec)
        return _new_abstract_array(self._trace, node_id, spec, owned=False)

    def __getattr__(self, name: str) -> str | Callable[..., Any]:
        if not self._prefix and name in _DTYPE_NAMES:
            return name
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
        spec = ArraySpec((), self.dtype, device=self.device)
        node_id = self._trace.add_constant(
            _StrongScalarConstant(1.0, self.dtype),
            spec,
        )
        return _new_abstract_array(self._trace, node_id, spec, owned=False)

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
        return self.spec.dtype

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
        index = normalize_item_index(args, ndim=self.ndim)
        if index is None:
            if self.size != 1:
                raise ValueError("can only convert an array of size 1 to a scalar")
            if self.shape == ():
                return self
            return self[tuple(0 for _dimension in self.shape)]
        if isinstance(index, tuple):
            return self[index]
        return self.reshape((-1,))[index]

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


def _binary_method(
    name: str,
    *,
    reverse: bool = False,
) -> Callable[[AbstractArray, object], AbstractArray]:
    def method(self: AbstractArray, other: object) -> AbstractArray:
        args = (other, self) if reverse else (self, other)
        return _apply_array(self._trace, name, args, {})

    return method


for _dunder, _operation in {
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "truediv": "divide",
    "floordiv": "floor_divide",
    "mod": "remainder",
    "pow": "pow",
    "matmul": "matmul",
    "and": "bitwise_and",
    "or": "bitwise_or",
    "xor": "bitwise_xor",
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
    "eq": "equal",
    "ne": "not_equal",
}.items():
    setattr(AbstractArray, f"__{_dunder}__", _binary_method(_operation))
    setattr(AbstractArray, f"__r{_dunder}__", _binary_method(_operation, reverse=True))


def _unary_method(name: str) -> Callable[[AbstractArray], AbstractArray]:
    def method(self: AbstractArray) -> AbstractArray:
        return _apply_array(self._trace, name, (self,), {})

    return method


def _inplace_method(name: str) -> Callable[[AbstractArray, object], object]:
    def method(self: AbstractArray, other: object) -> object:
        self._require()
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
            replacement = _apply_array(self._trace, name, (self, other), {})
            if replacement.shape != self.shape or _dtype_name(replacement.dtype) != _dtype_name(
                self.dtype
            ):
                raise MutationError(
                    f"Augmented {name} would change shape or dtype from {self.spec!r} to "
                    f"{replacement.spec!r}"
                )
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
        replacement = _apply_array(self._trace, name, (self, other), {})
        if replacement.shape != self.shape or _dtype_name(replacement.dtype) != _dtype_name(
            self.dtype
        ):
            raise MutationError(
                f"Augmented {name} would change shape or dtype from {self.spec!r} to "
                f"{replacement.spec!r}"
            )
        self._commit(replacement)
        return self

    return method


AbstractArray.__neg__ = _unary_method("negative")
AbstractArray.__pos__ = _unary_method("positive")
AbstractArray.__abs__ = _unary_method("absolute")
for _dunder in (
    ("iadd", "add"),
    ("isub", "subtract"),
    ("imul", "multiply"),
    ("itruediv", "divide"),
    ("ifloordiv", "floor_divide"),
    ("imod", "remainder"),
    ("ipow", "pow"),
    ("imatmul", "matmul"),
    ("iand", "bitwise_and"),
    ("ior", "bitwise_or"),
    ("ixor", "bitwise_xor"),
):
    setattr(AbstractArray, f"__{_dunder[0]}__", _inplace_method(_dunder[1]))


def _lift(trace: AbstractTrace, value: object) -> AbstractArray:
    if isinstance(value, AbstractArray):
        value._require()
        if value._trace is not trace:
            raise TracingError("Cannot mix values from different abstract traces")
        return value
    spec = (
        _nested_sequence_spec(value, None)
        if isinstance(value, (tuple, list))
        else _scalar_spec(value)
    )
    return _new_abstract_array(
        trace,
        trace.add_constant(value, spec),
        spec,
        owned=False,
    )


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
    values: Sequence[AbstractArray],
    attrs: Mapping[str, Any],
) -> tuple[ArraySpec, ...]:
    """Return the fixed output contract for one abstract operation."""
    definition = get_registry().get(op)
    evaluator = definition.abstract_evaluator
    if evaluator is None:
        raise AssertionError(f"Operation {op!r} has no abstract evaluator")
    return evaluator([value.spec for value in values], attrs)


def _record_abstract_op(
    trace: AbstractTrace,
    op: str,
    raw_operands: Sequence[object],
    raw_attrs: Mapping[str, object],
    *,
    abstract_attrs: Mapping[str, object] | None = None,
    graph_attrs: Mapping[str, object] | None = None,
) -> AbstractArray | tuple[AbstractArray, ...]:
    """Record one canonical operation after a frontend has bound its call."""
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
    missing = rule.required_attrs - public_attrs
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
    specs = _result_specs(op, operands, evaluation_attrs)
    emitted_attrs = attrs if graph_attrs is None else {**attrs, **graph_attrs}
    if len(specs) > 1:
        return _emit_outputs(trace, op, operands, emitted_attrs, specs)
    result = _emit(trace, op, operands, emitted_attrs, specs[0])
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
) -> AbstractArray:
    """Apply one single-output provider-neutral Array API operation."""
    result = _apply_array_api(trace, raw_name, raw_args, raw_kwargs)
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


def _array_api_asarray(  # noqa: C901 - one constructor contract
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
    dtype = kwargs.get("dtype")

    def contains_abstract(value: object) -> bool:
        if isinstance(value, AbstractArray):
            return True
        if isinstance(value, (tuple, list)):
            return any(contains_abstract(item) for item in value)
        return False

    def assemble(value: object) -> AbstractArray:
        if isinstance(value, AbstractArray):
            return value
        if not isinstance(value, (tuple, list)):
            return _lift(trace, value)
        if not value:
            spec = _nested_sequence_spec(value, dtype)
            return _new_abstract_array(
                trace,
                trace.add_constant(value, spec),
                spec,
                owned=False,
            )
        children = tuple(assemble(item) for item in value)
        return cast(
            "AbstractArray",
            _apply_array_api(trace, "stack", (children,), {"axis": 0}),
        )

    if isinstance(raw_value, (tuple, list)):
        if copy is False:
            raise ValueError("asarray(copy=False) cannot construct an array from a sequence")
        if contains_abstract(raw_value):
            raw_value = assemble(raw_value)
        else:
            spec = _nested_sequence_spec(raw_value, dtype)
            raw_value = _new_abstract_array(
                trace,
                trace.add_constant(raw_value, spec),
                spec,
                owned=False,
            )
    if dtype is not None and isinstance(raw_value, (bool, int, float, complex)):
        spec = ArraySpec((), _dtype_name(dtype))
        raw_value = _new_abstract_array(
            trace,
            trace.add_constant(_StrongScalarConstant(raw_value, spec.dtype), spec),
            spec,
            owned=False,
        )
    value = _lift(trace, raw_value)
    target_dtype = value.dtype if dtype is None else _dtype_name(dtype)
    device = kwargs.get("device")
    target_device = value.device if device is None else str(device)
    if copy is False and (
        _dtype_name(target_dtype) != _dtype_name(value.dtype)
        or (value.device is not None and target_device != value.device)
    ):
        raise ValueError("asarray(copy=False) cannot satisfy the requested dtype or device")
    attrs: dict[str, object] = {
        "_advect_array_api_asarray": True,
        "copy": copy,
        "dtype": _dtype_name(target_dtype),
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
    unchanged = (
        _dtype_name(target_dtype) == _dtype_name(value.dtype) and target_device == value.device
    )
    return _alias_result(value, result) if copy is not True and unchanged else result


def _array_api_cumulative_with_initial(
    trace: AbstractTrace,
    path: str,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    if not raw_args or len(raw_args) > 2:
        raise TypeError(f"{path}() expects an array and optional axis")
    if len(raw_args) == 2 and "axis" in raw_kwargs:
        raise TypeError(f"{path}() received 'axis' twice")
    source = _lift(trace, raw_args[0])
    axis_value = raw_args[1] if len(raw_args) == 2 else raw_kwargs.get("axis")
    if axis_value is None:
        if source.ndim != 1:
            raise ValueError(
                "cumulative operations require axis= for inputs with more than one dimension"
            )
        axis = 0
    else:
        axis = _normalize_axis(axis_value, source.ndim)
    options = dict(raw_kwargs)
    options["axis"] = axis
    options.pop("include_initial", None)
    base = cast("AbstractArray", _apply_array_api(trace, path, (source,), options))
    seed_shape = list(base.shape)
    seed_shape[axis] = 1
    fill_value = 1 if path == "cumulative_prod" else 0
    seed = cast(
        "AbstractArray",
        _apply_array_api(
            trace,
            "full",
            (tuple(seed_shape), fill_value),
            {"dtype": base.dtype},
        ),
    )
    return cast(
        "AbstractArray",
        _apply_array_api(trace, "concat", ((seed, base),), {"axis": axis}),
    )


def _array_api_diff(
    trace: AbstractTrace,
    raw_args: tuple[Any, ...],
    raw_kwargs: dict[str, Any],
) -> AbstractArray:
    if not raw_args or len(raw_args) > 5:
        raise TypeError("diff() expects (a, n, axis, prepend, append)")
    values = dict(raw_kwargs)
    unexpected = set(values) - {"append", "axis", "n", "prepend"}
    if unexpected:
        raise TypeError(
            f"Abstract staging of diff() does not support {tuple(sorted(unexpected))!r}"
        )
    source_raw = raw_args[0]
    for name, value in zip(("n", "axis", "prepend", "append"), raw_args[1:], strict=False):
        if name in values:
            raise TypeError(f"diff() received {name!r} twice")
        values[name] = value
    n = values.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("diff n must be a non-negative integer")
    source = _lift(trace, source_raw)
    axis = _normalize_axis(values.get("axis", -1), source.ndim)

    def emit_diff(value: AbstractArray) -> AbstractArray:
        return cast(
            "AbstractArray",
            _record_abstract_op(
                trace,
                "array.diff",
                (value,),
                {"axis": axis, "n": n},
                abstract_attrs={"_advect_array_api_version": trace.array_api_version},
            ),
        )

    if n == 0 or (values.get("prepend") is None and values.get("append") is None):
        return emit_diff(source)
    boundary_shape = list(source.shape)
    boundary_shape[axis] = 1

    def lift_boundary(raw_value: object) -> AbstractArray:
        boundary = _lift(trace, raw_value)
        if boundary.shape == ():
            return cast(
                "AbstractArray",
                _apply_array_api(
                    trace,
                    "broadcast_to",
                    (boundary, tuple(boundary_shape)),
                    {},
                ),
            )
        return boundary

    parts = [lift_boundary(values["prepend"])] if values.get("prepend") is not None else []
    parts.append(source)
    if values.get("append") is not None:
        parts.append(lift_boundary(values["append"]))
    joined = cast(
        "AbstractArray",
        _apply_array_api(trace, "concat", (tuple(parts),), {"axis": axis}),
    )
    return emit_diff(joined)


def _apply_array_api(
    trace: AbstractTrace,
    path: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 - Array API calls may return structured results
    """Bind one provider-neutral call and record its canonical operation."""
    trace.require_open()
    if path == "asarray":
        return _array_api_asarray(trace, args, kwargs)
    if path in {"cumulative_prod", "cumulative_sum"} and bool(kwargs.get("include_initial", False)):
        return _array_api_cumulative_with_initial(trace, path, args, kwargs)
    if path == "diff":
        return _array_api_diff(trace, args, kwargs)
    if path == "searchsorted" and kwargs.get("sorter") is not None:
        if len(args) != 2:
            raise TypeError("searchsorted() expects two positional array arguments")
        options = dict(kwargs)
        sorter = options.pop("sorter")
        sorted_source = _apply_array_api(trace, "take", (args[0], sorter), {"axis": 0})
        return _apply_array_api(trace, path, (sorted_source, args[1]), options)

    binding = bind_array_api_call(path, args, kwargs)
    result = _record_abstract_op(
        trace,
        binding.op,
        binding.operands,
        binding.attrs,
        abstract_attrs={"_advect_array_api_version": trace.array_api_version},
    )
    if not isinstance(result, tuple):
        return result
    return restore_array_api_result(path, result)


def _basic_index_spec(
    value: AbstractArray,
    index: object,
    *,
    allow_pending: bool = False,
) -> tuple[object, ArraySpec]:
    value._require(allow_pending=allow_pending)
    encoded = encode_basic_index(index)
    items = index if isinstance(index, tuple) else (index,)
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
        if not isinstance(item, slice):
            raise TracingError(f"Unsupported basic index component {type(item).__name__}")
        start, stop, step = item.indices(size)
        shape.append(len(range(start, stop, step)))
    return encoded, ArraySpec(tuple(shape), source_spec.dtype)


def _apply_getitem(value: AbstractArray, index: object) -> AbstractArray:
    encoded, result_spec = _basic_index_spec(value, index)
    result = _emit(value._trace, "advect.getitem", (value,), {"index": encoded}, result_spec)
    items = index if isinstance(index, tuple) else (index,)
    is_alias = bool(result_spec.shape) or any(not isinstance(item, int) for item in items)
    if not is_alias:
        return result
    return _alias_result(value, result, index=encoded if value._view is None else None)


__all__ = [
    "AbstractArray",
    "AbstractNamespace",
    "AbstractTrace",
    "AbstractValue",
    "ArraySpec",
]
