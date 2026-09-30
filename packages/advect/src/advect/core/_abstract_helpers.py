# ruff: noqa: PLR2004
"""Shape, axis, and dtype helpers shared by abstract-evaluation domains."""

from __future__ import annotations

import functools
import math
import string
from collections.abc import Iterable
from typing import TYPE_CHECKING

from advect.core._abstract_model import ArraySpec

if TYPE_CHECKING:
    from collections.abc import Sequence

_SINGLE_PRECISION_BITS = 32
_DOUBLE_PRECISION_BITS = 64
# NumPy 2's default dtype for each Python scalar type.
_PYTHON_SCALAR_DTYPES: dict[type, str] = {
    bool: "bool",
    int: "int64",
    float: "float64",
    complex: "complex128",
}
# Exact types with weak-scalar semantics; built-in subclasses need isinstance.
PYTHON_SCALAR_TYPES = tuple(_PYTHON_SCALAR_DTYPES)
# The name of the Array API namespace of abstract staged values.
ABSTRACT_NAMESPACE_NAME = "advect.array_api"

# NumPy's "safe" cast targets for every staged dtype, in promotion order. The
# first target shared by every strong operand is their NumPy 2 result type.
_INEXACT = "float16 float32 float64 complex64 complex128"
_SAFE_CASTS: dict[str, tuple[str, ...]] = {
    source: tuple(targets.split())
    for source, targets in {
        "bool": f"bool uint8 int8 uint16 int16 uint32 int32 uint64 int64 {_INEXACT}",
        "uint8": f"uint8 uint16 int16 uint32 int32 uint64 int64 {_INEXACT}",
        "int8": f"int8 int16 int32 int64 {_INEXACT}",
        "uint16": "uint16 uint32 int32 uint64 int64 float32 float64 complex64 complex128",
        "int16": "int16 int32 int64 float32 float64 complex64 complex128",
        "uint32": "uint32 uint64 int64 float64 complex128",
        "int32": "int32 int64 float64 complex128",
        "uint64": "uint64 float64 complex128",
        "int64": "int64 float64 complex128",
        "float16": _INEXACT,
        "float32": "float32 float64 complex64 complex128",
        "float64": "float64 complex128",
        "complex64": "complex64 complex128",
        "complex128": "complex128",
    }.items()
}
DTYPE_NAMES = frozenset(_SAFE_CASTS)
_UNSUPPORTED_DTYPE = (
    "Unsupported staged dtype {!r}; staged programs support only the canonical bool "
    f"and numeric dtypes: {', '.join(_SAFE_CASTS)}"
)
_KIND_BITS = {
    name: (name.rstrip(string.digits), int(name.lstrip(string.ascii_lowercase) or 8))
    for name in _SAFE_CASTS
}
_WEAK_FLOAT = ArraySpec((), "float64", weak=True)
_WEAK_COMPLEX = ArraySpec((), "complex128", weak=True)
# Kind order of NEP 50 weak (Python scalar) promotion.
_KIND_RANK = {"bool": 0, "uint": 1, "int": 1, "float": 2, "complex": 3}
# Categories within which the Array API admits lossless casts.
_CAST_CATEGORY = {"bool": 0, "uint": 1, "int": 1, "float": 2, "complex": 2}


@functools.cache
def _result_type(dtypes: frozenset[str]) -> str:
    return next(
        target
        for target in _SAFE_CASTS["bool"]
        if all(target in _SAFE_CASTS[dtype] for dtype in dtypes)
    )


def _weak_result_type(result: str, scalar: str) -> str:
    result_kind = _KIND_BITS[result][0]
    scalar_kind = _KIND_BITS[scalar][0]
    if _KIND_RANK[scalar_kind] <= _KIND_RANK[result_kind]:
        return result
    if scalar_kind == "complex" and result_kind == "float":
        return _result_type(frozenset((result, "complex64")))
    return scalar


def dtype_name(dtype: object) -> str:
    """Return the stable dtype spelling stored in graph metadata."""
    if isinstance(dtype, type) and dtype in _PYTHON_SCALAR_DTYPES:
        return _PYTHON_SCALAR_DTYPES[dtype]
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    text = str(dtype)
    if text.startswith("<class '") and text.endswith("'>"):
        text = text[8:-2].rsplit(".", 1)[-1]
    elif "." in text:
        suffix = text.rsplit(".", 1)[-1]
        if suffix.startswith(("bool", "int", "uint", "float", "complex")):
            text = suffix
    return text


def value_spec(value: object) -> ArraySpec:
    """Return the staged spec of one concrete operand.

    Python scalars are weak. NumPy scalars such as ``float64`` subclass Python
    types but carry a dtype, so they are strong rank-zero arrays.
    """
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return ArraySpec(tuple(int(size) for size in shape), dtype)
    for python_type, python_dtype in _PYTHON_SCALAR_DTYPES.items():
        if isinstance(value, python_type):
            return ArraySpec((), python_dtype, weak=True)
    raise TypeError(f"Cannot stage concrete operand of type {type(value).__name__}")


@functools.lru_cache(maxsize=256)
def _typed_dtype_name(_dtype_type: type, dtype: object) -> str:
    # A provider dtype may compute its name on every access (NumPy's is a
    # Python property), and staging reads one per sequence leaf. Keying on the
    # type first keeps a cache hit from comparing dtypes across providers.
    return dtype_name(dtype)


def _staged_dtype(dtype: object) -> str:
    name = dtype.lower() if type(dtype) is str else ""
    if name not in DTYPE_NAMES:
        try:
            name = _typed_dtype_name(type(dtype), dtype).lower()
        except TypeError:  # An unhashable dtype object.
            name = dtype_name(dtype).lower()
        if name not in DTYPE_NAMES:
            raise TypeError(_UNSUPPORTED_DTYPE.format(dtype))
    return name


def dtype_kind_bits(dtype: object) -> tuple[str, int]:
    """Return Advect's staged promotion category and precision."""
    return _KIND_BITS[_staged_dtype(dtype)]


def safely_casts(source: object, target: object) -> bool:
    """Return whether NumPy's "safe" casting admits *source* to *target*."""
    return _staged_dtype(target) in _SAFE_CASTS[_staged_dtype(source)]


def can_cast_dtype(source: str, target: str) -> bool:
    """Return the Array API lossless-cast relation between two staged dtypes."""
    return (
        target in _SAFE_CASTS[source]
        and _CAST_CATEGORY[_KIND_BITS[source][0]] == _CAST_CATEGORY[_KIND_BITS[target][0]]
    )


def _operand_dtypes(specs: Sequence[ArraySpec]) -> frozenset[str]:
    """Return the operand dtypes after NEP 50 resolves each weak scalar."""
    if not specs:
        raise TypeError("A staged operation requires at least one typed operand")
    strong = frozenset(_staged_dtype(spec.dtype) for spec in specs if not spec.weak)
    weak = {_staged_dtype(spec.dtype) for spec in specs if spec.weak}
    if not strong:
        return frozenset(weak)
    result = _result_type(strong)
    return strong | {_weak_result_type(result, dtype) for dtype in weak}


def promote_dtype(specs: Sequence[ArraySpec]) -> str:
    """Return NumPy 2's result type, with weak specs promoting as Python scalars."""
    return _result_type(_operand_dtypes(specs))


def strong_result_dtype(specs: Sequence[ArraySpec]) -> str:
    """Return NumPy 2's result type once array coercion makes every weak spec strong.

    ``stack`` and the contractions such as ``dot`` coerce a Python scalar to
    an array at its default dtype before promoting, unlike ufuncs.
    """
    return _result_type(frozenset(_staged_dtype(spec.dtype) for spec in specs))


def discovered_dtype(value: object) -> object:
    """Return the dtype NumPy's array coercion discovers for one sequence leaf.

    Coercion reads a Python scalar at its default dtype, not weakly, except that
    it reads a Python int by value: int64, else uint64, else an object array,
    which staging does not support.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if -(2**63) <= value < 2**63:
            return "int64"
        return "uint64" if 0 <= value < 2**64 else "object"
    return value_spec(value).dtype


def coerced_dtype(dtypes: Iterable[object]) -> str:
    """Return the dtype NumPy's array coercion discovers for a sequence's leaves.

    Coercion folds ``promote_types`` over the leaves in order, so unlike the
    result type it depends on their order: int8, uint8, float16 is float32, but
    float16, uint8, int8 is float16. An empty sequence is float64.
    """
    result: str | None = None
    for dtype in dtypes:
        name = _staged_dtype(dtype)
        result = name if result is None else _result_type(frozenset((result, name)))
    return "float64" if result is None else result


def inexact_dtype(specs: Sequence[ArraySpec]) -> str:
    """Return the dtype of a NumPy ufunc with only floating-point and complex loops."""
    return _result_type(_operand_dtypes(specs) | {"float16"})


def division_dtype(dtype: object) -> str:
    """Return NumPy's true-division and linalg dtype: exact dtypes compute in float64."""
    name = _staged_dtype(dtype)
    return "float64" if _KIND_BITS[name][0] in {"bool", "int", "uint"} else name


def real_dtype(dtype: object) -> str:
    """Return the real tangent-space dtype corresponding to *dtype*."""
    kind, bits = dtype_kind_bits(dtype)
    if kind != "complex":
        return dtype_name(dtype)
    return "float32" if bits == _DOUBLE_PRECISION_BITS else "float64"


def fft_dtype(dtype: object, *, real_output: bool) -> str:
    """Return NumPy's FFT result dtype for one input dtype.

    Complex transforms return ``result_type(x, 1j)`` and inverse real
    transforms ``result_type(x.real, 1.0)``, so exact inputs become float64.
    """
    if real_output:
        return promote_dtype((ArraySpec((), real_dtype(dtype)), _WEAK_FLOAT))
    return promote_dtype((ArraySpec((), dtype), _WEAK_COMPLEX))


def accumulation_dtype(
    dtype: object,
    *,
    array_api_version: str | None = None,
) -> str:
    """Return the default dtype for an accumulation."""
    kind, bits = dtype_kind_bits(dtype)
    if kind in {"bool", "int"}:
        return "int64"
    if kind == "uint":
        return "uint64"
    single_precision = (kind == "float" and bits <= _SINGLE_PRECISION_BITS) or (
        kind == "complex" and bits <= _DOUBLE_PRECISION_BITS
    )
    if array_api_version == "2022.12" and single_precision:
        return "complex128" if kind == "complex" else "float64"
    return dtype_name(dtype)


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return the broadcast result shape or fail deterministically."""
    result: list[int] = []
    width = max((len(shape) for shape in shapes), default=0)
    for offset in range(1, width + 1):
        dimensions = [shape[-offset] for shape in shapes if len(shape) >= offset]
        non_unit = {dimension for dimension in dimensions if dimension != 1}
        if len(non_unit) > 1:
            raise ValueError(f"Shapes are not broadcast-compatible: {shapes!r}")
        target = next(iter(non_unit), 1)
        result.append(target)
    return tuple(reversed(result))


def normalize_axis(axis: object, ndim: int, *, insertion: bool = False) -> int:
    """Normalize one axis under ordinary or insertion bounds."""
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError(f"Axis must be an integer, got {axis!r}")
    width = ndim + 1 if insertion else ndim
    if not -width <= axis < width:
        raise ValueError(f"Axis {axis} is out of bounds for rank {ndim}")
    return axis + width if axis < 0 else axis


def normalize_axes(axis: object, ndim: int) -> tuple[int, ...]:
    """Normalize an optional axis collection and reject duplicates."""
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        raw = (axis,)
    elif isinstance(axis, Iterable):
        raw = tuple(axis)
    else:
        raise TypeError(f"Axis must be an integer or iterable of integers, got {axis!r}")
    normalized = tuple(normalize_axis(item, ndim) for item in raw)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Repeated axis in {axis!r}")
    return normalized


def reduction_shape(
    shape: tuple[int, ...],
    axis: object,
    *,
    keepdims: bool,
) -> tuple[int, ...]:
    """Return the result shape for a reduction."""
    axes = set(normalize_axes(axis, len(shape)))
    if keepdims:
        return tuple(1 if index in axes else size for index, size in enumerate(shape))
    return tuple(size for index, size in enumerate(shape) if index not in axes)


def matmul_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return the Array API matmul result shape."""
    if not left or not right:
        raise ValueError("matmul inputs must have at least one dimension")
    left_vector = len(left) == 1
    right_vector = len(right) == 1
    left_matrix = (1, left[0]) if left_vector else left[-2:]
    right_matrix = (right[0], 1) if right_vector else right[-2:]
    if left_matrix[1] != right_matrix[0]:
        raise ValueError(f"matmul core dimensions disagree: {left!r} and {right!r}")
    batch = broadcast_shape(
        left[:-2] if not left_vector else (),
        right[:-2] if not right_vector else (),
    )
    tail = (left_matrix[0], right_matrix[1])
    if left_vector:
        tail = tail[1:]
    if right_vector:
        tail = tail[:-1]
    return (*batch, *tail)


def shape_tuple(value: object) -> tuple[int, ...]:
    """Normalize shape-like static metadata."""
    if isinstance(value, int):
        raw = (value,)
    elif isinstance(value, Iterable):
        raw = tuple(value)
    else:
        raise TypeError(f"Shape must be an integer or iterable of integers, got {value!r}")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in raw):
        raise TypeError(f"Shape must contain integers, got {value!r}")
    return raw


def replace_axis(shape: tuple[int, ...], axis: int, size: int) -> tuple[int, ...]:
    """Replace one axis length after validating an FFT size."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"FFT transform length must be a positive integer, got {size!r}")
    result = list(shape)
    result[axis] = size
    return tuple(result)


def fft_shape(
    shape: tuple[int, ...],
    *,
    n: object,
    axis: object,
    real_output: bool,
    inverse_real: bool,
) -> tuple[int, ...]:
    """Return the shape of a one-dimensional FFT family operation."""
    normalized_axis = normalize_axis(axis, len(shape))
    source_size = shape[normalized_axis]
    if n is None:
        size = 2 * (source_size - 1) if inverse_real else source_size
    elif isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"FFT transform length must be an integer or None, got {n!r}")
    else:
        size = n
    if real_output:
        size = size // 2 + 1
    return replace_axis(shape, normalized_axis, size)


def fftn_shape(
    shape: tuple[int, ...],
    *,
    sizes: object,
    axes: object,
    real_output: bool,
    inverse_real: bool,
) -> tuple[int, ...]:
    """Return the shape of an n-dimensional FFT family operation."""
    normalized_axes = tuple(range(len(shape))) if axes is None else normalize_axes(axes, len(shape))
    if not normalized_axes:
        raise ValueError("FFT transform axes must be non-empty")
    if sizes is None:
        target_sizes = [shape[axis] for axis in normalized_axes]
        if inverse_real:
            target_sizes[-1] = 2 * (target_sizes[-1] - 1)
    else:
        target_sizes = list(shape_tuple(sizes))
        if len(target_sizes) != len(normalized_axes):
            raise ValueError("FFT transform sizes and axes must have equal length")
    if real_output:
        target_sizes[-1] = target_sizes[-1] // 2 + 1
    result = shape
    for axis, size in zip(normalized_axes, target_sizes, strict=True):
        result = replace_axis(result, axis, size)
    return result


def reshape_shape(source: tuple[int, ...], target_value: object) -> tuple[int, ...]:
    """Resolve a static reshape target."""
    target = list(shape_tuple(target_value))
    unknown = [index for index, size in enumerate(target) if size == -1]
    if len(unknown) > 1 or any(size < -1 for size in target):
        raise ValueError(f"Invalid reshape target {tuple(target)!r}")
    source_size = math.prod(source)
    known_size = math.prod(size for size in target if size != -1)
    if unknown:
        if known_size == 0 or source_size % known_size:
            raise ValueError(f"reshape changes element count: {source!r} -> {tuple(target)!r}")
        target[unknown[0]] = source_size // known_size
    elif known_size != source_size:
        raise ValueError(f"reshape changes element count: {source!r} -> {tuple(target)!r}")
    return tuple(target)


def moveaxis_shape(
    shape: tuple[int, ...],
    source: object,
    destination: object,
) -> tuple[int, ...]:
    """Return the shape after moving axes."""
    sources = normalize_axes(source, len(shape))
    destinations = normalize_axes(destination, len(shape))
    if len(sources) != len(destinations):
        raise ValueError("moveaxis source and destination must have equal length")
    order = [axis for axis in range(len(shape)) if axis not in sources]
    for destination_axis, source_axis in sorted(zip(destinations, sources, strict=True)):
        order.insert(destination_axis, source_axis)
    return tuple(shape[axis] for axis in order)


def diagonal_size(rows: int, columns: int, offset: int) -> int:
    """Return the diagonal length for a matrix and offset."""
    if offset >= 0:
        return min(rows, max(columns - offset, 0))
    return min(max(rows + offset, 0), columns)


def tensordot_shape(
    left: tuple[int, ...],
    right: tuple[int, ...],
    axes_value: object,
) -> tuple[int, ...]:
    """Return the static output shape of tensordot."""
    if isinstance(axes_value, bool):
        raise TypeError("tensordot axes must be an integer or a pair of axis sequences")
    if isinstance(axes_value, int):
        if axes_value < 0 or axes_value > min(len(left), len(right)):
            raise ValueError(f"Invalid tensordot axes count {axes_value}")
        left_axes = tuple(range(len(left) - axes_value, len(left)))
        right_axes = tuple(range(axes_value))
    else:
        if not isinstance(axes_value, Iterable):
            raise TypeError("tensordot axes must be an integer or a pair of axis sequences")
        raw = tuple(axes_value)
        if len(raw) != 2:
            raise ValueError("tensordot axes must contain two axis sequences")
        left_axes = normalize_axes(raw[0], len(left))
        right_axes = normalize_axes(raw[1], len(right))
        if len(left_axes) != len(right_axes):
            raise ValueError("tensordot contraction axis lists must have equal length")
    if any(
        left[left_axis] != right[right_axis]
        for left_axis, right_axis in zip(left_axes, right_axes, strict=True)
    ):
        raise ValueError("tensordot contraction dimensions disagree")
    return (
        *(size for axis, size in enumerate(left) if axis not in left_axes),
        *(size for axis, size in enumerate(right) if axis not in right_axes),
    )


def arange_length(start: object, stop: object, step: object) -> int:
    """Return the concrete length of a staged arange."""
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        raise TypeError("arange start, stop, and step must be concrete real scalars")
    if isinstance(stop, bool) or not isinstance(stop, (int, float)):
        raise TypeError("arange start, stop, and step must be concrete real scalars")
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise TypeError("arange start, stop, and step must be concrete real scalars")
    start_value = float(start)
    stop_value = float(stop)
    step_value = float(step)
    if step_value == 0:
        raise ValueError("arange step must be nonzero")
    return max(0, math.ceil((stop_value - start_value) / step_value))
