# ruff: noqa: EM101, EM102, PLR2004, TRY003
"""Shape, axis, and dtype helpers shared by abstract-evaluation domains."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from advect.core._abstract_model import ArraySpec

_SINGLE_PRECISION_BITS = 32
_DOUBLE_PRECISION_BITS = 64
_COMPLEX128_BITS = 128


def dtype_name(dtype: object) -> str:
    """Return the stable dtype spelling stored in graph metadata."""
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


def dtype_kind_bits(dtype: object) -> tuple[str, int]:
    """Return Advect's staged promotion category and precision."""
    name = dtype_name(dtype).lower()
    if "complex" in name:
        return "complex", 128 if "128" in name else 64
    if "float" in name:
        return "float", 64 if "64" in name else 32
    if "uint" in name:
        return "uint", int("".join(char for char in name if char.isdigit()) or 64)
    if "int" in name:
        return "int", int("".join(char for char in name if char.isdigit()) or 64)
    if "bool" in name:
        return "bool", 8
    raise TypeError(f"Unsupported staged dtype {dtype!r}")


def promote_dtype(specs: Sequence[ArraySpec]) -> str:
    """Apply the staged weak-scalar promotion contract."""
    if not specs:
        raise TypeError("A staged operation requires at least one typed operand")
    strong = [spec for spec in specs if not spec.weak]
    effective = strong or list(specs)
    effective_kinds_and_bits = [dtype_kind_bits(spec.dtype) for spec in effective]
    effective_kinds = {kind for kind, _bits in effective_kinds_and_bits}
    if effective_kinds == {"int", "uint"}:
        signed_bits = max(bits for kind, bits in effective_kinds_and_bits if kind == "int")
        unsigned_bits = max(bits for kind, bits in effective_kinds_and_bits if kind == "uint")
        for candidate_bits in (8, 16, 32, 64):
            if candidate_bits >= signed_bits and candidate_bits > unsigned_bits:
                return f"int{candidate_bits}"
        # NumPy defines the otherwise-unrepresentable int64/uint64 pair as
        # float64. Array API providers do not admit that pair for promotion.
        return "float64"
    base = strong[0] if strong else specs[0]
    kind, bits = dtype_kind_bits(base.dtype)
    rank = {"bool": 0, "uint": 1, "int": 2, "float": 3, "complex": 4}
    for spec in specs:
        candidate_kind, candidate_bits = dtype_kind_bits(spec.dtype)
        if spec.weak and strong:
            if candidate_kind == "complex" and rank[kind] < rank["complex"]:
                kind = "complex"
                bits = (
                    _DOUBLE_PRECISION_BITS if bits <= _SINGLE_PRECISION_BITS else _COMPLEX128_BITS
                )
            elif candidate_kind == "float" and rank[kind] < rank["float"]:
                kind = "float"
                bits = max(bits, _SINGLE_PRECISION_BITS)
            continue
        comparison_bits = candidate_bits
        if kind == "complex" and candidate_kind != "complex":
            comparison_bits *= 2
        elif kind != "complex" and candidate_kind == "complex":
            bits *= 2
        if rank[candidate_kind] > rank[kind]:
            kind = candidate_kind
        bits = max(bits, comparison_bits)
    if kind == "complex":
        bits = _DOUBLE_PRECISION_BITS if bits <= _DOUBLE_PRECISION_BITS else _COMPLEX128_BITS
    elif kind == "float":
        bits = _SINGLE_PRECISION_BITS if bits <= _SINGLE_PRECISION_BITS else _DOUBLE_PRECISION_BITS
    return "bool" if kind == "bool" else f"{kind}{bits}"


def real_dtype(dtype: object) -> str:
    """Return the real tangent-space dtype corresponding to *dtype*."""
    kind, bits = dtype_kind_bits(dtype)
    if kind != "complex":
        return dtype_name(dtype)
    return "float32" if bits == _DOUBLE_PRECISION_BITS else "float64"


def complex_dtype(dtype: object) -> str:
    """Return the complex FFT dtype corresponding to *dtype*."""
    kind, bits = dtype_kind_bits(dtype)
    if kind not in {"float", "complex"}:
        raise TypeError(f"FFT input must be floating-point or complex, got {dtype!r}")
    if kind == "complex":
        return "complex64" if bits == _DOUBLE_PRECISION_BITS else "complex128"
    return "complex64" if bits == _SINGLE_PRECISION_BITS else "complex128"


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
    raw = (axis,) if isinstance(axis, int) else tuple(axis)  # type: ignore[arg-type]
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
    raw = (value,) if isinstance(value, int) else tuple(value)  # type: ignore[arg-type]
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
        raw = tuple(axes_value)  # type: ignore[arg-type]
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
