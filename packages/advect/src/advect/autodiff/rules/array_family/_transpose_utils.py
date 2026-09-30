"""Shared transpose and tangent helpers for array-family JVP/VJP synthesis."""

from __future__ import annotations

from typing import Any, Literal, cast

from advect.autodiff.rules.array_family._backend_runtime import (
    _array_constructor_like,
    _scalar_like,
    current_array_backend_provider,
    xp,
)
from advect.core._protocols import _innermost

type FFTNorm = Literal["backward", "ortho", "forward"]


def _shape_of(value: Any) -> tuple[int, ...]:
    """Return the concrete shape of an array, a traced leaf, or a Python scalar."""
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = xp.asarray(_innermost(value)).shape
    return tuple(int(dimension) for dimension in shape)


def _normalize_axis(axis: int, *, ndim: int, op_name: str) -> int:
    """Map ``axis`` into ``[0, ndim)``, naming ``op_name`` when it is out of bounds."""
    if not -ndim <= axis < ndim:
        msg = f"{op_name} axis {axis} is out of bounds for rank {ndim}"
        raise ValueError(msg)
    return axis % ndim


def _axis_slice(
    *,
    ndim: int,
    axis: int,
    start: int | None = None,
    stop: int | None = None,
    index: int | None = None,
) -> tuple[int | slice, ...]:
    """Index ``axis`` with ``index`` or ``start:stop`` and every other axis whole."""
    result: list[int | slice] = [slice(None)] * ndim
    result[axis] = slice(start, stop) if index is None else index
    return tuple(result)


def _difference(value: Any, *, order: int, axis: int) -> Any:
    """Take ``order`` forward differences along ``axis`` by slicing.

    ``diff`` entered the Array API in 2024.12.
    """
    ndim = len(_shape_of(value))
    later = _axis_slice(ndim=ndim, axis=axis, start=1)
    earlier = _axis_slice(ndim=ndim, axis=axis, stop=-1)
    for _ in range(order):
        value = xp.subtract(value[later], value[earlier])
    return value


def _tile_layout(
    shape: tuple[int, ...],
    reps: int | tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``tile``'s rank-aligned source shape and its ``(copies, extent)`` pairs.

    The pairs, flattened, shape the tiled result with each copy count before
    the source extent it repeats.
    """
    repetitions = (reps,) if isinstance(reps, int) else tuple(reps)
    rank = max(len(shape), len(repetitions))
    source = (1,) * (rank - len(shape)) + tuple(shape)
    counts = (1,) * (rank - len(repetitions)) + repetitions
    return source, tuple(extent for pair in zip(counts, source, strict=True) for extent in pair)


def _dtype_of(value: Any) -> Any:
    """Return the dtype of an array, a traced leaf, or a Python scalar."""
    dtype = getattr(value, "dtype", None)
    return xp.asarray(value).dtype if dtype is None else dtype


def _dtype_is_complex(dtype: object) -> bool:
    kind = getattr(dtype, "kind", None)
    return kind == "c" if kind is not None else "complex" in str(dtype).lower()


def _conjugate_if_complex(value: Any) -> Any:
    if isinstance(value, complex):
        return value.conjugate()
    return xp.conj(value) if _dtype_is_complex(getattr(value, "dtype", None)) else value


def _layout_order(value: Any, order: str | None) -> str | None:
    """Resolve NumPy's ``order='A'`` against the layout of the concrete primal.

    ``'A'`` means Fortran order for a Fortran-contiguous array that is not
    C-contiguous, and C order otherwise. A tangent or cotangent can be laid out
    differently from its primal, so rules must not pass ``'A'`` on to it.
    Values without layout flags, such as Array API or staged arrays, read in C
    order.
    """
    if order != "A":
        return order
    flags = getattr(_innermost(value), "flags", None)
    fortran = flags is not None and bool(flags.f_contiguous) and not bool(flags.c_contiguous)
    return "F" if fortran else "C"


def _memory_order_axes(shape: tuple[int, ...], strides: tuple[int, ...]) -> tuple[int, ...]:
    """Return the axes, outermost first, in which NumPy's ``order='K'`` reads.

    This follows NumPy's iterator: starting from the innermost axis, a stable
    insertion sort moves axes with larger absolute strides outward. A zero
    stride, which NumPy also gives every length-one axis, compares with
    nothing, so a broadcast axis keeps its C-order place. Negative strides are
    read as they are, not reversed.
    """
    magnitudes = [
        0 if extent == 1 else abs(stride) for extent, stride in zip(shape, strides, strict=True)
    ]
    innermost_first: list[int] = []
    for axis in reversed(range(len(shape))):
        position = len(innermost_first)
        for candidate in reversed(range(position) if magnitudes[axis] else ()):
            inner = magnitudes[innermost_first[candidate]]
            if inner == 0:
                continue
            if inner <= magnitudes[axis]:
                break
            position = candidate
        innermost_first.insert(position, axis)
    return tuple(reversed(innermost_first))


def _ravel_axes(value: Any, order: str | None) -> tuple[int, ...]:
    """Return the axis order whose C-order ravel is NumPy's ``ravel(value, order)``.

    ``'A'`` and ``'K'`` depend on the layout of the concrete primal, never on
    that of a tangent or cotangent. Values without strides, such as Array API
    or staged arrays, read ``'K'`` in C order.
    """
    concrete = _innermost(value)
    shape = tuple(int(extent) for extent in concrete.shape)
    resolved = _layout_order(value, order)
    strides = getattr(concrete, "strides", None)
    if resolved == "F":
        return tuple(reversed(range(len(shape))))
    if resolved == "K" and strides is not None:
        return _memory_order_axes(shape, tuple(int(stride) for stride in strides))
    return tuple(range(len(shape)))


def _reshape_in_order(value: Any, shape: tuple[int, ...], order: str | None) -> xp.ndarray:
    """Reshape in NumPy's C or F ``order`` through portable primitives.

    Array API reshape has no ``order``. Column-major reshaping reverses the
    axes, reshapes to the reversed shape, and reverses the axes back.
    """
    if order != "F":
        return xp.reshape(value, shape)
    reversed_value = xp.permute_dims(value, tuple(reversed(range(value.ndim))))
    reshaped = xp.reshape(reversed_value, tuple(reversed(shape)))
    return xp.permute_dims(reshaped, tuple(reversed(range(len(shape)))))


def _tangent_type_operand(value: Any) -> Any:
    """Preserve Python-scalar weakness while reducing array tangents to dtypes."""
    unwrapped = _innermost(value)
    if type(unwrapped) in {bool, complex, float, int}:
        return unwrapped
    return xp.asarray(unwrapped).dtype


def _adjoint_fft_norm(norm: FFTNorm | None) -> FFTNorm:
    if norm in {None, "backward"}:
        return "forward"
    if norm == "forward":
        return "backward"
    return "ortho"


def _conjugate_transpose(value: xp.ndarray) -> xp.ndarray:
    """Conjugate-transpose the final two axes."""
    return xp.conj(xp.swapaxes(value, -1, -2))


def _diagonal_matrix(values: xp.ndarray, *, dtype: xp.dtype[Any]) -> xp.ndarray:
    size = int(values.shape[-1])
    eye = _array_constructor_like(values, "eye", size, dtype=dtype)
    return cast("xp.ndarray", eye * values[..., None, :])


def _lower_triangular_halfdiag(value: xp.ndarray) -> xp.ndarray:
    """Project to the lower triangle and halve its diagonal."""
    lower = xp.tril(value)
    diagonal = xp.diagonal(lower, axis1=-2, axis2=-1)
    return cast(
        "xp.ndarray",
        lower - _scalar_like(0.5, lower) * _diagonal_matrix(diagonal, dtype=xp.dtype(lower.dtype)),
    )


def _normalize_uplo(value: str) -> Literal["L", "U"]:
    normalized = value.upper()
    if normalized not in {"L", "U"}:
        msg = f"expected UPLO='L' or 'U', got {value!r}"
        raise ValueError(msg)
    return cast("Literal['L', 'U']", normalized)


def _right_solve(a: xp.ndarray, b: xp.ndarray) -> xp.ndarray:
    """Solve ``result @ a = b`` over the final two axes."""
    return cast(
        "xp.ndarray",
        xp.swapaxes(
            xp.linalg.solve(xp.swapaxes(a, -1, -2), xp.swapaxes(b, -1, -2)),
            -1,
            -2,
        ),
    )


def _uses_standard_linalg_contract() -> bool:
    provider = current_array_backend_provider()
    return provider is not None and provider.backend.split(".", 1)[0] != "numpy"


def dtype_is_inexact(dtype: object) -> bool:
    """Return whether a provider dtype has a real or complex tangent space."""
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return kind in {"c", "f"}
    name = str(dtype).lower()
    return "float" in name or "complex" in name


def infer_tangent_dtype(ans: Any, tangents: tuple[Any | None, ...]) -> xp.dtype[Any]:
    """Infer a dtype for tangent computations from ans and non-None tangents.

    An integer or boolean answer has no tangent space of its own, so it does
    not promote the tangents. Array API providers need not promote it with a
    floating-point dtype at all.
    """
    answer_dtype = xp.asarray(_innermost(ans)).dtype
    operands = [_tangent_type_operand(tangent) for tangent in tangents if tangent is not None]
    if dtype_is_inexact(answer_dtype) or not operands:
        operands.insert(0, answer_dtype)
    return xp.result_type(*operands)


def zeros_output_tangent(ans: Any, tangents: tuple[Any | None, ...]) -> xp.ndarray:
    """Create a zero tangent matching ``ans`` with inferred tangent dtype."""
    dtype = infer_tangent_dtype(ans, tangents)
    return xp.zeros_like(xp.asarray(_innermost(ans)), dtype=dtype)


def infer_output_tangent_dtype(ans: Any, tangents: tuple[Any | None, ...]) -> xp.dtype[Any]:
    """Infer tangent dtype for outputs, including tuple-output structures.

    As in ``infer_tangent_dtype``, an integer or boolean leaf does not promote
    the tangents.
    """
    leaf_dtypes: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, tuple):
            for item in value:
                collect(item)
            return
        leaf_dtypes.append(xp.asarray(_innermost(value)).dtype)

    collect(ans)
    tangent_dtypes = [_tangent_type_operand(tangent) for tangent in tangents if tangent is not None]
    dtypes = [dtype for dtype in leaf_dtypes if dtype_is_inexact(dtype) or not tangent_dtypes]
    dtypes.extend(tangent_dtypes)
    if not dtypes:
        return cast("xp.dtype[Any]", xp.float64)
    return xp.result_type(*dtypes)
