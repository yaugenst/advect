"""Adapt real Python scalars to the ordinary array transformation path."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, TypeGuard, cast

from advect.core._pytree import tree_flatten, tree_unflatten


def _dtype_name(value: object) -> str:
    dtype = getattr(value, "dtype", None)
    return "" if dtype is None else str(dtype).lower()


def _is_boolean_numeric(value: object) -> bool:
    if isinstance(value, bool):
        return True
    dtype = getattr(value, "dtype", None)
    kind = getattr(dtype, "kind", None)
    return kind == "b" if kind is not None else "bool" in _dtype_name(value)


def _is_complex_numeric(value: object) -> bool:
    if isinstance(value, complex):
        return True
    dtype = getattr(value, "dtype", None)
    kind = getattr(dtype, "kind", None)
    return kind == "c" if kind is not None else "complex" in _dtype_name(value)


def _is_real_python_scalar(value: object) -> TypeGuard[int | float]:
    """Return whether ``value`` is a selected scalar primal Advect can lift."""
    return type(value) in {int, float}


def _lift_scalar_to_array(value: object, *, namespace: Any | None) -> Any:
    """Create a provider-backed rank-zero float64 array for one scalar primal."""
    if not _is_real_python_scalar(value):
        msg = f"Cannot lift {type(value).__name__} as a real scalar primal"
        raise TypeError(msg)

    if namespace is None:
        import numpy as np  # noqa: PLC0415 - NumPy is the default scalar provider

        namespace = np
    asarray = getattr(namespace, "asarray", None)
    if not callable(asarray):
        msg = f"Array provider {namespace!r} cannot lift a Python scalar"
        raise TypeError(msg)

    dtype = getattr(namespace, "float64", None)
    lifted = asarray(value, dtype=dtype) if dtype is not None else asarray(float(value))
    if tuple(int(dimension) for dimension in getattr(lifted, "shape", ())) != ():
        msg = "Array provider did not produce a rank-zero scalar primal"
        raise TypeError(msg)
    return lifted


def _coerce_scalar_tangent_like(tangent: object, primal: object) -> Any:
    """Normalize a scalar-boundary tangent through the primal's provider."""
    if _is_boolean_numeric(tangent) or _is_complex_numeric(tangent):
        msg = "Scalar JVP tangents must be real numbers or rank-zero real arrays"
        raise TypeError(msg)

    tangent_shape = getattr(tangent, "shape", None)
    if tangent_shape is not None and tuple(int(dimension) for dimension in tangent_shape) != ():
        msg = f"Scalar JVP tangent shape mismatch: expected (), got {tuple(tangent_shape)}."
        raise ValueError(msg)
    if tangent_shape is None and not _is_real_python_scalar(tangent):
        msg = (
            "Scalar JVP tangents must be real numbers or rank-zero real arrays; "
            f"got {type(tangent).__name__}."
        )
        raise TypeError(msg)

    del primal
    item = getattr(tangent, "item", None)
    scalar = item() if callable(item) else tangent
    return float(cast("Any", scalar))


def _unlift_scalar_array(value: Any) -> Any:
    """Return a Python scalar for a concrete rank-zero array when possible."""
    if callable(getattr(value, "_advect_snapshot", None)):
        return value
    if getattr(value, "shape", None) != ():
        return value

    item = getattr(value, "item", None)
    if callable(item):
        with suppress(Exception):
            return item()

    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "kind", None) == "c" or "complex" in str(dtype).lower():
        with suppress(Exception):
            return complex(value)
    with suppress(Exception):
        return float(value)
    return value


def _unlift_scalar_tree_by_mask(value: Any, *, mask: tuple[bool, ...]) -> Any:
    """Unlift only output leaves that depend on lifted scalar primals."""
    if not any(mask):
        return value
    leaves, treedef = tree_flatten(value)
    if len(leaves) != len(mask):
        msg = (
            "Scalar output restoration mask does not match the output pytree: "
            f"expected {len(leaves)} entries, got {len(mask)}."
        )
        raise RuntimeError(msg)
    restored = [
        _unlift_scalar_array(leaf) if restore else leaf
        for leaf, restore in zip(leaves, mask, strict=True)
    ]
    return tree_unflatten(treedef, restored)
