"""JVP tangent validation, input seeding, and output materialization."""

from __future__ import annotations

from typing import Any

from advect.autodiff.api._scalar_boundary import _coerce_scalar_tangent_like
from advect.core._array_namespace import _get_array_namespace
from advect.core._pytree import tree_flatten


def _normalize_jvp_tangents(
    tangents: Any,
    *,
    single_argnum: bool,
    expected: int,
) -> tuple[Any, ...]:
    if single_argnum:
        return (tangents,)

    if not isinstance(tangents, tuple):
        msg = (
            "jvp requires tangents as a tuple when differentiating multiple arguments "
            f"(expected {expected} tangents)."
        )
        raise TypeError(msg)
    if len(tangents) != expected:
        msg = f"jvp tangent arity mismatch: expected {expected}, got {len(tangents)}"
        raise ValueError(msg)
    return tangents


def _coerce_tangent_like(
    tangent: Any,
    primal: Any,
    *,
    restore_python_scalar: bool,
) -> Any:
    if tangent is None:
        return None

    if restore_python_scalar:
        return _coerce_scalar_tangent_like(tangent, primal)

    if hasattr(primal, "shape"):
        if hasattr(tangent, "shape"):
            tangent_shape = tuple(int(d) for d in tangent.shape)
            primal_shape = tuple(int(d) for d in primal.shape)
            if tangent_shape != primal_shape:
                msg = f"JVP tangent shape mismatch: expected {primal_shape}, got {tangent_shape}."
                raise ValueError(msg)
            return tangent

        xp = _get_array_namespace(primal)
        if xp is not None and hasattr(xp, "asarray"):
            coerced = xp.asarray(tangent)
            coerced_shape = tuple(int(d) for d in coerced.shape)
            primal_shape = tuple(int(d) for d in primal.shape)
            if coerced_shape != primal_shape:
                msg = (
                    "JVP tangent shape mismatch after coercion: "
                    f"expected {primal_shape}, got {coerced_shape}."
                )
                raise ValueError(msg)
            return coerced

    return tangent


def _build_input_tangent_seeds(
    *,
    positional_specs: list[Any],
    tangents: Any,
    single_argnum: bool,
) -> dict[Any, Any]:
    tangent_args = _normalize_jvp_tangents(
        tangents,
        single_argnum=single_argnum,
        expected=len(positional_specs),
    )

    tangent_seeds: dict[Any, Any] = {}
    for spec, tangent_arg in zip(positional_specs, tangent_args, strict=True):
        tangent_leaves, tangent_treedef = tree_flatten(tangent_arg)
        if tangent_treedef != spec.treedef:
            msg = "JVP tangent pytree structure does not match the selected primal input."
            raise ValueError(msg)

        for leaf_spec, tangent_leaf in zip(spec.leaf_specs, tangent_leaves, strict=True):
            node_id = leaf_spec.node_id
            if node_id is None:
                if tangent_leaf is not None:
                    msg = (
                        "JVP tangent provided for a static/untraceable input leaf. "
                        "Use None for non-differentiable leaves."
                    )
                    raise TypeError(msg)
                continue

            tangent_val = _coerce_tangent_like(
                tangent_leaf,
                leaf_spec.primal,
                restore_python_scalar=leaf_spec.restore_python_scalar,
            )
            if tangent_val is not None:
                tangent_seeds[node_id] = tangent_val
    return tangent_seeds
