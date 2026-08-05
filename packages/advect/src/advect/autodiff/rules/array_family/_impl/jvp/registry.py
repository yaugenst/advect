"""Canonical array-family JVP rule registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from advect.autodiff.rules.array_family._backend_runtime import wrap_array_family_jvp_rule
from advect.autodiff.rules.array_family._impl.jvp import (
    basic,
    creation,
    elementwise,
    elementwise_partials_a,
    elementwise_partials_b,
    fft,
    gather,
    indexing,
    interp,
    linalg,
    multi_input,
    multi_output,
    reductions,
    shape_advanced,
    shape_ops,
)
from advect.autodiff.rules.array_family._impl.jvp.common import make_diagonal_jvp_from_partials
from advect.core._array_family_ops import _canonical_array_family_op_name

__all__ = ["jvp_rule_items"]

_JVPFn = Callable[..., Any]


def _named_callables(module: object, prefix: str) -> tuple[tuple[str, _JVPFn], ...]:
    return tuple(
        (name.removeprefix(prefix), function)
        for name, function in vars(module).items()
        if name.startswith(prefix) and callable(function)
    )


def _manual_jvp_items() -> tuple[tuple[str, _JVPFn], ...]:
    rules: list[tuple[str, _JVPFn]] = []
    for module in (
        basic,
        creation,
        elementwise,
        reductions,
        shape_ops,
        shape_advanced,
        multi_input,
        gather,
        multi_output,
        interp,
    ):
        for name, function in _named_callables(module, "_jvp_"):
            if module is basic and name == "copy":
                continue
            op_name = (
                f"array.{name}"
                if module is shape_ops and (name == "swapaxes" or name.startswith("atleast_"))
                else _canonical_array_family_op_name(name)
            )
            rules.append((op_name, function))
    rules.extend(
        (f"array_ext.fft.{name}", function) for name, function in _named_callables(fft, "_jvp_")
    )
    for name, function in _named_callables(linalg, "_jvp_"):
        op_name = f"array_ext.linalg.{name.removeprefix('linalg_')}"
        rules.append(("array.vecdot" if name == "vecdot" else op_name, function))
    rules.extend(
        (f"advect.{name}", function)
        for name, function in (
            ("copy", dict(_named_callables(basic, "_jvp_"))["copy"]),
            *_named_callables(indexing, "_jvp_"),
        )
    )

    by_name = dict(rules)
    if len(by_name) != len(rules):
        msg = "Duplicate manually authored array-family JVP rule"
        raise RuntimeError(msg)
    by_name["array_ext.amax"] = by_name["array.max"]
    by_name["array_ext.amin"] = by_name["array.min"]
    return tuple(by_name.items())


def _formula_partial_items() -> tuple[tuple[str, _JVPFn], ...]:
    partial_items = (
        *_named_callables(elementwise_partials_a, "_partials_"),
        *_named_callables(elementwise_partials_b, "_partials_"),
    )
    partials = dict(partial_items)
    if len(partials) != len(partial_items):
        msg = "Duplicate elementwise partial definition"
        raise RuntimeError(msg)
    rules = [
        (_canonical_array_family_op_name(name), function)
        for name, function in partials.items()
        if name != "zero"
    ]
    rules.extend(
        (
            ("array_ext.radians", partials["deg2rad"]),
            ("array_ext.degrees", partials["rad2deg"]),
            *(
                (_canonical_array_family_op_name(name), partials["zero"])
                for name in ("floor", "ceil", "trunc", "rint", "spacing")
            ),
        )
    )
    return tuple(rules)


def jvp_rule_items() -> tuple[tuple[str, _JVPFn], ...]:
    """Build canonical JVP payloads for the built-in operation definitions."""
    rules: dict[str, _JVPFn] = {}
    for op_name, partials_fn in _formula_partial_items():
        if op_name in rules:
            msg = f"Duplicate array-family JVP rule for {op_name!r}"
            raise RuntimeError(msg)
        rules[op_name] = wrap_array_family_jvp_rule(
            make_diagonal_jvp_from_partials(op_name, partials_fn)
        )

    for op_name, jvp_fn in _manual_jvp_items():
        if op_name in rules:
            msg = f"Duplicate array-family JVP rule for {op_name!r}"
            raise RuntimeError(msg)
        rules[op_name] = wrap_array_family_jvp_rule(jvp_fn)
    return tuple(rules.items())
