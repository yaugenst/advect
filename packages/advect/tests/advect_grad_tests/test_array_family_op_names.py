"""Tests for canonical array-family op name helpers."""

from __future__ import annotations

from advect._builtin_ops import _output_arities
from advect.autodiff.rules.array_family.jvp.registry import jvp_rule_items
from advect.autodiff.rules.array_family.vjp.registry import (
    non_differentiable_items,
    vjp_rule_items,
)
from advect.core._abstract_domains import operation_semantics
from advect.core._array_api.frontend import (
    _ARRAY_API_COMPOSITES,
    _ARRAY_API_META_FUNCTIONS,
    _FUNCTION_SPECS,
)
from advect.core._registry import get_registry
from advect.numpy._array_function.registry import ARRAY_FUNCTION_RUNTIME
from advect.numpy._op_bindings import canonicalize_numpy_op
from advect.numpy._protocol_eval import NUMPY_EVAL_RUNTIME
from advect_conformance_tests._builtin_cases import BUILTIN_INVOCATIONS
from advect_conformance_tests._raw_rule_cases import RAW_RULE_OPS


def test_numpy_op_binding_emits_canonical_ids() -> None:
    assert canonicalize_numpy_op("numpy.add") == "array.add"
    assert canonicalize_numpy_op("numpy.linalg.svd") == "array_ext.linalg.svd"
    assert canonicalize_numpy_op("array.multiply") == "array.multiply"
    assert canonicalize_numpy_op("advect.copy") == "advect.copy"


def test_array_family_identity_comes_from_registry_and_frontend_forms() -> None:
    definitions = get_registry().definitions()
    abstract_operation_names = {name for name, _schema, _evaluator in operation_semantics()}
    abstract_array_suffixes = {
        name.removeprefix("array.")
        for name in abstract_operation_names
        if name.startswith("array.")
    }
    registered_array_family_names = {
        definition.name
        for definition in definitions
        if definition.name.startswith(("array.", "array_ext."))
    }
    registered_array_suffixes = {
        definition.name.removeprefix("array.")
        for definition in definitions
        if definition.name.startswith("array.")
    }
    registered_extension_suffixes = {
        definition.name.removeprefix("array_ext.")
        for definition in definitions
        if definition.name.startswith("array_ext.")
    }
    assert registered_array_suffixes.isdisjoint(registered_extension_suffixes)
    assert len(abstract_array_suffixes) == 121
    assert len(registered_array_suffixes) == 127
    derivative_names = {
        *(name for name, _rule in jvp_rule_items()),
        *(name for name, _reason in non_differentiable_items()),
    }
    derivative_names = {
        name for name in derivative_names if name.startswith(("array.", "array_ext."))
    }
    dynamic_only_names = registered_array_family_names - abstract_operation_names
    assert len(dynamic_only_names) == 46
    assert dynamic_only_names == derivative_names - abstract_operation_names

    for suffix in abstract_array_suffixes:
        assert canonicalize_numpy_op(f"numpy.{suffix}") == f"array.{suffix}"
    for suffix in registered_extension_suffixes:
        assert canonicalize_numpy_op(f"numpy.{suffix}") == f"array_ext.{suffix}"

    rule_only_names = {
        f"array.{suffix}" for suffix in registered_array_suffixes - abstract_array_suffixes
    }
    declared_numpy_lowerings = {
        lowering
        for handler in ARRAY_FUNCTION_RUNTIME.handlers.values()
        if isinstance((lowering := getattr(handler, "__advect_lowering__", None)), str)
    }
    declared_array_api_bindings = {spec.op for spec in _FUNCTION_SPECS.values()} | {
        f"array.{path}" for path in _ARRAY_API_COMPOSITES if "." not in path
    }
    assert len(rule_only_names) == 6
    assert rule_only_names <= declared_numpy_lowerings | declared_array_api_bindings
    for name in rule_only_names:
        definition = get_registry().get(name)
        assert definition.jvp is not None or definition.non_differentiable_reason is not None

    declared_frontend_only = {
        path
        for path in {*_ARRAY_API_COMPOSITES, *_ARRAY_API_META_FUNCTIONS, "from_dlpack"}
        if "." not in path and path not in registered_array_suffixes
    }
    assert declared_frontend_only == {
        "broadcast_arrays",
        "can_cast",
        "finfo",
        "from_dlpack",
        "iinfo",
        "isdtype",
        "meshgrid",
        "nonzero",
        "result_type",
        "unique_all",
        "unique_values",
        "unstack",
    }


def test_derivative_declarations_name_known_canonical_operations() -> None:
    declarations = {
        "JVP": tuple(name for name, _rule in jvp_rule_items()),
        "VJP": tuple(name for name, _rule, _needs_inputs, _needs_output in vjp_rule_items()),
        "non-differentiability": tuple(name for name, _reason in non_differentiable_items()),
    }
    declared_lowerings = {
        lowering
        for handler in ARRAY_FUNCTION_RUNTIME.handlers.values()
        if isinstance((lowering := getattr(handler, "__advect_lowering__", None)), str)
    }
    evaluator_operations = {
        canonicalize_numpy_op(name) for name in NUMPY_EVAL_RUNTIME.special_evaluators
    }
    known = {
        *(name for name, _schema, _evaluator in operation_semantics()),
        *(case.op for case in BUILTIN_INVOCATIONS),
        *RAW_RULE_OPS,
        *declared_lowerings,
        *evaluator_operations,
        *_output_arities(),
        "advect.input",
        "advect.const",
        "advect.getitem",
        "advect.getoutput",
        "advect.index_update",
        "advect.copy",
    }
    assert {label: len(names) for label, names in declarations.items()} == {
        "JVP": 179,
        "VJP": 99,
        "non-differentiability": 33,
    }
    for label, names in declarations.items():
        assert len(names) == len(set(names)), f"duplicate {label} declarations"
        unknown = set(names) - known
        assert not unknown, f"{label} declarations name unknown operations: {sorted(unknown)}"
        assert all(name.startswith(("advect.", "array.", "array_ext.")) for name in names)
