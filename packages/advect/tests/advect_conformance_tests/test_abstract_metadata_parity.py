"""Abstract staging declares the concrete result metadata for every dtype.

The staged law compares values on each invocation's floating-point domain. This
law reuses the same invocations but draws every argument's dtype from all
staged dtypes, and requires the staged graph's output shapes and dtypes to
equal the concrete provider result exactly whenever the provider has one.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

import advect as ad
from advect.core._pytree import tree_flatten
from advect_conformance_tests._builtin_cases import INVOCATIONS_BY_ID, STAGED_ONLY_INVOCATIONS
from advect_conformance_tests._harness import Law, argument_tuples
from advect_conformance_tests._harness._frontends import is_python_number, to_numpy, wrap_for

if TYPE_CHECKING:
    from hypothesis.strategies import DataObject

_DTYPES = st.sampled_from(
    (
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
    )
)
# NumPy chooses these result dtypes from the eigenvalues of a real input, so
# staging accepts only complex inputs and must refuse the others explicitly.
_DATA_DEPENDENT_DTYPE_OPS = frozenset({"array_ext.linalg.eig", "array_ext.linalg.eigvals"})
_CASE_ITEMS = [
    *(
        (identifier, case)
        for identifier, case in INVOCATIONS_BY_ID.items()
        if Law.STAGED in case.laws
    ),
    *((f"{case.op}[{case.frontend.value}]", case) for case in STAGED_ONLY_INVOCATIONS),
]
_CASES = dict(_CASE_ITEMS)
# A repeated identifier would silently drop a case from the law.
assert len(_CASES) == len(_CASE_ITEMS), "abstract metadata case identifiers collide"
# Metadata is exact and staging only traces, so each example is cheap.
_EXAMPLES = max(10, settings.default.max_examples // 10)


def _concrete_metadata(result: object) -> list[tuple[tuple[int, ...], str]]:
    leaves, _treedef = tree_flatten(result)
    arrays = [np.asarray(to_numpy(leaf)) for leaf in leaves]
    return [(array.shape, array.dtype.name) for array in arrays]


@pytest.mark.parametrize("identifier", sorted(_CASES))
@given(data=st.data())
@settings(max_examples=_EXAMPLES, deadline=None)
def test_staged_metadata_matches_the_concrete_result(identifier: str, data: DataObject) -> None:
    case = _CASES[identifier]
    variant = data.draw(st.integers(0, case.variant_count - 1), label="variant")
    drawn = data.draw(argument_tuples(case, variant), label="arguments")

    def call(*arguments: Any) -> Any:
        return case.call(*arguments, **dict(case.static))

    # Lossy casts warn, and so do providers that SciPy abstract rules rerun.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        values = tuple(
            value
            if is_python_number(value)
            else np.asarray(value).astype(data.draw(_DTYPES, label="dtype"))
            for value in drawn
        )
        try:
            concrete = call(*(wrap_for(case.frontend, value) for value in values))
        except (ArithmeticError, RuntimeError, TypeError, ValueError):
            return  # The law constrains only calls with a concrete result.
        specs = tuple(
            ad.ArraySpec(np.shape(value), np.asarray(value).dtype, weak=is_python_number(value))
            for value in values
        )
        if case.op in _DATA_DEPENDENT_DTYPE_OPS and values[0].dtype.kind != "c":
            with pytest.raises(TypeError, match="requires a complex input"):
                ad.stage(call, specs=specs)
            return
        graph = ad.stage(call, specs=specs).graph
    staged = [
        (tuple(graph.get_node(node_id).shape), graph.get_node(node_id).dtype)
        for node_id in graph.outputs
    ]
    assert staged == _concrete_metadata(concrete)
