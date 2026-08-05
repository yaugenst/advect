"""Raw rule invocations for operations without differentiable frontends."""

from __future__ import annotations

import numpy as np

from advect_conformance_tests._harness._rules import RawRuleCase

RAW_RULE_CASES = (
    RawRuleCase(
        op="array.full",
        operands=(np.array(1.25),),
        tangents=(np.array(0.75),),
        attrs={"shape": (2, 3), "dtype": np.dtype("float64")},
    ),
    RawRuleCase(
        op="array_ext.true_divide",
        operands=(np.array([1.0, -2.0]), np.array([2.0, 3.0])),
        tangents=(np.array([0.25, -0.5]), np.array([0.5, 0.25])),
        attrs={},
    ),
    RawRuleCase(
        op="array.zeros_like",
        operands=(np.array([1.0, 2.0]),),
        tangents=(np.array([0.3, -0.2]),),
        attrs={"dtype": None, "shape": None},
    ),
    RawRuleCase(
        op="array.ones_like",
        operands=(np.array([1.0, 2.0]),),
        tangents=(np.array([0.3, -0.2]),),
        attrs={"dtype": None, "shape": None},
    ),
    RawRuleCase(
        op="array.empty_like",
        operands=(np.array([1.0, 2.0]),),
        tangents=(np.array([0.3, -0.2]),),
        attrs={"dtype": None, "shape": None},
        numerical=False,
    ),
    RawRuleCase(
        op="array.full_like",
        operands=(np.array([1.0, 2.0]), np.array(2.5)),
        tangents=(np.array([0.3, -0.2]), None),
        attrs={"dtype": None, "shape": None},
    ),
    RawRuleCase(
        op="advect.getoutput",
        operands=((np.array([1.0, 2.0]), np.array([-1.0, 3.0])),),
        tangents=((np.array([0.3, -0.2]), np.array([0.5, 0.1])),),
        attrs={"index": 1, "num_outputs": 2},
    ),
)

RAW_RULE_OPS = frozenset(case.op for case in RAW_RULE_CASES)
