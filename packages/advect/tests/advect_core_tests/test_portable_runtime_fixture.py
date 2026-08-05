"""Cross-language staged graph fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

import advect as ad

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "advect-runtime"
    / "tests"
    / "fixtures"
    / "python_staged_add_multiply_v2.json"
)


def test_python_staging_emits_and_loads_the_rust_envelope_fixture() -> None:
    fixture_text = _FIXTURE_PATH.read_text(encoding="utf-8").strip()
    fixture = json.loads(fixture_text)
    kernel = np.array([2.0, -1.0], dtype=np.float32)
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, coefficient: (x + coefficient, x * coefficient),
            specs=(
                ad.ArraySpec((2,), "float32"),
                ad.ArraySpec((2,), "float32"),
            ),
        ),
    )
    payload = cast("dict[str, Any]", program.to_dict())

    assert payload == fixture

    restored = ad.StagedProgram.from_dict(fixture)
    assert restored.to_dict() == fixture

    added, multiplied = restored(
        np.array([3.0, 4.0], dtype=np.float32),
        kernel,
    )
    np.testing.assert_array_equal(added, np.array([5.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(multiplied, np.array([6.0, -4.0], dtype=np.float32))
