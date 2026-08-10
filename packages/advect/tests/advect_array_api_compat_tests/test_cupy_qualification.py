"""Optional single-device CuPy qualification for Advect's Array API path."""

from __future__ import annotations

import pytest
from scripts import qualify_array_providers

from advect.core._array_api.profiles import SUPPORTED_ARRAY_API_VERSIONS

cp = pytest.importorskip("cupy")


def _reduction_dtype(array_api_version: str) -> object:
    return cp.float64 if array_api_version == "2022.12" else cp.float32


@pytest.mark.parametrize("array_api_version", SUPPORTED_ARRAY_API_VERSIONS)
def test_cupy_scientific_provider_qualification(array_api_version: str) -> None:
    result = qualify_array_providers._qualify_provider(
        qualify_array_providers._cupy_provider(),
        qualify_array_providers._build_programs(array_api_version),
        array_api_version=array_api_version,
    )

    assert result.report["name"] == "cupy"
    assert result.report["selected_array_api_version"] == array_api_version
    assert result.report["captured_constant"]["result_dtype"] == str(
        _reduction_dtype(array_api_version)
    )
