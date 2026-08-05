"""Frozen Array API revision profiles and call-level negotiation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._array_api_profiles import (
    SUPPORTED_ARRAY_API_VERSIONS,
    materialize_array_api_profile,
    minimum_array_api_version,
)
from advect.core._array_namespace import _negotiate_array_namespace_for_call
from advect.core._context import _get_active_array_api_version


def test_profiles_fold_thin_monotonic_deltas() -> None:
    profile_2022 = materialize_array_api_profile("2022.12")
    profile_2023 = materialize_array_api_profile("2023.12")
    profile_2024 = materialize_array_api_profile("2024.12")

    assert set(profile_2022.signatures) < set(profile_2023.signatures)
    assert set(profile_2023.signatures) < set(profile_2024.signatures)
    assert profile_2022.signatures["sin"] == profile_2024.signatures["sin"]
    assert "cumulative_sum" not in profile_2022.signatures
    assert "cumulative_sum" in profile_2023.signatures
    assert "cumulative_prod" not in profile_2023.signatures
    assert "cumulative_prod" in profile_2024.signatures
    assert minimum_array_api_version("cumulative_sum") == "2023.12"
    assert minimum_array_api_version("cumulative_prod") == "2024.12"

    with pytest.raises(TypeError):
        profile_2022.signatures["future"] = "(x, /)"  # type: ignore[index]


def test_unknown_profile_is_rejected_clearly() -> None:
    with pytest.raises(ValueError, match=r"Unsupported Array API revision '2025\.12'"):
        materialize_array_api_profile("2025.12")


class _VersionedArray:
    __advect_namespace_is_instance_specific__ = True
    shape = (1,)
    dtype = np.dtype("float64")

    def __init__(
        self,
        *supported: str,
        backend: str = "versioned_array",
        reported: str | None = None,
        namespace_info: bool = True,
    ) -> None:
        self.supported = frozenset(supported)
        self.backend = backend
        self.reported = reported
        self.namespace_info = namespace_info
        self.requests: list[str | None] = []

    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        self.requests.append(api_version)
        if api_version not in self.supported:
            message = f"unsupported revision {api_version}"
            raise ValueError(message)
        namespace = SimpleNamespace(
            __name__=self.backend,
            __array_api_version__=self.reported or api_version,
            asarray=lambda value: value,
        )
        if self.namespace_info:
            namespace.__array_namespace_info__ = lambda: object()
        return namespace


def test_negotiation_selects_newest_revision_served_by_every_leaf() -> None:
    first = _VersionedArray(*SUPPORTED_ARRAY_API_VERSIONS)
    second = _VersionedArray("2022.12", "2023.12")

    resolution = _negotiate_array_namespace_for_call(
        args=({"first": first, "nested": [second]},),
        kwargs={},
    )

    assert resolution is not None
    assert resolution.requested_version == "2023.12"
    assert first.requests == ["2024.12", "2023.12"]
    assert second.requests == ["2024.12", "2023.12"]


def test_negotiation_accepts_a_newer_provider_revision() -> None:
    value = _VersionedArray(*SUPPORTED_ARRAY_API_VERSIONS, reported="2025.12")

    resolution = _negotiate_array_namespace_for_call(args=(value,), kwargs={})

    assert resolution is not None
    assert resolution.requested_version == "2024.12"
    assert resolution.raw_namespace.__array_api_version__ == "2025.12"


def test_2022_negotiation_does_not_require_later_namespace_info_protocol() -> None:
    value = _VersionedArray("2022.12", namespace_info=False)

    resolution = _negotiate_array_namespace_for_call(args=(value,), kwargs={})

    assert resolution is not None
    assert resolution.requested_version == "2022.12"
    assert not hasattr(resolution.raw_namespace, "__array_namespace_info__")


def test_negotiation_rejects_mixed_providers_before_tracing() -> None:
    left = _VersionedArray(*SUPPORTED_ARRAY_API_VERSIONS, backend="left_array")
    right = _VersionedArray(*SUPPORTED_ARRAY_API_VERSIONS, backend="right_array")

    with pytest.raises(TypeError, match="different array providers"):
        _negotiate_array_namespace_for_call(args=(left, right), kwargs={})


def test_negotiation_reports_every_attempted_revision() -> None:
    value = _VersionedArray()

    with pytest.raises(
        TypeError,
        match=r"attempted 2024\.12, 2023\.12, 2022\.12",
    ):
        _negotiate_array_namespace_for_call(args=(value,), kwargs={})

    assert value.requests == ["2024.12", "2023.12", "2022.12"]


def test_dynamic_numpy_negotiation_uses_provider_declared_revision() -> None:
    expected_array_api_version = np.__array_api_version__
    value = np.asarray([1.0, 2.0])

    traced = trace_call(
        lambda x: np.sum(x * x),
        args=(value,),
        kwargs={},
        argnums=(0,),
        argnames=None,
    )
    try:
        assert traced.array_api_version == expected_array_api_version
    finally:
        traced.tape.release_payloads()


def test_jacobian_replays_the_negotiated_numpy_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "__array_api_version__", "2022.12")
    matrix = np.asarray([[1.0, 2.0], [-0.5, 1.5], [2.0, -1.0]])
    value = np.asarray([0.3, -0.7])

    actual = ad.jacobian(lambda argument: matrix @ argument)(value)

    np.testing.assert_allclose(actual, matrix)


def test_nested_dynamic_transforms_preserve_the_enclosing_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "__array_api_version__", "2022.12")
    observed: list[str | None] = []

    def outer(x: object) -> object:
        def inner(y: object) -> object:
            observed.append(_get_active_array_api_version())
            return np.sum(y * y)

        return np.sum(ad.grad(inner)(x))

    ad.grad(outer)(np.asarray([1.0, 2.0]))

    assert observed == ["2022.12"]


def test_older_staged_profile_hides_later_callable() -> None:
    def cumulative_sum(x: object) -> object:
        namespace = x.__array_namespace__()
        return namespace.cumulative_sum(x, axis=0)

    with pytest.raises(
        AttributeError,
        match=r"cumulative_sum.*not available.*2022\.12",
    ):
        ad.stage(
            cumulative_sum,
            specs=(ad.ArraySpec((2,), "float64"),),
            array_api_version="2022.12",
        )

    program = ad.stage(
        cumulative_sum,
        specs=(ad.ArraySpec((2,), "float64"),),
        array_api_version="2023.12",
    )
    np.testing.assert_array_equal(program(np.asarray([1.0, 2.0])), np.asarray([1.0, 3.0]))


def test_stage_selects_explicit_inferred_and_specification_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = ad.stage(
        lambda x: x + 1,
        specs=(ad.ArraySpec((2,), "float64"),),
        array_api_version="2022.12",
    )
    specification_default = ad.stage(
        lambda x: x + 1,
        specs=(ad.ArraySpec((2,), "float64"),),
    )
    monkeypatch.setattr(np, "__array_api_version__", "2023.12")
    inferred = ad.stage(lambda x: x + 1, np.asarray([1.0, 2.0]))

    assert explicit.array_api_version == "2022.12"
    assert inferred.array_api_version == "2023.12"
    assert specification_default.array_api_version == "2024.12"


def test_explicit_stage_target_must_be_served_by_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "__array_api_version__", "2022.12")

    with pytest.raises(TypeError, match=r"cannot serve required Array API 2024\.12"):
        ad.stage(
            lambda x: x + 1,
            np.asarray([1.0, 2.0]),
            array_api_version="2024.12",
        )


def test_staged_target_survives_derivation_and_serialization() -> None:
    primal = ad.stage(
        lambda x: np.sum(x * x),
        specs=(ad.ArraySpec((2,), "float64"),),
        array_api_version="2022.12",
    )
    derived = (ad.grad(primal), ad.value_and_grad(primal), ad.vjp_program(primal))

    for program in (primal, *derived):
        assert program.array_api_version == "2022.12"
        restored = ad.StagedProgram.from_dict(program.to_dict())
        assert restored.array_api_version == program.array_api_version


def test_older_staged_target_runs_on_newer_numpy() -> None:
    program = ad.stage(
        lambda x: x + 1,
        specs=(ad.ArraySpec((2,), "float64"),),
        array_api_version="2022.12",
    )

    np.testing.assert_array_equal(program(np.asarray([1.0, 2.0])), np.asarray([2.0, 3.0]))


def test_newer_staged_target_rejects_older_numpy_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = ad.stage(
        lambda x: x + 1,
        specs=(ad.ArraySpec((2,), "float64"),),
        array_api_version="2024.12",
    )
    monkeypatch.setattr(np, "__array_api_version__", "2022.12")

    with pytest.raises(TypeError, match=r"cannot serve required Array API 2024\.12"):
        program(np.asarray([1.0, 2.0]))
