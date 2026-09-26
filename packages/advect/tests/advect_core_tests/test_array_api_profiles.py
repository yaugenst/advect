"""Frozen Array API revision profiles and call-level negotiation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import example, given, settings, strategies as st

import advect as ad
from advect.autodiff._ephemeral import trace_call
from advect.core._array_api.profiles import (
    LATEST_ARRAY_API_VERSION,
    SUPPORTED_ARRAY_API_VERSIONS,
    materialize_array_api_profile,
    minimum_array_api_version,
)
from advect.core._array_api.providers import (
    ResolvedArrayNamespace,
    _negotiate_array_namespace_for_call,
)
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


# Reported revisions one step below and above the supported range.
_REPORTED_VERSIONS = ("2021.12", *SUPPORTED_ARRAY_API_VERSIONS, "2025.12")


@dataclass(frozen=True, slots=True)
class _Leaf:
    """One provider leaf and the protocol report it gives per request."""

    served: frozenset[str]
    reported: int | str | None  # offset from the request, a malformed string, or none
    info_from: int  # first supported-revision index exposing namespace info
    backend: str | None
    asarray: bool

    def serves(self, version: str) -> bool:
        return (
            version in self.served
            and isinstance(self.reported, int)
            and self.reported >= 0
            and (
                version == "2022.12"
                or SUPPORTED_ARRAY_API_VERSIONS.index(version) >= self.info_from
            )
            and self.backend is not None
            and self.asarray
        )


class _NegotiatedArray:
    __advect_namespace_is_instance_specific__ = True
    shape = (1,)
    dtype = np.dtype("float64")

    def __init__(self, leaf: _Leaf) -> None:
        self.leaf = leaf
        self.requests: list[str | None] = []

    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        self.requests.append(api_version)
        leaf = self.leaf
        if api_version not in leaf.served:
            message = f"unsupported revision {api_version}"
            raise ValueError(message)
        index = SUPPORTED_ARRAY_API_VERSIONS.index(api_version)
        attributes: dict[str, object] = {}
        if leaf.backend is not None:
            attributes["__name__"] = leaf.backend
        if isinstance(leaf.reported, int):
            attributes["__array_api_version__"] = _REPORTED_VERSIONS[index + 1 + leaf.reported]
        elif leaf.reported is not None:
            attributes["__array_api_version__"] = leaf.reported
        if index >= leaf.info_from:
            attributes["__array_namespace_info__"] = object
        if leaf.asarray:
            attributes["asarray"] = lambda value: value
        return SimpleNamespace(**attributes)


_LEAVES = st.builds(
    _Leaf,
    served=st.frozensets(st.sampled_from(SUPPORTED_ARRAY_API_VERSIONS)),
    reported=st.sampled_from((None, "future", -1, 0, 1)),
    info_from=st.integers(0, len(SUPPORTED_ARRAY_API_VERSIONS)),
    backend=st.sampled_from(("a", "b", None)),
    asarray=st.booleans(),
)
_COMPLETE = frozenset(SUPPORTED_ARRAY_API_VERSIONS)


@settings(derandomize=True, deadline=None)
@given(
    leaves=st.lists(_LEAVES, min_size=1, max_size=4),
    required=st.none() | st.sampled_from(SUPPORTED_ARRAY_API_VERSIONS),
)
# Regression: a leaf reporting no revision used to negotiate successfully.
@example(leaves=[_Leaf(_COMPLETE, None, 0, "a", asarray=True)], required=None)
@example(leaves=[_Leaf(_COMPLETE, 1, 0, "a", asarray=True)], required=None)
@example(leaves=[_Leaf(frozenset({"2022.12"}), 0, 3, "a", asarray=True)], required=None)
def test_negotiation_selects_the_newest_revision_every_leaf_serves(
    leaves: list[_Leaf],
    required: str | None,
) -> None:
    arrays = [_NegotiatedArray(leaf) for leaf in leaves]
    attempted = tuple(reversed(SUPPORTED_ARRAY_API_VERSIONS)) if required is None else (required,)
    common = [version for version in attempted if all(leaf.serves(version) for leaf in leaves)]

    def negotiate() -> object:
        return _negotiate_array_namespace_for_call(
            args=({"leaves": arrays},),
            kwargs={},
            required_version=required,
        )

    if not common:
        with pytest.raises(TypeError, match=f"attempted {re.escape(', '.join(attempted))}$"):
            negotiate()
        assert arrays[0].requests == list(attempted)
        return
    selected = common[0]
    if len({leaf.backend for leaf in leaves}) > 1:
        with pytest.raises(TypeError, match="different array providers"):
            negotiate()
    else:
        resolution = negotiate()
        assert isinstance(resolution, ResolvedArrayNamespace)
        assert resolution.requested_version == selected
    # Requests descend and stop at the first revision every leaf serves.
    assert arrays[0].requests == list(attempted[: attempted.index(selected) + 1])


def test_dynamic_numpy_negotiation_uses_provider_declared_revision() -> None:
    expected_array_api_version = min(np.__array_api_version__, LATEST_ARRAY_API_VERSION)
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
