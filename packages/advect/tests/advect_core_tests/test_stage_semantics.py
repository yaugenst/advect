"""Focused contracts for conservative abstract staging semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import array_api_strict as strict
import numpy as np
import pytest
from hypothesis import given, strategies as st

import advect as ad
from advect.core._abstract_helpers import broadcast_shape

_SHAPES = st.lists(
    st.integers(min_value=0, max_value=4),
    min_size=0,
    max_size=4,
).map(tuple)


@given(st.lists(_SHAPES, min_size=1, max_size=4))
def test_abstract_broadcast_shape_matches_numpy(shapes: list[tuple[int, ...]]) -> None:
    try:
        expected = np.broadcast_shapes(*shapes)
    except ValueError:
        with pytest.raises(ValueError, match="Shapes are not broadcast-compatible"):
            broadcast_shape(*shapes)
    else:
        assert broadcast_shape(*shapes) == expected


def _graph_nodes(program: ad.StagedProgram) -> list[Any]:
    graph = program.graph
    return [graph.get_node(node_id) for node_id in graph.node_ids()]


class _FutureNamespace:
    __name__ = "future_array"
    __array_api_version__ = "2099.12"

    @staticmethod
    def __array_namespace_info__() -> object:
        return object()

    @staticmethod
    def asarray(value: object) -> np.ndarray[Any, Any]:
        return np.asarray(value)


class _FutureArray:
    shape = (1,)
    dtype = np.dtype("float64")

    def __array_namespace__(self) -> _FutureNamespace:
        return _FutureNamespace()


class _UnversionedNamespace:
    __name__ = "unversioned_array"


class _UnversionedArray:
    shape = (1,)
    dtype = np.dtype("float64")

    def __array_namespace__(self) -> _UnversionedNamespace:
        return _UnversionedNamespace()


class _PinnedNamespace:
    __name__ = "multi_version_array"
    __array_api_version__ = "2024.12"

    @staticmethod
    def __array_namespace_info__() -> object:
        return object()

    @staticmethod
    def asarray(value: object) -> np.ndarray[Any, Any]:
        return np.asarray(value)


class _DefaultFutureNamespace(_PinnedNamespace):
    __array_api_version__ = "2099.12"


class _MultiVersionArray:
    shape = (1,)
    dtype = np.dtype("float64")

    def __init__(self) -> None:
        self.requests: list[str | None] = []

    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        self.requests.append(api_version)
        return _PinnedNamespace() if api_version == "2024.12" else _DefaultFutureNamespace()


def _contains_python_index(value: object) -> bool:
    if isinstance(value, (slice, type(Ellipsis))):
        return True
    if isinstance(value, Mapping):
        return any(_contains_python_index(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_python_index(item) for item in value)
    return False


def test_expand_dims_and_squeeze_have_explicit_shapes_and_canonical_ids() -> None:
    def transform(x: Any) -> Any:
        xp = x.__array_namespace__()
        return xp.squeeze(xp.expand_dims(x, axis=(0, 2)), axis=(0, 2))

    program = cast(
        "ad.StagedProgram",
        ad.stage(transform, specs=(ad.ArraySpec((2, 3), "float32"),)),
    )
    operation_nodes = [node for node in _graph_nodes(program) if node.op.startswith("array.")]

    assert [(node.op, tuple(node.shape)) for node in operation_nodes] == [
        ("array.expand_dims", (1, 2, 1, 3)),
        ("array.squeeze", (2, 3)),
    ]
    value = strict.reshape(strict.arange(6, dtype=strict.float32), (2, 3))
    result = ad.StagedProgram.from_dict(program.to_dict())(value)
    assert result.shape == (2, 3)


def test_basic_index_is_structural_serializable_and_round_trips() -> None:
    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x[1:, None, ..., ::-2],
            specs=(ad.ArraySpec((3, 4, 5), "float32"),),
        ),
    )
    index_node = next(node for node in _graph_nodes(program) if node.op == "advect.getitem")

    assert tuple(index_node.shape) == (2, 1, 4, 3)
    assert not _contains_python_index(index_node.attrs["index"])
    value = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_array_equal(restored(value), value[1:, None, ..., ::-2])


def test_unknown_operation_has_no_broadcast_guess() -> None:
    with pytest.raises(NotImplementedError, match="no abstract staging rule"):
        ad.stage(
            lambda x: np.partition(x, 1),
            specs=(ad.ArraySpec((3,), "float32"),),
        )


def test_runtime_result_must_match_declared_abstract_spec() -> None:
    @ad.primitive(name="tests.stage_wrong_result")
    def wrong(x: Any) -> Any:
        return x[:1].astype(np.float64)

    @wrong.def_abstract
    def wrong_abstract(x: ad.AbstractValue) -> ad.ArraySpec:
        return x.spec

    program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: wrong(x),  # noqa: PLW0108 - explicit trace boundary
            specs=(ad.ArraySpec((2,), "float32"),),
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"declared shape=\(2,\), dtype=float32; produced shape=\(1,\), dtype=float64",
    ):
        program(np.ones(2, dtype=np.float32))


def test_staged_calls_validate_device_and_weak_scalar_contracts() -> None:
    device_program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x,
            specs=(ad.ArraySpec((1,), "float32", device="cuda:0"),),
        ),
    )
    with pytest.raises(ValueError, match=r"device=cuda:0.*device=cpu"):
        device_program(np.ones(1, dtype=np.float32))

    weak_program = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x: x,
            specs=(ad.ArraySpec((), "float64", weak=True),),
        ),
    )
    with pytest.raises(ValueError, match=r"weak=True.*weak=False"):
        weak_program(np.asarray(1.0, dtype=np.float64))
    assert weak_program(2.0) == 2.0


def test_abstract_staging_rejects_input_mutation_and_input_out() -> None:
    def inplace(x: Any) -> Any:
        x += 1
        return x

    def setitem(x: Any) -> Any:
        x[0] = 1
        return x

    def out_call(x: Any) -> Any:
        return np.add(x, 1, out=x)

    for function, match in (
        (inplace, "staged input"),
        (setitem, "staged input"),
        (out_call, "staged input"),
    ):
        with pytest.raises(ad.MutationError, match=match):
            ad.stage(function, specs=(ad.ArraySpec((2,), "float32"),))


def test_staged_stencil_functionalizes_basic_updates_and_round_trips() -> None:
    def step(u: Any, dt: float) -> Any:
        u = u.copy()
        lap = u[2:] - 2 * u[1:-1] + u[:-2]
        u[1:-1] += dt * lap
        u[0] = -1
        return u

    program = cast(
        "ad.StagedProgram",
        ad.stage(
            step,
            specs=(ad.ArraySpec((6,), "float32"), ad.StaticSpec(0.1)),
        ),
    )
    update_nodes = [node for node in _graph_nodes(program) if node.op == "advect.index_update"]
    assert len(update_nodes) == 2

    value = np.arange(6, dtype=np.float32) ** 2
    expected = value.copy()
    lap = value[2:] - 2 * value[1:-1] + value[:-2]
    expected[1:-1] += np.float32(0.1) * lap
    expected[0] = -1
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_allclose(restored(value, 0.1), expected)
    np.testing.assert_allclose(
        ad.grad(lambda u: np.sum(restored(u, 0.1)))(value),
        np.array([0.1, 0.9, 1.0, 1.0, 0.9, 1.1], dtype=np.float32),
    )


def test_staged_ufunc_out_preserves_identity_where_and_round_trips() -> None:
    identities: list[bool] = []

    def update(x: Any) -> Any:
        destination = x.copy()
        result = np.add(x, 1, out=destination, where=x > 0)
        identities.append(result is destination)
        return destination

    program = cast(
        "ad.StagedProgram",
        ad.stage(update, specs=(ad.ArraySpec((3,), "float32"),)),
    )
    assert identities == [True]
    assert any(node.op == "array.where" for node in _graph_nodes(program))

    value = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    restored = ad.StagedProgram.from_dict(program.to_dict())
    np.testing.assert_array_equal(restored(value), np.array([-1.0, 0.0, 3.0]))


def test_staged_views_detect_stale_and_support_named_basic_view_mutation() -> None:
    def stale(x: Any) -> Any:
        x = x.copy()
        view = x[::2]
        x += 1
        return view + 1

    with pytest.raises(ad.StaleViewError, match="used after its base changed"):
        ad.stage(stale, specs=(ad.ArraySpec((4,), "float32"),))

    def named(x: Any) -> Any:
        x = x.copy()
        view = x[::2]
        view += 1
        view *= 2
        return x

    program = cast(
        "ad.StagedProgram",
        ad.stage(named, specs=(ad.ArraySpec((4,), "float32"),)),
    )
    np.testing.assert_array_equal(
        program(np.arange(4, dtype=np.float32)),
        np.array([2, 1, 6, 3], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "key",
    [
        0,
        (slice(None), 1),
        Ellipsis,
        (None, Ellipsis),
        slice(None, None, 2),
    ],
    ids=("integer", "tuple", "ellipsis", "newaxis", "step-slice"),
)
def test_staged_named_view_update_supports_each_basic_index_form(key: object) -> None:
    def update(x: Any) -> Any:
        current = x.copy()
        view = current[key]
        view += 2
        return current

    program = cast(
        "ad.StagedProgram",
        ad.stage(update, specs=(ad.ArraySpec((3, 4), "float32"),)),
    )
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = source.copy()
    expected[key] += 2

    np.testing.assert_array_equal(program(source), expected)


def test_staged_runtime_enforces_pinned_array_api_version() -> None:
    program = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: x, specs=(ad.ArraySpec((1,), "float64"),)),
    )

    with pytest.raises(TypeError, match=r"required Array API 2024\.12"):
        program(_FutureArray())
    with pytest.raises(TypeError, match=r"required Array API 2024\.12"):
        program(_UnversionedArray())


def test_staged_runtime_validates_every_provider_and_negotiates_the_pin() -> None:
    identity = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: x, specs=(ad.ArraySpec((1,), "float64"),)),
    )
    negotiated = _MultiVersionArray()
    assert identity(negotiated) is negotiated
    assert negotiated.requests == ["2024.12"]

    first_of_two = cast(
        "ad.StagedProgram",
        ad.stage(
            lambda x, _y: x,
            specs=(
                ad.ArraySpec((1,), "float64"),
                ad.ArraySpec((1,), "float64"),
            ),
        ),
    )
    with pytest.raises(TypeError, match=r"required Array API 2024\.12"):
        first_of_two(np.ones(1), _FutureArray())


def test_staged_numpy_uses_its_separate_frontend_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "__array_api_version__", "2099.12")
    program = cast(
        "ad.StagedProgram",
        ad.stage(lambda x: x + 1, specs=(ad.ArraySpec((2,), "float64"),)),
    )

    np.testing.assert_array_equal(program(np.arange(2.0)), np.array([1.0, 2.0]))
