"""Tests for concrete NumPy ufunc graph recording."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from advect.core._context import _set_active_recorder
from advect.core._native import DynamicTape
from advect.numpy._protocol_ufunc import UFUNC_RUNTIME

if TYPE_CHECKING:
    from collections.abc import Iterator

    from advect.numpy._protocol_ufunc import UfuncLike


@dataclass(slots=True)
class _DummyTraced:
    value: Any
    node_id: int
    recorder: DynamicTape

    def _advect_snapshot(self) -> tuple[int, Any]:
        return self.node_id, self.value


@dataclass(slots=True)
class _ForeignArray:
    """Structural ArrayLike that deliberately is not a NumPy ndarray."""

    values: np.ndarray[Any, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def size(self) -> int:
        return self.values.size

    def __len__(self) -> int:
        return len(self.values)

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray[Any, Any]:
        return np.asarray(self.values, dtype=dtype)

    def copy(self) -> _ForeignArray:
        return _ForeignArray(self.values.copy())


def _record_input(recorder: DynamicTape, value: np.ndarray[Any, Any]) -> int:
    return recorder.record_input(value, value.shape, value.dtype)


@contextmanager
def _active(recorder: DynamicTape) -> Iterator[None]:
    _set_active_recorder(recorder, trace_kind="autodiff_dynamic")
    try:
        yield
    finally:
        _set_active_recorder(None)


def test_handle_ufunc_records_single_output_node() -> None:
    recorder = DynamicTape()
    x_value = np.array([1.0, 2.0])
    y_value = np.array([3.0, 4.0])
    x = _DummyTraced(x_value, _record_input(recorder, x_value), recorder)
    y = _DummyTraced(y_value, _record_input(recorder, y_value), recorder)

    with _active(recorder):
        result, node_id = UFUNC_RUNTIME.handle_ufunc(
            ufunc=cast("UfuncLike", np.add),
            recorder=recorder,
            traced_type=_DummyTraced,
            inputs=(x, y),
            kwargs={},
        )

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([4.0, 6.0]))
    assert isinstance(node_id, int)
    assert recorder.op_names == ["advect.input", "array.add"]
    assert recorder.node_count == 3


def test_handle_ufunc_normalizes_structural_array_to_backend_literal() -> None:
    recorder = DynamicTape()
    x_value = np.array([1.0, 2.0])
    x = _DummyTraced(x_value, _record_input(recorder, x_value), recorder)
    foreign = _ForeignArray(np.array([3.0, 4.0]))

    with _active(recorder):
        result, node_id = UFUNC_RUNTIME.handle_ufunc(
            ufunc=cast("UfuncLike", np.add),
            recorder=recorder,
            traced_type=_DummyTraced,
            inputs=(x, foreign),
            kwargs={},
        )

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([4.0, 6.0]))
    assert isinstance(node_id, int)
    assert recorder.stats()["literal_count"] == 1


def test_handle_ufunc_multi_output_creates_getoutput_nodes() -> None:
    recorder = DynamicTape()
    x_value = np.array([1.5, -2.2])
    x = _DummyTraced(x_value, _record_input(recorder, x_value), recorder)

    with _active(recorder):
        result, node_ids = UFUNC_RUNTIME.handle_ufunc(
            ufunc=cast("UfuncLike", np.modf),
            recorder=recorder,
            traced_type=_DummyTraced,
            inputs=(x,),
            kwargs={},
        )

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(node_ids, tuple)
    assert len(node_ids) == 2
    assert recorder.op_names == ["advect.input", "array_ext.modf", "advect.getoutput"]
    assert recorder.node_count == 4
