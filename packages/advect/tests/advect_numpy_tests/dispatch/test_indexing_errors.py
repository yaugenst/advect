"""Error handling tests for array indexing operations."""

from __future__ import annotations

import numpy as np
import pytest

import advect as ad


class TestIndexingErrors:
    """Tests for indexing error handling."""

    def test_getitem_advanced_indexing_float_array_raises(self) -> None:
        """Advanced indexing rejects non-integer/non-bool index arrays."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        idx = np.array([0.0, 1.0])

        with pytest.raises(ad.TracingError, match="only supported for integer/bool arrays"):
            ad.jvp(lambda x: x[idx])(arr, tangents=np.ones_like(arr))

    def test_getitem_advanced_indexing_object_array_raises(self) -> None:
        """Advanced indexing rejects object-dtype index arrays."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        idx = np.array([0, "a"], dtype=object)

        with pytest.raises(ad.TracingError, match="object arrays"):
            ad.jvp(lambda x: x[idx])(arr, tangents=np.ones_like(arr))

    def test_getitem_multiple_array_components_with_invalid_dtype_raises(self) -> None:
        """Multiple advanced index arrays still reject non-integer/non-bool dtypes."""
        arr = np.arange(9.0).reshape(3, 3)
        idx0 = np.array([0, 2], dtype=np.int64)
        idx1 = np.array([0.0, 1.0], dtype=np.float64)

        with pytest.raises(ad.TracingError, match="only supported for integer/bool arrays"):
            ad.jvp(lambda x: x[idx0, idx1])(arr, tangents=np.ones_like(arr))

    def test_getitem_traced_array_index_freezes_selection(self) -> None:
        """Traced discrete indices select source tangents and have zero tangent."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        idx_arr = np.array([1, 0])

        value, tangent = ad.jvp(lambda x, idx: x[idx], argnums=(0, 1))(
            arr,
            idx_arr,
            tangents=(np.arange(4.0).reshape(2, 2), np.zeros_like(idx_arr)),
        )

        np.testing.assert_array_equal(value, arr[idx_arr])
        np.testing.assert_array_equal(tangent, np.arange(4.0).reshape(2, 2)[idx_arr])

    def test_getitem_outside_trace_raises(self) -> None:
        """Test that indexing outside trace context raises an error."""
        arr = np.array([1.0, 2.0, 3.0])

        escaped = []

        def capture(x):
            escaped.append(x)
            return x

        ad.jvp(capture)(arr, tangents=np.ones_like(arr))

        with pytest.raises(ad.TracingError, match="escaped its Advect transform"):
            _ = escaped[0][0]
