# Handler signatures mirror NumPy APIs
"""Additional ``__array_function__`` handlers for NumPy coverage expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from advect.numpy._array_functions_extra_algorithms import register_algorithm_handlers
from advect.numpy._array_functions_extra_aliases import register_alias_handlers
from advect.numpy._array_functions_extra_composite import register_composite_handlers
from advect.numpy._array_functions_extra_creation import (
    _full_handler,
    _full_like_handler,
    _linspace_handler,
    _pad_handler,
    register_creation_handlers,
)
from advect.numpy._array_functions_extra_linalg import (
    _cross_handler,
    _einsum_handler,
    _inner_handler,
    _kron_handler,
    _lstsq_handler,
    _matrix_rank_handler,
    _outer_handler,
    _tensordot_handler,
)
from advect.numpy._array_functions_extra_misc import (
    _angle_handler,
    _copy_handler,
    _gradient_handler,
    _nan_to_num_handler,
    _partition_handler,
    _sinc_handler,
    _sort_handler,
    _take_along_axis_handler,
    _take_handler,
)
from advect.numpy._array_functions_extra_ordering import register_ordering_handlers
from advect.numpy._array_functions_extra_polynomial import register_polynomial_handlers
from advect.numpy._array_functions_extra_predicates import register_predicate_handlers
from advect.numpy._array_functions_extra_scimath import register_scimath_handlers
from advect.numpy._array_functions_extra_signal import register_signal_handlers
from advect.numpy._array_functions_extra_split import (
    _array_split_handler,
    _dsplit_handler,
    _hsplit_handler,
    _split_handler,
    _vsplit_handler,
)
from advect.numpy._array_functions_extra_statistics import register_statistics_handlers
from advect.numpy._array_functions_extra_unique import register_unique_handlers

if TYPE_CHECKING:
    from collections.abc import Callable


def register_extra_handlers(handlers: dict[Callable[..., Any], Callable[..., Any]]) -> None:
    """Register expanded array-function handlers."""
    register_alias_handlers(handlers)
    register_algorithm_handlers(handlers)
    register_composite_handlers(handlers)
    register_creation_handlers(handlers)
    register_ordering_handlers(handlers)
    register_polynomial_handlers(handlers)
    register_predicate_handlers(handlers)
    register_scimath_handlers(handlers)
    register_signal_handlers(handlers)
    register_statistics_handlers(handlers)
    register_unique_handlers(handlers)
    handlers[np.angle] = _angle_handler
    handlers[np.nan_to_num] = _nan_to_num_handler
    handlers[np.sinc] = _sinc_handler
    handlers[np.copy] = _copy_handler
    handlers[np.full] = _full_handler
    handlers[np.full_like] = _full_like_handler
    handlers[np.inner] = _inner_handler
    handlers[np.outer] = _outer_handler
    handlers[np.cross] = _cross_handler
    handlers[np.kron] = _kron_handler
    handlers[np.tensordot] = _tensordot_handler
    handlers[np.einsum] = _einsum_handler
    handlers[np.linalg.lstsq] = _lstsq_handler
    handlers[np.linalg.matrix_rank] = _matrix_rank_handler
    handlers[np.linspace] = _linspace_handler
    handlers[np.sort] = _sort_handler
    handlers[np.partition] = _partition_handler
    handlers[np.take] = _take_handler
    handlers[np.take_along_axis] = _take_along_axis_handler
    handlers[np.pad] = _pad_handler
    handlers[np.split] = _split_handler
    handlers[np.array_split] = _array_split_handler
    handlers[np.hsplit] = _hsplit_handler
    handlers[np.vsplit] = _vsplit_handler
    handlers[np.dsplit] = _dsplit_handler
    handlers[np.gradient] = _gradient_handler
