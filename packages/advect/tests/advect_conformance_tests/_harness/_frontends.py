"""Frontend adapters for conformance cases.

Advect reaches the same primitive through more than one frontend, and they are
genuinely different code paths: the NumPy frontend traces through
``__array_ufunc__``/``__array_function__``, while the Array API frontend
resolves operations through ``__array_namespace__``. A rule can be correct
through one and unreachable -- or wrong -- through the other, so a case says
which frontend it exercises and the runner keeps every numeric comparison on
plain NumPy regardless.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

__all__ = ["Frontend", "is_python_number", "to_numpy", "wrap_for"]

#: ``numpy.float64`` subclasses ``float``, so an ``isinstance`` check would
#: quietly pass NumPy scalars through as if they were Python numbers. Under a
#: non-NumPy provider that leaks a foreign array type into the trace, which
#: surfaces far away as an attribute error inside a transpose rule.
_PYTHON_NUMBER_TYPES = (bool, int, float, complex)


def is_python_number(value: Any) -> bool:
    """Report whether a value is a genuine Python number, never a NumPy scalar."""
    return type(value) in _PYTHON_NUMBER_TYPES


class Frontend(Enum):
    """Which tracing entry point a case exercises."""

    #: ``numpy`` free functions and operators on ``numpy.ndarray`` inputs.
    NUMPY = "numpy"
    #: ``x.__array_namespace__()`` on Array API inputs. Several primitives are
    #: only reachable this way, because the NumPy frontend does not bind them.
    ARRAY_API = "array_api"


def wrap_for(frontend: Frontend, value: Any) -> Any:
    """Convert a NumPy sample into the array type the frontend expects."""
    if frontend is Frontend.NUMPY:
        return value
    if is_python_number(value):
        return value
    import array_api_strict  # noqa: PLC0415 - optional, test-time dependency only

    return array_api_strict.asarray(np.asarray(value))


def to_numpy(value: Any) -> Any:
    """Bring any frontend's array back to NumPy for comparison."""
    if is_python_number(value):
        return value
    return np.asarray(value)
