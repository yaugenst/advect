"""Container-category preservation at concrete SciPy callback boundaries."""

from __future__ import annotations

import numpy as np


def _restore_container(value: object, template: object) -> object:
    """Restore Python scalar, NumPy scalar, or ndarray shape from ``template``."""
    restored = np.asarray(value).reshape(np.asarray(template).shape)
    if type(template) in (bool, int, float, complex):
        return restored.item()
    if isinstance(template, np.generic):
        return restored[()]
    return restored


__all__ = ["_restore_container"]
