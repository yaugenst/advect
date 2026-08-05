"""Optional SciPy integration for Advect."""

from __future__ import annotations

try:
    from advect.scipy import ndimage, optimize, sparse, special
except ModuleNotFoundError as error:
    if error.name != "scipy":
        raise
    msg = (
        "advect.scipy requires the optional SciPy dependency; "
        "install it with `pip install 'advect[scipy]'`."
    )
    raise ModuleNotFoundError(msg, name="scipy") from None

__all__ = ["ndimage", "optimize", "sparse", "special"]
