"""First-order host-framework bridges for NumPy-backed Advect callables.

Import ``wrap`` from exactly one optional boundary: ``advect.interop.jax``,
``advect.interop.torch``, or ``advect.interop.autograd`` for HIPS Autograd.
The returned callable accepts positional or keyword floating or complex array
pytrees; close over static configuration. Importing :mod:`advect` or this
package never imports an optional framework dependency.
"""

from __future__ import annotations

__all__: list[str] = []
