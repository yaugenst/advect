"""Private mechanism modules behind :mod:`advect.scipy.ndimage`.

The public signatures remain in :mod:`advect.scipy.ndimage`.  This boundary
module intentionally exports nothing and performs no primitive registration;
the public facade imports each implementation directly from its owning leaf.
"""
