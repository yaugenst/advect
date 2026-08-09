# NumPy Frontend

`advect.numpy` is a transparent companion namespace for ordinary NumPy code.
It overrides only constructors that must preserve live Advect values and
forwards other attributes to the installed NumPy module. NumPy's own ufunc and
array-function protocols perform tracing; users do not register operations in
this namespace.

The installed NumPy minor determines exact signatures. Dynamic, staged,
serialized, and derivative support remain explicit per callable, so consult the
generated [NumPy compatibility page](../compatibility/numpy.md) rather than
inferring support from attribute availability.

::: advect.numpy
